"""
FastAPI wrapper around the LangGraph dermatology pipeline, with Supabase
auth + Postgres-backed chat history.
"""

import asyncio
import os
from typing import List, Literal, Optional

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from pipeline.graph import app as graph_app
from pipeline.graph import get_embedder, get_reranker, get_collection

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]  
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]

# Service-role client: trusted backend context. We bypass RLS here but always
# filter explicitly by the verified user_id below, so a bug in one query
# can't leak another user's rows purely by omission.
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

api = FastAPI(title="DermBot API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "https://dermbot-ten.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup warmup.
#
# get_embedder/get_reranker/get_collection (in pipeline/graph.py) are lazy
# singletons: they load once, on first call, and cache the result. Without
# this startup hook, "first call" would be the first real user's chat
# request -- which meant that request raced against Render's own proxy
# timeout while loading a multi-hundred-MB model, and lost (502, connection
# just dies, no clean error).
#
# Triggering the same loads here instead means:
#   - uvicorn still binds the port immediately (this event runs after that),
#     so Render's port-scan still succeeds fast and deploys don't stall.
#   - The heavy loading happens once, right after deploy, off the request
#     path -- so a user's first message doesn't pay for it.
#   - run_in_executor keeps this off the main event loop, so it doesn't
#     block other startup work or make the process look hung.
#   - If this now fails (OOM, timeout, etc.), it fails loudly in the deploy
#     logs right after "Running 'uvicorn ...'" instead of showing up later
#     as a confusing per-request 502.
# ---------------------------------------------------------------------------

@api.on_event("startup")
async def warm_up_models():
    print("[startup] warming up embedder/reranker/chroma in background...", flush=True)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, get_embedder)
        await loop.run_in_executor(None, get_reranker)
        await loop.run_in_executor(None, get_collection)
        print("[startup] warmup complete", flush=True)
    except Exception as e:
        # Don't let a warmup failure crash the whole app -- if this happens,
        # the lazy singletons will just retry on first actual use (same as
        # before this change), so we still want the server up so you can see
        # this error clearly rather than the process dying silently.
        print(f"[startup] warmup FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()


# ---- Auth dependency -------------------------------------------------------

def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        print(f"[auth] token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = user_response.user
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user.id


# ---- Request/response models ------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None  # omitted on the first message of a new chat
    # A Settings preference, not auto-detected from the message — see
    # pipeline/graph.py's prepare_language node for why.
    language: Literal["en", "roman_ur"] = "en"


class ChatResponse(BaseModel):
    answer: str
    chat_id: str
    clarification_questions: Optional[List[dict]] = None


class ChatSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    is_clarification: bool
    created_at: str


# ---- Helpers ----------------------------------------------------------------

def _get_owned_chat(chat_id: str, user_id: str) -> dict:
    """Fetch a chat row, raising 404 if it doesn't exist or isn't owned by user_id."""
    res = supabase.table("chats").select("*").eq("id", chat_id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Chat not found")
    return res.data[0]


def _load_history(chat_id: str) -> List[dict]:
    res = (
        supabase.table("messages")
        .select("role, content, is_clarification")
        .eq("chat_id", chat_id)
        .order("created_at")
        .execute()
    )
    return res.data or []


# ---- Routes -------------------------------------------------------------

@api.get("/api/health")
def health():
    return {"status": "ok"}


@api.get("/api/chats", response_model=List[ChatSummary])
def list_chats(user_id: str = Depends(get_current_user)):
    res = (
        supabase.table("chats")
        .select("id, title, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


@api.post("/api/chats", response_model=ChatSummary)
def create_chat(user_id: str = Depends(get_current_user)):
    res = supabase.table("chats").insert({"user_id": user_id, "title": "New conversation"}).execute()
    return res.data[0]


@api.get("/api/chats/{chat_id}/messages", response_model=List[MessageOut])
def get_chat_messages(chat_id: str, user_id: str = Depends(get_current_user)):
    _get_owned_chat(chat_id, user_id)  # 404s if not this user's chat
    res = (
        supabase.table("messages")
        .select("role, content, is_clarification, created_at")
        .eq("chat_id", chat_id)
        .order("created_at")
        .execute()
    )
    return res.data or []


@api.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    _get_owned_chat(chat_id, user_id)
    supabase.table("chats").delete().eq("id", chat_id).execute()
    return {"status": "deleted"}


@api.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user_id: str = Depends(get_current_user)):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    if payload.chat_id:
        chat_row = _get_owned_chat(payload.chat_id, user_id)  # 404s if not owned by this user
        chat_id = chat_row["id"]
    else:
        insert_res = supabase.table("chats").insert({"user_id": user_id, "title": message[:40]}).execute()
        chat_id = insert_res.data[0]["id"]

    history = _load_history(chat_id)

    try:
        result = graph_app.invoke({
            "query": message,
            "original_query": message,
            "response_language": payload.language,
            "messages": history,
            "is_greeting": False,
            "triage_tier": "NONE",
            "is_dermatology": False,
            "special_population": "NONE",
            "is_medication_query": False,
            "is_comparison": False,
            "comparison_subjects": [],
            "retrieved_chunks": [],
            "retrieval_ok": False,
            "is_grounded": False,
            "retry_count": 0,
            "final_answer": "",
            "needs_personalization": False,
            "clarification_asked": False,
            "clarification_rounds": 0,
            "clarification_questions": [],
            "retrieval_fallback_used": False,
            "has_systemic_symptoms": False,
            "forbidden_ingredients_found": [],
        })
        answer = result["final_answer"]
        is_clarification = result.get("clarification_asked", False)
        # Only surface the structured MCQ payload when this turn actually asked
        # a clarifying question — otherwise leave it None so the frontend falls
        # back to rendering `answer` as a normal markdown bubble.
        clarification_questions = result.get("clarification_questions") if is_clarification else None
    except Exception as e:
        # Log the REAL error server-side instead of swallowing it silently —
        # this is what made the earlier Groq model-deprecation issue invisible.
        import traceback
        print(f"[chat] pipeline error: {e}")
        traceback.print_exc()
        if payload.language == "roman_ur":
            answer = "Maazrat, abhi is sawal ko process karne mein masla ho raha hai. Baraye meharbani thori dair baad dobara koshish karein."
        else:
            answer = "Sorry, I'm having trouble processing that right now. Please try again in a moment."
        is_clarification = False
        clarification_questions = None

    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "user",
        "content": message,
        "is_clarification": False,
    }).execute()
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "assistant",
        "content": answer,
        "is_clarification": is_clarification,
    }).execute()
    supabase.table("chats").update({"updated_at": "now()"}).eq("id", chat_id).execute()

    return ChatResponse(answer=answer, chat_id=chat_id, clarification_questions=clarification_questions)
