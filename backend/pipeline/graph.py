import os
import re
import time
import json
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Lazy-loaded model/DB singletons.
#
# IMPORTANT: These used to be loaded at module level (i.e. the instant this
# file was imported). Since api.py does `from pipeline.graph import app as
# graph_app` at the top of the file, that meant uvicorn couldn't bind its
# port until the embedding model, cross-encoder, and Chroma client had all
# finished loading -- on Render this took long enough (and used enough RAM)
# to blow past both the port-scan timeout and the 512MB memory limit.
#
# Now each of these loads once, on first actual use, and is cached in a
# module-level variable for reuse. `app = graph.compile()` below stays cheap
# (it just wires up the graph structure), so importing this file is now fast
# and uvicorn can bind the port within seconds. The heavy loading happens
# lazily the first time a request actually needs retrieval.
# ---------------------------------------------------------------------------

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(SCRIPT_DIR, "chroma_db_bge")

_embedder = None
_reranker = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("[lazy_load] loading embedder BAAI/bge-base-en-v1.5 ...", flush=True)
        t0 = time.time()
        _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print(f"[lazy_load] embedder loaded ({time.time() - t0:.1f}s)", flush=True)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        print("[lazy_load] loading reranker cross-encoder/ms-marco-MiniLM-L-6-v2 ...", flush=True)
        t0 = time.time()
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"[lazy_load] reranker loaded ({time.time() - t0:.1f}s)", flush=True)
    return _reranker


def get_collection():
    global _collection
    if _collection is None:
        print(f"[lazy_load] connecting to chroma at {db_path} ...", flush=True)
        t0 = time.time()
        chroma_client = chromadb.PersistentClient(path=db_path)
        _collection = chroma_client.get_collection(name="dermatology")
        print(f"[lazy_load] chroma collection ready ({time.time() - t0:.1f}s)", flush=True)
    return _collection


load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_groq_with_retry(messages, temperature=0, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            is_quota_error = "rate_limit_exceeded" in err_str and "tokens per day" in err_str
            print(f"[groq_retry] attempt {attempt + 1} failed: {e}")
            if is_quota_error:
                # Daily quota exhaustion won't resolve within seconds — retrying is pointless.
                raise RuntimeError("GROQ_DAILY_QUOTA_EXCEEDED") from e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise

class GraphState(TypedDict):
    query: str
    original_query: str
    response_language: str
    messages: list
    is_greeting: bool
    triage_tier: str
    is_dermatology: bool
    special_population: str
    is_medication_query: bool
    is_comparison: bool
    comparison_subjects: list
    retrieved_chunks: list
    retrieval_ok: bool
    is_grounded: bool
    retry_count: int
    final_answer: str
    needs_personalization: bool
    clarification_asked: bool
    clarification_rounds: int
    retrieval_fallback_used: bool
    has_systemic_symptoms: bool
    clarification_questions: list
    forbidden_ingredients_found: list


# ---------------------------------------------------------------------------
# Language support
#
# response_language is supplied by the caller (a Settings preference — see
# api.py's ChatRequest.language), never auto-detected from the query. This
# keeps behavior predictable for users who code-switch mid-sentence.
#
# Everything downstream of prepare_language (classifiers, retrieval, the
# reranker) keeps operating on English text exactly as before. Only the
# final response text — generated or fixed — needs to vary by language.
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ("en", "roman_ur")

LANGUAGE_RESPONSE_INSTRUCTIONS = {
    "roman_ur": (
        "\nLANGUAGE REQUIREMENT (applies to the ENTIRE response, every sentence, from the first "
        "word to the last): write only in Roman Urdu (Urdu written in Latin/English script, casual "
        "conversational tone). Do not write any part of the response in English and do not switch to "
        "Urdu script (اردو). Medical/condition names may stay in their standard English/Latin form "
        "(e.g. \"Eczema\", \"Tinea corporis\"), but every surrounding sentence, explanation, and label "
        "around them must be in Roman Urdu."
    ),
}

def language_instruction(state: GraphState) -> str:
    """Prompt-injectable instruction line for the current response_language. Empty for English."""
    lang = state.get("response_language", "en")
    return LANGUAGE_RESPONSE_INSTRUCTIONS.get(lang, "")


def prepare_language(state: GraphState) -> GraphState:
    """
    Entry node. Runs before check_greeting.

    If response_language is non-English, translates the incoming query to
    English so every existing classifier/retriever/reranker downstream needs
    zero changes. The original text is preserved in original_query in case a
    later node ever needs it (e.g. debugging, logging).

    If response_language is English (or unset), this is a no-op pass-through.
    """
    query = state["query"]
    response_language = state.get("response_language", "en")
    if response_language not in SUPPORTED_LANGUAGES:
        response_language = "en"

    if response_language == "en":
        print(f"[prepare_language] response_language=en, no translation needed")
        return {**state, "original_query": query, "response_language": response_language}

    prompt = f"""Translate the following message into clear, natural English. The message may be in Roman Urdu (Urdu written in Latin script), English, or a mix of both.

Output ONLY the English translation — no preamble, no notes, nothing else.

Message: {query}

English translation:"""

    translated = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip()
    print(f"[prepare_language] '{query}' -> '{translated}' (response_language={response_language})")
    return {**state, "query": translated, "original_query": query, "response_language": response_language}


# ---------------------------------------------------------------------------
# Fixed (non-LLM-generated) response strings, translated per language.
# ---------------------------------------------------------------------------

FIXED_RESPONSES = {
    "respond_greeting": {
        "en": "Hello! I'm a dermatology assistant. Ask me about a wide range of skin conditions.",
        "roman_ur": "Assalam o Alaikum! Main aik dermatology assistant hoon. Mujh se skin, baal ya nails se related kisi bhi masle ke baare mein pooch sakte hain.",
    },
    "respond_out_of_scope": {
        "en": "I can only help with dermatology-related questions. Try asking about a skin condition.",
        "roman_ur": "Main sirf dermatology (skin) se related sawalat mein madad kar sakta hoon. Kisi skin condition ke baare mein poochain.",
    },
    "respond_no_info": {
        "en": "I don't have enough information in my knowledge base to answer that confidently.",
        "roman_ur": "Mere paas is sawal ka wazeh jawab dene ke liye kaafi maloomat nahi hai.",
    },
    "respond_emergency": {
        "en": (
            " Based on what you've described, this could be a medical emergency or urgent condition. "
            "Please seek in-person medical evaluation right away — go to an urgent care clinic, emergency room, "
            "or call your local emergency number. This is not something I can safely assess or advise on through chat."
        ),
        "roman_ur": (
            " Jo aap ne bataya hai us ke mutabiq yeh aik medical emergency ya urgent condition ho sakti hai. "
            "Foran kisi doctor se in-person check karwayen — urgent care clinic ya emergency room jayen, "
            "ya apne local emergency number par call karein. Main is cheez ko chat par mehfooz tareeqe se assess ya advise nahi kar sakta."
        ),
    },
}

def fixed_response(node_name: str, state: GraphState) -> str:
    lang = state.get("response_language", "en")
    table = FIXED_RESPONSES[node_name]
    return table.get(lang, table["en"])


# ---------------------------------------------------------------------------
# Forbidden-ingredient safety net.
#
# generate()'s prompt already tells the model never to recommend these
# ingredients (see the SAFETY RULE block below), but that instruction can get
# crowded out in a long, rule-dense prompt — same failure mode we saw with
# the language instruction. Rather than relying only on the model reading and
# obeying the rule, this is a deterministic keyword check on the *output*:
# - check_groundedness scans the generated answer and, if a forbidden
#   ingredient shows up, forces a retry (same retry path as a groundedness
#   failure) with an explicit callout of what was wrong last time.
# - safety_check is the last node before END on the RAG/comparison paths, so
#   it does one final scan; if a forbidden ingredient is STILL present after
#   retries are exhausted, the answer is replaced with a safe fallback
#   instead of ever being shipped to the user.
#
# This is intentionally a coarse keyword match, not semantic understanding —
# it can false-positive on a legitimate mention (e.g. a question about
# garlic-handling contact dermatitis, which isn't a home-remedy recommendation).
# That's an acceptable trade-off here: a false positive just costs an extra
# retry or a slightly-too-cautious fallback, while a false negative would
# ship unsafe advice. It is not a substitute for fixing the underlying
# prompt-crowding issue, just a backstop for when that fix isn't enough.
# ---------------------------------------------------------------------------

FORBIDDEN_INGREDIENT_PATTERNS = [
    (r"\btea[\s-]*tree\b", "tea tree oil"),
    (r"\bessential\s+oils?\b", "undiluted essential oil"),
    (r"\b(raw|undiluted)?\s*lemon\s*juice\b|\bnimbu\b", "raw/undiluted lemon juice"),
    (r"\bbaking\s*soda\b", "baking soda"),
    (r"\braw\s*garlic\b|\blehsun\b", "raw garlic"),
    (r"\btoothpaste\b|\bmanjan\b", "toothpaste"),
]

def find_forbidden_ingredients(text: str) -> list:
    text_lower = text.lower()
    found = []
    for pattern, label in FORBIDDEN_INGREDIENT_PATTERNS:
        if re.search(pattern, text_lower) and label not in found:
            found.append(label)
    return found

FORBIDDEN_INGREDIENT_FALLBACK = {
    "en": (
        "I can't safely recommend a home remedy here — some ingredients that might otherwise come up "
        "(like tea tree oil, lemon juice, baking soda, raw garlic, or toothpaste) aren't safe to apply "
        "directly to skin. Please use a gentle, fragrance-free moisturizer instead, and see a dermatologist "
        "for a personalized recommendation."
    ),
    "roman_ur": (
        "Main yahan koi ghar ka nuskha safely recommend nahi kar sakta — kuch ingredients (jaise tea tree "
        "oil, lemon juice, baking soda, kacha lehsun, ya toothpaste) seedha skin par lagana safe nahi hota. "
        "Iske bajaye aik gentle, fragrance-free moisturizer istemal karein, aur behtar mashware ke liye "
        "dermatologist se rabta karein."
    ),
}


def check_greeting(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""Is this message just a greeting or small talk (e.g. "hi", "hello", "how are you", "thanks")?
Reply with only YES or NO.

Message: {query}"""
    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0)
    is_greeting = "YES" in answer.strip().upper()
    print(f"[check_greeting] '{query}' -> is_greeting={is_greeting}")
    return {**state, "is_greeting": is_greeting}

def check_domain(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]) if history else "(no prior conversation)"

    prompt = f"""Is this question related to dermatology, skin, hair, or nails in any way? This includes medical skin conditions AND general skincare/cosmetic topics.

Examples of IN-SCOPE questions (answer YES):
- Medical/pathology: melanoma, moles, rashes, skin cancer, eczema, psoriasis, acne, treatments for skin issues
- Medications applied to skin: steroid creams, antifungals, retinoids, and questions about using them
- General skincare and cosmetic topics: homemade face masks, skincare routines, moisturizers, cleansers, sunscreen, product recommendations, hair or nail care, "what's good for oily skin", "how do I get rid of dark circles"

If the message is short and looks like it could be answering a previous question, check the conversation history below — if the conversation was already about a dermatology or skincare topic, treat this as a continuation and answer YES, even if the message alone (like "yes", "bare hands", "no allergies") doesn't obviously mention skin.

Conversation history:
{history_text}

Question: {query}

Reply with only YES or NO."""
    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0)
    is_dermatology = "YES" in answer.strip().upper()
    print(f"[check_domain] '{query}' -> is_dermatology={is_dermatology}")
    return {**state, "is_dermatology": is_dermatology}

def check_special_population(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""Does this question involve a pregnant person, someone trying to conceive, breastfeeding, an infant, or a child/minor?

Reply with ONLY one of: PREGNANCY, PEDIATRIC, NONE

Question: {query}"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip().upper()
    print(f"[check_special_population] '{query}' -> {answer}")
    result = answer if answer in ("PREGNANCY", "PEDIATRIC") else "NONE"
    return {**state, "special_population": result}

def check_medication_query(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) if history else "(no prior conversation)"

    prompt = f"""You are a medical safety classifier. Flag YES if the user is asking about USING, CONTINUING, STOPPING, or the APPROPRIATENESS/DURATION/FREQUENCY/TAPERING of a specific named prescription medication, topical steroid, or other prescription-strength treatment on themselves or their child — regardless of how much clinical-sounding detail (skin condition, history, other medications) they've provided. This also includes questions about using leftover medication, medication prescribed to someone else, or a named over-the-counter medication (e.g. hydrocortisone) on a described symptom.

This includes questions like:
- "Can I use [drug] on [body part] for [duration]?"
- "How long should I use [drug]?"
- "Is it safe to use [drug] for two weeks?"
- "Should I taper off [drug]?"
- "Can I use [drug] on my face / around my eyes / on a child?"
- "Can I use my leftover [drug] for this?"
- "Can I put [OTC drug, e.g. hydrocortisone] on this rash?"

This is regardless of whether the user describes their condition, prior use, pregnancy status, or other medications — providing that history does NOT change the answer to this classification.

Answer NO for general questions about a drug class or condition that do NOT ask about the user's own personal use, duration, or dosing (e.g. "what is clobetasol used for", "what are side effects of topical steroids", "difference between clobetasol and hydrocortisone").

Conversation history:
{history_text}

Reply with ONLY YES or NO.

Question: {query}"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0)
    is_medication_query = "YES" in answer.strip().upper()
    print(f"[check_medication_query] '{query}' -> is_medication_query={is_medication_query}")
    return {**state, "is_medication_query": is_medication_query}

MAX_CLARIFICATION_ROUNDS = 2

def check_needs_personalization(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])

    # Structural round count — based on a flag set by respond_ask_personalization itself,
    # NOT string-matching the LLM's (temperature>0, wording-variable) intro sentence.
    prior_asks = sum(
        1 for m in history
        if m["role"] == "assistant" and m.get("is_clarification", False)
    )
    if prior_asks >= MAX_CLARIFICATION_ROUNDS:
        print(f"[check_needs_personalization] '{query}' -> hit max clarification rounds ({prior_asks}), proceeding to answer")
        return {**state, "needs_personalization": False}

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) if history else "(no prior conversation)"

    prompt = f"""Does this question need more information from the user before it can be answered well? This applies to two different cases:

1. A request for a PERSONALIZED RECOMMENDATION (mask, routine, product) where skin type or allergies would change the answer.
2. A description of a SYMPTOM OR REACTION happening to the user, where key details would meaningfully change or narrow the answer. This includes onset/duration, triggers, exposure history, what makes it better/worse, AND morphology and spread details: whether lesions are raised/flat/scaly/blistered/ring-shaped, how many there are, whether they're spreading, whether they're symmetrical, whether anyone nearby has similar symptoms, and any systemic symptoms (fever, feeling unwell, swelling, pain, warmth, pus, bleeding).

Check the conversation history — has the user already provided enough of the relevant details to meaningfully narrow down what this could be? A few generic details (e.g. only location, duration, and "itchy") are usually NOT enough on their own for a symptom question — many unrelated conditions share those features. Only mark HAS_INFO if the details given actually help distinguish between likely conditions.

IMPORTANT EXCEPTION: if the user has already identified a clear, specific likely trigger with a plausible temporal link (e.g. symptoms started after a new product, food, medication, or exposure, with no red-flag features present), that alone is often enough to give useful general guidance (e.g. discontinue the suspected trigger, general care, when to see a dermatologist) even without full morphology detail. In that case, prefer HAS_INFO — don't ask for description-refining details (exact shape, borders, symmetry, count) if they wouldn't change the near-term guidance you'd give. Ask again only if a genuinely new question would materially change the guidance (e.g. worsening despite stopping the trigger, or a red-flag feature appearing).

IMPORTANT EXCEPTION 2: if the user has responded to the most recent round of questions with repeated denials (e.g. several consecutive "no" answers, or "none of those"), treat that as HAS_INFO rather than NEEDS_INFO. Repeated denial of additional distinguishing features means the picture is simple, not that more questions will help — asking again is unlikely to surface anything new.

Conversation history:
{history_text}

Question: {query}

Reply with exactly one of:
NEEDS_INFO — more detail would meaningfully help and the user hasn't given enough of it yet
HAS_INFO — the user has already given enough distinguishing detail in the history, OR has repeatedly indicated there's nothing more to add
NOT_APPLICABLE — this question can be answered well as-is, no distinguishing detail needed"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip().upper()
    print(f"[check_needs_personalization] '{query}' -> {answer} (round {prior_asks + 1})")
    needs_info = "NEEDS_INFO" in answer
    return {**state, "needs_personalization": needs_info}

def check_comparison(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""Is this question asking to compare two or more skin conditions (e.g. "difference between X and Y", "X vs Y", "how is X different from Y")?

If YES, reply in exactly this format on one line:
COMPARISON: subject1, subject2

If NO, reply with exactly:
COMPARISON: none

Question: {query}"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip()
    print(f"[check_comparison] '{query}' -> '{answer}'")

    if answer.upper().startswith("COMPARISON: NONE") or "none" in answer.lower():
        return {**state, "is_comparison": False, "comparison_subjects": []}

    subjects_part = answer.split(":", 1)[-1].strip()
    subjects = [s.strip() for s in subjects_part.split(",") if s.strip()]
    return {**state, "is_comparison": len(subjects) >= 2, "comparison_subjects": subjects}


COMPARISON_ASPECT_TEMPLATES = {
    "cause": "{subject} cause",
    "appearance": "{subject} appearance",
    "symptoms": "{subject} symptoms",
    "treatment": "{subject} treatment or management or removal",
}

def check_systemic_symptoms(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) if history else "(no prior conversation)"

    prompt = f"""Does this message describe BOTH (a) a skin finding (patch, rash, lesion, discoloration, texture change, etc.) AND (b) a non-skin systemic symptom that could indicate an underlying medical condition — e.g. increased thirst, frequent urination, unexplained weight change, fatigue, excessive hunger, palpitations, hair thinning/loss, menstrual irregularity, cold/heat intolerance?

Check the conversation history as well as the latest message — the two pieces of information may have been given in different turns.

Conversation history:
{history_text}

Message: {query}

Reply with ONLY YES or NO."""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0)
    has_systemic = "YES" in answer.strip().upper()
    print(f"[check_systemic_symptoms] '{query}' -> has_systemic_symptoms={has_systemic}")
    return {**state, "has_systemic_symptoms": has_systemic}

def retrieve_comparison(state: GraphState) -> GraphState:
    subjects = state["comparison_subjects"]
    all_chunks = []
    seen_texts = set()
    embedder = get_embedder()
    reranker = get_reranker()
    collection = get_collection()

    for subject in subjects:
        for aspect, template in COMPARISON_ASPECT_TEMPLATES.items():
            aspect_query = template.format(subject=subject)
            query_embedding = embedder.encode([BGE_QUERY_PREFIX + aspect_query], normalize_embeddings=True).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=10)
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            if not docs:
                continue

            pairs = [[aspect_query, doc] for doc in docs]
            scores = reranker.predict(pairs)
            reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)

            added = 0
            for doc, meta, score in reranked:
                if doc in seen_texts:
                    continue
                all_chunks.append({"text": doc, "source": meta["source"], "score": float(score), "subject": subject, "aspect": aspect})
                seen_texts.add(doc)
                added += 1
                if added >= 2:
                    break

        subject_chunk_count = len([c for c in all_chunks if c["subject"] == subject])
        print(f"[retrieve_comparison] '{subject}' -> {subject_chunk_count} chunks")

    return {**state, "retrieved_chunks": all_chunks}

def retrieve_fallback(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""The first search for this dermatology question returned only weak matches. Rewrite it as a short, clinical, keyword-style search query — not a narrative sentence — using standard dermatology terminology, so it matches medical reference text better.

Focus on: the likely condition category (e.g. contact dermatitis, irritant reaction, allergic reaction), the affected area, and the key symptoms/trigger. Strip out conversational phrasing like "I'm experiencing" or "it started after".

Question: {query}

Output (short clinical search query only, nothing else):"""

    simplified = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip()
    print(f"[retrieve_fallback] '{query}' -> simplified '{simplified}'")

    embedder = get_embedder()
    reranker = get_reranker()
    collection = get_collection()

    query_embedding = embedder.encode([BGE_QUERY_PREFIX + simplified], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=20)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        print(f"[retrieve_fallback] no chunks found for '{simplified}'")
        return {**state, "retrieved_chunks": [], "retrieval_fallback_used": True}

    pairs = [[simplified, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    top_chunks = [{"text": doc, "source": meta["source"], "score": float(score)} for doc, meta, score in reranked[:4]]

    print(f"[retrieve_fallback] '{simplified}' -> {len(top_chunks)} chunks, top score={top_chunks[0]['score']:.3f}")
    return {**state, "retrieved_chunks": top_chunks, "retrieval_fallback_used": True}

def generate_comparison(state: GraphState) -> GraphState:
    subjects = state["comparison_subjects"]
    chunks = state["retrieved_chunks"]
    retry_count = state.get("retry_count", 0)

    context_by_subject = {}
    for subject in subjects:
        subject_chunks = [c for c in chunks if c["subject"] == subject]
        context_by_subject[subject] = "\n\n".join(c["text"] for c in subject_chunks)

    context_text = "\n\n".join(f"=== {s} ===\n{context_by_subject[s]}" for s in subjects)

    stricter = "\nIMPORTANT: Only state facts explicitly present in the context. Do not add outside information." if retry_count > 0 else ""
    lang_line = language_instruction(state)

    prompt = f"""You are a dermatology information assistant. Using ONLY the context below, compare {" and ".join(subjects)}.{stricter}{lang_line}

Respond with a markdown table with these rows: Cause, Appearance, Common Symptoms, Treatment. If the context lacks information for a cell, write "Not specified in available information" rather than guessing. Keep the table's row labels (Cause, Appearance, Common Symptoms, Treatment) and the subject column headers in English even when the rest of the response is in another language, so the table stays readable.

After the table, add 1-2 sentences on the single most important distinguishing feature, if supported by the context.

Context:
{context_text}
{lang_line}
Comparison:"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0.2)
    print(f"[generate_comparison] produced answer ({len(answer)} chars, attempt {retry_count + 1})")
    return {**state, "final_answer": answer, "retry_count": retry_count + 1}

def rewrite_query(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])

    recent = history[-4:] if history else []
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent) if recent else "(no prior conversation)"

    prompt = f"""You are preparing a dermatology question for a medical search system. Given the question below:

1. Fix any misspelled dermatology terms (e.g. "melinoma" -> "melanoma", "soriasis" -> "psoriasis", "eczma" -> "eczema").
2. Expand medical abbreviations to their full term (e.g. "BCC" -> "basal cell carcinoma", "AK" -> "actinic keratosis", "SCC" -> "squamous cell carcinoma").
3. If the question contains an ambiguous reference (like "it", "that", "this one") that depends on the conversation history, resolve it using the history.
4. IMPORTANT: If this message looks like it's answering a clarifying question the assistant just asked (e.g. it states skin type, allergies, or symptoms without naming a topic), look at the conversation history to find what the ORIGINAL request was, and rewrite this into a single combined, standalone question that includes both the original request AND these new details. For example, if the assistant previously asked about skin type for a mask recommendation, and this message says "oily skin, no allergies", rewrite it as "recommend a homemade face mask for oily, acne-prone skin with no known allergies".
5. Otherwise, do NOT add any new topic, assumption, or detail that wasn't in the original question or implied by rule 4.
6. If the question is already clear, standalone, and correctly spelled, return it unchanged (aside from any abbreviation expansion).

Conversation history:
{history_text}

Question: {query}

Output (corrected/expanded/merged question only, nothing else):"""

    rewritten = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip()
    print(f"[rewrite_query] '{query}' -> '{rewritten}'")
    return {**state, "query": rewritten}

def retrieve(state: GraphState) -> GraphState:
    query = state["query"]
    embedder = get_embedder()
    reranker = get_reranker()
    collection = get_collection()

    query_embedding = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=20)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        print(f"[retrieve] no chunks found for '{query}'")
        return {**state, "retrieved_chunks": []}

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
    top_chunks = [{"text": doc, "source": meta["source"], "score": float(score)} for doc, meta, score in reranked[:4]]

    print(f"[retrieve] '{query}' -> {len(top_chunks)} chunks, top score={top_chunks[0]['score']:.3f}")
    return {**state, "retrieved_chunks": top_chunks}

RETRIEVAL_SCORE_THRESHOLD = -4.0

def retrieve_systemic_supplement(state: GraphState) -> GraphState:
    query = state["query"]
    existing_chunks = state["retrieved_chunks"]
    seen_texts = {c["text"] for c in existing_chunks}
    embedder = get_embedder()
    reranker = get_reranker()
    collection = get_collection()

    supplement_query = (
        "skin findings associated with underlying systemic or metabolic disease, "
        "e.g. acanthosis nigricans, thyroid disease, diabetes, insulin resistance, hormonal disorders"
    )
    query_embedding = embedder.encode([BGE_QUERY_PREFIX + supplement_query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        print("[retrieve_systemic_supplement] no supplemental chunks found")
        return state

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)

    added = []
    for doc, meta, score in reranked:
        if doc in seen_texts:
            continue
        added.append({"text": doc, "source": meta["source"], "score": float(score)})
        seen_texts.add(doc)
        if len(added) >= 2:
            break

    # Merge and re-sort by score so a strong supplemental match can actually become
    # chunks[0] and pass check_retrieval_quality — appending alone never achieves that.
    merged = sorted(existing_chunks + added, key=lambda c: c["score"], reverse=True)
    if merged:
        print(f"[retrieve_systemic_supplement] added {len(added)} supplemental chunks, merged top score={merged[0]['score']:.3f}")
    else:
        print("[retrieve_systemic_supplement] merge resulted in empty chunk list")
    return {**state, "retrieved_chunks": merged}

def check_emergency(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]) if history else "(no prior conversation)"

    prompt = f"""You are a medical safety triage classifier. Classify the person's (or their child's/dependent's) described symptom into exactly one tier.

Use the conversation history to understand context — a short follow-up message may describe a symptom introduced earlier. Evaluate the FULL picture, not just the latest message alone.

Conversation history:
{history_text}

TIER: EMERGENCY — needs ER/urgent care RIGHT NOW. Reserve this for acute, rapidly dangerous presentations:
- Active bleeding from a skin lesion
- High fever combined with a rash, especially in a child or infant
- A rash that is purple, dark red, or non-blanching (doesn't fade when pressed) — classic sign of serious infection, ALWAYS this tier with fever
- Skin turning black/purple or showing tissue death
- Facial or throat swelling
- Difficulty breathing
- A rapidly spreading rash (spreading over hours, not months)
- Signs of severe allergic reaction
- A blistering rash near/affecting the eye
- Honey-colored/yellow crusting with pus, increasing pain, warmth/swelling, fever, or rapid spreading (suggests bacterial superinfection)
- Any "should I wait" question about something matching this tier — don't let "wait" soften the answer

TIER: URGENT_DERM_REFERRAL — concerning, needs a dermatologist evaluation soon (days, not months), but is NOT an acute emergency. Only use this tier if the person explicitly describes a CHANGE over time (new, growing, evolving, or altered in color/shape/size/border) in a specific mole, lesion, or spot:
- A mole or pigmented lesion that has changed in color, shape, size, or border over WEEKS TO MONTHS, with no acute symptoms
- A new or changing lesion the person is worried might be skin cancer, without emergency features
- A sore or lesion that hasn't healed over weeks

Do NOT use this tier just because a lesion, patch, or spot is mentioned — only when the person describes it CHANGING or being NEW. A general question about an existing, stable skin finding (e.g. "what could cause a dark patch," "does this mean I have X condition") is NOT this tier — route those to NONE so they get answered with real information.

TIER: NONE — routine informational question, general symptom question, or something not urgent.

Reply with ONLY one word: EMERGENCY, URGENT_DERM_REFERRAL, or NONE.

Latest message: {query}"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0).strip().upper()
    print(f"[check_emergency] '{query}' -> {answer}")
    if "EMERGENCY" in answer and "URGENT_DERM" not in answer:
        tier = "EMERGENCY"
    elif "URGENT_DERM" in answer:
        tier = "URGENT_DERM_REFERRAL"
    else:
        tier = "NONE"
    return {**state, "triage_tier": tier}

def check_retrieval_quality(state: GraphState) -> GraphState:
    chunks = state["retrieved_chunks"]
    if not chunks:
        print("[check_retrieval_quality] empty retrieval -> FAIL")
        return {**state, "retrieval_ok": False}
    top_score = chunks[0]["score"]
    ok = top_score > RETRIEVAL_SCORE_THRESHOLD
    print(f"[check_retrieval_quality] top_score={top_score:.3f} -> {'OK' if ok else 'FAIL'}")
    return {**state, "retrieval_ok": ok}

def respond_emergency(state: GraphState) -> GraphState:
    return {**state, "final_answer": fixed_response("respond_emergency", state)}

def respond_urgent_derm(state: GraphState) -> GraphState:
    query = state["query"]
    lang_line = language_instruction(state)
    prompt = f"""The user described a skin finding that's concerning enough to need a dermatologist's evaluation soon, but is not an acute emergency. Write a brief, warm 2-3 sentence response that:
1. Acknowledges specifically what they described (don't use generic language — refer to their actual symptom/finding)
2. Explains this is worth having a dermatologist look at promptly (days, not an emergency, but not something to leave for months)
3. Makes clear you can't assess or diagnose it through chat

Question: {query}
{lang_line}
Response:"""
    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0.3)
    print(f"[respond_urgent_derm] generated response for '{query}'")
    return {**state, "final_answer": answer}

def respond_medication_caution(state: GraphState) -> GraphState:
    query = state["query"]
    lang_line = language_instruction(state)

    prompt = f"""The user is asking about using a specific named medication or potent treatment on themselves — including its duration, frequency, tapering, appropriateness for a body site, or whether they can use medication they already have on hand or one commonly available over-the-counter.

First, determine which case this is:

CASE A — LEFTOVER OR REPURPOSED MEDICATION: the user has medication left over from a previous prescription (their own or someone else's), or is asking to use a medication for a condition/purpose other than what it was originally prescribed for.
For this case, be direct: tell them NOT to use it. Explain briefly that it was prescribed for a specific past condition/course, not this one, that using leftover or repurposed prescription medication without a clinician evaluating the CURRENT problem can be ineffective, mask the real issue, cause side effects, or (for antibiotics specifically) contribute to antibiotic resistance. Recommend they see a clinician or pharmacist for the current issue.

CASE B — CURRENTLY PRESCRIBED MEDICATION: the user is currently on a course of this medication, prescribed for their current condition, and is asking about duration, tapering, frequency, or appropriateness for where they're applying/taking it.
Do NOT confirm, suggest, or imply any specific duration, frequency, tapering schedule, or dosing regimen, even in general terms. State that duration/tapering decisions need to come from the prescribing clinician, and tell them to follow the prescriber's instructions exactly or contact the prescriber/pharmacist if unclear.

CASE C — GENERAL / OTC QUESTION, NO STATED PRESCRIPTION: the user is asking generally whether they can use a named medication (which may be over-the-counter, like hydrocortisone) on a symptom they've described, without saying it was prescribed to them for this.
Don't assume it was prescribed. Advise against self-treating with it before a clinician or pharmacist has looked at the symptom, and explain why in terms of THIS specific situation if possible (see below).

SYMPTOM-BASED CAUTION (apply to any case where relevant): If the skin issue the user describes has features classically associated with a condition where this type of medication could worsen, mask, or be the wrong treatment for the underlying problem, briefly explain that specific reasoning as part of your answer — without diagnosing confidently or naming a single certain condition. Well-known examples: a ring-shaped, scaly, or spreading itchy rash can be consistent with a fungal infection (e.g. ringworm/tinea), and topical corticosteroids (like hydrocortisone) can worsen or mask fungal infections, making them harder to recognize and treat, so they're best avoided until a clinician has assessed it. Only include this reasoning if it's genuinely applicable to what the user described — don't force it in when there's no relevant pattern.

Only mention body-site risk (face, near the eyes, on a child, during pregnancy) if it's actually relevant to how THIS medication is taken/applied based on what the user described — do not add site-application language that doesn't match the medication's actual route (e.g. don't discuss facial application for an oral tablet).

Keep the response brief — a few sentences, not a full consultation.

Question: {query}
{lang_line}
Response:"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0.2)
    print(f"[respond_medication_caution] generated caution response")
    return {**state, "final_answer": answer}

def respond_ask_personalization(state: GraphState) -> GraphState:
    query = state["query"]
    history = state.get("messages", [])
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) if history else "(no prior conversation)"
    lang = state.get("response_language", "en")

    json_lang_note = (
        '\nWrite the "intro" text and every question/option string in Roman Urdu (Urdu written in Latin script, casual conversational tone) — not Urdu script, not English. Keep the JSON keys themselves ("intro", "questions", "text", "options") in English exactly as shown.'
        if lang == "roman_ur" else ""
    )

    prompt = f"""The user asked a dermatology question that needs more detail to answer well. Write 2-4 short, specific follow-up questions that would actually help answer THIS question — not a generic intake form. Base the questions on what's actually relevant to what they described, and on what they HAVEN'T already told you (check the conversation history so you don't repeat a question they already answered).

For a rash/spots/lesion description, consider asking about: whether it's raised/flat/scaly/blistered/ring-shaped, how many spots, whether it's spreading, whether it's symmetrical, any new product/food/medication/plant/insect exposure, whether anyone nearby has similar symptoms, and any fever, pain, warmth, swelling, or pus.

For EACH question, also provide 3-5 short clickable answer options (a few words each) covering the most likely answers. Include an option like "Not sure" or "None of these" when genuinely applicable, so the user isn't forced into a wrong-sounding choice.
{json_lang_note}

Conversation history:
{history_text}

Question: {query}

Respond with ONLY valid JSON, no markdown code fences, no extra text, in exactly this shape:
{{
  "intro": "<one warm sentence>",
  "questions": [
    {{"text": "<question 1>", "options": ["<option>", "<option>", "..."]}},
    {{"text": "<question 2>", "options": ["<option>", "<option>", "..."]}}
  ]
}}"""

    raw = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0.3)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        intro = parsed.get("intro", "Sure thing! I just need a bit more info.")
        questions = parsed.get("questions", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[respond_ask_personalization] JSON parse failed ({e}), falling back to plain text")
        intro = raw.strip()
        questions = []

    # Plain-text version for chat history/storage — every other prompt in this
    # graph reads history as flat "role: content" text, so this keeps that
    # working unchanged even though the frontend renders `questions` as chips.
    bullet_lines = "\n".join(f"- {q.get('text', '')}" for q in questions if q.get("text"))
    final_text = f"{intro}\n\n{bullet_lines}" if bullet_lines else intro

    print(f"[respond_ask_personalization] generated {len(questions)} structured questions")
    return {
        **state,
        "final_answer": final_text,
        "clarification_asked": True,
        "clarification_questions": questions,
    }

DIFFERENTIAL_RULE = """
DIFFERENTIAL vs. SINGLE-CONDITION RULE: Before naming a specific condition, check whether the described symptoms genuinely point toward one condition more than plausible alternatives, based on distinguishing features actually present in the context (not just shared generic features like "itchy" or "red" that apply to many conditions).

- If multiple conditions in the context share the described features with no clearly distinguishing detail present, do NOT settle on one as "the" answer. Instead, list 2-4 plausible possibilities from the context as a differential, and say clearly that it cannot be narrowed down further without an in-person exam.
- Only name a single specific condition confidently if the context provides a genuinely distinguishing feature the user described (e.g. a specific shape, distribution pattern, or trigger that's characteristic of that one condition and not the others).
- When staying in differential mode, general care advice should still be relevant to the body site and condition category described (e.g. for a scalp complaint, mention anti-dandruff/medicated shampoo approaches if supported by the context, rather than generic facial-skincare advice like moisturizer). Don't default to the same generic advice regardless of where on the body the symptom is.
- Do NOT include condition-specific PRESCRIPTION-level treatment information (e.g. specific corticosteroid names) in differential mode — general, widely-available approaches are fine if grounded in the context.
- The CONDITION-SPECIFIC TREATMENT RULE below only applies once you've determined a single condition is genuinely well-supported, not as a default for every symptom question.
"""

SYSTEMIC_SYMPTOM_RULE = """
SKIN–SYSTEMIC SYMPTOM RULE: If the user describes a skin finding alongside systemic symptoms (e.g. increased thirst, frequent urination, unexplained weight change, fatigue, palpitations, hair changes), do NOT state or imply that the skin finding and the systemic symptoms are unrelated or "separate" — that is a specific medical judgment call this assistant cannot make. Some skin findings (e.g. a dark, thickened, velvety patch, especially on the neck, armpits, or groin — a pattern known as acanthosis nigricans) are well-established as being potentially associated with insulin resistance or diabetes, even if this specific association isn't in the retrieved context — this is well-established general medical knowledge, similar to the established caution about steroids and fungal infections. If the description is consistent with this pattern, mention the possible association explicitly, make clear the skin finding alone cannot diagnose a systemic condition, and recommend the user seek medical evaluation that addresses BOTH the skin finding and the systemic symptoms together, rather than treating them as two separate, unconnected questions.
"""
CONDITION_SPECIFIC_TREATMENT_RULE = """
CONDITION-SPECIFIC TREATMENT RULE: If the context identifies or strongly suggests a specific condition (e.g. tinea corporis/ringworm, eczema, contact dermatitis, psoriasis), include the GENERAL category of treatment typically used for that condition, if the context supports it (e.g. "topical antifungal creams such as clotrimazole or terbinafine are commonly used for ringworm" or "topical corticosteroids are often used for eczema flares"). Do NOT give a specific dose, duration, brand recommendation, or prescription-strength guidance — keep it at the level of general information, and note that a dermatologist should confirm the diagnosis and appropriate treatment, since this can't be visually confirmed through chat.

EXCEPTION: If special_population is PEDIATRIC, do not name any treatment class at all for an undiagnosed rash in a child — not even at the general-category level. The pediatric population_note above takes precedence: recommend evaluation before any treatment, and limit suggestions to low-risk supportive care (e.g. fragrance-free moisturizer) if the context supports it.

Do NOT substitute generic skincare advice (gentle cleanser, moisturizer, sunscreen) for condition-specific treatment information when a likely condition has been identified — only use generic skincare advice when no specific condition is indicated, or when the user is asking for general skin maintenance.
"""

def generate(state: GraphState) -> GraphState:
    query = state["query"]
    chunks = state["retrieved_chunks"]
    history = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    population = state.get("special_population", "NONE")
    has_systemic = state.get("has_systemic_symptoms", False)
    lang_line = language_instruction(state)
    prior_forbidden = state.get("forbidden_ingredients_found", [])

    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    stricter = "\nIMPORTANT: Only state facts explicitly present in the context. Do not add outside information." if retry_count > 0 else ""

    forbidden_retry_note = ""
    if prior_forbidden:
        forbidden_retry_note = (
            f"\nCRITICAL: Your previous answer incorrectly recommended one or more forbidden ingredients: "
            f"{', '.join(prior_forbidden)}. These are NEVER allowed, under any circumstances, regardless of "
            f"what's in the context. Do not mention or recommend them in any form this time — suggest a "
            f"fragrance-free moisturizer/emollient instead if you would otherwise reach for a DIY ingredient."
        )

    population_note = ""
    if population == "PREGNANCY":
        population_note = "\nIMPORTANT: This question involves pregnancy or breastfeeding. Many dermatology treatments (especially retinoids/tretinoin, isotretinoin, and some other medications) are contraindicated or require caution during pregnancy. If the context mentions pregnancy safety, include it explicitly. If it doesn't, clearly state that pregnancy-specific safety should be confirmed with a doctor before use."
    elif population == "PEDIATRIC":
        population_note = (
            "\nIMPORTANT: This question involves a child. Do NOT name a specific treatment class "
            "(e.g. 'hydrocortisone', 'topical steroid', 'antifungal cream') as something to start using, "
            "and do NOT give any potency, frequency, or duration guidance — not even in general terms — "
            "for an undiagnosed rash in a child. Many pediatric rashes (eczema, contact dermatitis, "
            "ringworm, impetigo) look similar but need different treatments, and steroid strength/duration "
            "guidance that's fine for adults isn't automatically fine for a child, especially on the face. "
            "Instead, recommend the child be evaluated by a pediatrician or dermatologist before starting "
            "any treatment. You may suggest low-risk supportive care only, such as a fragrance-free "
            "moisturizer/emollient, if the context supports it."
        )

    systemic_block = SYSTEMIC_SYMPTOM_RULE if has_systemic else ""

    prompt = f"""You are a dermatology information assistant. Answer the question using ONLY the context provided below. If the context doesn't contain enough information, say so clearly.{stricter}{population_note}{lang_line}

If the answer discusses more than one possible condition, format each as a bullet point with the condition name in bold (e.g. "- **Psoriasis**: ..."). Use standard markdown (bold with **, bullets with -). Keep medical/condition names themselves in their standard English/Latin form even when writing the surrounding sentence in another language, since that's how patients will need to search for or discuss them with a doctor.

Do NOT end your answer with a disclaimer sentence like "this is general information, not medical advice" or "consult a dermatologist for diagnosis" — a disclaimer is added automatically after your answer. You may still recommend seeing a dermatologist as a concrete next step, but don't phrase it as a generic closing disclaimer.
{DIFFERENTIAL_RULE}
{systemic_block}
SAFETY RULE (DIY / home remedies): If the user is asking for a home remedy, DIY recipe, or homemade mask/treatment, you may ONLY suggest recipes or ingredients that are explicitly present in the retrieved context below. Do NOT add other ingredients from general knowledge, even ones commonly believed to be gentle (e.g. honey, avocado, oatmeal, yogurt), unless they actually appear in the context. If the context does not contain a recipe suitable for this person's described skin type/condition, do not invent one — say so clearly, and recommend a safer non-DIY alternative such as a fragrance-free moisturizer or emollient instead.

Never recommend applying raw/undiluted lemon juice, undiluted essential oils (tea tree, etc.), baking soda, raw garlic, or toothpaste to skin, even if such content appears in the retrieved context — these remain excluded regardless of what's in the context.

EXTRA CAUTION FOR SENSITIVE/REACTIVE SKIN: If the user describes their skin as extremely sensitive, reactive, easily irritated, or prone to allergic reactions, be conservative even about a context-validated recipe. Prefer advising against experimenting with homemade treatments altogether and recommend a fragrance-free moisturizer/emollient instead, unless the context explicitly states the recipe is formulated or tested for sensitive skin. Do not use "patch test first" as a way to make an otherwise unvalidated DIY treatment sound acceptable for this population — a patch test does not substitute for starting with a suitable, already-gentle product.
{CONDITION_SPECIFIC_TREATMENT_RULE}
If the user has described a specific trigger...

If the user has described a specific trigger, exposure, or timeline for their symptom, use that context to give a more specific, relevant answer (e.g. connecting a new product exposure to a likely irritant/allergic contact reaction) rather than a generic answer. For a likely irritant/allergic contact reaction tied to a specific product, explicitly suggest discontinuing the suspected product and, once resolved, reintroducing potential triggers one at a time (patch testing) to help identify the cause, rather than only offering generic skincare advice.

{f"Recent conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Context:
{context}

Question: {query}
{lang_line}{forbidden_retry_note}
Answer:"""

    answer = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0.2)
    print(f"[generate] produced answer ({len(answer)} chars, attempt {retry_count + 1})")
    return {**state, "final_answer": answer, "retry_count": retry_count + 1}

MAX_RETRIES = 2

def check_groundedness(state: GraphState) -> GraphState:
    answer = state["final_answer"]
    chunks = state["retrieved_chunks"]
    context = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""Does the ANSWER below rely only on facts present in the CONTEXT, with no unsupported or invented information? The ANSWER may be written in a different language than the CONTEXT (e.g. Roman Urdu vs English) — judge the underlying factual content across languages, not the wording.

Important: connecting the user's own described symptoms to a matching condition's symptoms IS a legitimate use of the context, not an unsupported claim — this is the correct behavior for symptom-based questions. Only flag the answer if it states a fact about a disease (a cause, mechanism, statistic, treatment, or claim) OR recommends a specific home remedy, DIY ingredient, or recipe that is NOT actually present anywhere in the context.

Reply with ONLY YES or NO.

CONTEXT:
{context}

ANSWER:
{answer}

Grounded (YES/NO):"""

    result = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0)
    is_grounded = "YES" in result.strip().upper()

    forbidden = find_forbidden_ingredients(answer)
    if forbidden:
        print(f"[check_groundedness] forbidden ingredient(s) detected despite exclusion rule: {forbidden} -> forcing retry")
        is_grounded = False

    print(f"[check_groundedness] is_grounded={is_grounded} (retry_count={state.get('retry_count', 0)})")
    return {**state, "is_grounded": is_grounded, "forbidden_ingredients_found": forbidden}

MEDICAL_KEYWORDS = ["treatment", "treated", "surgery", "dose", "dosage", "prescri", "medication", "biopsy", "diagnos"]

# Roman Urdu keyword variants for the same medical-disclaimer trigger words, since
# answers in that language won't contain the English MEDICAL_KEYWORDS above.
MEDICAL_KEYWORDS_ROMAN_UR = [
    "ilaj", "dawa", "dawai", "surgery", "operation", "dose", "khuraak",
    "tajweez", "prescription", "biopsy", "tashkhees", "diagnos",
]

DISCLAIMERS = {
    "pregnancy": {
        "en": "\n\n This involves pregnancy or breastfeeding — some treatments considered safe generally may not be safe in this context. Please confirm with your doctor or OB before using any product or medication.",
        "roman_ur": "\n\n Is mein pregnancy ya breastfeeding shamil hai — kuch treatments jo aam tor par safe samjhi jati hain, woh is soorat-e-haal mein safe nahi ho sakti. Koi bhi product ya dawa istemal karne se pehle apne doctor ya OB se tasdeeq zaroor kar lein.",
    },
    "pediatric": {
        "en": "\n\n This involves a child — dosing and product safety can differ significantly from adults. Please consult a pediatrician or dermatologist before treating a child's skin condition.",
        "roman_ur": "\n\n Is mein aik bacha shamil hai — khuraak aur product ki safety adults se kaafi mukhtalif ho sakti hai. Bachay ki skin condition ka ilaj karne se pehle pediatrician ya dermatologist se mashwara zaroor karein.",
    },
    "general": {
        "en": "\n\n This is general information, not medical advice. Please consult a dermatologist for diagnosis or treatment decisions.",
        "roman_ur": "\n\n Yeh general maloomat hai, medical advice nahi hai. Tashkhees ya ilaj ke faislon ke liye dermatologist se mashwara zaroor karein.",
    },
}

def _disclaimer(kind: str, lang: str) -> str:
    table = DISCLAIMERS[kind]
    return table.get(lang, table["en"])

def safety_check(state: GraphState) -> GraphState:
    answer = state["final_answer"]
    population = state.get("special_population", "NONE")
    lang = state.get("response_language", "en")

    # Final backstop: even after check_groundedness's retry loop, confirm the
    # answer that's actually about to ship doesn't contain a forbidden
    # ingredient. If it still does (e.g. MAX_RETRIES was exhausted), never
    # ship it — replace with a safe fallback instead.
    still_forbidden = find_forbidden_ingredients(answer)
    if still_forbidden:
        print(f"[safety_check] forbidden ingredient(s) still present after retries: {still_forbidden} -> overriding with safe fallback")
        fallback = FORBIDDEN_INGREDIENT_FALLBACK.get(lang, FORBIDDEN_INGREDIENT_FALLBACK["en"])
        return {**state, "final_answer": fallback}

    answer_lower = answer.lower()

    if population == "PREGNANCY":
        answer += _disclaimer("pregnancy", lang)
        print("[safety_check] added pregnancy caution")
    elif population == "PEDIATRIC":
        answer += _disclaimer("pediatric", lang)
        print("[safety_check] added pediatric caution")
    elif any(kw in answer_lower for kw in MEDICAL_KEYWORDS) or any(kw in answer_lower for kw in MEDICAL_KEYWORDS_ROMAN_UR):
        answer += _disclaimer("general", lang)
        print("[safety_check] added medical disclaimer")
    else:
        print("[safety_check] no disclaimer needed")

    return {**state, "final_answer": answer}

def respond_greeting(state: GraphState) -> GraphState:
    return {**state, "final_answer": fixed_response("respond_greeting", state)}

def respond_out_of_scope(state: GraphState) -> GraphState:
    return {**state, "final_answer": fixed_response("respond_out_of_scope", state)}

def respond_no_info(state: GraphState) -> GraphState:
    return {**state, "final_answer": fixed_response("respond_no_info", state)}

def route_after_greeting(state: GraphState) -> Literal["respond_greeting", "check_emergency"]:
    return "respond_greeting" if state["is_greeting"] else "check_emergency"

def route_after_domain(state: GraphState) -> Literal["respond_out_of_scope", "check_special_population"]:
    return "check_special_population" if state["is_dermatology"] else "respond_out_of_scope"

def route_after_medication_query(state: GraphState) -> Literal["respond_medication_caution", "check_systemic_symptoms"]:
    return "respond_medication_caution" if state["is_medication_query"] else "check_systemic_symptoms"

def route_after_retrieve(state: GraphState) -> Literal["retrieve_systemic_supplement", "check_retrieval_quality"]:
    return "retrieve_systemic_supplement" if state.get("has_systemic_symptoms", False) else "check_retrieval_quality"

def route_after_quality_check(state: GraphState) -> Literal["respond_no_info", "generate", "retrieve_fallback"]:
    if state["retrieval_ok"]:
        return "generate"
    if not state.get("retrieval_fallback_used", False):
        return "retrieve_fallback"
    return "respond_no_info"

def route_after_comparison(state: GraphState) -> Literal["retrieve_comparison", "rewrite_query"]:
    return "retrieve_comparison" if state["is_comparison"] else "rewrite_query"

def route_after_emergency(state: GraphState) -> Literal["respond_emergency", "respond_urgent_derm", "check_domain"]:
    tier = state.get("triage_tier", "NONE")
    if tier == "EMERGENCY":
        return "respond_emergency"
    if tier == "URGENT_DERM_REFERRAL":
        return "respond_urgent_derm"
    return "check_domain"

def route_after_personalization_check(state: GraphState) -> Literal["respond_ask_personalization", "check_comparison"]:
    return "respond_ask_personalization" if state["needs_personalization"] else "check_comparison"

def route_after_groundedness(state: GraphState) -> Literal["generate", "generate_comparison", "safety_check"]:
    retry_count = state.get("retry_count", 0)
    if state["is_grounded"]:
        return "safety_check"
    if retry_count < MAX_RETRIES:
        return "generate_comparison" if state.get("is_comparison") else "generate"
    print("[route_after_groundedness] max retries hit, proceeding anyway")
    return "safety_check"

graph = StateGraph(GraphState)

# --- Register every node exactly once ---
graph.add_node("prepare_language", prepare_language)
graph.add_node("check_greeting", check_greeting)
graph.add_node("check_domain", check_domain)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("respond_greeting", respond_greeting)
graph.add_node("respond_out_of_scope", respond_out_of_scope)
graph.add_node("retrieve", retrieve)
graph.add_node("check_retrieval_quality", check_retrieval_quality)
graph.add_node("retrieve_fallback", retrieve_fallback)
graph.add_node("respond_no_info", respond_no_info)
graph.add_node("generate", generate)
graph.add_node("check_groundedness", check_groundedness)
graph.add_node("safety_check", safety_check)

graph.add_node("check_comparison", check_comparison)
graph.add_node("retrieve_comparison", retrieve_comparison)
graph.add_node("generate_comparison", generate_comparison)

graph.add_node("check_special_population", check_special_population)
graph.add_node("check_medication_query", check_medication_query)
graph.add_node("respond_medication_caution", respond_medication_caution)

graph.add_node("check_systemic_symptoms", check_systemic_symptoms)
graph.add_node("retrieve_systemic_supplement", retrieve_systemic_supplement)

graph.add_node("check_needs_personalization", check_needs_personalization)
graph.add_node("respond_ask_personalization", respond_ask_personalization)

graph.add_node("check_emergency", check_emergency)
graph.add_node("respond_emergency", respond_emergency)
graph.add_node("respond_urgent_derm", respond_urgent_derm)

# --- Entry point ---
# prepare_language runs first so every downstream node — classifiers,
# retrieval, the reranker — keeps operating on English text unchanged.
graph.set_entry_point("prepare_language")
graph.add_edge("prepare_language", "check_greeting")
graph.add_conditional_edges("check_greeting", route_after_greeting)
graph.add_edge("respond_greeting", END)

# --- Emergency/urgent triage ---
graph.add_conditional_edges("check_emergency", route_after_emergency)
graph.add_edge("respond_emergency", END)
graph.add_edge("respond_urgent_derm", END)

# --- Domain check ---
graph.add_conditional_edges("check_domain", route_after_domain)
graph.add_edge("respond_out_of_scope", END)

# --- Special population -> medication gate -> systemic-symptom check -> personalization ---
graph.add_edge("check_special_population", "check_medication_query")
graph.add_conditional_edges("check_medication_query", route_after_medication_query)
graph.add_edge("respond_medication_caution", END)

graph.add_edge("check_systemic_symptoms", "check_needs_personalization")

graph.add_conditional_edges("check_needs_personalization", route_after_personalization_check)
graph.add_edge("respond_ask_personalization", END)

# --- Comparison branch ---
graph.add_conditional_edges("check_comparison", route_after_comparison)
graph.add_edge("retrieve_comparison", "generate_comparison")
graph.add_edge("generate_comparison", "check_groundedness")

# --- Standard RAG branch ---
graph.add_edge("rewrite_query", "retrieve")
graph.add_conditional_edges("retrieve", route_after_retrieve)
graph.add_edge("retrieve_systemic_supplement", "check_retrieval_quality")
graph.add_conditional_edges("check_retrieval_quality", route_after_quality_check)
graph.add_edge("retrieve_fallback", "check_retrieval_quality")
graph.add_edge("respond_no_info", END)

graph.add_edge("generate", "check_groundedness")
graph.add_conditional_edges("check_groundedness", route_after_groundedness)
graph.add_edge("safety_check", END)

app = graph.compile()

def ask(query, history=None, response_language="en"):
    """
    history: list of {"role": ..., "content": ..., "is_clarification": ...} dicts
             for this specific chat/session — NOT global state.
    response_language: "en" or "roman_ur" — a Settings preference, not auto-detected.
    Returns: (final_answer: str, is_clarification: bool, clarification_questions: list)
    Caller is responsible for persisting the new turn.
    """
    history = history or []
    print(f"\n{'='*50}")
    try:
        result = app.invoke({
            "query": query,
            "original_query": query,
            "response_language": response_language,
            "messages": history,
            "is_greeting": False,
            "is_emergency": False,
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
            "clarification_rounds": 0,
            "clarification_asked": False,
            "clarification_questions": [],
            "retrieval_fallback_used": False,
            "has_systemic_symptoms": False,
            "forbidden_ingredients_found": [],
        })
        final_answer = result["final_answer"]
        is_clarification = result.get("clarification_asked", False)
        clarification_questions = result.get("clarification_questions", [])
    except RuntimeError as e:
        if str(e) == "GROQ_DAILY_QUOTA_EXCEEDED":
            print("[ask] Groq daily quota exhausted — returning degraded response")
            if response_language == "roman_ur":
                final_answer = (
                    "Filhaal main naye sawalat process nahi kar pa raha — aaj ki usage limit mukammal ho chuki hai. "
                    "Baraye meharbani kuch minute baad dobara koshish karein. Agar yeh urgent hai to dermatologist se "
                    "mashwara karein ya in-person ilaj karwayen."
                )
            else:
                final_answer = (
                    "I'm temporarily unable to process new questions — I've hit my daily usage limit. "
                    "Please try again in a few minutes. If this is urgent, please consult a dermatologist "
                    "or seek in-person care."
                )
            is_clarification = False
            clarification_questions = []
        else:
            raise

    print(f"USER: {query}")
    print(f"BOT: {final_answer}")
    return final_answer, is_clarification, clarification_questions

if __name__ == "__main__":
    history = []
    answer, is_clar, _ = ask("can you recommend a mask I can make at home", history)
    history.append({"role": "user", "content": "can you recommend a mask I can make at home"})
    history.append({"role": "assistant", "content": answer, "is_clarification": is_clar})

    answer, is_clar, _ = ask("I have oily, acne-prone skin, no allergies", history)

    # Roman Urdu smoke test
    ur_history = []
    answer, is_clar, _ = ask("mujhe acha facewash recommend karo", ur_history, response_language="roman_ur")
