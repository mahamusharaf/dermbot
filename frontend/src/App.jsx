import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Sun, Moon, Sparkles, Plus, MessageSquare, PanelLeftClose, PanelLeft, Trash2, Settings, LogOut, Check, Languages, Copy, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/* ------------------------------------------------------------------
   DESIGN TOKENS
------------------------------------------------------------------- */

const palette = {
    sky: "#D98CA3",
    sage: "#C97C92",
    pale: "#F0C6D3",
    deep: "#3A222A",
    core: "#B85C74",
    mist: "#F7DEE5",
};

const THEMES = {
    light: {
        bg: "#FBF2F4",
        panel: "#FFFFFF",
        sidebarBg: "#F6E6EA",
        sidebarText: "#3A222A",
        sidebarSubtext: "#8B6570",
        text: "#3A222A",
        subtext: "#8B6570",
        bubbleBot: "#F7E5EA",
        bubbleBotBorder: "#F0D2DA",
        // Two pinks, no brown/black "deep" tone — that gradient reads as
        // dark/muddy against the light theme's otherwise airy palette.
        bubbleUser: `linear-gradient(135deg, ${palette.sky}, ${palette.core})`,
        inputBg: "#FFFFFF",
        inputBorder: "#F0D2DA",
        headerWash: `radial-gradient(120% 140% at 10% -10%, ${palette.mist} 0%, rgba(240,198,211,0) 60%)`,
        scrollTrack: "#F3DEE4",
        activeChatBg: "#F0D2DA",
        menuPanelBg: "#FFFFFF",
        menuPanelBorder: "#F0D2DA",
    },
    dark: {
        bg: "#1B1A1D",
        panel: "#1B1A1D",
        sidebarBg: "#3A222A",
        sidebarText: "#F9DCE2",
        sidebarSubtext: "#C99AA6",
        text: "#EDEBEC",
        subtext: "#9C989B",
        bubbleBot: "#242226",
        bubbleBotBorder: "#332F33",
        bubbleUser: `linear-gradient(135deg, ${palette.core}, ${palette.deep})`,
        inputBg: "#242226",
        inputBorder: "#332F33",
        headerWash: `radial-gradient(120% 140% at 10% -10%, rgba(184,92,116,0.14) 0%, rgba(27,26,29,0) 60%)`,
        scrollTrack: "#242226",
        activeChatBg: "#5C3A46",
        menuPanelBg: "#2A2429",
        menuPanelBorder: "#3A2E33",
    },
};

// Languages the backend's response_language field understands. Keep this in
// sync with pipeline/graph.py's SUPPORTED_LANGUAGES and api.py's
// ChatRequest.language Literal.
const LANGUAGES = [
    { code: "en", label: "English" },
    { code: "roman_ur", label: "Roman Urdu" },
];

const WELCOME_MESSAGES = {
    en: "Hi, I'm DermBot. Ask me about a skin condition — moles, rashes, treatments, anything dermatology-related.",
    roman_ur: "Hi, main DermBot hoon. Mujh se kisi bhi skin condition ke baare mein poochain — moles, rashes, treatments, kuch bhi dermatology se related.",
};

function makeWelcomeMessage(language) {
    return {
        id: "m0",
        role: "bot",
        text: WELCOME_MESSAGES[language] || WELCOME_MESSAGES.en,
    };
}

function makeDraftChat(language) {
    return {
        id: `draft-${Date.now()}`,
        title: "New conversation",
        isDraft: true,
        messages: [makeWelcomeMessage(language)],
    };
}

const SUGGESTED_QUESTIONS = {
    en: [
        "How do I treat acne?",
        "What causes eczema flare-ups?",
        "Difference between a mole and melanoma?",
        "Best routine for oily skin?",
    ],
    roman_ur: [
        "Acne ka ilaj kaise karoon?",
        "Eczema flare-ups ki wajah kya hai?",
        "Mole aur melanoma mein farq?",
        "Oily skin ke liye best routine?",
    ],
};

const PLACEHOLDER_TEXT = {
    en: "Ask about a skin condition…",
    roman_ur: "Kisi skin condition ke baare mein poochain…",
};

const FOOTER_DISCLAIMER = {
    en: "General information only — not a substitute for a dermatologist.",
    roman_ur: "Sirf general maloomat — dermatologist ka متبادل نہیں۔",
};

const API_BASE_URL = "http://127.0.0.1:8000";

// Single shared breakpoint for the whole app. Below this width the sidebar
// becomes a slide-over drawer instead of a permanent column, and spacing /
// type sizes tighten up so the chat is usable on a phone screen.
const MOBILE_BREAKPOINT = 768;

function useIsMobile(breakpoint = MOBILE_BREAKPOINT) {
    const [isMobile, setIsMobile] = useState(() =>
        typeof window !== "undefined" ? window.innerWidth <= breakpoint : false
    );

    useEffect(() => {
        if (typeof window === "undefined") return;
        const mq = window.matchMedia(`(max-width: ${breakpoint}px)`);
        const handler = (e) => setIsMobile(e.matches);
        setIsMobile(mq.matches);
        if (mq.addEventListener) mq.addEventListener("change", handler);
        else mq.addListener(handler); // Safari < 14 fallback
        return () => {
            if (mq.removeEventListener) mq.removeEventListener("change", handler);
            else mq.removeListener(handler);
        };
    }, [breakpoint]);

    return isMobile;
}

function TypingDots({ color }) {
    return (
        <div style={{ display: "flex", gap: 4, padding: "2px 2px" }}>
            {[0, 1, 2].map((i) => (
                <span
                    key={i}
                    style={{
                        width: 6,
                        height: 6,
                        borderRadius: 999,
                        background: color,
                        display: "inline-block",
                        animation: `derm-bounce 1.1s ${i * 0.15}s infinite ease-in-out`,
                    }}
                />
            ))}
        </div>
    );
}

function initialFor(user) {
    const name = user?.user_metadata?.name || user?.email || "?";
    return name.trim().charAt(0).toUpperCase();
}

/* ------------------------------------------------------------------
   CLARIFICATION BUBBLE
   Renders one follow-up question at a time (no "1 of 4" counter),
   collapses answered ones into small receipt lines, and always
   offers a Skip chip. Progress lives on the message object itself
   (message.clarification) so it survives switching chats.
------------------------------------------------------------------- */

function ClarificationBubble({ theme, message, onAnswer, isMobile }) {
    const t = theme;
    const state = message.clarification || { index: 0, answers: [] };
    const question = message.questions[state.index];
    const answeredSoFar = state.answers.filter((a) => a.answer);

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {answeredSoFar.map((a, i) => (
                <div
                    key={i}
                    style={{
                        fontSize: isMobile ? 12 : 12.5,
                        color: t.subtext,
                        padding: "5px 12px",
                        borderRadius: 12,
                        background: t.bubbleBot,
                        border: `1px solid ${t.bubbleBotBorder}`,
                        alignSelf: "flex-start",
                    }}
                >
                    {a.question}{" "}
                    <span style={{ color: t.text, fontWeight: 600 }}>{a.answer}</span>
                </div>
            ))}

            {question && (
                <>
                    <div
                        style={{
                            padding: isMobile ? "10px 14px" : "11px 16px",
                            borderRadius: "16px 16px 16px 4px",
                            background: t.bubbleBot,
                            border: `1px solid ${t.bubbleBotBorder}`,
                            color: t.text,
                            fontSize: isMobile ? 14 : 14.5,
                            lineHeight: 1.5,
                        }}
                    >
                        {question.text}
                    </div>

                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                        {question.options.map((opt) => (
                            <button
                                key={opt}
                                onClick={() => onAnswer(opt)}
                                style={{
                                    padding: "7px 14px",
                                    borderRadius: 999,
                                    border: `1px solid ${palette.core}`,
                                    background: "transparent",
                                    color: palette.core,
                                    fontSize: 13,
                                    fontWeight: 600,
                                    cursor: "pointer",
                                }}
                            >
                                {opt}
                            </button>
                        ))}
                        <button
                            onClick={() => onAnswer(null)}
                            style={{
                                padding: "7px 14px",
                                borderRadius: 999,
                                border: `1px solid ${t.inputBorder}`,
                                background: "transparent",
                                color: t.subtext,
                                fontSize: 13,
                                cursor: "pointer",
                            }}
                        >
                            Skip
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

/* ------------------------------------------------------------------
   SETTINGS MENU
   Small popover with a language selector. Language is lifted up to
   App (and, ideally, one level further to AppGate alongside theme —
   see the language/onLanguageChange props below) so it survives
   navigating between chats and can be sent with every /api/chat call.
------------------------------------------------------------------- */

function SettingsMenu({ theme, language, onSelectLanguage, onClose }) {
    const t = theme;
    return (
        <>
            <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 10 }} />
            <div
                style={{
                    position: "absolute",
                    bottom: "calc(100% + 6px)",
                    left: 0,
                    right: 0,
                    background: t.menuPanelBg,
                    border: `1px solid ${t.menuPanelBorder}`,
                    borderRadius: 10,
                    padding: 6,
                    zIndex: 11,
                    boxShadow: "0 8px 24px rgba(0,0,0,0.18)",
                }}
            >
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "6px 10px 4px",
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.02em",
                        textTransform: "uppercase",
                        color: t.sidebarSubtext,
                    }}
                >
                    <Languages size={12} />
                    Response language
                </div>
                {LANGUAGES.map((lang) => (
                    <button
                        key={lang.code}
                        onClick={() => {
                            onSelectLanguage(lang.code);
                            onClose();
                        }}
                        className="derm-chat-item"
                        style={{
                            width: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 8,
                            padding: "9px 10px",
                            borderRadius: 8,
                            border: "none",
                            background: "transparent",
                            color: t.sidebarText,
                            fontSize: 13,
                            textAlign: "left",
                            cursor: "pointer",
                        }}
                    >
                        {lang.label}
                        {language === lang.code && <Check size={14} color={palette.core} />}
                    </button>
                ))}
            </div>
        </>
    );
}

export default function App({
    accessToken,
    user,
    onSignOut,
    mode = "light",
    onToggleMode,
    // Lifted the same way `mode` is: parent (e.g. AppGate) can own this and
    // pass language/onLanguageChange down. If the parent doesn't wire these
    // up yet, App falls back to its own local state so it still works
    // standalone.
    language: languageProp,
    onLanguageChange,
}) {
    const isMobile = useIsMobile();
    const [localLanguage, setLocalLanguage] = useState(languageProp || "en");
    const language = onLanguageChange ? (languageProp || "en") : localLanguage;
    const setLanguage = onLanguageChange || setLocalLanguage;

    const [chats, setChats] = useState([makeDraftChat(language)]);
    const [activeChatId, setActiveChatId] = useState(() => chats[0]?.id);
    const [draft, setDraft] = useState("");
    const [isThinking, setIsThinking] = useState(false);
    // Sidebar starts open on desktop, closed on mobile (it's a drawer there,
    // so opening by default would cover the whole screen on first load).
    const [sidebarOpen, setSidebarOpen] = useState(() =>
        typeof window !== "undefined" ? window.innerWidth > MOBILE_BREAKPOINT : true
    );
    const [accountMenuOpen, setAccountMenuOpen] = useState(false);
    const [settingsMenuOpen, setSettingsMenuOpen] = useState(false);
    const [loadingChats, setLoadingChats] = useState(true);
    const [copiedMessageId, setCopiedMessageId] = useState(null);
    const scrollRef = useRef(null);
    const t = THEMES[mode];

    const authHeaders = {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
    };

    const activeChat = chats.find((c) => c.id === activeChatId) || chats[0];
    const isEmptyChat = activeChat?.messages?.length === 1 && activeChat.messages[0].id === "m0";

    // If the viewport crosses the mobile breakpoint (e.g. rotating a tablet,
    // resizing a browser window), keep the sidebar behavior sane: force it
    // closed the moment we cross into mobile so it doesn't linger as a
    // full-screen drawer, and reopen it as a fixed column once we're back
    // on desktop.
    const prevIsMobile = useRef(isMobile);
    useEffect(() => {
        if (prevIsMobile.current !== isMobile) {
            setSidebarOpen(!isMobile);
            prevIsMobile.current = isMobile;
        }
    }, [isMobile]);

    // Prevent the page behind the drawer from scrolling while it's open on
    // mobile.
    useEffect(() => {
        if (!isMobile) return;
        document.body.style.overflow = sidebarOpen ? "hidden" : "";
        return () => {
            document.body.style.overflow = "";
        };
    }, [isMobile, sidebarOpen]);

    // Load the persisted chat list once on mount / whenever the token refreshes.
    useEffect(() => {
        let cancelled = false;
        async function loadChats() {
            setLoadingChats(true);
            try {
                const res = await fetch(`${API_BASE_URL}/api/chats`, { headers: authHeaders });
                if (!res.ok) throw new Error(`Failed to load chats (${res.status})`);
                const data = await res.json();
                if (cancelled) return;
                const loaded = data.map((c) => ({
                    id: c.id,
                    title: c.title,
                    isDraft: false,
                    messages: null, // fetched lazily when the chat is opened
                }));
                if (loaded.length > 0) {
                    setChats(loaded);
                    setActiveChatId(loaded[0].id);
                } else {
                    const fresh = makeDraftChat(language);
                    setChats([fresh]);
                    setActiveChatId(fresh.id);
                }
            } catch (err) {
                console.error("Failed to load chat list", err);
            } finally {
                if (!cancelled) setLoadingChats(false);
            }
        }
        loadChats();
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [accessToken]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [activeChat?.messages, isThinking, activeChatId]);

    const updateChat = (chatId, updater) => {
        setChats((prev) => prev.map((c) => (c.id === chatId ? updater(c) : c)));
    };

    const closeSidebarOnMobile = () => {
        if (isMobile) setSidebarOpen(false);
    };

    const handleNewChat = () => {
        const fresh = makeDraftChat(language);
        setChats((prev) => [fresh, ...prev]);
        setActiveChatId(fresh.id);
        setDraft("");
        closeSidebarOnMobile();
    };

    const handleSelectChat = async (chatId) => {
        setActiveChatId(chatId);
        closeSidebarOnMobile();
        const chat = chats.find((c) => c.id === chatId);
        if (!chat || chat.isDraft || chat.messages !== null) return; // already loaded, or a local draft

        try {
            const res = await fetch(`${API_BASE_URL}/api/chats/${chatId}/messages`, { headers: authHeaders });
            if (!res.ok) throw new Error(`Failed to load messages (${res.status})`);
            const data = await res.json();
            const mapped = data.map((m, i) => ({
                id: `${chatId}-${i}`,
                role: m.role === "assistant" ? "bot" : "user",
                text: m.content,
            }));
            updateChat(chatId, (c) => ({ ...c, messages: [makeWelcomeMessage(language), ...mapped] }));
        } catch (err) {
            console.error("Failed to load chat messages", err);
            updateChat(chatId, (c) => ({ ...c, messages: [makeWelcomeMessage(language)] }));
        }
    };

    const handleDeleteChat = async (e, chatId) => {
        e.stopPropagation();
        const chat = chats.find((c) => c.id === chatId);

        if (chat && !chat.isDraft) {
            try {
                await fetch(`${API_BASE_URL}/api/chats/${chatId}`, {
                    method: "DELETE",
                    headers: authHeaders,
                });
            } catch (err) {
                console.error("Failed to delete chat", err);
            }
        }

        setChats((prev) => {
            const remaining = prev.filter((c) => c.id !== chatId);
            if (remaining.length === 0) {
                const fresh = makeDraftChat(language);
                setActiveChatId(fresh.id);
                return [fresh];
            }
            if (chatId === activeChatId) {
                setActiveChatId(remaining[0].id);
            }
            return remaining;
        });
    };

    const handleSend = async (overrideText) => {
        const text = (overrideText ?? draft).trim();
        if (!text || isThinking || !activeChat) return;

        const userMsg = { id: `u-${Date.now()}`, role: "user", text };
        const wasDraft = activeChat.isDraft;
        const sendingChatId = activeChatId;

        updateChat(sendingChatId, (c) => ({
            ...c,
            messages: [...(c.messages || [makeWelcomeMessage(language)]), userMsg],
            title: c.title === "New conversation" ? text.slice(0, 40) : c.title,
        }));
        setDraft("");
        setIsThinking(true);

        try {
            const res = await fetch(`${API_BASE_URL}/api/chat`, {
                method: "POST",
                headers: authHeaders,
                body: JSON.stringify({
                    message: text,
                    chat_id: wasDraft ? null : sendingChatId,
                    language,
                }),
            });
            if (res.status === 401) {
                throw new Error("Session expired — please sign in again.");
            }
            if (!res.ok) {
                throw new Error("Failed to communicate with API");
            }
            const data = await res.json();

            const hasQuestions =
                Array.isArray(data.clarification_questions) && data.clarification_questions.length > 0;

            // If this was a draft, the backend just created a real chat row —
            // swap the local draft id for the real one so future turns and
            // deletes target the persisted chat.
            setChats((prev) =>
                prev.map((c) =>
                    c.id === sendingChatId
                        ? {
                            ...c,
                            id: data.chat_id,
                            isDraft: false,
                            messages: [
                                ...(c.messages || []),
                                {
                                    id: `b-${Date.now()}`,
                                    role: "bot",
                                    text: data.answer,
                                    questions: hasQuestions ? data.clarification_questions : null,
                                },
                            ],
                        }
                        : c
                )
            );
            setActiveChatId(data.chat_id);
        } catch (error) {
            console.error(error);
            updateChat(sendingChatId, (c) => ({
                ...c,
                messages: [
                    ...(c.messages || []),
                    {
                        id: `b-${Date.now()}`,
                        role: "bot",
                        text: error.message === "Session expired — please sign in again."
                            ? error.message
                            : "Sorry, I encountered an error. Please try again.",
                    },
                ],
            }));
        } finally {
            setIsThinking(false);
        }
    };

    // Advances a clarification message one step. Passing `null` counts as
    // "Skip" for that question. On the last question, composes a single
    // labeled message ("question answer" per line) and sends it through
    // the normal flow — this is still just one backend round-trip total,
    // since all questions were already generated in the original response.
    const handleClarificationAnswer = (chatId, msgId, optionText) => {
        const chat = chats.find((c) => c.id === chatId);
        if (!chat) return;
        const msg = (chat.messages || []).find((m) => m.id === msgId);
        if (!msg || !msg.questions) return;

        const state = msg.clarification || { index: 0, answers: [] };
        const question = msg.questions[state.index];
        if (!question) return;

        const newAnswers = [...state.answers, { question: question.text, answer: optionText }];
        const nextIndex = state.index + 1;
        const isDone = nextIndex >= msg.questions.length;

        updateChat(chatId, (c) => ({
            ...c,
            messages: (c.messages || []).map((m) =>
                m.id === msgId
                    ? { ...m, clarification: { index: nextIndex, answers: newAnswers, done: isDone } }
                    : m
            ),
        }));

        if (isDone) {
            const composed = newAnswers
                .filter((a) => a.answer)
                .map((a) => `${a.question} ${a.answer}`)
                .join("\n");
            handleSend(composed || "No additional details to add.");
        }
    };

    const handleCopy = async (messageId, text) => {
        try {
            await navigator.clipboard.writeText(text);
        } catch (err) {
            console.error("Failed to copy message", err);
            return;
        }
        setCopiedMessageId(messageId);
        setTimeout(() => setCopiedMessageId((current) => (current === messageId ? null : current)), 1500);
    };

    const handleKeyDown = (e) => {
        // On mobile, Enter should insert a newline (there's no physical
        // keyboard convention forcing send-on-enter, and the on-screen
        // keyboard's return key is the easiest way to add a line break).
        // Sending stays a deliberate tap of the send button on touch
        // devices; on desktop, Enter still sends as before.
        if (isMobile) return;
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const displayName = user?.user_metadata?.name || user?.email || "Account";
    const suggestedQuestions = SUGGESTED_QUESTIONS[language] || SUGGESTED_QUESTIONS.en;
    const placeholderText = PLACEHOLDER_TEXT[language] || PLACEHOLDER_TEXT.en;
    const footerDisclaimer = FOOTER_DISCLAIMER[language] || FOOTER_DISCLAIMER.en;

    const composerBar = (
        <div
            style={{
                display: "flex",
                alignItems: "flex-end",
                gap: isMobile ? 8 : 10,
                background: t.inputBg,
                border: `1px solid ${t.inputBorder}`,
                borderRadius: 26,
                padding: isMobile ? "8px 8px 8px 16px" : "10px 10px 10px 18px",
                animation: "derm-glow 3.2s infinite ease-in-out",
            }}
        >
            <textarea
                rows={1}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholderText}
                style={{
                    flex: 1,
                    resize: "none",
                    border: "none",
                    outline: "none",
                    background: "transparent",
                    color: t.text,
                    fontSize: isMobile ? 16 : 14.5, // 16px avoids iOS Safari auto-zoom on focus
                    fontFamily: "inherit",
                    maxHeight: 120,
                    padding: "6px 0",
                }}
            />
            <button
                onClick={() => handleSend()}
                disabled={!draft.trim() || isThinking}
                className="derm-send"
                aria-label="Send message"
                style={{
                    width: isMobile ? 38 : 40,
                    height: isMobile ? 38 : 40,
                    borderRadius: 999,
                    border: "none",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: draft.trim() && !isThinking ? "pointer" : "default",
                    background:
                        draft.trim() && !isThinking
                            ? `linear-gradient(135deg, ${palette.sky}, ${palette.core})`
                            : t.bubbleBot,
                    transition: "transform 0.15s ease, background 0.2s ease",
                    flexShrink: 0,
                }}
            >
                <ArrowUp
                    size={18}
                    color={draft.trim() && !isThinking ? "#fff" : t.subtext}
                    strokeWidth={2.4}
                />
            </button>
        </div>
    );

    return (
        <div
            style={{
                width: "100%",
                height: "100dvh",
                minHeight: 480,
                display: "flex",
                background: t.bg,
                fontFamily:
                    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                transition: "background 0.35s ease",
                overflow: "hidden",
                position: "relative",
            }}
        >
            <style>{`
    @keyframes derm-bounce {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
      30% { transform: translateY(-4px); opacity: 1; }
    }
    @keyframes derm-glow {
      0%, 100% { box-shadow: 0 0 0 0 rgba(62,174,177,0.25); }
      50% { box-shadow: 0 0 0 8px rgba(62,174,177,0); }
    }
    @keyframes derm-slide-in {
      from { transform: translateX(-100%); }
      to { transform: translateX(0); }
    }
    @keyframes derm-fade-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    * { box-sizing: border-box; }
    .derm-scroll::-webkit-scrollbar { width: 6px; }
    .derm-scroll::-webkit-scrollbar-thumb { background: ${t.scrollTrack}; border-radius: 999px; }
    .derm-icon-btn:hover { background: ${t.bubbleBot}; }
    .derm-send:hover { transform: translateY(-1px); }
    .derm-send:active { transform: translateY(0); }
    .derm-chat-item:hover { background: ${t.bubbleBot}; }
    .derm-scroll table {
      border-collapse: collapse;
      width: 100%;
      margin: 8px 0;
      font-size: 13.5px;
      display: block;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    .derm-scroll th, .derm-scroll td {
      border: 1px solid ${t.inputBorder};
      padding: 6px 10px;
      text-align: left;
    }
    .derm-scroll th {
      background: ${t.bubbleBot};
      font-weight: 600;
    }
    .derm-scroll p {
      margin: 0 0 8px 0;
    }
    .derm-scroll p:last-child {
      margin-bottom: 0;
    }
    @media (max-width: ${MOBILE_BREAKPOINT}px) {
      .derm-sidebar-drawer { animation: derm-slide-in 0.22s ease; }
      .derm-sidebar-backdrop { animation: derm-fade-in 0.22s ease; }
    }
  `}</style>

            {/* Sidebar backdrop (mobile drawer mode only) */}
            {sidebarOpen && isMobile && (
                <div
                    onClick={() => setSidebarOpen(false)}
                    className="derm-sidebar-backdrop"
                    style={{
                        position: "fixed",
                        inset: 0,
                        background: "rgba(0,0,0,0.45)",
                        zIndex: 19,
                    }}
                />
            )}

            {/* Sidebar */}
            {sidebarOpen && (
                <div
                    className={isMobile ? "derm-sidebar-drawer" : undefined}
                    style={{
                        width: isMobile ? "min(82vw, 300px)" : 260,
                        flexShrink: 0,
                        background: t.sidebarBg,
                        borderRight: `1px solid ${t.inputBorder}`,
                        display: "flex",
                        flexDirection: "column",
                        padding: "16px 12px",
                        position: isMobile ? "fixed" : "relative",
                        top: 0,
                        left: 0,
                        height: isMobile ? "100%" : "auto",
                        zIndex: isMobile ? 20 : "auto",
                        boxShadow: isMobile ? "4px 0 24px rgba(0,0,0,0.22)" : "none",
                    }}
                >
                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            marginBottom: 16,
                            padding: "0 4px",
                        }}
                    >
                        <span style={{ fontWeight: 700, fontSize: 14, color: t.sidebarText }}>
                            Chats
                        </span>
                        <button
                            onClick={() => setSidebarOpen(false)}
                            className="derm-icon-btn"
                            aria-label={isMobile ? "Close sidebar" : "Collapse sidebar"}
                            style={{
                                width: 30,
                                height: 30,
                                borderRadius: 8,
                                border: "none",
                                background: "transparent",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                cursor: "pointer",
                            }}
                        >
                            {isMobile ? (
                                <X size={16} color={t.sidebarSubtext} />
                            ) : (
                                <PanelLeftClose size={16} color={t.sidebarSubtext} />
                            )}
                        </button>
                    </div>

                    <button
                        onClick={handleNewChat}
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            padding: "10px 12px",
                            borderRadius: 10,
                            border: `1px solid ${palette.core}`,
                            background: palette.core,
                            color: "#fff",
                            fontSize: 13.5,
                            fontWeight: 600,
                            cursor: "pointer",
                            marginBottom: 14,
                        }}
                    >
                        <Plus size={16} />
                        New chat
                    </button>

                    <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
                        {loadingChats ? (
                            <div style={{ fontSize: 13, color: t.sidebarSubtext, padding: "8px 10px" }}>
                                Loading…
                            </div>
                        ) : (
                            chats.map((c) => (
                                <div
                                    key={c.id}
                                    onClick={() => handleSelectChat(c.id)}
                                    className="derm-chat-item"
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                        padding: "9px 8px 9px 10px",
                                        borderRadius: 8,
                                        background: c.id === activeChatId ? t.activeChatBg : "transparent",
                                        color: t.sidebarText,
                                        fontSize: 13,
                                        cursor: "pointer",
                                    }}
                                >
                                    <MessageSquare size={14} color={t.sidebarSubtext} style={{ flexShrink: 0 }} />
                                    <span
                                        style={{
                                            flex: 1,
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                            whiteSpace: "nowrap",
                                            minWidth: 0,
                                        }}
                                    >
                                        {c.title}
                                    </span>
                                    <button
                                        onClick={(e) => handleDeleteChat(e, c.id)}
                                        aria-label="Delete conversation"
                                        className="derm-icon-btn"
                                        style={{
                                            width: 24,
                                            height: 24,
                                            borderRadius: 6,
                                            border: "none",
                                            background: "transparent",
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                            cursor: "pointer",
                                            flexShrink: 0,
                                        }}
                                    >
                                        <Trash2 size={13} color={t.sidebarSubtext} />
                                    </button>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Settings / Account footer */}
                    <div
                        style={{
                            borderTop: `1px solid ${t.inputBorder}`,
                            marginTop: 12,
                            paddingTop: 10,
                            display: "flex",
                            flexDirection: "column",
                            gap: 2,
                            position: "relative",
                        }}
                    >
                        <div style={{ position: "relative" }}>
                            <button
                                onClick={() => setSettingsMenuOpen((o) => !o)}
                                className="derm-chat-item"
                                style={{
                                    width: "100%",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    gap: 8,
                                    padding: "9px 10px",
                                    borderRadius: 8,
                                    border: "none",
                                    background: "transparent",
                                    color: t.sidebarText,
                                    fontSize: 13,
                                    textAlign: "left",
                                    cursor: "pointer",
                                }}
                            >
                                <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                    <Settings size={15} color={t.sidebarSubtext} />
                                    Settings
                                </span>
                                <span style={{ fontSize: 11.5, color: t.sidebarSubtext }}>
                                    {LANGUAGES.find((l) => l.code === language)?.label}
                                </span>
                            </button>

                            {settingsMenuOpen && (
                                <SettingsMenu
                                    theme={t}
                                    language={language}
                                    onSelectLanguage={setLanguage}
                                    onClose={() => setSettingsMenuOpen(false)}
                                />
                            )}
                        </div>

                        <button
                            onClick={() => setAccountMenuOpen((o) => !o)}
                            className="derm-chat-item"
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 8,
                                padding: "9px 10px",
                                borderRadius: 8,
                                border: "none",
                                background: "transparent",
                                color: t.sidebarText,
                                fontSize: 13,
                                textAlign: "left",
                                cursor: "pointer",
                            }}
                        >
                            <div
                                style={{
                                    width: 22,
                                    height: 22,
                                    borderRadius: 999,
                                    background: `linear-gradient(135deg, ${palette.sky}, ${palette.core})`,
                                    color: "#fff",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    fontSize: 11,
                                    fontWeight: 700,
                                    flexShrink: 0,
                                }}
                            >
                                {initialFor(user)}
                            </div>
                            <span
                                style={{
                                    flex: 1,
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                    minWidth: 0,
                                }}
                            >
                                {displayName}
                            </span>
                        </button>

                        {accountMenuOpen && (
                            <>
                                {/* click-outside catcher */}
                                <div
                                    onClick={() => setAccountMenuOpen(false)}
                                    style={{ position: "fixed", inset: 0, zIndex: 10 }}
                                />
                                <div
                                    style={{
                                        position: "absolute",
                                        bottom: "calc(100% + 6px)",
                                        left: 0,
                                        right: 0,
                                        background: t.menuPanelBg,
                                        border: `1px solid ${t.menuPanelBorder}`,
                                        borderRadius: 10,
                                        padding: 6,
                                        zIndex: 11,
                                        boxShadow: "0 8px 24px rgba(0,0,0,0.18)",
                                    }}
                                >
                                    <div
                                        style={{
                                            padding: "8px 10px",
                                            fontSize: 12,
                                            color: t.sidebarSubtext,
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                            whiteSpace: "nowrap",
                                        }}
                                    >
                                        {user?.email}
                                    </div>
                                    <button
                                        onClick={onSignOut}
                                        className="derm-chat-item"
                                        style={{
                                            width: "100%",
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 8,
                                            padding: "9px 10px",
                                            borderRadius: 8,
                                            border: "none",
                                            background: "transparent",
                                            color: t.sidebarText,
                                            fontSize: 13,
                                            textAlign: "left",
                                            cursor: "pointer",
                                        }}
                                    >
                                        <LogOut size={15} color={t.sidebarSubtext} />
                                        Log out
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}

            {/* Main column */}
            <div
                style={{
                    flex: 1,
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    background: t.panel,
                    minWidth: 0,
                }}
            >
                {/* Header */}
                <div
                    style={{
                        position: "relative",
                        padding: isMobile ? "14px 16px" : "20px 24px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        borderBottom: `1px solid ${t.inputBorder}`,
                        background: t.headerWash,
                        flexShrink: 0,
                    }}
                >
                    <div style={{ display: "flex", alignItems: "center", gap: isMobile ? 8 : 12, minWidth: 0 }}>
                        {!sidebarOpen && (
                            <button
                                onClick={() => setSidebarOpen(true)}
                                className="derm-icon-btn"
                                aria-label="Open sidebar"
                                style={{
                                    width: 34,
                                    height: 34,
                                    borderRadius: 8,
                                    border: "none",
                                    background: "transparent",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    cursor: "pointer",
                                    marginRight: 4,
                                    flexShrink: 0,
                                }}
                            >
                                <PanelLeft size={18} color={t.text} />
                            </button>
                        )}
                        <div
                            style={{
                                width: isMobile ? 34 : 40,
                                height: isMobile ? 34 : 40,
                                borderRadius: isMobile ? 11 : 14,
                                background: `linear-gradient(135deg, ${palette.sky}, ${palette.deep})`,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                            }}
                        >
                            <Sparkles size={isMobile ? 16 : 18} color="#fff" strokeWidth={2} />
                        </div>
                        <div style={{ minWidth: 0 }}>
                            <div
                                style={{
                                    fontWeight: 700,
                                    fontSize: isMobile ? 15 : 16,
                                    color: t.text,
                                    letterSpacing: "-0.01em",
                                    whiteSpace: "nowrap",
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                }}
                            >
                                DermBot
                            </div>
                            {!isMobile && (
                                <div style={{ fontSize: 12.5, color: t.subtext }}>
                                    Skin health, answered carefully
                                </div>
                            )}
                        </div>
                    </div>

                    <button
                        onClick={onToggleMode}
                        aria-label="Toggle dark mode"
                        className="derm-icon-btn"
                        style={{
                            width: isMobile ? 34 : 38,
                            height: isMobile ? 34 : 38,
                            borderRadius: 999,
                            border: `1px solid ${t.inputBorder}`,
                            background: "transparent",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            cursor: "pointer",
                            transition: "background 0.2s ease",
                            flexShrink: 0,
                        }}
                    >
                        {mode === "light" ? (
                            <Moon size={isMobile ? 15 : 17} color={t.text} />
                        ) : (
                            <Sun size={isMobile ? 15 : 17} color={t.text} />
                        )}
                    </button>
                </div>

                {/* Centered content wrapper */}
                <div
                    style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        width: "100%",
                        maxWidth: 820,
                        margin: "0 auto",
                        minHeight: 0,
                    }}
                >
                    {isEmptyChat ? (
                        /* Empty-state hero */
                        <div
                            style={{
                                flex: 1,
                                display: "flex",
                                flexDirection: "column",
                                alignItems: "center",
                                justifyContent: "center",
                                padding: isMobile ? "20px 16px" : "24px",
                                gap: isMobile ? 18 : 24,
                                overflowY: "auto",
                            }}
                        >
                            <div style={{ textAlign: "center" }}>
                                <div
                                    style={{
                                        fontSize: isMobile ? 19 : 24,
                                        fontWeight: 700,
                                        color: t.text,
                                        letterSpacing: "-0.01em",
                                        marginBottom: 6,
                                    }}
                                >
                                    Ask me anything about your skin
                                </div>
                                <div style={{ fontSize: isMobile ? 13 : 14, color: t.subtext }}>
                                    Moles, rashes, treatments, routines — I'm here to help.
                                </div>
                            </div>

                            <div style={{ width: "100%", maxWidth: 640 }}>{composerBar}</div>

                            <div
                                style={{
                                    width: "100%",
                                    maxWidth: 640,
                                    display: "flex",
                                    flexWrap: "wrap",
                                    gap: 8,
                                    justifyContent: "center",
                                }}
                            >
                                {suggestedQuestions.map((q) => (
                                    <button
                                        key={q}
                                        onClick={() => handleSend(q)}
                                        disabled={isThinking}
                                        style={{
                                            padding: isMobile ? "7px 12px" : "8px 14px",
                                            borderRadius: 999,
                                            border: `1px solid ${t.inputBorder}`,
                                            background: t.bubbleBot,
                                            color: t.text,
                                            fontSize: isMobile ? 12.5 : 13,
                                            cursor: isThinking ? "default" : "pointer",
                                        }}
                                    >
                                        {q}
                                    </button>
                                ))}
                            </div>

                            <div
                                style={{
                                    fontSize: 11,
                                    color: t.subtext,
                                    textAlign: "center",
                                    padding: "0 8px",
                                }}
                            >
                                {footerDisclaimer}
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* Messages */}
                            <div
                                ref={scrollRef}
                                className="derm-scroll"
                                style={{
                                    flex: 1,
                                    overflowY: "auto",
                                    padding: isMobile ? "16px" : "24px",
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: 14,
                                }}
                            >
                                {(activeChat?.messages || []).map((m) => {
                                    const showClarification =
                                        m.role === "bot" &&
                                        m.questions &&
                                        m.questions.length > 0 &&
                                        !(m.clarification && m.clarification.done);

                                    return (
                                        <div
                                            key={m.id}
                                            style={{
                                                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                                                maxWidth: isMobile ? "90%" : "78%",
                                                display: "flex",
                                                flexDirection: "column",
                                                gap: 4,
                                            }}
                                        >
                                            {showClarification ? (
                                                <ClarificationBubble
                                                    theme={t}
                                                    message={m}
                                                    isMobile={isMobile}
                                                    onAnswer={(opt) =>
                                                        handleClarificationAnswer(activeChatId, m.id, opt)
                                                    }
                                                />
                                            ) : (
                                                <>
                                                    <div
                                                        style={{
                                                            padding: isMobile ? "10px 14px" : "11px 16px",
                                                            borderRadius:
                                                                m.role === "user"
                                                                    ? "16px 16px 4px 16px"
                                                                    : "16px 16px 16px 4px",
                                                            background: m.role === "user" ? t.bubbleUser : t.bubbleBot,
                                                            border:
                                                                m.role === "user"
                                                                    ? "none"
                                                                    : `1px solid ${t.bubbleBotBorder}`,
                                                            color: m.role === "user" ? "#F3FBFB" : t.text,
                                                            fontSize: isMobile ? 14 : 14.5,
                                                            lineHeight: 1.5,
                                                            overflowWrap: "break-word",
                                                            wordBreak: "break-word",
                                                        }}
                                                    >
                                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                                                    </div>
                                                    <button
                                                        onClick={() => handleCopy(m.id, m.text)}
                                                        className="derm-icon-btn"
                                                        aria-label="Copy message"
                                                        style={{
                                                            alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                                                            display: "flex",
                                                            alignItems: "center",
                                                            gap: 4,
                                                            padding: "3px 6px",
                                                            borderRadius: 6,
                                                            border: "none",
                                                            background: "transparent",
                                                            color: t.subtext,
                                                            fontSize: 11,
                                                            cursor: "pointer",
                                                        }}
                                                    >
                                                        {copiedMessageId === m.id ? (
                                                            <>
                                                                <Check size={11} />
                                                                Copied
                                                            </>
                                                        ) : (
                                                            <>
                                                                <Copy size={11} />
                                                                Copy
                                                            </>
                                                        )}
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    );
                                })}

                                {isThinking && (
                                    <div
                                        style={{
                                            alignSelf: "flex-start",
                                            padding: "12px 16px",
                                            borderRadius: "16px 16px 16px 4px",
                                            background: t.bubbleBot,
                                            border: `1px solid ${t.bubbleBotBorder}`,
                                        }}
                                    >
                                        <TypingDots color={palette.core} />
                                    </div>
                                )}
                            </div>

                            {/* Input bar */}
                            <div style={{ padding: isMobile ? "10px 12px 14px" : "16px 20px 20px", flexShrink: 0 }}>
                                {composerBar}
                                <div
                                    style={{
                                        fontSize: 11,
                                        color: t.subtext,
                                        textAlign: "center",
                                        marginTop: 10,
                                        padding: "0 8px",
                                    }}
                                >
                                    {footerDisclaimer}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}