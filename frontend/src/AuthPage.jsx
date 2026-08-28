import React, { useEffect, useState } from "react";
import { Sun, Moon, Sparkles, Eye, EyeOff } from "lucide-react";
import { supabase } from "./supabaseClient";

/* ------------------------------------------------------------------
   Reuses the same palette/theme tokens as App.jsx so this page matches
   DermBot's look automatically in both light and dark mode.
   If you already export `palette`/`THEMES` from App.jsx, you can delete
   this block and import them instead:
     import { palette, THEMES } from "./App";
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
        text: "#3A222A",
        subtext: "#8B6570",
        inputBg: "#FFFFFF",
        inputBorder: "#F0D2DA",
        divider: "#F0D2DA",
    },
    dark: {
        bg: "#1B1A1D",
        panel: "#1B1A1D",
        text: "#EDEBEC",
        subtext: "#9C989B",
        inputBg: "#242226",
        inputBorder: "#332F33",
        divider: "#332F33",
    },
};

// Same breakpoint App.jsx uses — below this the two-column layout stacks
// into a short brand banner on top and the form underneath.
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

function GoogleGlyph() {
    return (
        <svg width="18" height="18" viewBox="0 0 18 18">
            <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.87 2.7-6.62z" />
            <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.84.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.9v2.33A9 9 0 0 0 9 18z" />
            <path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 0 1 3.67 9c0-.59.1-1.16.28-1.7V4.97H.9A9 9 0 0 0 0 9c0 1.45.35 2.83.9 4.03l3.05-2.33z" />
            <path fill="#EA4335" d="M9 3.58c1.32 0 2.51.46 3.44 1.35l2.58-2.58A8.62 8.62 0 0 0 9 0 9 9 0 0 0 .9 4.97l3.05 2.33C4.66 5.17 6.65 3.58 9 3.58z" />
        </svg>
    );
}

function StatBlock({ value, label }) {
    return (
        <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#fff" }}>{value}</div>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.65)" }}>{label}</div>
        </div>
    );
}

export default function AuthPage({ mode = "light", onToggleMode }) {
    // Note: no onAuthenticated prop anymore. Auth state is driven entirely by
    // Supabase's onAuthStateChange listener in the root gate component
    // (AppGate.jsx) — once supabase.auth.signInWithPassword/signUp/OAuth
    // succeeds, that listener fires and the gate swaps this page out for
    // <App />. This component only needs to trigger the Supabase calls.
    //
    // `mode`/`onToggleMode` are passed down from AppGate so the theme choice
    // made here carries over into <App /> after sign-in, instead of each
    // page tracking its own independent light/dark state.
    const isMobile = useIsMobile();
    const [view, setView] = useState("signin"); // "signin" | "signup"
    const [showPassword, setShowPassword] = useState(false);
    const [form, setForm] = useState({ name: "", email: "", password: "" });
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState("");
    const [infoMsg, setInfoMsg] = useState("");
    const t = THEMES[mode];
    const isSignup = view === "signup";

    const handleChange = (field) => (e) => {
        setForm((prev) => ({ ...prev, [field]: e.target.value }));
        if (errorMsg) setErrorMsg("");
        if (infoMsg) setInfoMsg("");
    };

    const switchView = (nextView) => {
        setView(nextView);
        setErrorMsg("");
        setInfoMsg("");
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg("");
        setInfoMsg("");
        setLoading(true);

        try {
            if (isSignup) {
                const { error } = await supabase.auth.signUp({
                    email: form.email,
                    password: form.password,
                    options: { data: { name: form.name } },
                });
                if (error) throw error;
                // If email confirmation is enabled in your Supabase project (default),
                // there's no session yet at this point — the user needs to click the
                // confirmation link before onAuthStateChange fires. If you've turned
                // confirmation off, a session is created immediately and the gate
                // will switch to <App /> on its own.
                setInfoMsg("Check your email to confirm your account before signing in.");
            } else {
                const { error } = await supabase.auth.signInWithPassword({
                    email: form.email,
                    password: form.password,
                });
                if (error) throw error;
                // Success: onAuthStateChange in AppGate picks this up automatically.
            }
        } catch (err) {
            setErrorMsg(err.message || "Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleSignIn = async () => {
        setErrorMsg("");
        setInfoMsg("");
        const { error } = await supabase.auth.signInWithOAuth({
            provider: "google",
            options: { redirectTo: window.location.origin },
        });
        // On success this redirects the browser away to Google, so there's
        // nothing further to do here. Only reachable on immediate failure
        // (e.g. Google provider not enabled in Supabase yet).
        if (error) setErrorMsg(error.message);
    };

    const handleForgotPassword = async () => {
        setErrorMsg("");
        setInfoMsg("");
        if (!form.email) {
            setErrorMsg("Enter your email above first, then click 'Forgot password?'");
            return;
        }
        const { error } = await supabase.auth.resetPasswordForEmail(form.email, {
            redirectTo: window.location.origin,
        });
        if (error) {
            setErrorMsg(error.message);
        } else {
            setInfoMsg("Password reset email sent — check your inbox.");
        }
    };

    return (
        <div
            style={{
                width: "100%",
                minHeight: "100dvh",
                height: isMobile ? "auto" : "100vh",
                display: "flex",
                flexDirection: isMobile ? "column" : "row",
                background: t.bg,
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            }}
        >
            <style>{`
                * { box-sizing: border-box; }
                .derm-auth-input::placeholder { color: ${t.subtext}; opacity: 0.8; }
                .derm-auth-input:focus { outline: none; border-color: ${palette.core} !important; }
                .derm-auth-btn:hover { filter: brightness(1.04); }
                .derm-auth-btn:disabled { opacity: 0.6; cursor: default !important; filter: none; }
                .derm-auth-link:hover { text-decoration: underline; }
            `}</style>

            {/* Left brand panel — full hero on desktop. On mobile it's dropped
                entirely (rather than shrunk into a banner) so the form itself
                is the first thing in view, no scrolling past a hero needed. */}
            {!isMobile && (
                <div
                    style={{
                        flex: "0 0 46%",
                        minWidth: 360,
                        position: "relative",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                        padding: "40px 48px",
                        background: `radial-gradient(140% 120% at 15% 10%, ${palette.core} 0%, ${palette.deep} 55%, #1B1015 100%)`,
                        overflow: "hidden",
                    }}
                >
                    <div
                        style={{
                            position: "absolute",
                            width: 420,
                            height: 420,
                            borderRadius: "50%",
                            background: `radial-gradient(circle, ${palette.mist}22 0%, transparent 70%)`,
                            top: -140,
                            right: -140,
                            pointerEvents: "none",
                        }}
                    />

                    <div style={{ display: "flex", alignItems: "center", gap: 10, position: "relative" }}>
                        <div
                            style={{
                                width: 34,
                                height: 34,
                                borderRadius: 10,
                                background: "rgba(255,255,255,0.14)",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                            }}
                        >
                            <Sparkles size={17} color="#fff" strokeWidth={2} />
                        </div>
                        <span style={{ fontWeight: 700, fontSize: 15, color: "#fff" }}>DermBot</span>
                    </div>

                    <div style={{ position: "relative" }}>
                        <div
                            style={{
                                fontSize: 38,
                                fontWeight: 800,
                                color: "#fff",
                                lineHeight: 1.15,
                                letterSpacing: "-0.02em",
                                marginBottom: 14,
                            }}
                        >
                            Clearer skin
                            <br />
                            starts with
                            <br />
                            good answers.
                        </div>
                        <div style={{ fontSize: 14.5, color: "rgba(255,255,255,0.75)", maxWidth: 340 }}>
                            Ask about a condition, get a considered answer, and know when it's time to see a dermatologist.
                        </div>
                    </div>

                    <div style={{ display: "flex", gap: 28, position: "relative" }}>
                        <StatBlock value="24/7" label="Available" />
                        <StatBlock value="Private" label="Your conversations" />
                    </div>
                </div>
            )}

            {/* Right form panel */}
            <div
                style={{
                    flex: 1,
                    position: "relative",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: isMobile ? "64px 20px 40px" : 24,
                    minHeight: isMobile ? "100dvh" : "auto",
                }}
            >
                <button
                    onClick={onToggleMode}
                    aria-label="Toggle dark mode"
                    style={{
                        position: isMobile ? "absolute" : "absolute",
                        top: isMobile ? 20 : 24,
                        right: isMobile ? 20 : 28,
                        width: 38,
                        height: 38,
                        borderRadius: 999,
                        border: `1px solid ${t.inputBorder}`,
                        background: t.panel,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: "pointer",
                        zIndex: 2,
                    }}
                >
                    {mode === "light" ? <Moon size={17} color={t.text} /> : <Sun size={17} color={t.text} />}
                </button>

                <div style={{ width: "100%", maxWidth: 380 }}>
                    {isMobile && (
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 22 }}>
                            <div
                                style={{
                                    width: 30,
                                    height: 30,
                                    borderRadius: 9,
                                    background: `linear-gradient(135deg, ${palette.sky}, ${palette.deep})`,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    flexShrink: 0,
                                }}
                            >
                                <Sparkles size={15} color="#fff" strokeWidth={2} />
                            </div>
                            <span style={{ fontWeight: 700, fontSize: 14.5, color: t.text }}>DermBot</span>
                        </div>
                    )}
                    <div style={{ fontSize: isMobile ? 22 : 26, fontWeight: 800, color: t.text, marginBottom: 6 }}>
                        {isSignup ? "Create your account" : "Welcome back"}
                    </div>
                    <div style={{ fontSize: 14, color: t.subtext, marginBottom: 26 }}>
                        {isSignup
                            ? "Sign up to save your conversations and preferences."
                            : "Sign in to keep your preferences and history."}
                    </div>

                    <button
                        type="button"
                        onClick={handleGoogleSignIn}
                        disabled={loading}
                        className="derm-auth-btn"
                        style={{
                            width: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: 10,
                            padding: "12px 16px",
                            borderRadius: 12,
                            border: `1px solid ${t.inputBorder}`,
                            background: t.inputBg,
                            color: t.text,
                            fontSize: 14,
                            fontWeight: 600,
                            cursor: loading ? "default" : "pointer",
                            marginBottom: 20,
                        }}
                    >
                        <GoogleGlyph />
                        Continue with Google
                    </button>

                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
                        <div style={{ flex: 1, height: 1, background: t.divider }} />
                        <span style={{ fontSize: 12, color: t.subtext }}>or</span>
                        <div style={{ flex: 1, height: 1, background: t.divider }} />
                    </div>

                    {errorMsg && (
                        <div
                            style={{
                                fontSize: 13,
                                color: "#B3261E",
                                background: "#FBE9E7",
                                border: "1px solid #F3C6C0",
                                borderRadius: 10,
                                padding: "10px 12px",
                                marginBottom: 16,
                            }}
                        >
                            {errorMsg}
                        </div>
                    )}
                    {infoMsg && (
                        <div
                            style={{
                                fontSize: 13,
                                color: palette.deep,
                                background: palette.mist,
                                border: `1px solid ${palette.pale}`,
                                borderRadius: 10,
                                padding: "10px 12px",
                                marginBottom: 16,
                            }}
                        >
                            {infoMsg}
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        {isSignup && (
                            <div style={{ marginBottom: 16 }}>
                                <label style={{ fontSize: 12.5, color: t.subtext, display: "block", marginBottom: 6 }}>
                                    Name
                                </label>
                                <input
                                    type="text"
                                    required
                                    value={form.name}
                                    onChange={handleChange("name")}
                                    placeholder="Jane Doe"
                                    className="derm-auth-input"
                                    style={inputStyle(t)}
                                />
                            </div>
                        )}

                        <div style={{ marginBottom: 16 }}>
                            <label style={{ fontSize: 12.5, color: t.subtext, display: "block", marginBottom: 6 }}>
                                Email
                            </label>
                            <input
                                type="email"
                                required
                                value={form.email}
                                onChange={handleChange("email")}
                                placeholder="you@example.com"
                                className="derm-auth-input"
                                style={inputStyle(t)}
                            />
                        </div>

                        <div style={{ marginBottom: isSignup ? 20 : 10 }}>
                            <label style={{ fontSize: 12.5, color: t.subtext, display: "block", marginBottom: 6 }}>
                                Password
                            </label>
                            <div style={{ position: "relative" }}>
                                <input
                                    type={showPassword ? "text" : "password"}
                                    required
                                    minLength={6}
                                    value={form.password}
                                    onChange={handleChange("password")}
                                    placeholder="Enter your password"
                                    className="derm-auth-input"
                                    style={{ ...inputStyle(t), paddingRight: 40 }}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword((s) => !s)}
                                    aria-label={showPassword ? "Hide password" : "Show password"}
                                    style={{
                                        position: "absolute",
                                        right: 10,
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                        border: "none",
                                        background: "transparent",
                                        cursor: "pointer",
                                        display: "flex",
                                        alignItems: "center",
                                    }}
                                >
                                    {showPassword ? (
                                        <EyeOff size={16} color={t.subtext} />
                                    ) : (
                                        <Eye size={16} color={t.subtext} />
                                    )}
                                </button>
                            </div>
                        </div>

                        {!isSignup && (
                            <div style={{ textAlign: "right", marginBottom: 20 }}>
                                <button
                                    type="button"
                                    onClick={handleForgotPassword}
                                    className="derm-auth-link"
                                    style={{
                                        fontSize: 12.5,
                                        color: t.subtext,
                                        background: "none",
                                        border: "none",
                                        padding: 0,
                                        cursor: "pointer",
                                    }}
                                >
                                    Forgot password?
                                </button>
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className="derm-auth-btn"
                            style={{
                                width: "100%",
                                padding: "13px 16px",
                                borderRadius: 12,
                                border: "none",
                                background: `linear-gradient(135deg, ${palette.sky}, ${palette.core})`,
                                color: "#fff",
                                fontSize: 14.5,
                                fontWeight: 700,
                                cursor: loading ? "default" : "pointer",
                            }}
                        >
                            {loading ? "Please wait…" : isSignup ? "Create account" : "Sign in"}
                        </button>
                    </form>

                    <div style={{ textAlign: "center", marginTop: 20, fontSize: 13.5, color: t.subtext }}>
                        {isSignup ? "Already have an account? " : "Don't have an account? "}
                        <button
                            onClick={() => switchView(isSignup ? "signin" : "signup")}
                            className="derm-auth-link"
                            style={{
                                background: "none",
                                border: "none",
                                padding: 0,
                                color: palette.core,
                                fontWeight: 600,
                                fontSize: 13.5,
                                cursor: "pointer",
                            }}
                        >
                            {isSignup ? "Sign in" : "Sign up"}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

function inputStyle(t) {
    return {
        width: "100%",
        boxSizing: "border-box",
        padding: "11px 14px",
        borderRadius: 10,
        border: `1px solid ${t.inputBorder}`,
        background: t.inputBg,
        color: t.text,
        fontSize: 16, // 16px avoids iOS Safari auto-zooming the page on input focus
        fontFamily: "inherit",
    };
}