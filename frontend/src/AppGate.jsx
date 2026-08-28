import React, { useEffect, useState } from "react";
import { supabase } from "./supabaseClient";
import AuthPage from "./AuthPage";
import App from "./App";

/*
 * This is the actual entry point — point your main.jsx / index.jsx at
 * <AppGate /> instead of <App /> directly.
 *
 * It owns the single source of truth for "is anyone logged in": it checks
 * for an existing session on load, then subscribes to Supabase's
 * onAuthStateChange so sign-in/sign-up/OAuth/sign-out/token-refresh in
 * AuthPage.jsx (or anywhere else) are all picked up automatically without
 * AuthPage needing to know anything about App, or vice versa.
 *
 * It also owns light/dark mode now, for the same reason: AuthPage and App
 * used to each keep their own local `mode` state, so toggling dark mode on
 * the login screen had no effect once AppGate swapped you into <App /> —
 * that was a brand-new component instance starting back at "light". Theme
 * lives here instead and is passed down as a prop, and persisted to
 * localStorage so a refresh doesn't flash back to light mode either.
 */

const THEME_STORAGE_KEY = "dermbot-theme";

export default function AppGate() {
    const [session, setSession] = useState(undefined); // undefined = "still checking"
    const [mode, setMode] = useState(() => {
        try {
            return localStorage.getItem(THEME_STORAGE_KEY) || "light";
        } catch {
            return "light";
        }
    });

    useEffect(() => {
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session);
        });

        const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session);
        });

        return () => listener.subscription.unsubscribe();
    }, []);

    useEffect(() => {
        try {
            localStorage.setItem(THEME_STORAGE_KEY, mode);
        } catch {
            // localStorage unavailable (e.g. private browsing) — theme just
            // won't persist across reloads, which is fine to fail silently on.
        }
    }, [mode]);

    const toggleMode = () => setMode((m) => (m === "light" ? "dark" : "light"));

    if (session === undefined) {
        // Still checking for an existing session (e.g. page refresh with a
        // valid stored token) — avoid a flash of the sign-in page.
        return null; // or a small loading spinner if you prefer
    }

    if (!session) {
        return <AuthPage mode={mode} onToggleMode={toggleMode} />;
    }

    // session.access_token is the JWT api.py's get_current_user() verifies.
    // session.user.email / session.user.id are handy for a profile menu later.
    return (
        <App
            accessToken={session.access_token}
            user={session.user}
            onSignOut={() => supabase.auth.signOut()}
            mode={mode}
            onToggleMode={toggleMode}
        />
    );
}