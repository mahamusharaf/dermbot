import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// A single shared client — import this everywhere instead of creating new clients.
export const supabase = createClient(supabaseUrl, supabaseAnonKey);