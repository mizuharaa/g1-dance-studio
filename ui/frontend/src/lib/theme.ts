// Light/dark theme, class-based (Tailwind `darkMode: "class"` + `.dark` on <html>).
// An inline script in index.html applies the stored/system theme BEFORE React mounts to
// avoid a flash; this hook keeps React in sync and persists the operator's choice.
import { useCallback, useEffect, useState } from "react"

export type Theme = "light" | "dark"
const STORAGE_KEY = "g1-theme"

export function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === "light" || stored === "dark") return stored
  } catch {
    /* localStorage blocked — fall through to system preference */
  }
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return "dark"
  }
  return "light"
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement
  root.classList.toggle("dark", theme === "dark")
  root.style.colorScheme = theme
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    /* ignore persistence failure */
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)
  useEffect(() => applyTheme(theme), [theme])
  const toggle = useCallback(() => setTheme((current) => (current === "dark" ? "light" : "dark")), [])
  return { theme, toggle, setTheme }
}
