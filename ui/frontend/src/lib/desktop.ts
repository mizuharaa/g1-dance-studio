// Bridge to the native desktop shell (ui/desktop.py). pywebview injects
// `window.pywebview.api` once the native window is ready (after a `pywebviewready`
// event). The bundled PySide6 QtWebEngine has NO H.264/AAC codecs, so inline <video>
// previews cannot decode there — `openExternal` hands the preview URL to the operator's
// real system browser (which has the codecs) so previews are always watchable.
import { useEffect, useState } from "react"

declare global {
  interface Window {
    pywebview?: {
      api?: {
        open_external?: (url: string) => Promise<boolean>
        is_desktop?: () => Promise<boolean>
      }
    }
  }
}

export function isDesktopShell(): boolean {
  return typeof window !== "undefined" && !!window.pywebview?.api?.open_external
}

/** Turn a server-relative path (`/previews/x.mp4`) into an absolute URL the system
 *  browser can open. Absolute http(s) URLs pass through unchanged. */
export function absoluteUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path
  const origin = typeof window !== "undefined" ? window.location.origin : ""
  return `${origin}${path.startsWith("/") ? "" : "/"}${path}`
}

/** Open a preview in the real system browser. Uses the pywebview bridge in the desktop
 *  shell; falls back to a normal new tab in a plain browser (dev / Playwright). */
export async function openExternal(path: string): Promise<void> {
  const url = absoluteUrl(path)
  const api = typeof window !== "undefined" ? window.pywebview?.api : undefined
  if (api?.open_external) {
    try {
      await api.open_external(url)
      return
    } catch {
      /* bridge failed — fall through to a normal open */
    }
  }
  window.open(url, "_blank", "noopener,noreferrer")
}

/** True once the native desktop bridge is available. Tracks the async `pywebviewready`
 *  injection so components re-render when the shell finishes wiring up. */
export function useDesktopShell(): boolean {
  const [ready, setReady] = useState(isDesktopShell)
  useEffect(() => {
    if (ready) return
    const mark = () => isDesktopShell() && setReady(true)
    window.addEventListener("pywebviewready", mark)
    const timer = window.setInterval(mark, 500)
    return () => {
      window.removeEventListener("pywebviewready", mark)
      window.clearInterval(timer)
    }
  }, [ready])
  return ready
}
