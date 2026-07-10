import type { Dance } from "@/lib/api"

export function normalizePreviewUrl(value?: string | null) {
  if (!value) return null
  if (value.startsWith("/")) return value
  const normalized = value.replaceAll("\\", "/")
  const marker = "/previews/"
  const index = normalized.indexOf(marker)
  if (index >= 0) return normalized.slice(index)
  return `/previews/${normalized.split("/").pop()}`
}

export function dancePreviewUrl(dance?: Dance | null) {
  if (!dance) return null
  const muxed = dance.audio?.muxed_preview
  return normalizePreviewUrl(typeof muxed === "string" ? muxed : dance.preview)
}
