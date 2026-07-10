import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const fmtDuration = (seconds?: number | null) => {
  if (!seconds && seconds !== 0) return "—"
  const rounded = Math.max(0, Math.round(seconds))
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`
}

export const fmtMoney = (amount?: number | null) => {
  if (amount == null) return "—"
  if (amount >= 1_000_000) return `${(amount / 1_000_000).toFixed(2)}M ₫`
  if (amount >= 1_000) return `${Math.round(amount / 1_000)}K ₫`
  return `${Math.round(amount)} ₫`
}

export const fmtPercent = (value?: number | null, digits = 0) =>
  value == null ? "—" : `${(value * 100).toFixed(digits)}%`

export const fmtDate = (epoch?: number | null, withTime = true) => {
  if (!epoch) return "—"
  const date = new Date(epoch * 1000)
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  }).format(date)
}

export const shortHash = (value?: string | null) => value ? value.slice(0, 10) : "—"
