import { useMutation, useQueryClient } from "@tanstack/react-query"
import { RefreshCw } from "lucide-react"
import { api, type SystemStatus } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/** Manual refresh for the box telemetry: forces an immediate ssh snapshot on
 * the backend (POST /api/system/refresh, bypassing its 8 s cache loop) and
 * writes the result straight into the ["system"] query cache. */
export function RefreshSystemButton({ className }: { className?: string }) {
  const queryClient = useQueryClient()
  const refresh = useMutation({
    mutationFn: () => api.send<SystemStatus>("/api/system/refresh", "POST"),
    onSuccess: (snap) => queryClient.setQueryData(["system"], snap),
  })
  return (
    <Button
      variant="outline"
      size="sm"
      className={className}
      disabled={refresh.isPending}
      onClick={() => refresh.mutate()}
      title="Poll the GPU box now (bypasses the cached snapshot)"
    >
      <RefreshCw className={cn(refresh.isPending && "animate-spin")} />
      {refresh.isPending ? "Polling box…" : "Refresh"}
    </Button>
  )
}
