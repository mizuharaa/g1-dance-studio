import { useQuery } from "@tanstack/react-query"
import { api, type Dance, type PipelineJob, type RunStatus, type SetList, type Show, type SystemStatus, type Venue } from "@/lib/api"

export interface ShowPhase { phase: string; owner: string; note: string }
export interface VenueResponse { venues: Venue[]; active: Venue }

export function useConsoleData() {
  const dances = useQuery({ queryKey: ["dances"], queryFn: () => api.get<Dance[]>("/api/dances"), refetchInterval: 15_000 })
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api.get<PipelineJob[]>("/api/jobs"), refetchInterval: 5_000 })
  const shows = useQuery({ queryKey: ["shows"], queryFn: () => api.get<Show[]>("/api/shows"), refetchInterval: 7_500 })
  const setlists = useQuery({ queryKey: ["setlists"], queryFn: () => api.get<SetList[]>("/api/setlists"), refetchInterval: 15_000 })
  const system = useQuery({ queryKey: ["system"], queryFn: () => api.get<SystemStatus>("/api/system"), refetchInterval: 20_000 })
  const venues = useQuery({ queryKey: ["venues"], queryFn: () => api.get<VenueResponse>("/api/venues"), refetchInterval: 30_000 })
  const phases = useQuery({ queryKey: ["show-phases"], queryFn: () => api.get<{ phases: ShowPhase[] }>("/api/show-phases"), staleTime: 300_000 })
  const run = useQuery({
    queryKey: ["current-run"],
    queryFn: () => api.get<RunStatus>("/api/shows/runs/current"),
    refetchInterval: (query) => query.state.data?.running ? 1_000 : 4_000,
  })

  return {
    dances: dances.data ?? [],
    jobs: jobs.data ?? [],
    shows: shows.data ?? [],
    setlists: setlists.data ?? [],
    system: system.data,
    venues: venues.data,
    phases: phases.data?.phases ?? [],
    run: run.data ?? { running: false, phase: "idle" },
    loading: dances.isLoading || jobs.isLoading || shows.isLoading,
    error: dances.error || jobs.error || shows.error || system.error,
    refetchAll: () => Promise.all([dances.refetch(), jobs.refetch(), shows.refetch(), setlists.refetch(), system.refetch(), venues.refetch(), run.refetch()]),
  }
}

export type ConsoleData = ReturnType<typeof useConsoleData>
