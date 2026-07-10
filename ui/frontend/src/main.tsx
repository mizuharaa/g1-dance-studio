import React from "react"
import ReactDOM from "react-dom/client"
import * as Tooltip from "@radix-ui/react-tooltip"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Toaster } from "sonner"
import App from "./App"
import "./index.css"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 5_000, retry: 1, refetchOnWindowFocus: false },
  },
})

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <Tooltip.Provider delayDuration={250}>
        <App />
        <Toaster theme="light" richColors position="top-right" closeButton />
      </Tooltip.Provider>
    </QueryClientProvider>
  </React.StrictMode>,
)
