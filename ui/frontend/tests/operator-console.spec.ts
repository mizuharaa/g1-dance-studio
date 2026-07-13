import { expect, test, type Page } from "@playwright/test"
import { fileURLToPath } from "node:url"

const evidence = (name: string) => fileURLToPath(new URL(`../../../docs/ui_revamp/${name}`, import.meta.url))

async function openConsole(page: Page) {
  await page.goto("/")
  await expect(page.getByRole("heading", { name: "Operator overview" })).toBeVisible()
  await page.waitForTimeout(1_500)
}

test("real dashboard renders at 1440", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await openConsole(page)
  await expect(page.getByTestId("live-run-card")).toBeVisible()
  await page.screenshot({ path: evidence("dashboard-1440.png"), fullPage: true })
})

test("tablet and half-screen layouts remain usable", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 900 })
  await openConsole(page)
  await page.getByRole("button", { name: "Dances & stats" }).click()
  await expect(page.getByRole("heading", { name: "Dances & stats" })).toBeVisible()
  await page.screenshot({ path: evidence("dances-1024.png"), fullPage: true })

  await page.setViewportSize({ width: 768, height: 900 })
  await page.getByRole("button", { name: "Refresh data" }).click()
  await page.getByRole("button", { name: "Menu" }).click()
  await page.getByRole("button", { name: "Pipeline studio" }).click()
  await expect(page.getByRole("heading", { name: "Pipeline studio" })).toBeVisible()
  await page.screenshot({ path: evidence("pipeline-768.png"), fullPage: true })
})

test("upload interaction posts multipart and selects the new job", async ({ page }) => {
  await page.route("**/api/jobs/upload", async (route) => {
    expect(route.request().method()).toBe("POST")
    await route.fulfill({ json: {
      id: "ui-e2e-upload",
      name: "venue-test",
      created_at: Date.now() / 1000,
      input: { type: "video", source: "venue-test.mp4" },
      current_stage: "extract",
      stages: {
        extract: { state: "running", progress: 0.15, message: "normalizing video" },
        retarget: { state: "pending", progress: 0 }, train: { state: "pending", progress: 0 },
        verify: { state: "pending", progress: 0 }, export: { state: "pending", progress: 0 },
      },
    } })
  })
  await page.route("**/api/jobs/ui-e2e-upload", (route) => route.fulfill({ status: 404, json: { detail: "mock summary only" } }))
  await openConsole(page)
  await page.getByRole("button", { name: "Pipeline studio" }).click()
  const picker = page.locator('input[type="file"]')
  await picker.setInputFiles({ name: "venue-test.mp4", mimeType: "video/mp4", buffer: Buffer.from("mock-video") })
  await expect(page.getByText("Pipeline job created")).toBeVisible()
})

test("running state keeps an oversized STOP visible and sends stop", async ({ page }) => {
  await page.route("**/api/shows/runs/current", async (route) => {
    if (route.request().method() === "POST") {
      return route.fulfill({ json: { stopped: true, was_running: true, detail: "STOP sent — robot damping" } })
    }
    return route.fulfill({ json: {
      running: true,
      show_id: "ui-e2e-show",
      dance_id: "20260704-18f65bbd",
      mode: "live",
      phase: "performing",
      fall_detected: false,
      started_at: Date.now() / 1000 - 12,
      last_lines: ["SHOW RUN", "starting leg-odometry policy"],
    } })
  })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await openConsole(page)
  await expect(page.getByTestId("global-stop")).toBeVisible()
  await expect(page.getByTestId("stop-show")).toBeVisible()
  await page.screenshot({ path: evidence("dashboard-running-stop.png"), fullPage: true })
  const stopRequest = page.waitForRequest((request) => request.url().includes("/api/shows/runs/current") && request.method() === "POST")
  await page.getByTestId("global-stop").getByRole("button", { name: "STOP SHOW" }).click()
  await expect((await stopRequest).method()).toBe("POST")
})

test("typed confirmation stays locked until exact phrase", async ({ page }) => {
  await page.route("**/api/shows", (route) => route.fulfill({ json: [] }))
  // Deterministic: guarantee a show-ready dance so the run gate renders regardless of the
  // mutable local library state (Thriller drifts between sim-verified and show-ready).
  await page.route("**/api/dances", (route) => route.fulfill({ json: [{
    id: "ui-e2e-ready", name: "E2E Ready", created_at: Date.now() / 1000, updated_at: Date.now() / 1000,
    status: "show-ready", duration_s: 30, audio: { track: "data/audio/song.wav" },
    policy_sha256: "abc1234567", repeatability: { consecutive_clean: 3 }, repeatability_target: 3,
  }] }))
  await openConsole(page)
  await page.getByRole("button", { name: "Show mode" }).click()
  await expect(page.getByRole("heading", { name: "Show mode" })).toBeVisible()
  await expect(page.getByTestId("show-warning-gate")).toBeVisible()
  await page.getByPlaceholder("Operator name").fill("Venue operator")
  const confirmation = page.getByTestId("run-confirmation")
  const start = page.getByTestId("start-show")
  await confirmation.fill("I HAVE THE REMOTE")
  await expect(start).toBeDisabled()
  await confirmation.fill("I AM PRESENT WITH THE DAMPING REMOTE")
  await expect(start).toBeEnabled()
  await page.screenshot({ path: evidence("show-mode-1280.png"), fullPage: true })
})

test("simulation stage opens the preview video", async ({ page }) => {
  await openConsole(page)
  await page.getByRole("button", { name: "Dances & stats" }).click()
  await expect(page.getByRole("heading", { name: "Dances & stats" })).toBeVisible()
  const preview = page.getByTestId("preview-open").first()
  await expect(preview).toBeVisible()
  await preview.click()
  await expect(page.getByRole("dialog")).toBeVisible()
  await expect(page.getByTestId("preview-video")).toBeVisible()
  await page.screenshot({ path: evidence("preview-video-open.png"), fullPage: true })
})

test("dark mode toggle flips the theme and persists", async ({ page }) => {
  await openConsole(page)
  const html = page.locator("html")
  await expect(html).not.toHaveClass(/dark/)
  await page.getByRole("button", { name: "Toggle dark mode" }).click()
  await expect(html).toHaveClass(/dark/)
  await page.screenshot({ path: evidence("dashboard-dark-1280.png"), fullPage: true })
  // persisted choice survives a reload
  await page.reload()
  await expect(page.locator("html")).toHaveClass(/dark/)
  // and back to light
  await page.getByRole("button", { name: "Toggle dark mode" }).click()
  await expect(page.locator("html")).not.toHaveClass(/dark/)
})

test("safety screen shows robot state and fires the E-STOP", async ({ page }) => {
  await page.route("**/api/safety/status", (route) =>
    route.fulfill({ json: { robot_reachable: false, run: { running: false, phase: "idle", last_lines: [] } } }))
  let estopHit = false
  await page.route("**/api/safety/estop", (route) => {
    estopHit = true
    return route.fulfill({ json: { stopped: false, tracked_stopped: false, strays_signaled: [], was_running: false, detail: "No app-launched policy run is active — use the remote B-damp." } })
  })
  await openConsole(page)
  await page.getByRole("button", { name: "Safety & E-stop" }).click()
  await expect(page.getByRole("heading", { name: "Safety & E-stop" })).toBeVisible()
  await expect(page.getByTestId("robot-state-viz")).toBeVisible()
  await expect(page.getByText("Feet flat and fully loaded on the ground BEFORE you arm")).toBeVisible()
  await page.screenshot({ path: evidence("safety-1280.png"), fullPage: true })
  const estopRequest = page.waitForRequest((r) => r.url().includes("/api/safety/estop") && r.method() === "POST")
  await page.getByTestId("estop-button").click()
  await estopRequest
  expect(estopHit).toBe(true)
})

test("an always-available compact E-STOP sits in the top bar", async ({ page }) => {
  await openConsole(page)
  await expect(page.getByTestId("estop-compact")).toBeVisible()
})

test("preview dialog offers an open-in-browser fallback", async ({ page }) => {
  await openConsole(page)
  await page.getByRole("button", { name: "Dances & stats" }).click()
  const preview = page.getByTestId("preview-open").first()
  await expect(preview).toBeVisible()
  await preview.click()
  await expect(page.getByRole("dialog")).toBeVisible()
  await expect(page.getByTestId("preview-video")).toBeVisible()
  // the codec-independent escape hatch is always present in the dialog footer
  await expect(page.getByRole("dialog").getByRole("button", { name: "Open in browser" }).first()).toBeVisible()
})

test("audit filters incidents and per-dance records", async ({ page }) => {
  await openConsole(page)
  await page.getByRole("button", { name: "Audit log" }).click()
  await expect(page.getByRole("heading", { name: "Audit log" })).toBeVisible()
  await page.getByTestId("audit-type-filter").selectOption("incident")
  await expect(page.getByText(/incident/i).first()).toBeVisible()
  const danceOptions = await page.getByTestId("audit-dance-filter").locator("option").count()
  expect(danceOptions).toBeGreaterThan(1)
  await page.getByTestId("audit-search").fill("thriller")
  await expect(page.getByText("Operational evidence")).toBeVisible()
})
