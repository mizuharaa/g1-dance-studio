import { defineConfig } from "@playwright/test"

export default defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  expect: { timeout: 8_000 },
  use: {
    baseURL: process.env.PW_BASE_URL ?? "http://127.0.0.1:8735",
    colorScheme: "light",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  reporter: [["list"]],
})
