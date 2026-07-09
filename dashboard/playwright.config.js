import { defineConfig, devices } from "@playwright/test";

// Smoke tests + screenshots. Locally, if the dockerized dashboard is already on
// :3000 (full stack with live API), Playwright reuses it and captures real data.
// In CI (nothing on :3000), it starts `npm run preview` to serve the built SPA —
// the app renders graceful loading/empty states without the API, which is enough
// to smoke-test that the build is healthy and the pages render.
export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  expect: { timeout: 8000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: process.env.BASE_URL || "http://localhost:3000",
    viewport: { width: 1366, height: 900 },
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run preview",
    port: 3000,
    reuseExistingServer: true,
    timeout: 60000,
  },
});
