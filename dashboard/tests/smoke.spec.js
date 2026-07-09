import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

// Screenshots are written into the repo's docs/img so the README can embed them.
const SHOTS = resolve(dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "../../docs/img");
try { mkdirSync(SHOTS, { recursive: true }); } catch { /* exists */ }

test.describe("dashboard smoke", () => {
  test("showcase page renders", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/Multi-Agent Trading System/i);
    // Hero headline + nav.
    await expect(page.getByRole("heading", { name: /autonomous multi-agent/i })).toBeVisible();
    await expect(page.getByRole("link", { name: "Live Dashboard", exact: true })).toBeVisible();
    // Architecture diagram (SVG) is present.
    await expect(page.getByRole("img", { name: /architecture diagram/i })).toBeVisible();
    // Sections.
    await expect(page.getByRole("heading", { name: /^Architecture$/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: /How it works/i })).toBeVisible();
    await page.waitForTimeout(1200); // let any eval fetch settle
    await page.screenshot({ path: `${SHOTS}/showcase.png`, fullPage: true });
  });

  test("live dashboard page renders", async ({ page }) => {
    await page.goto("/live");
    await expect(page.getByRole("heading", { name: /Live paper-trading dashboard/i })).toBeVisible();
    // Stat cards + sections render even before data arrives.
    await expect(page.getByText(/Equity/i).first()).toBeVisible();
    await expect(page.getByRole("heading", { name: /Agent health/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Live activity/i })).toBeVisible();
    await page.waitForTimeout(1500); // allow first poll + ws connect
    await page.screenshot({ path: `${SHOTS}/live.png`, fullPage: true });
  });
});
