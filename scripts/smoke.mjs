// Browser smoke test: every implemented simulation page must reach the "ready" state on a real
// WebGPU device without GPU errors or console errors.
//
//   pnpm build && pnpm start &            # serve the production build on :3000
//   node scripts/smoke.mjs                # BASE_URL=http://localhost:3000 by default
//
// Needs a Chromium build (`npx playwright-core install chromium`). Without a GPU the CPU software
// Vulkan renderer is used (SwiftShader); the full "chromium" channel (new headless mode) is required —
// the older headless shell cannot run vgpu's pipelines.
import { chromium } from "playwright-core";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:3000";
const PAGES = ["/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata"];

const browser = await chromium.launch({
  channel: "chromium",
  args: [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
    "--enable-features=Vulkan",
    "--use-vulkan=swiftshader",
    "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader",
  ],
});

let failed = 0;
for (const path of PAGES) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const problems = [];
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(m.text());
  });
  page.on("pageerror", (e) => problems.push(e.message));

  await page.goto(BASE_URL + path);
  const canvas = page.locator("canvas");
  await canvas.waitFor({ timeout: 15_000 });
  // Give the GPU time to initialise and draw some frames; the overlay disappears once ready.
  await page.waitForTimeout(4000);
  const overlay = (await page.locator("[role=status], [role=alert]").allTextContents()).join(" ").trim();
  const ok = !overlay && problems.length === 0;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${path}${overlay ? `  overlay: ${overlay}` : ""}${problems.length ? `\n     ${problems.join("\n     ")}` : ""}`);
  await page.close();
}
await browser.close();
process.exit(failed ? 1 : 0);
