// End-to-end browser suite. Runs every simulation in a real Chromium (new headless mode, WebGPU via
// the SwiftShader/Vulkan software renderer when there is no GPU), plus a "no WebGPU" and a phone-sized
// pass. Works against a production server, a dev server (React strict mode double-mounts effects!) or
// a static export served by any file server.
//
//   pnpm build && pnpm start &        # then:
//   BASE_URL=http://localhost:3000 pnpm e2e
//
// Needs a Chromium build:  npx playwright-core install chromium
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { chromium } from "playwright-core";
import zlib from "node:zlib";

const BASE_URL = (process.env.BASE_URL ?? "http://localhost:3000").replace(/\/$/, "");
const SHOTS = process.env.SHOTS_DIR; // optional: save screenshots here
// Set E2E_DEV=1 when testing `next dev`: its bundles are unminified, so the size budgets do not apply.
const IS_DEV = process.env.E2E_DEV === "1";
// E2E_GREP=<regex> runs only the scenarios whose name matches (handy while working on one simulation).
const GREP = process.env.E2E_GREP ? new RegExp(process.env.E2E_GREP, "i") : undefined;
// Static exports use trailing slashes; harmless elsewhere.
const url = (p) => `${BASE_URL}${p}`;

const browser = await chromium.launch({
  channel: "chromium", // full Chromium: the old headless shell cannot run vgpu's pipelines
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

const results = [];
const IGNORED_ERRORS = [/WebGPU is experimental/i, /Download the React DevTools/i];

/** Runs `body` in a fresh page; fails the test on any console error or uncaught exception. */
async function test(name, body, { context = {}, initScript, allowErrors = [], adaptive = false } = {}) {
  if (GREP && !GREP.test(name)) return;
  const ctx = await browser.newContext({
    viewport: { width: 1360, height: 860 },
    acceptDownloads: true,
    permissions: ["clipboard-read", "clipboard-write"],
    ...context,
  });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !IGNORED_ERRORS.some((r) => r.test(m.text())) && !allowErrors.some((r) => r.test(m.text()))) {
      errors.push(m.text());
    }
  });
  page.on("pageerror", (e) => errors.push(`uncaught: ${e.message}`));
  // Pixel-exact checks need a stable resolution, so pin full quality unless the scenario is about adaptivity.
  if (!adaptive) await page.addInitScript(() => localStorage.setItem("edusim:quality", "full"));
  if (initScript) await page.addInitScript(initScript);
  const started = Date.now();
  let failure;
  try {
    await body(page);
    if (errors.length) failure = `console errors:\n      ${[...new Set(errors)].slice(0, 5).join("\n      ")}`;
  } catch (e) {
    failure = e.message.split("\n").slice(0, 6).join("\n      ");
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, `FAIL-${name.replace(/\W+/g, "_")}.png`) }).catch(() => {});
  }
  results.push({ name, ok: !failure });
  console.log(`${failure ? "FAIL" : "ok  "} ${name} (${((Date.now() - started) / 1000).toFixed(1)}s)${failure ? `\n      ${failure}` : ""}`);
  await ctx.close();
}

const assert = (cond, message) => {
  if (!cond) throw new Error(message);
};
const shot = (page, name) => (SHOTS ? page.screenshot({ path: path.join(SHOTS, `${name}.png`) }) : undefined);

// The overlay is identified by test id: Next.js adds its own (empty) role=alert route announcer.
const gpuProblem = (page) => page.locator("[data-testid=gpu-status][data-state=error], [data-testid=gpu-status][data-state=unsupported]");
async function assertNoGpuProblem(page, where = "") {
  const found = await gpuProblem(page).allTextContents();
  assert(found.length === 0, `GPU overlay shown${where}: ${found.join(" ")}`);
}

/** Navigates and waits for the GPU overlay to go away (the sim is ready and drawing). */
async function open(page, route) {
  await page.goto(url(route));
  await page.locator("canvas").first().waitFor({ timeout: 20_000 });
  await page.waitForSelector("[data-testid=gpu-status][data-state=loading]", { state: "detached", timeout: 30_000 });
  await page.waitForTimeout(600);
  await assertNoGpuProblem(page);
}

const canvasPixels = (page) => page.locator("canvas").first().screenshot();
// By role: a label match is a substring match and would also hit e.g. the chart's accessible name.
const slider = (page, name) => page.getByRole("slider", { name, exact: true });
/**
 * True when two PNGs show the same picture. Byte equality is too strict for a software renderer, whose
 * output can differ by one colour level between presented buffers, so small differences are tolerated.
 */
async function same(page, a, b, tolerance = 3) {
  if (Buffer.compare(a, b) === 0) return true;
  return page.evaluate(
    async ([x, y, tol]) => {
      const load = async (b64) => {
        const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
        const c = new OffscreenCanvas(img.width, img.height);
        const g = c.getContext("2d");
        g.drawImage(img, 0, 0);
        return g.getImageData(0, 0, img.width, img.height);
      };
      const A = await load(x);
      const B = await load(y);
      if (A.width !== B.width || A.height !== B.height) return false;
      for (let i = 0; i < A.data.length; i += 4) {
        const d = Math.abs(A.data[i] - B.data[i]) + Math.abs(A.data[i + 1] - B.data[i + 1]) + Math.abs(A.data[i + 2] - B.data[i + 2]);
        if (d > tol) return false;
      }
      return true;
    },
    [a.toString("base64"), b.toString("base64"), tolerance],
  );
}

/**
 * Waits until the canvas stops changing and returns that image. On a software renderer frames take
 * long enough that work submitted before a state change can still land after it, so "unchanged for
 * a moment" is only meaningful once the pipeline has drained.
 */
async function settle(page, timeout = 12_000) {
  let prev = await canvasPixels(page);
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    await page.waitForTimeout(300);
    const cur = await canvasPixels(page);
    if (await same(page, prev, cur)) return cur;
    prev = cur;
  }
  throw new Error("canvas never settled (still animating?)");
}

/** Polls until the canvas differs from `reference`. */
async function waitForChange(page, reference, message, timeout = 10_000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (!await same(page, reference, await canvasPixels(page))) return;
    await page.waitForTimeout(250);
  }
  throw new Error(message);
}
const text = async (page, selector) => (await page.locator(selector).first().textContent()) ?? "";

// ───────────────────────────── home ─────────────────────────────
await test("home: compact editorial layout — plates link to the 4 simulations, planned ones are plain text", async (page) => {
  await page.goto(url("/"));
  assert((await page.title()).includes("EduSim"), `title: ${await page.title()}`);
  assert((await page.locator("h1").count()) === 1, "exactly one h1");
  assert((await text(page, "h1")).includes("reach into"), "headline");
  const links = await page.locator("main a[href^='/']").evaluateAll((as) => as.map((a) => a.getAttribute("href")));
  assert(links.length === 4, `expected 4 plate links, got ${links.join(",")}`);
  for (const l of await page.locator("main a").all()) assert(((await l.textContent()) ?? "").trim().length > 3, "link without text");
  assert((await page.locator("h2").count()) >= 4, "plate headings + index heading");
  for (const subject of ["Physics", "Chemistry", "Computer Science", "Biology"]) {
    assert(await page.getByRole("heading", { name: subject }).first().isVisible(), `missing index column ${subject}`);
  }
  assert(await page.getByText("Double Pendulum Chaos").first().isVisible(), "planned item not shown");
  assert((await page.locator("a:has-text('Double Pendulum')").count()) === 0, "planned simulations must not be links");
  await page.getByRole("link", { name: /Hodgkin/ }).click();
  await page.waitForURL(/hodgkin-huxley/);
  await page.getByRole("link", { name: "← All simulations" }).click();
  await page.waitForURL((u) => u.pathname === "/" || u.pathname === "");
  await shot(page, "home");
});

await test("home: compact — about one screen on desktop, under 1.4 on a phone", async (page) => {
  const height = async () => page.evaluate(() => document.documentElement.scrollHeight);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(url("/"));
  await page.waitForTimeout(500);
  const desktop = (await height()) / 900;
  assert(desktop <= 1.15, `desktop page is ${desktop.toFixed(2)} screens tall (budget 1.15)`);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  const phone = (await height()) / 844;
  assert(phone <= 1.4, `phone page is ${phone.toFixed(2)} screens tall (budget 1.4)`);
  console.log(`      (desktop ${desktop.toFixed(2)} screens, phone ${phone.toFixed(2)} screens)`);
});

await test("home: no horizontal overflow at 320, 390, 768, 1024 and 1440 px", async (page) => {
  for (const width of [320, 390, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto(url("/"));
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    assert(overflow <= 1, `${width}px: horizontal overflow ${overflow}px`);
  }
});

await test("home: phone index is collapsed and expands on tap", async (page) => {
  await page.goto(url("/"));
  const details = page.locator("details");
  assert((await details.count()) === 4, "four accordions");
  assert((await page.locator("details[open]").count()) === 0, "accordions should start collapsed");
  await page.locator("details summary").first().tap();
  assert((await page.locator("details[open]").count()) === 1, "tap should open one");
  assert(await page.getByText("Double Pendulum Chaos").last().isVisible(), "item visible after opening");
}, { context: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } });

await test("home: does not touch the GPU on load, and stays within size budgets", async (page) => {
  await page.addInitScript(() => {
    window.__gpuCalls = 0;
    if (navigator.gpu) {
      const original = GPU.prototype.requestAdapter;
      GPU.prototype.requestAdapter = function (...args) {
        window.__gpuCalls++;
        return original.apply(this, args);
      };
    }
  });
  await page.goto(url("/"), { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  const calls = await page.evaluate(() => window.__gpuCalls);
  assert(calls === 0, `home requested a GPU adapter ${calls} time(s) before any interaction`);

  // Sizes are measured by downloading exactly the scripts the page loaded, from Node: deterministic, unlike
  // reading response bodies inside the browser (which can fail and silently under-count).
  const scripts = await page.evaluate(() => performance.getEntriesByType("resource").map((e) => e.name).filter((n) => /\.js(\?|$)/.test(n)));
  const script = (await Promise.all(scripts.map(async (u) => (await (await fetch(u)).arrayBuffer()).byteLength))).reduce((a, b) => a + b, 0);
  const html = zlib.gzipSync(Buffer.from(await (await fetch(url("/"))).arrayBuffer())).length;
  // Baseline before the redesign: 456.6 KB (almost all of it the React + Next runtime). Must not grow.
  assert(IS_DEV || script <= 470 * 1024, `home JavaScript is ${(script / 1024).toFixed(1)} KB over ${scripts.length} files (budget 470 KB; 456.6 KB when the redesign started)`);
  assert(IS_DEV || html <= 14 * 1024, `home HTML is ${(html / 1024).toFixed(1)} KB gzipped (budget 14 KB)`);
  console.log(`      (JS ${(script / 1024).toFixed(1)} KB over ${scripts.length} files, HTML ${(html / 1024).toFixed(1)} KB gzipped, GPU adapter requests: ${calls})`);
});

await test("home: automated accessibility audit (axe) finds no violations, desktop and phone", async (page) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.goto(url("/"));
    await page.waitForTimeout(800);
    const result = await new AxeBuilder({ page }).analyze();
    const summary = result.violations.map((v) => `${v.id} (${v.nodes.length}): ${v.help}`);
    assert(summary.length === 0, `${viewport.width}px: ${summary.join(" | ")}`);
  }
});

// ───────────────────────────── home: live plates ─────────────────────────────
const PLATES = ["wave-interference", "n-body", "cellular-automata", "hodgkin-huxley"]; // catalog (= page) order
const liveCanvases = (page) => page.locator("[data-testid=plate-live]");
const waitLive = (page, id, timeout = 45_000) => page.waitForSelector(`[data-plate="${id}"] [data-testid=plate-live][data-state=ready]`, { timeout });
/** Fraction of lit pixels in a plate's live canvas (a blank canvas would be ~0). */
const litFraction = (page, id) =>
  page.evaluate(async (id) => {
    const canvas = document.querySelector(`[data-plate="${id}"] canvas`);
    const blob = await new Promise((resolve) => requestAnimationFrame(() => canvas.toBlob(resolve)));
    const bitmap = await createImageBitmap(blob);
    const ctx = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    let lit = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i] + data[i + 1] + data[i + 2] > 90) lit++;
    return lit / (bitmap.width * bitmap.height);
  }, id);

await test("home plates: hover runs the real simulation (all four), one at a time, and leaving restores the poster", async (page) => {
  await page.goto(url("/"));
  assert((await liveCanvases(page).count()) === 0, "no live canvas before any intent");
  for (const id of PLATES) {
    await page.locator(`[data-plate="${id}"]`).hover();
    await waitLive(page, id);
    assert((await liveCanvases(page).count()) === 1, `${id}: more than one live plate`);
    assert((await page.locator(`[data-plate="${id}"] img`).count()) === 1, `${id}: poster should stay underneath`);
    // "ready" arrives a tick before the first frame is presented, so poll for real content.
    let lit = 0;
    for (let i = 0; i < 30 && lit <= 0.01; i++) {
      lit = await litFraction(page, id);
      if (lit <= 0.01) await page.waitForTimeout(300);
    }
    assert(lit > 0.01, `${id}: live canvas looks blank (lit fraction ${lit.toFixed(4)})`);
  }
  // Moving from the last plate to the first hands over: the first starts, the last stops.
  await page.locator(`[data-plate="${PLATES[0]}"]`).hover();
  await waitLive(page, PLATES[0]);
  await page.waitForFunction((id) => document.querySelectorAll(`[data-plate="${id}"] canvas`).length === 0, PLATES.at(-1), { timeout: 10_000 });
  assert((await liveCanvases(page).count()) === 1, "exactly one live canvas after hand-over");
  await page.mouse.move(5, 5);
  await page.waitForFunction(() => document.querySelectorAll("[data-testid=plate-live]").length === 0, null, { timeout: 10_000 });
  await shot(page, "home-live");
});

await test("home plates: keyboard focus starts the live preview", async (page) => {
  await page.goto(url("/"));
  await page.keyboard.press("Tab"); // masthead has no links; the first tab stop is the first plate
  const focused = await page.evaluate(() => document.activeElement?.getAttribute("data-plate"));
  assert(focused === PLATES[0], `first tab stop is ${focused}`);
  await waitLive(page, PLATES[0]);
  await page.keyboard.press("Tab");
  await waitLive(page, PLATES[1]);
  await page.waitForFunction((id) => document.querySelectorAll(`[data-plate="${id}"] canvas`).length === 0, PLATES[0], { timeout: 10_000 });
});

await test("home plates: reduced motion never starts a preview and never requests the GPU", async (page) => {
  await page.addInitScript(() => {
    window.__gpuCalls = 0;
    const original = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = function (...args) {
      window.__gpuCalls++;
      return original.apply(this, args);
    };
  });
  await page.goto(url("/"));
  await page.locator(`[data-plate="${PLATES[0]}"]`).hover();
  await page.waitForTimeout(3000);
  assert((await liveCanvases(page).count()) === 0, "live preview started despite prefers-reduced-motion");
  assert((await page.evaluate(() => window.__gpuCalls)) === 0, "GPU was requested despite prefers-reduced-motion");
  assert(await page.locator(`[data-plate="${PLATES[0]}"] img`).isVisible(), "poster should remain");
}, { context: { reducedMotion: "reduce" } });

await test("home plates: without WebGPU hovering does nothing and raises no errors", async (page) => {
  await page.goto(url("/"));
  await page.locator(`[data-plate="${PLATES[1]}"]`).hover();
  await page.waitForTimeout(2000);
  assert((await liveCanvases(page).count()) === 0, "preview started without WebGPU");
  assert(await page.locator(`[data-plate="${PLATES[1]}"] img`).isVisible(), "poster should remain");
}, { initScript: () => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined, configurable: true }) });

await test("home plates: a GPU failure falls back to the poster and is not retried", async (page) => {
  await page.addInitScript(() => {
    window.__gpuCalls = 0;
    GPU.prototype.requestAdapter = async function () {
      window.__gpuCalls++;
      return null; // WebGPU exists but yields no adapter, as on some blocklisted machines
    };
  });
  await page.goto(url("/"));
  await page.locator(`[data-plate="${PLATES[0]}"]`).hover();
  await page.waitForFunction(() => window.__gpuCalls >= 1, null, { timeout: 15_000 });
  await page.waitForFunction(() => document.querySelectorAll("[data-testid=plate-live]").length === 0, null, { timeout: 10_000 });
  assert(await page.locator(`[data-plate="${PLATES[0]}"] img`).isVisible(), "poster should remain after a failure");
  const calls = await page.evaluate(() => window.__gpuCalls);
  await page.mouse.move(5, 5);
  await page.waitForTimeout(400);
  await page.locator(`[data-plate="${PLATES[0]}"]`).hover();
  await page.waitForTimeout(1500);
  assert((await page.evaluate(() => window.__gpuCalls)) === calls, "a failed preview must not be retried on every hover");
});

await test("home plates: scrolling a live plate out of view stops it", async (page) => {
  await page.setViewportSize({ width: 1280, height: 420 });
  await page.goto(url("/"));
  await page.locator(`[data-plate="${PLATES[0]}"]`).hover();
  await waitLive(page, PLATES[0]);
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await page.waitForFunction(() => document.querySelectorAll("[data-testid=plate-live]").length === 0, null, { timeout: 10_000 });
});

await test("routes: unknown simulations return 404", async (page) => {
  for (const p of ["/physics/nope", "/chemistry/titration", "/biology/wave-interference"]) {
    const res = await page.goto(url(p));
    assert(res.status() === 404, `${p} -> ${res.status()}`);
  }
  // The 404 page is on-brand, explains itself, offers the way home and passes the accessibility audit.
  assert((await text(page, "h1")).includes("doesn’t exist"), "404 headline");
  assert(await page.getByRole("link", { name: "← All simulations" }).isVisible(), "404 should link home");
  const violations = (await new AxeBuilder({ page }).analyze()).violations.map((v) => `${v.id}: ${v.help}`);
  assert(violations.length === 0, `404 page a11y: ${violations.join(" | ")}`);
}, { allowErrors: [/404/] });

// ───────────────────────────── wave interference ─────────────────────────────
await test("wave: URL -> state, state -> URL, restore, defaults", async (page) => {
  await open(page, "/physics/wave-interference?view=intensity&frequency=6&separation=0.5&slitWidth=0.1&damping=0");
  assert(await page.getByLabel("Intensity").isChecked(), "view from URL not applied");
  await page.getByRole("button", { name: "Single Slit" }).click();
  await page.waitForTimeout(200);
  assert(page.url().includes("mode=single-slit"), `URL not updated: ${page.url()}`);
  await page.goto(url("/physics/wave-interference"));
  await page.locator("canvas").waitFor();
  assert((await page.getByRole("button", { name: "Single Slit" }).getAttribute("aria-pressed")) === "true", "settings not restored from storage");
  await page.getByRole("button", { name: "Defaults" }).click();
  assert(!page.url().includes("?"), `URL not cleared: ${page.url()}`);
  assert((await page.getByRole("button", { name: "Double Slit" }).getAttribute("aria-pressed")) === "true", "defaults not applied");
});

await test("wave: garbage in the URL is ignored/clamped, never crashes", async (page) => {
  await open(page, "/physics/wave-interference?frequency=9999&mode=banana&view=&separation=abc&detectorX=-50");
  assert((await page.getByRole("button", { name: "Double Slit" }).getAttribute("aria-pressed")) === "true", "invalid mode not defaulted");
  const freq = await page.getByLabel("Frequency").inputValue();
  assert(freq === "10", `frequency should clamp to 10, got ${freq}`);
});

await test("wave: copy link, save image (non-blank PNG)", async (page) => {
  await open(page, "/physics/wave-interference");
  await page.getByRole("button", { name: "Two Points" }).click();
  await page.getByRole("button", { name: "Copy link" }).click();
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  assert(clip === page.url() && clip.includes("mode=two-points"), `clipboard: ${clip}`);
  // Repeated, because a snapshot taken in a tick the (deliberately skipping) render loop did not draw would be blank.
  for (let i = 1; i <= 4; i++) {
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Save image" }).click()]);
    const file = path.join(os.tmpdir(), `edusim-${Date.now()}-${i}.png`);
    await download.saveAs(file);
    const size = fs.statSync(file).size;
    fs.rmSync(file);
    assert(download.suggestedFilename().endsWith(".png") && size > 20_000, `download ${i}: PNG too small (blank?): ${size}B`);
  }
});

await test("wave: detector graph, fringe measurement matches theory, probe, ruler, drag", async (page) => {
  await open(page, "/physics/wave-interference?view=intensity&frequency=6&separation=0.5&slitWidth=0.1&damping=0");
  const card = await text(page, "text=/Fringe spacing/");
  const [measured, theory] = card.match(/([\d.]+) measured · ([\d.]+) λL\/d/).slice(1).map(Number);
  assert(Math.abs(measured / theory - 1) < 0.1, `measured ${measured} vs theory ${theory}`);
  const box = await page.locator("canvas").boundingBox();

  await page.getByRole("button", { name: "Probe", exact: true }).click();
  await page.mouse.click(box.x + box.width * 0.6, box.y + box.height / 2);
  assert((await page.locator("svg text", { hasText: /I = / }).count()) === 1, "probe label missing");

  await page.getByRole("button", { name: "Ruler", exact: true }).click();
  await page.mouse.move(box.x + 300, box.y + 300);
  await page.mouse.down();
  await page.mouse.move(box.x + 500, box.y + 300, { steps: 6 });
  await page.mouse.up();
  const ruler = parseFloat(await text(page, "svg text:has-text('units')"));
  assert(Math.abs(ruler - (200 / box.height) * 4) < 0.03, `ruler ${ruler}`);

  await page.getByRole("button", { name: "None", exact: true }).click();
  const before = await text(page, "text=/Detector x = /");
  const handle = page.getByTestId("detector-handle");
  const hb = await handle.boundingBox();
  await page.mouse.move(hb.x + hb.width / 2, hb.y + 200);
  await page.mouse.down();
  await page.mouse.move(hb.x + 140, hb.y + 200, { steps: 8 });
  await page.mouse.up();
  assert(before !== (await text(page, "text=/Detector x = /")), "detector did not move");
  await shot(page, "wave-tools");
});

await test("wave: every mode x view renders without GPU errors", async (page) => {
  await open(page, "/physics/wave-interference");
  for (const mode of ["Single Point", "Two Points", "Single Slit", "Double Slit"]) {
    for (const view of ["Amplitude", "Intensity", "3D Water"]) {
      await page.getByRole("button", { name: mode }).click();
      await page.getByRole("radio", { name: view }).check();
      await page.waitForTimeout(120);
    }
  }
  await assertNoGpuProblem(page);
});

await test("wave: pause freezes the animation, play resumes it", async (page) => {
  await open(page, "/physics/wave-interference");
  await page.getByRole("button", { name: "Pause" }).click();
  // The wave shader is heavy for a CPU-emulated GPU: frames queued before the pause drain slowly.
  const frozen = await settle(page, 60_000);
  await page.waitForTimeout(800);
  assert(await same(page, frozen, await canvasPixels(page)), "canvas changed while paused");
  await page.getByRole("button", { name: "Play" }).click();
  await waitForChange(page, frozen, "canvas static after Play", 20_000);
}, { context: { viewport: { width: 900, height: 560 } } });

await test("frame pacing: at full resolution on a very slow GPU, Pause still takes effect within seconds", async (page) => {
  // ~700 ms per frame on this software renderer. Without a bounded frame queue the backlog took >12 s to drain.
  await open(page, "/physics/wave-interference");
  await page.getByRole("button", { name: "Pause" }).click();
  const t0 = Date.now();
  await settle(page, 20_000);
  const seconds = (Date.now() - t0) / 1000;
  assert(seconds < 8, `canvas took ${seconds.toFixed(1)}s to freeze after Pause`);
  console.log(`      (froze ${seconds.toFixed(1)}s after Pause)`);
});

await test("adaptive quality: a GPU too slow for full resolution lowers it, converges, and shows a badge", async (page) => {
  await open(page, "/physics/wave-interference");
  const width = () => page.evaluate(() => document.querySelector("canvas").width);
  const initial = await width();
  await page.waitForSelector("[data-testid=quality-badge]", { timeout: 60_000 });
  assert(/Resolution \d+%/.test(await text(page, "[data-testid=quality-badge]")), "badge text");
  // Converges: the resolution stops changing (no oscillation) within a minute.
  let last = await width();
  let stableSince = Date.now();
  const deadline = Date.now() + 60_000;
  while (Date.now() - stableSince < 10_000) {
    assert(Date.now() < deadline, "resolution kept changing for 60s (oscillating?)");
    await page.waitForTimeout(500);
    const w = await width();
    if (w !== last) { last = w; stableSince = Date.now(); }
  }
  assert(last < initial, `resolution not reduced: ${initial} -> ${last}`);
  console.log(`      (canvas ${initial}px -> ${last}px wide, then stable)`);
  await shot(page, "adaptive-quality");
}, { adaptive: true });

await test("adaptive quality: can be pinned to full resolution (teacher/projector setting)", async (page) => {
  await open(page, "/physics/wave-interference");
  const initial = await page.evaluate(() => document.querySelector("canvas").width);
  await page.waitForTimeout(15_000);
  assert((await page.evaluate(() => document.querySelector("canvas").width)) === initial, "resolution changed although pinned");
  assert((await page.locator("[data-testid=quality-badge]").count()) === 0, "badge shown although pinned");
});

await test("wave: challenge flow (predict → setup → goal → explanation) and persistence", async (page) => {
  await open(page, "/physics/wave-interference");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Squeeze the fringes/ }).click();
  assert(await page.getByRole("button", { name: /Lock in/ }).isDisabled(), "lock-in enabled without a prediction");
  await page.getByLabel(/closer together/).check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  assert(page.url().includes("separation=0.5"), "setup not applied");
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met too early");
  const sep = page.getByLabel("Slit Separation");
  await sep.focus();
  for (let i = 0; i < 5; i++) await page.keyboard.press("ArrowRight");
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓");
  assert(await page.locator("text=Your prediction was right.").isVisible(), "no feedback");
  await shot(page, "wave-challenge");
  await page.reload();
  await page.locator("canvas").waitFor();
  await page.getByRole("tab", { name: "Challenges" }).click();
  assert((await page.locator("text=/1 of 3 done/").count()) === 1, "completion not persisted");
});

await test("wave: keyboard shortcuts (Space, R, ?) and no hijack while typing/clicking controls", async (page) => {
  await open(page, "/physics/wave-interference");
  const toggle = page.getByRole("button", { name: /^(Pause|Play)$/ });
  await page.mouse.click(20, 400); // neutral spot
  const t0 = await toggle.textContent();
  await page.keyboard.press("Space");
  assert((await toggle.textContent()) !== t0, "Space did not toggle");
  await page.keyboard.press("?");
  assert(await page.getByRole("dialog", { name: "Keyboard shortcuts" }).isVisible(), "help not shown");
  await page.keyboard.press("Escape");
  await page.getByLabel("Frequency").focus();
  const t1 = await toggle.textContent();
  await page.keyboard.press("Space");
  assert((await toggle.textContent()) === t1, "Space hijacked while a control was focused");
});

// ───────────────────────────── hodgkin-huxley ─────────────────────────────
await test("hodgkin-huxley: runs, pauses, presets and sliders respond", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  const a = await canvasPixels(page);
  await page.waitForTimeout(500);
  assert(!await same(page, a, await canvasPixels(page)), "trace not animating");
  await page.getByRole("button", { name: "Pause" }).click();
  const p1 = await settle(page);
  await page.waitForTimeout(600);
  assert(await same(page, p1, await canvasPixels(page)), "trace moved while paused");
  await page.getByRole("button", { name: "Play" }).click();
  for (const preset of ["TTX Block", "TEA Block", "Anode Break", "Normal AP"]) {
    await page.getByRole("button", { name: preset }).click();
    await page.waitForTimeout(400);
  }
  await slider(page, "Playback speed").fill("120");
  await slider(page, "Temperature").fill("20");
  await page.getByRole("radio", { name: "Twin" }).check();
  await page.waitForTimeout(500);
  await shot(page, "hh");
});

await test("hodgkin-huxley: TTX flattens the spike (pixel check on the voltage trace)", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  // Count sky-blue (voltage trace) pixels where spikes peak: the top quarter, in the stretch of the canvas
  // between the legend (left) and the readout card (right), so overlay text is not counted.
  const spikeCount = async () => {
    const png = await canvasPixels(page);
    return page.evaluate(async (b64) => {
      const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const c = new OffscreenCanvas(img.width, img.height);
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0);
      const x0 = Math.floor(img.width * 0.45);
      const { data } = g.getImageData(x0, 0, Math.floor(img.width * 0.72) - x0, Math.floor(img.height * 0.25));
      let n = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i + 2] > 180 && data[i] < 140 && data[i + 1] > 120 && data[i + 1] < 225) n++; // sky-blue voltage trace
      return n;
    }, png.toString("base64"));
  };
  await page.getByRole("button", { name: "Normal AP" }).click();
  await page.waitForTimeout(4000); // let spikes fill the 102 ms window
  const normal = await spikeCount();
  await page.getByRole("button", { name: "TTX Block" }).click();
  await page.waitForTimeout(4000);
  const ttx = await spikeCount();
  assert(normal > 15 && ttx < normal / 4, `voltage pixels near the top: normal=${normal}, ttx=${ttx}`);
});

await test("hodgkin-huxley: URL state, restore, defaults, copy link, save image", async (page) => {
  await open(page, "/biology/hodgkin-huxley?I_inj=12&g_Na=0&temperature=20&pulse_mode=1");
  assert((await slider(page, "Injected current").inputValue()) === "12", "I_inj from URL");
  assert((await slider(page, "Na⁺ conductance (TTX)").inputValue()) === "0", "g_Na from URL");
  assert(await page.getByRole("radio", { name: "Pulse" }).isChecked(), "stimulus from URL");
  await slider(page, "Temperature").fill("25");
  await page.waitForTimeout(200);
  assert(page.url().includes("temperature=25"), `URL not updated: ${page.url()}`);
  await page.goto(url("/biology/hodgkin-huxley"));
  await page.locator("canvas").waitFor();
  assert((await slider(page, "Temperature").inputValue()) === "25", "settings not restored from storage");
  await page.getByRole("button", { name: "Copy link" }).click();
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  assert(clip === page.url() && clip.includes("g_Na=0"), `clipboard: ${clip}`);
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Save image" }).click()]);
  const file = path.join(os.tmpdir(), `edusim-hh-${Date.now()}.png`);
  await download.saveAs(file);
  const size = fs.statSync(file).size;
  fs.rmSync(file);
  assert(size > 10_000, `PNG too small (blank?): ${size}B`);
  await page.getByRole("button", { name: "Defaults" }).click();
  assert(!page.url().includes("?"), `URL not cleared: ${page.url()}`);
  assert((await slider(page, "Na⁺ conductance (TTX)").inputValue()) === "120", "defaults not applied");
});

await test("hodgkin-huxley: firing-rate readout, firing-rate curve and threshold", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  const rate = async () => (await text(page, "[data-testid=hh-rate]")).trim();
  const hz = parseFloat(await rate());
  assert(hz > 60 && hz < 76, `expected ~68 Hz at I=10, got "${await rate()}"`);
  const note = await text(page, "text=/Starts firing at/");
  const threshold = parseFloat(note.match(/≈ ([\d.]+)/)[1]);
  assert(threshold > 5 && threshold < 6.5, `threshold ${threshold}`);
  assert((await page.locator("figure svg[role=img]").count()) === 1, "f–I chart missing");
  await slider(page, "Injected current").fill("3");
  await page.waitForTimeout(300);
  assert((await rate()) === "silent", `expected silent at I=3, got "${await rate()}"`);
  await slider(page, "Na⁺ conductance (TTX)").fill("0");
  await page.waitForFunction(() => document.body.innerText.includes("No repetitive firing"), null, { timeout: 15_000 });
  await page.getByRole("radio", { name: "Pulse" }).check();
  await slider(page, "Injected current").fill("10");
  await page.waitForTimeout(300);
  assert(/0 spikes per 25 ms/.test(await rate()), `TTX pulse readout: "${await rate()}"`);
  await page.getByRole("button", { name: "Normal AP" }).click();
  await page.waitForTimeout(300);
  assert(/^1 spike per 25 ms/.test(await rate()), `normal pulse readout: "${await rate()}"`);
  await shot(page, "hh-measure");
});

await test("hodgkin-huxley: hover cursor reads plausible voltage and gate values", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  await page.getByRole("button", { name: "Pause" }).click();
  await page.waitForTimeout(800);
  const box = await page.locator("canvas").boundingBox();
  const read = async (fraction) => {
    await page.mouse.move(box.x + box.width * fraction, box.y + box.height / 2);
    await page.waitForSelector("[data-testid=hh-cursor]");
    await page.waitForTimeout(450); // let a fresh GPU read-back arrive
    const t = await text(page, "[data-testid=hh-cursor]");
    const num = (re) => parseFloat(t.match(re)[1]);
    return { ago: num(/([\d.]+) ms ago/), V: num(/V = (-?[\d.]+)/), m: num(/m = ([\d.]+)/), h: num(/h = ([\d.]+)/), n: num(/n = ([\d.]+)/) };
  };
  const left = await read(0.05);
  const right = await read(0.95);
  for (const r of [left, right]) {
    assert(r.V >= -100 && r.V <= 60, `V out of range: ${r.V}`);
    for (const g of [r.m, r.h, r.n]) assert(g >= 0 && g <= 1, `gate out of range: ${g}`);
  }
  assert(left.ago > right.ago + 80, `time axis wrong: ${left.ago} vs ${right.ago} ms ago`);
  assert(right.ago < 8, `right edge should be "now": ${right.ago} ms ago`);
  await page.mouse.move(box.x - 5, box.y - 5);
  await page.waitForTimeout(200);
  assert((await page.locator("[data-testid=hh-cursor]").count()) === 0, "cursor readout should disappear when not hovering");
  await shot(page, "hh-cursor");
});

await test("hodgkin-huxley: challenge flow (threshold) with persistence", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Find the firing threshold/ }).click();
  await page.getByLabel(/stay silent until a threshold/).check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  assert((await slider(page, "Injected current").inputValue()) === "0", "setup should set current to 0");
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met too early");
  const current = slider(page, "Injected current");
  await current.focus();
  for (let i = 0; i < 4; i++) await page.keyboard.press("ArrowRight"); // 2 µA/cm²: still silent
  await page.waitForTimeout(300);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met below threshold");
  for (let i = 0; i < 9; i++) await page.keyboard.press("ArrowRight"); // 6.5 µA/cm²: just over threshold
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓", { timeout: 15_000 });
  assert(await page.locator("text=Your prediction was right.").isVisible(), "no feedback");
  await shot(page, "hh-challenge");
  await page.reload();
  await page.locator("canvas").waitFor();
  await page.getByRole("tab", { name: "Challenges" }).click();
  assert((await page.locator("text=/1 of 3 done/").count()) === 1, "completion not persisted");
});

await test("hodgkin-huxley: challenge flow (TTX) cannot be solved by removing the current", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Silence it with a toxin/ }).click();
  await page.getByLabel(/spikes disappear/).check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  await slider(page, "Injected current").fill("0");
  await page.waitForTimeout(500);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "shortcut accepted");
  await slider(page, "Injected current").fill("15");
  await slider(page, "Na⁺ conductance (TTX)").fill("30");
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓", { timeout: 15_000 });
});

// ───────────────────────────── cellular automata ─────────────────────────────
const generation = async (page) => Number((await text(page, "text=Generation >> xpath=following-sibling::span")).trim());
const caStatus = async (page) => (await text(page, "[data-testid=ca-status]")).trim();
const caPopulation = async (page) => {
  const t = (await text(page, "[data-testid=ca-population]")).trim();
  return t === "—" ? NaN : Number(t.replace(/,/g, ""));
};
/** Waits until the GPU-measured status line matches (the measurement arrives a moment after the cells change). */
const waitStatus = (page, re, timeout = 30_000) =>
  page.waitForFunction((src) => new RegExp(src).test(document.querySelector("[data-testid=ca-status]")?.textContent ?? ""), re.source, { timeout });
const waitPopulation = (page, expected, timeout = 30_000) =>
  page.waitForFunction((n) => Number((document.querySelector("[data-testid=ca-population]")?.textContent ?? "").replace(/,/g, "")) === n, expected, { timeout });
/**
 * Zooms in around the middle of the canvas, where grid cell (128, 128) meets its neighbours, and returns that
 * point plus the on-screen size of one cell, so a test can click exactly on cells.
 */
async function zoomToCells(page, wheelEvents = 20) {
  const box = await page.locator("canvas").boundingBox();
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  for (let i = 0; i < wheelEvents; i++) await page.mouse.wheel(0, -100);
  await page.waitForTimeout(300);
  const zoom = parseFloat((await text(page, "text=/Zoom:/")).match(/Zoom: ([\d.]+)/)[1]);
  return { cx, cy, cell: (Math.min(box.width, box.height) / 256) * zoom };
}
async function clickAt(page, x, y) {
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.up();
}

await test("cellular-automata: play, pause, step, reset", async (page) => {
  await open(page, "/cs/cellular-automata");
  assert((await generation(page)) === 0, "should start at generation 0");
  await page.getByRole("button", { name: "Play" }).click();
  await page.waitForTimeout(2500);
  const g = await generation(page);
  assert(g > 5, `generation only reached ${g}`);
  await page.getByRole("button", { name: "Pause" }).click();
  const paused = await generation(page);
  await page.waitForTimeout(700);
  assert((await generation(page)) === paused, "generation advanced while paused");
  await page.getByRole("button", { name: "Step" }).click();
  await page.waitForTimeout(300);
  assert((await generation(page)) === paused + 1, "Step should advance exactly one generation");
  await page.getByRole("button", { name: "Reset" }).click();
  await page.waitForTimeout(300);
  assert((await generation(page)) === 0, "Reset should return to generation 0");
});

await test("cellular-automata: draw and erase strokes, wheel zoom, pan, grid sizes, rules", async (page) => {
  await open(page, "/cs/cellular-automata");
  await page.getByRole("button", { name: "Clear" }).click();
  await waitStatus(page, /^Extinct$/); // the on-canvas population panel is part of the picture: let it catch up
  const empty = await settle(page);
  const box = await page.locator("canvas").boundingBox();
  const stroke = async () => {
    await page.mouse.move(box.x + 300, box.y + 300);
    await page.mouse.down();
    await page.mouse.move(box.x + 360, box.y + 340, { steps: 10 });
    await page.mouse.up();
  };
  await page.getByRole("button", { name: "Draw", exact: true }).click();
  await stroke();
  assert(!await same(page, empty, await settle(page)), "drawing changed nothing");
  await page.getByRole("button", { name: "Erase", exact: true }).click();
  await stroke();
  // Back to the Pan tool first: the on-canvas caption changes with the tool and is part of the image.
  await page.getByRole("button", { name: "Pan", exact: true }).click();
  await waitStatus(page, /^Extinct$/);
  assert(await same(page, empty, await settle(page)), "erasing the same stroke did not restore the empty grid");

  await page.mouse.move(box.x + 400, box.y + 300);
  await page.mouse.wheel(0, -800);
  await page.waitForTimeout(300);
  assert(!(await text(page, "text=/Zoom:/")).includes("1.00"), "zoom readout unchanged");

  for (const size of ["512", "1024", "2048", "256"]) {
    await page.getByLabel("Grid size").selectOption(size);
    await page.getByRole("button", { name: "Random" }).click();
    await page.getByRole("button", { name: "Step" }).click();
    await page.waitForTimeout(250);
    await assertNoGpuProblem(page, ` at ${size}`);
  }
  await page.locator("select[aria-label='Rule preset']").selectOption({ label: "HighLife (B36/S23)" });
  assert((await page.locator("input[inputmode=numeric]").first().inputValue()) === "36", "preset not applied to B");
  await shot(page, "ca");
});

await test("cellular-automata: Gosper gun is alive after 100 generations (rule correctness)", async (page) => {
  await open(page, "/cs/cellular-automata");
  await page.getByLabel("Speed").fill("60");
  const before = await canvasPixels(page);
  await page.getByRole("button", { name: "Play" }).click();
  await page.waitForFunction(() => Number([...document.querySelectorAll("span")].find((s) => s.textContent === "Generation")?.nextElementSibling?.textContent ?? 0) >= 100, null, { timeout: 30_000 });
  assert(!await same(page, before, await canvasPixels(page)), "pattern did not evolve");
});

await test("cellular-automata: URL state, restore, defaults, copy link, save image", async (page) => {
  await open(page, "/cs/cellular-automata?speed=30&gridSize=512&theme=3&birth=36&pattern=Pulsar&showGraph=0");
  assert((await slider(page, "Speed").inputValue()) === "30", "speed from URL");
  assert((await page.getByLabel("Grid size").inputValue()) === "512", "grid size from URL");
  assert((await page.getByLabel("Color theme").inputValue()) === "3", "theme from URL");
  assert((await page.getByLabel("Starting pattern").inputValue()) === "Pulsar", "starting pattern from URL");
  assert((await page.locator("input[inputmode=numeric]").first().inputValue()) === "36", "birth rule from URL");
  assert((await page.locator("select[aria-label='Rule preset']").inputValue()) === "HighLife", "matching rule preset not shown");
  assert(!(await page.getByLabel("Population graph").isChecked()), "graph switch from URL");
  assert((await generation(page)) === 0, "a shared link starts at generation 0");
  await waitPopulation(page, 48); // the Pulsar has 48 cells
  await slider(page, "Speed").fill("45");
  await page.waitForTimeout(200);
  assert(page.url().includes("speed=45") && page.url().includes("pattern=Pulsar"), `URL not updated: ${page.url()}`);
  await page.goto(url("/cs/cellular-automata"));
  await page.locator("canvas").waitFor();
  assert((await slider(page, "Speed").inputValue()) === "45", "settings not restored from storage");
  assert((await page.getByLabel("Starting pattern").inputValue()) === "Pulsar", "starting pattern not restored");
  await page.getByRole("button", { name: "Copy link" }).click();
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  assert(clip === page.url() && clip.includes("birth=36"), `clipboard: ${clip}`);
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Save image" }).click()]);
  const file = path.join(os.tmpdir(), `edusim-ca-${Date.now()}.png`);
  await download.saveAs(file);
  const size = fs.statSync(file).size;
  fs.rmSync(file);
  assert(size > 10_000, `PNG too small (blank?): ${size}B`);
  await page.getByRole("button", { name: "Defaults" }).click();
  assert(!page.url().includes("?"), `URL not cleared: ${page.url()}`);
  assert((await slider(page, "Speed").inputValue()) === "10", "defaults not applied");
  assert((await page.getByLabel("Starting pattern").inputValue()) === "Glider Gun (Gosper)", "default pattern not applied");
  await waitPopulation(page, 36); // ...and the grid really was reloaded: the Gosper gun has 36 cells
});

await test("cellular-automata: garbage in the URL is ignored/clamped, never crashes", async (page) => {
  await open(page, "/cs/cellular-automata?speed=9999&gridSize=7&theme=x&pattern=banana&birth=zz&survive=9&showGraph=maybe");
  assert((await slider(page, "Speed").inputValue()) === "60", "speed should clamp to 60");
  assert((await page.getByLabel("Grid size").inputValue()) === "256", "grid size should default");
  assert((await page.getByLabel("Color theme").inputValue()) === "0", "theme should default");
  assert((await page.getByLabel("Starting pattern").inputValue()) === "Glider Gun (Gosper)", "pattern should default");
  assert((await page.locator("input[inputmode=numeric]").first().inputValue()) === "3", "birth rule should default");
  assert((await page.locator("select[aria-label='Rule preset']").inputValue()) === "Conway's Life", "rule should be Conway's");
  assert(await page.getByLabel("Population graph").isChecked(), "graph switch should default on");
  await waitPopulation(page, 36);
});

await test("cellular-automata: the GPU recognises still lifes, oscillators, moving patterns and extinction", async (page) => {
  await open(page, "/cs/cellular-automata?pattern=Block&speed=60");
  await waitPopulation(page, 4);
  assert(/^Evolving/.test(await caStatus(page)), `a single generation proves nothing yet: "${await caStatus(page)}"`);
  await page.getByRole("button", { name: "Play" }).click();
  await waitStatus(page, /^Still life$/);
  assert((await caPopulation(page)) === 4, "a block keeps its 4 cells");
  const expectations = [
    ["Blinker", /^Oscillator, period 2$/, [3]],
    ["Pulsar", /^Oscillator, period 3$/, [48, 56, 72]], // its three phases have different sizes
    ["Glider", /^Moving pattern/, [5]],
  ];
  for (const [name, status, cells] of expectations) {
    await page.getByLabel("Starting pattern").selectOption(name); // restarts the history; each verdict differs from the last one
    await waitStatus(page, status);
    const n = await caPopulation(page);
    assert(cells.includes(n), `${name}: ${n} cells, expected one of ${cells.join("/")}`);
  }
  await page.getByRole("button", { name: "Pause" }).click();
  await page.getByRole("button", { name: "Clear" }).click();
  await waitStatus(page, /^Extinct$/);
  assert((await caPopulation(page)) === 0, "empty grid has no cells");
  await page.getByRole("button", { name: "Random" }).click();
  await waitStatus(page, /^Evolving/);
  assert((await caPopulation(page)) > 5000, "a random soup fills about a fifth of 65,536 cells"); // 20 %
  await shot(page, "ca-status");
});

await test("cellular-automata: population graph follows the run and can be switched off", async (page) => {
  await open(page, "/cs/cellular-automata?pattern=Blinker&speed=60");
  assert((await page.locator("figure svg[role=img]").count()) === 0, "no graph before there is any history");
  await page.getByRole("button", { name: "Play" }).click();
  await page.waitForSelector("figure svg[role=img]", { timeout: 20_000 });
  await waitStatus(page, /^Oscillator, period 2$/);
  await page.waitForFunction(() => /3 live cells now\. Oscillator, period 2/.test(document.querySelector("figure svg[role=img]")?.getAttribute("aria-label") ?? ""), null, { timeout: 15_000 });
  assert(/Live cells per generation/.test((await page.locator("figure svg[role=img]").getAttribute("aria-label")) ?? ""), "graph needs an accessible name");
  await shot(page, "ca-graph");
  await page.getByLabel("Population graph").uncheck();
  assert((await page.locator("figure svg[role=img]").count()) === 0, "graph still shown after switching it off");
  assert(page.url().includes("showGraph=0"), `URL not updated: ${page.url()}`);
});

await test("cellular-automata: editing the grid restarts the measurement from what is on screen", async (page) => {
  await open(page, "/cs/cellular-automata?pattern=Blinker&speed=60");
  await page.getByRole("button", { name: "Play" }).click();
  await waitStatus(page, /^Oscillator, period 2$/);
  await page.getByRole("button", { name: "Pause" }).click();
  await page.waitForTimeout(400);
  await page.getByRole("button", { name: "Draw", exact: true }).click();
  const { cx, cy, cell } = await zoomToCells(page, 0); // the blinker already opens framed (zoomed in)
  await clickAt(page, cx + 6 * cell, cy + 6 * cell); // one extra, isolated cell far from the blinker
  await waitPopulation(page, 4);
  assert(/^Evolving/.test(await caStatus(page)), `old verdict survived an edit: "${await caStatus(page)}"`);
  await page.getByRole("button", { name: "Erase", exact: true }).click();
  await clickAt(page, cx + 6 * cell, cy + 6 * cell); // and take it away again
  await waitPopulation(page, 3);
});

await test("cellular-automata: challenge flow (still life) — draw a 2×2 block, press Play, persist", async (page) => {
  await open(page, "/cs/cellular-automata");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Build something that never changes/ }).click();
  assert(await page.getByRole("button", { name: /Lock in/ }).isDisabled(), "lock-in enabled without a prediction");
  await page.getByLabel("A 2×2 square").check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  assert(page.url().includes("pattern=clear"), `setup not applied: ${page.url()}`);
  assert((await page.getByRole("button", { name: "Draw", exact: true }).getAttribute("aria-pressed")) === "true", "setup should hand over the Draw tool");
  assert((await page.getByRole("button", { name: /^(Play|Pause)$/ }).textContent()) === "Play", "experiments start paused");
  await waitStatus(page, /^Extinct$/);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met by an empty grid");

  const { cx, cy, cell } = await zoomToCells(page);
  for (const [dx, dy] of [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) await clickAt(page, cx + dx * cell, cy + dy * cell);
  await waitPopulation(page, 4);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met before pressing Play");

  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓", { timeout: 30_000 });
  assert((await caStatus(page)) === "Still life", `status: ${await caStatus(page)}`);
  assert(await page.locator("text=Your prediction was right.").isVisible(), "no feedback");
  await shot(page, "ca-challenge");
  await page.reload();
  await page.locator("canvas").waitFor();
  await page.getByRole("tab", { name: "Challenges" }).click();
  assert((await page.locator("text=/1 of 3 done/").count()) === 1, "completion not persisted");
});

await test("cellular-automata: challenge flow (oscillator) cannot be solved by changing the rules", async (page) => {
  await open(page, "/cs/cellular-automata");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Make it blink/ }).click();
  await page.getByLabel(/flip between a horizontal and a vertical row/).check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  await waitStatus(page, /^Extinct$/);
  const { cx, cy, cell } = await zoomToCells(page);
  for (const dx of [-0.5, 0.5, 1.5]) await clickAt(page, cx + dx * cell, cy - 0.5 * cell); // three in a row
  await waitPopulation(page, 3);
  // A blinker blinks under HighLife too, but that proves nothing about Conway's rules.
  await page.locator("select[aria-label='Rule preset']").selectOption({ label: "HighLife (B36/S23)" });
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await waitStatus(page, /^Oscillator, period 2$/);
  await page.waitForTimeout(500);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal accepted under non-Conway rules");
  await page.locator("select[aria-label='Rule preset']").selectOption({ label: "Conway's Life (B3/S23)" });
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓", { timeout: 30_000 });
  assert(await page.locator("text=Your prediction was right.").isVisible(), "no feedback");
});

await test("cellular-automata: challenge flow (glider gun) — the population climbs past 100", async (page) => {
  await open(page, "/cs/cellular-automata");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Grow without limit/ }).click();
  await page.getByLabel("keep growing").check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  await waitPopulation(page, 36);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met at 36 cells");
  // A dense random soup is far above 100 cells from the start; it must not count.
  await page.getByRole("button", { name: "Random" }).click();
  await page.waitForFunction(() => Number((document.querySelector("[data-testid=ca-population]")?.textContent ?? "").replace(/,/g, "")) > 5000);
  await page.waitForTimeout(500);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "a random soup solved the gun challenge");
  await page.getByLabel("Starting pattern").selectOption("Glider Gun (Gosper)");
  await waitPopulation(page, 36);
  await slider(page, "Speed").fill("60");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓", { timeout: 150_000 });
  // (No re-read of the population here: the gun's count swings by ±15 every 30 generations, so it can dip below 100
  // again right after the goal — which already checked "> 100" on the measured value.)
  assert(await page.locator("text=Your prediction was right.").isVisible(), "no feedback");
});

// ───────────────────────────── n-body ─────────────────────────────
const nbText = async (page, id) => (await text(page, `[data-testid=${id}]`)).trim();
/** Energy drift readout ("+0.0004 %", with a real minus sign when negative) as a fraction. */
const nbDrift = async (page) => parseFloat((await nbText(page, "nb-drift")).replace("−", "-")) / 100;
const nbPeriod = async (page) => parseFloat(((await nbText(page, "nb-period")).match(/period ([\d.]+)/) ?? [])[1]);
const waitPeriod = (page, timeout = 90_000) =>
  page.waitForFunction(() => /period \d/.test(document.querySelector("[data-testid=nb-period]")?.textContent ?? ""), null, { timeout });

await test("n-body: Earth's period is measured as 1 year on the GPU, energy conserved", async (page) => {
  await open(page, "/physics/n-body?speed=2");
  assert((await page.locator("canvas").count()) === 2, "WebGPU canvas + label overlay");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await waitPeriod(page);
  const T = await nbPeriod(page);
  assert(Math.abs(T - 1) < 0.005, `Earth period ${T} yr`);
  assert(parseFloat(await nbText(page, "nb-e")) < 0.01, `eccentricity ${await nbText(page, "nb-e")}`);
  assert(Math.abs(await nbDrift(page)) < 1e-4, `energy drift ${await nbText(page, "nb-drift")}`);
  await shot(page, "nbody");
});

await test("n-body: inner planets — Mercury's measured period and the Kepler plot", async (page) => {
  await open(page, "/physics/n-body?preset=inner-planets&speed=2");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await waitPeriod(page);
  const T = await nbPeriod(page); // Mercury is selected (body 1)
  assert(Math.abs(T - 0.2408) < 0.003, `Mercury period ${T} yr (88 days = 0.2408 yr)`);
  await page.waitForFunction(() => [...document.querySelectorAll("figure svg[role=img]")].some((s) => /T² against a³/.test(s.getAttribute("aria-label") ?? "")), null, { timeout: 60_000 });
  assert((await page.locator("figure svg circle").count()) >= 1, "Kepler plot has measured points");
  await shot(page, "nbody-kepler");
});

await test("n-body: URL state, hostile links, copy link, save image, defaults", async (page) => {
  await open(page, "/physics/n-body?preset=binary-star&speed=1.5&trails=0&integrator=euler");
  assert((await slider(page, "Speed").inputValue()) === "1.5", "speed from URL");
  assert(!(await page.getByLabel("Trails").isChecked()), "trails from URL");
  assert(await page.getByRole("radio", { name: "Euler" }).isChecked(), "integrator from URL");
  assert((await page.getByRole("button", { name: "Star A" }).count()) === 1, "preset bodies from URL");
  await page.getByRole("button", { name: "Copy link" }).click();
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  assert(clip === page.url() && clip.includes("preset=binary-star"), `clipboard: ${clip}`);
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Save image" }).click()]);
  const file = path.join(os.tmpdir(), `edusim-nbody-${Date.now()}.png`);
  await download.saveAs(file);
  const size = fs.statSync(file).size;
  fs.rmSync(file);
  assert(size > 5_000, `PNG too small (blank?): ${size}B`);
  await page.getByRole("button", { name: "Defaults" }).click();
  assert(!page.url().includes("?"), `URL not cleared: ${page.url()}`);
  assert((await page.getByRole("button", { name: "Earth" }).count()) === 1, "default preset not restored");

  await page.goto(url("/physics/n-body?mode=wormhole&speed=1e9&count=3&bodies=1,2,3&preset=nope"));
  await page.locator("canvas").first().waitFor();
  await page.waitForSelector("[data-testid=gpu-status][data-state=loading]", { state: "detached", timeout: 30_000 });
  assert((await slider(page, "Speed").inputValue()) === "3", "speed should clamp to the slider's 3 yr/s");
  assert((await page.getByRole("button", { name: "Earth" }).count()) === 1, "garbage setup should fall back to the default system");
  await assertNoGpuProblem(page);
});

await test("n-body: dragging a body and its velocity arrow edits the setup (and the link)", async (page) => {
  await open(page, "/physics/n-body");
  await page.waitForFunction(() => /\d/.test(document.querySelector("[data-testid=nb-time]")?.textContent ?? "")); // first measurement in: labels and arrows are placed
  const box = await page.locator("canvas").first().boundingBox();
  const k = Math.min(box.width, box.height) / 2 / 1.25; // CSS px per AU: the default view frames 1.25 AU
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  // Earth starts at (1, 0) AU: drag it out to 1.4 AU.
  await page.mouse.move(cx + k, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 1.4 * k, cy, { steps: 6 });
  await page.mouse.up();
  await page.waitForFunction(() => location.search.includes("bodies="));
  const moved = new URL(page.url()).searchParams.get("bodies").split(";")[1].split(",").map(Number);
  assert(Math.abs(moved[1] - 1.4) < 0.03 && Math.abs(moved[2]) < 0.03, `Earth moved to ${moved.slice(1, 3)}`);
  // Its arrow tip sits a quarter of r/v ahead: drag it twice as far, past escape speed.
  const v = Math.hypot(moved[3], moved[4]);
  const t = 0.25 * (1.4 / v);
  const tip = [cx + 1.4 * k, cy - v * t * k];
  const before = new URL(page.url()).searchParams.get("bodies");
  await page.mouse.move(tip[0], tip[1]);
  await page.mouse.down();
  await page.mouse.move(tip[0], cy - 2.2 * v * t * k, { steps: 6 });
  await page.mouse.up();
  await page.waitForFunction((b) => new URL(location.href).searchParams.get("bodies") !== b, before);
  const faster = new URL(page.url()).searchParams.get("bodies").split(";")[1].split(",").map(Number);
  assert(Math.hypot(faster[3], faster[4]) > 1.8 * v, `speed ${Math.hypot(faster[3], faster[4])} after dragging the arrow (was ${v})`);
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForSelector("[data-testid=nb-status]:has-text('escaping')", { timeout: 30_000 });
});

await test("n-body: sliders set distance and speed; the escape speed escapes", async (page) => {
  await open(page, "/physics/n-body");
  await slider(page, "Orbital speed").fill("8.6"); // below √2 × 6.28 ≈ 8.89: still bound
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForTimeout(1500);
  assert((await page.locator("[data-testid=nb-status]").count()) === 0, "8.6 AU/yr should stay bound");
  await slider(page, "Orbital speed").fill("9.1");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForSelector("[data-testid=nb-status]:has-text('escaping')", { timeout: 30_000 });
  await slider(page, "Distance from star").fill("4");
  await page.waitForTimeout(200);
  const [sun, earth] = new URL(page.url()).searchParams.get("bodies").split(";").map((b) => b.split(",").map(Number));
  assert(Math.abs(Math.hypot(earth[1] - sun[1], earth[2] - sun[2]) - 4) < 0.02, `distance in the link: ${page.url()}`);
});

await test("n-body: Euler pumps energy in; leapfrog keeps it", async (page) => {
  await open(page, "/physics/n-body?speed=3&integrator=euler");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForFunction(() => parseFloat((document.querySelector("[data-testid=nb-drift]")?.textContent ?? "0").replace("−", "-")) > 1, null, { timeout: 60_000 });
  await page.getByRole("radio", { name: "Leapfrog" }).check();
  await page.waitForTimeout(4000);
  assert(Math.abs(await nbDrift(page)) < 1e-4, `leapfrog drift ${await nbText(page, "nb-drift")}`);
});

await test("n-body: galaxies — every preset runs on the GPU and conserves energy", async (page) => {
  await open(page, "/physics/n-body?mode=galaxy&count=1024&galaxySpeed=2");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  for (const preset of ["Two colliding galaxies", "Rotating disk galaxy", "Star cluster (Plummer)"]) {
    await page.locator("select").first().selectOption({ label: preset });
    await page.waitForFunction(() => parseFloat(document.querySelector("[data-testid=nb-time]")?.textContent ?? "0") > 0.5, null, { timeout: 60_000 });
    assert(Math.abs(await nbDrift(page)) < 0.01, `${preset}: energy drift ${await nbText(page, "nb-drift")}`);
    await assertNoGpuProblem(page, ` (${preset})`);
  }
  await shot(page, "nbody-galaxy");
});

await test("n-body: challenge flow (circular orbit) with persistence", async (page) => {
  await open(page, "/physics/n-body");
  await page.getByRole("tab", { name: "Challenges" }).click();
  await page.getByRole("button", { name: /Make the orbit a circle/ }).click();
  await page.getByLabel("less speed than Earth").check();
  await page.getByRole("button", { name: /Lock in/ }).click();
  assert(page.url().includes("bodies="), `setup not applied: ${page.url()}`);
  assert(!(await text(page, "[role=status]:has-text('Goal')")).includes("✓"), "goal met too early");
  await slider(page, "Orbital speed").fill("4.44");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForSelector("[role=status]:has-text('Goal') >> text=✓", { timeout: 90_000 });
  assert(await page.locator("text=Your prediction was right.").isVisible(), "no feedback");
  await page.reload();
  await page.locator("canvas").first().waitFor();
  await page.getByRole("tab", { name: "Challenges" }).click();
  assert((await page.locator("text=/1 of 4 done/").count()) === 1, "completion not persisted");
});

// ───────────────────────────── resilience ─────────────────────────────
await test("navigation churn: open/close each simulation 3 times without errors", async (page) => {
  for (let i = 0; i < 3; i++) {
    for (const route of ["/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata", "/physics/n-body"]) {
      await open(page, route);
      await page.getByRole("link", { name: "← All simulations" }).click();
      await page.waitForURL((u) => u.pathname === "/");
    }
  }
});

await test("no WebGPU: every simulation explains the problem instead of a blank canvas", async (page) => {
  for (const route of ["/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata", "/physics/n-body"]) {
    await page.goto(url(route));
    await page.waitForSelector("[data-testid=gpu-status][data-state=unsupported]");
    assert((await text(page, "[data-testid=gpu-status]")).includes("WebGPU isn’t available"), `${route}: unsupported message missing`);
    assert(await page.getByRole("link", { name: "← All simulations" }).isVisible(), `${route}: can't navigate away`);
  }
}, { initScript: () => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined, configurable: true }) });

await test("phone viewport: no horizontal overflow, controls reachable, tap works", async (page) => {
  for (const route of ["/", "/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata", "/physics/n-body"]) {
    await page.goto(url(route));
    if (route !== "/") await page.locator("canvas").first().waitFor();
    await page.waitForTimeout(800);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    assert(overflow <= 1, `${route}: horizontal overflow ${overflow}px`);
    if (route !== "/") {
      const canvasHeight = (await page.locator("canvas").first().boundingBox()).height;
      assert(canvasHeight >= 200, `${route}: canvas only ${canvasHeight}px tall`);
      await page.getByRole("button", { name: /^(Pause|Play)$/ }).scrollIntoViewIfNeeded();
      await page.getByRole("button", { name: /^(Pause|Play)$/ }).tap();
    }
  }
  await shot(page, "phone");
}, { context: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 } });

const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 };

await test("phone: sidebar is a bottom drawer — tap or drag the handle, Play always within reach, compact header", async (page) => {
  for (const route of ["/physics/wave-interference", "/physics/n-body", "/cs/cellular-automata", "/biology/hodgkin-huxley"]) {
    await open(page, route);
    const header = (await page.locator("header").boundingBox()).height;
    assert(header <= 100, `${route}: header is ${header}px tall`);
    const drawer = () => page.getAttribute("[data-testid=sidebar]", "data-drawer");
    const canvasH = async () => (await page.locator("canvas").first().boundingBox()).height;
    assert((await drawer()) === "half", `${route}: starts half open`);
    const half = await canvasH();
    await page.getByRole("button", { name: "Expand controls" }).tap();
    assert((await drawer()) === "full", `${route}: tap should expand`);
    assert((await canvasH()) < half, `${route}: canvas should shrink when the drawer grows`);
    await page.getByRole("button", { name: "Collapse controls" }).tap();
    assert((await drawer()) === "peek", `${route}: tap should collapse`);
    const peek = await canvasH();
    assert(peek > 600, `${route}: collapsed drawer leaves the canvas only ${peek}px`);
    assert(await page.getByRole("button", { name: /^(Play|Pause)$/ }).isVisible(), `${route}: Play must stay visible when collapsed`);
    assert(!(await page.locator("#sim-sidebar-content").isVisible()), `${route}: content hidden when collapsed`);
    // Drag the handle up: back to a larger drawer.
    const handle = await page.getByRole("button", { name: "Show controls" }).boundingBox();
    const cdp = await page.context().newCDPSession(page);
    const at = (y) => [{ x: handle.x + handle.width / 2, y, id: 1 }];
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: at(handle.y + 10) });
    for (let k = 1; k <= 6; k++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: at(handle.y + 10 - k * 40) });
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await page.waitForTimeout(200);
    assert((await drawer()) !== "peek", `${route}: dragging the handle up should open the drawer`);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    assert(overflow <= 1, `${route}: horizontal overflow ${overflow}px`);
  }
  await shot(page, "phone-drawer");
}, { context: PHONE });

/** Two real touch points through the DevTools protocol: start at a/b, then move them to a2/b2. */
async function pinch(page, [a, b], [a2, b2], steps = 6) {
  const cdp = await page.context().newCDPSession(page);
  const pts = (p, q) => [{ x: p[0], y: p[1], id: 1 }, { x: q[0], y: q[1], id: 2 }];
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: pts(a, b) });
  for (let k = 1; k <= steps; k++) {
    const f = k / steps;
    const lerp = (p, q) => [p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f];
    await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: pts(lerp(a, a2), lerp(b, b2)) });
  }
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await page.waitForTimeout(250);
}

await test("phone: pinch-zoom and two-finger pan (cellular automata, with the Draw tool selected)", async (page) => {
  await open(page, "/cs/cellular-automata?pattern=clear");
  await page.getByRole("button", { name: "Draw", exact: true }).tap();
  const box = await page.locator("canvas").first().boundingBox();
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await pinch(page, [[cx - 50, cy], [cx + 50, cy]], [[cx - 100, cy], [cx + 100, cy]]);
  const zoom = parseFloat((await text(page, "text=/Zoom:/")).match(/Zoom: ([\d.]+)/)[1]);
  assert(Math.abs(zoom - 2) < 0.05, `pinching fingers apart 2× should zoom 2×, got ${zoom}`);
  await pinch(page, [[cx - 100, cy], [cx + 100, cy]], [[cx - 40, cy + 30], [cx + 160, cy + 30]]); // pan, same spread
  const after = parseFloat((await text(page, "text=/Zoom:/")).match(/Zoom: ([\d.]+)/)[1]);
  assert(Math.abs(after - zoom) < 0.05, `two-finger pan changed the zoom: ${zoom} → ${after}`);
  // With Draw selected, a pinch must not draw: the grid is still empty.
  await waitStatus(page, /^Extinct$/);
  // ...but a one-finger tap does draw a cell.
  await page.touchscreen.tap(cx, cy);
  await waitPopulation(page, 1);
}, { context: PHONE });

await test("phone: pinching over a planet zooms instead of dragging it (n-body)", async (page) => {
  await open(page, "/physics/n-body");
  const box = await page.locator("canvas").first().boundingBox();
  const k = Math.min(box.width, box.height) / 2 / 1.25;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await pinch(page, [[cx + k, cy], [cx - 60, cy]], [[cx + k + 40, cy], [cx - 100, cy]]); // first finger lands on Earth
  assert(!page.url().includes("bodies="), `a pinch edited the setup: ${page.url()}`);
  await assertNoGpuProblem(page);
}, { context: PHONE });

// ───────────────────────────── accessibility & polish (phase B) ─────────────────────────────
const SIM_ROUTES = ["/physics/wave-interference", "/physics/n-body", "/cs/cellular-automata", "/biology/hodgkin-huxley"];

await test("a11y: every simulation page passes the axe audit, desktop and phone", async (page) => {
  const problems = [];
  for (const viewport of [{ width: 1360, height: 860 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const route of SIM_ROUTES) {
      await open(page, route);
      const result = await new AxeBuilder({ page }).analyze();
      for (const v of result.violations) problems.push(`${route} at ${viewport.width}px: ${v.id} (${v.nodes.length}): ${v.help} — ${v.nodes.slice(0, 3).map((n) => n.target).join(", ")}`);
    }
  }
  assert(problems.length === 0, problems.join("\n      "));
});

await test("a11y: keyboard focus is always visible", async (page) => {
  for (const route of SIM_ROUTES) {
    await open(page, route);
    await page.locator("body").focus();
    for (let i = 0; i < 14; i++) {
      await page.keyboard.press("Tab");
      const style = await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body || el.tagName === "NEXTJS-PORTAL") return null; // Next's dev overlay
        const cs = getComputedStyle(el);
        return { tag: el.tagName, text: (el.textContent ?? "").trim().slice(0, 20), outline: cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0, ring: cs.boxShadow !== "none" };
      });
      if (!style) continue;
      assert(style.outline || style.ring, `${route}: focused ${style.tag} "${style.text}" has no visible focus indicator`);
    }
  }
});

await test("a11y: each simulation describes its canvas for screen readers (polite live region)", async (page) => {
  const expected = {
    "/physics/wave-interference": /Double slit, amplitude view\. Bright fringes .* units apart/,
    "/physics/n-body": /Year \d+\.\d\. Earth: /,
    "/cs/cellular-automata": /Generation \d+: 36 live cells\./,
    "/biology/hodgkin-huxley": /fires steadily at \d+ spikes per second/,
  };
  for (const route of SIM_ROUTES) {
    await open(page, route);
    await page.waitForFunction((src) => new RegExp(src).test(document.querySelector("[data-testid=sim-summary]")?.textContent ?? ""), expected[route].source, { timeout: 15_000 });
    assert((await page.locator("[aria-live=polite][data-testid=sim-summary]").count()) === 1, `${route}: live region`);
  }
});

await test("reduced motion: simulations start paused", async (page) => {
  for (const route of SIM_ROUTES) {
    await open(page, route);
    assert((await page.getByRole("button", { name: /^(Play|Pause)$/ }).textContent()) === "Play", `${route} should start paused`);
  }
  await open(page, "/biology/hodgkin-huxley");
  const frozen = await settle(page);
  await page.waitForTimeout(800);
  assert(await same(page, frozen, await canvasPixels(page)), "the trace moves although paused");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await waitForChange(page, frozen, "Play should still start it");
}, { context: { reducedMotion: "reduce" } });

await test("cellular-automata: opens framed on its pattern; Fit grid and Fit pattern", async (page) => {
  await open(page, "/cs/cellular-automata");
  const zoom = async () => parseFloat((await text(page, "text=/Zoom:/")).match(/Zoom: ([\d.]+)/)[1]);
  const opened = await zoom();
  assert(opened > 3 && opened < 6, `the Gosper gun (36 × 9 cells) should fill the view, zoom ${opened}`);
  await page.getByRole("button", { name: "Fit grid" }).click();
  assert((await zoom()) === 1, "Fit grid frames the whole grid");
  await page.getByRole("button", { name: "Fit pattern" }).click();
  await page.waitForFunction((z) => parseFloat(document.body.innerText.match(/Zoom: ([\d.]+)/)?.[1] ?? "0") > z, 3);
  const zoomIs = (test) => page.waitForFunction((src) => new Function("z", `return ${src}`)(parseFloat(document.body.innerText.match(/Zoom: ([\d.]+)/)?.[1] ?? "0")), test, { timeout: 10_000 });
  await page.getByLabel("Starting pattern").selectOption("Block");
  await zoomIs("z > 10"); // a 2×2 block opens close up
  await page.getByRole("button", { name: "Random" }).click();
  await zoomIs("z === 1"); // a random soup shows the whole grid
});

await test("phone: sliders have a finger-sized hit area", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  const h = (await slider(page, "Temperature").boundingBox()).height;
  assert(h >= 24, `slider is ${h}px tall`);
}, { context: PHONE });

// ───────────────────────────── summary ─────────────────────────────
await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} passed${failed.length ? `, FAILED: ${failed.map((f) => f.name).join("; ")}` : ""}`);
process.exit(failed.length ? 1 : 0);
