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
import { chromium } from "playwright-core";

const BASE_URL = (process.env.BASE_URL ?? "http://localhost:3000").replace(/\/$/, "");
const SHOTS = process.env.SHOTS_DIR; // optional: save screenshots here
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
  await page.locator("canvas").waitFor({ timeout: 20_000 });
  await page.waitForSelector("[data-testid=gpu-status][data-state=loading]", { state: "detached", timeout: 30_000 });
  await page.waitForTimeout(600);
  await assertNoGpuProblem(page);
}

const canvasPixels = (page) => page.locator("canvas").screenshot();
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
await test("home: renders, groups by subject, links to the 3 available sims", async (page) => {
  await page.goto(url("/"));
  assert((await page.title()).includes("EduSim"), `title: ${await page.title()}`);
  for (const s of ["Physics", "Chemistry", "Computer Science", "Biology"]) {
    assert(await page.getByRole("heading", { name: s, exact: true }).first().isVisible(), `missing ${s}`);
  }
  const links = await page.locator("main a[href^='/']").evaluateAll((as) => as.map((a) => a.getAttribute("href")));
  assert(links.length === 3, `expected 3 sim links, got ${links.join(",")}`);
  await page.getByRole("link", { name: /Hodgkin/ }).click();
  await page.waitForURL(/hodgkin-huxley/);
  await page.getByRole("link", { name: "← All simulations" }).click();
  await page.waitForURL((u) => u.pathname === "/" || u.pathname === "");
  await shot(page, "home");
});

await test("routes: unknown simulations return 404", async (page) => {
  for (const p of ["/physics/nope", "/chemistry/titration", "/biology/wave-interference"]) {
    const res = await page.goto(url(p));
    assert(res.status() === 404, `${p} -> ${res.status()}`);
  }
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
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Save image" }).click()]);
  const file = path.join(os.tmpdir(), `edusim-${Date.now()}.png`);
  await download.saveAs(file);
  const size = fs.statSync(file).size;
  fs.rmSync(file);
  assert(download.suggestedFilename().endsWith(".png") && size > 20_000, `PNG too small (blank?): ${size}B`);
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
  await page.getByLabel("Playback speed").fill("120");
  await page.getByLabel("Temperature").fill("20");
  await page.getByRole("radio", { name: "Twin" }).check();
  await page.waitForTimeout(500);
  await shot(page, "hh");
});

await test("hodgkin-huxley: TTX flattens the spike (pixel check on the voltage trace)", async (page) => {
  await open(page, "/biology/hodgkin-huxley");
  // Count bright-green (voltage trace) pixels in the top quarter of the canvas, where spikes peak.
  const spikeCount = async () => {
    const png = await canvasPixels(page);
    return page.evaluate(async (b64) => {
      const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const c = new OffscreenCanvas(img.width, img.height);
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0);
      const { data } = g.getImageData(0, 0, img.width, Math.floor(img.height * 0.2));
      let n = 0;
      for (let i = 0; i < data.length; i += 4) if (data[i + 1] > 180 && data[i] < 120 && data[i + 2] < 140) n++;
      return n;
    }, png.toString("base64"));
  };
  await page.getByRole("button", { name: "Normal AP" }).click();
  await page.waitForTimeout(4000); // let spikes fill the 102 ms window
  const normal = await spikeCount();
  await page.getByRole("button", { name: "TTX Block" }).click();
  await page.waitForTimeout(4000);
  const ttx = await spikeCount();
  assert(normal > 20 && ttx < normal / 4, `voltage pixels near the top: normal=${normal}, ttx=${ttx}`);
});

// ───────────────────────────── cellular automata ─────────────────────────────
const generation = async (page) => Number((await text(page, "text=Generation >> xpath=following-sibling::span")).trim());

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
  await page.waitForFunction(() => /Generation\s*\n?\s*(\d+)/.test(document.body.innerText) && Number(document.body.innerText.match(/Generation\s*\n?\s*(\d+)/)[1]) >= 100, null, { timeout: 30_000 });
  assert(!await same(page, before, await canvasPixels(page)), "pattern did not evolve");
});

// ───────────────────────────── resilience ─────────────────────────────
await test("navigation churn: open/close each simulation 3 times without errors", async (page) => {
  for (let i = 0; i < 3; i++) {
    for (const route of ["/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata"]) {
      await open(page, route);
      await page.getByRole("link", { name: "← All simulations" }).click();
      await page.waitForURL((u) => u.pathname === "/");
    }
  }
});

await test("no WebGPU: every simulation explains the problem instead of a blank canvas", async (page) => {
  for (const route of ["/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata"]) {
    await page.goto(url(route));
    await page.waitForSelector("[data-testid=gpu-status][data-state=unsupported]");
    assert((await text(page, "[data-testid=gpu-status]")).includes("WebGPU isn’t available"), `${route}: unsupported message missing`);
    assert(await page.getByRole("link", { name: "← All simulations" }).isVisible(), `${route}: can't navigate away`);
  }
}, { initScript: () => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined, configurable: true }) });

await test("phone viewport: no horizontal overflow, controls reachable, tap works", async (page) => {
  for (const route of ["/", "/physics/wave-interference", "/biology/hodgkin-huxley", "/cs/cellular-automata"]) {
    await page.goto(url(route));
    if (route !== "/") await page.locator("canvas").waitFor();
    await page.waitForTimeout(800);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    assert(overflow <= 1, `${route}: horizontal overflow ${overflow}px`);
    if (route !== "/") {
      const canvasHeight = (await page.locator("canvas").boundingBox()).height;
      assert(canvasHeight >= 200, `${route}: canvas only ${canvasHeight}px tall`);
      await page.getByRole("button", { name: /^(Pause|Play)$/ }).scrollIntoViewIfNeeded();
      await page.getByRole("button", { name: /^(Pause|Play)$/ }).tap();
    }
  }
  await shot(page, "phone");
}, { context: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 } });

// ───────────────────────────── summary ─────────────────────────────
await browser.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} passed${failed.length ? `, FAILED: ${failed.map((f) => f.name).join("; ")}` : ""}`);
process.exit(failed.length ? 1 : 0);
