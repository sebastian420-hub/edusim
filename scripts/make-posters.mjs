// Generates the home-page plate artwork from the REAL simulations, so the posters always show what
// the simulations actually look like. Run after changing a simulation's look:
//
//   pnpm posters                 # builds, serves, captures, writes public/plates/*.webp and the OG image
//   pnpm posters --skip-build    # reuse an existing .next build
//
// Each poster is captured from fixed, shareable-link settings (so it is reproducible), reading only
// the <canvas> pixels (no HTML overlays), cover-cropped to the plate's proportions and encoded as WebP.
// Needs a Chromium build and WebGPU (a real GPU, or the vgpu software renderer / SwiftShader).
import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";
import { withServer } from "./lib/serve.mjs";

const OUT_DIR = "public/plates";
const WIDTH = 640; // keep in sync with src/components/home/poster.ts
const HEIGHT = 704; // 4 : 4.4, the aspect ratio of a plate's art area

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** What to capture for each simulation. `focus` places the crop window within any overflow (0..1). */
const POSTERS = [
  {
    id: "wave-interference",
    route: "/physics/wave-interference?view=intensity&frequency=5&separation=0.7&slitWidth=0.12&damping=0.01&detector=0",
    canvasWidth: 1000, // wider than the poster, so the crop can start right at the barrier
    focus: { x: 0.6, y: 0.5 },
    async prepare() {
      await wait(4000); // intensity view is time-independent; just let one clean frame land
    },
  },
  {
    id: "n-body",
    route: "/physics/n-body?mode=galaxy&count=2048&galaxySpeed=4",
    focus: { x: 0.5, y: 0.5 },
    async prepare(page) {
      await page.getByRole("button", { name: "Play", exact: true }).click();
      // Let the galaxies pass each other and throw out tidal tails.
      await page.waitForFunction(() => Number(document.querySelector("[data-testid=nb-time]")?.textContent ?? 0) >= 16, null, { timeout: 240_000 });
      await page.getByRole("button", { name: "Pause", exact: true }).click();
      const box = await page.locator("canvas").first().boundingBox();
      await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
      for (let i = 0; i < 4; i++) await page.mouse.wheel(0, -100); // ~1.6x: the galaxies fill the plate
      await wait(1500);
    },
  },
  {
    id: "hodgkin-huxley",
    route: "/biology/hodgkin-huxley?I_inj=12&temperature=1&timeScale=200&showCurve=0",
    focus: { x: 0.5, y: 0.5 },
    async prepare(page) {
      await wait(9000); // let the 102 ms window fill with spikes
      await page.getByRole("button", { name: "Pause" }).click();
      await wait(2500);
    },
  },
  {
    id: "cellular-automata",
    route: "/cs/cellular-automata",
    focus: { x: 0.5, y: 0.5 },
    // The random soup uses Math.random(); seed it so regenerating gives the identical poster.
    init: () => {
      let a = 0x2f6e2b1;
      Math.random = () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    },
    async prepare(page) {
      await page.getByLabel("Color theme").selectOption("3"); // Amber, the computer-science accent
      await page.getByRole("button", { name: "Random" }).click(); // a dense, evolving soup fills the frame
      await page.getByRole("slider", { name: "Speed" }).fill("60");
      await page.getByRole("button", { name: "Play" }).click();
      // The sidebar counter (not the page text: the screen-reader summary also mentions the generation, throttled).
      await page.waitForFunction(() => Number([...document.querySelectorAll("span")].find((s) => s.textContent === "Generation")?.nextElementSibling?.textContent ?? 0) >= 14, null, { timeout: 60_000 });
      await page.getByRole("button", { name: "Pause" }).click();
      const box = await page.locator("canvas").boundingBox();
      await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
      for (let i = 0; i < 9; i++) await page.mouse.wheel(0, -100); // ~2.4x: cells large enough to read, still a rich texture
      await wait(1500);
    },
  },
];

/** Reads the canvas pixels in the frame they are presented, crops to the poster size, encodes WebP. */
async function grab(page, focus) {
  return page.evaluate(
    async ({ w, h, focus }) => {
      const canvas = document.querySelector("canvas");
      const blob = await new Promise((resolve) => requestAnimationFrame(() => canvas.toBlob(resolve, "image/png")));
      const bitmap = await createImageBitmap(blob);
      const scale = Math.max(w / bitmap.width, h / bitmap.height);
      const sw = w / scale;
      const sh = h / scale;
      const out = new OffscreenCanvas(w, h);
      const ctx = out.getContext("2d");
      ctx.drawImage(bitmap, (bitmap.width - sw) * focus.x, (bitmap.height - sh) * focus.y, sw, sh, 0, 0, w, h);
      // A blank/black capture (frame not ready) has almost no bright pixels.
      const { data } = ctx.getImageData(0, 0, w, h);
      let bright = 0;
      for (let i = 0; i < data.length; i += 4) if (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2] > 90) bright++;
      const webp = await out.convertToBlob({ type: "image/webp", quality: 0.84 });
      const bytes = new Uint8Array(await webp.arrayBuffer());
      let binary = "";
      for (const b of bytes) binary += String.fromCharCode(b);
      return { b64: btoa(binary), bright: bright / (w * h), source: [bitmap.width, bitmap.height] };
    },
    { w: WIDTH, h: HEIGHT, focus },
  );
}

await withServer(
  async (base) => {
    const browser = await chromium.launch({
      channel: "chromium",
      args: ["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"],
    });
    fs.mkdirSync(OUT_DIR, { recursive: true });

    for (const poster of POSTERS) {
      // Viewport chosen so the canvas is already close to the poster's proportions (sidebar is 384 px wide).
      const context = await browser.newContext({ viewport: { width: (poster.canvasWidth ?? WIDTH) + 384, height: HEIGHT + 64 }, deviceScaleFactor: 1 });
      await context.addInitScript(() => localStorage.setItem("edusim:quality", "full"));
      if (poster.init) await context.addInitScript(poster.init);
      const page = await context.newPage();
      await page.goto(base + poster.route);
      await page.locator("canvas").waitFor();
      await page.waitForSelector("[data-testid=gpu-status][data-state=loading]", { state: "detached", timeout: 60_000 });
      await poster.prepare(page);

      let result;
      for (let attempt = 1; attempt <= 4; attempt++) {
        result = await grab(page, poster.focus);
        if (result.bright > 0.004) break;
        await wait(2500);
      }
      if (result.bright <= 0.004) throw new Error(`${poster.id}: capture looks blank (bright fraction ${result.bright.toFixed(4)})`);
      const file = path.join(OUT_DIR, `${poster.id}.webp`);
      fs.writeFileSync(file, Buffer.from(result.b64, "base64"));
      console.log(`ok  ${file}  ${(fs.statSync(file).size / 1024).toFixed(0)} KB  (canvas ${result.source.join("×")}, bright ${(result.bright * 100).toFixed(1)}%)`);
      await context.close();
    }

    // Social-preview image: the posters and the headline, set in the site's own fonts and tokens.
    const context = await browser.newContext({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
    const page = await context.newPage();
    await page.goto(base + "/");
    const dataUrls = POSTERS.map((p) => `data:image/webp;base64,${fs.readFileSync(path.join(OUT_DIR, `${p.id}.webp`)).toString("base64")}`);
    await page.evaluate((images) => {
      document.body.innerHTML = `
        <div style="width:1200px;height:630px;background:radial-gradient(900px 420px at 78% -8%,rgba(94,234,212,.10),transparent 70%),#0a0c10;display:flex;align-items:center;padding:0 56px;gap:40px;font-family:var(--font-geist-sans),sans-serif;color:#e9ebef;overflow:hidden">
          <div style="width:360px;flex:none">
            <div style="font:500 14px var(--font-geist-mono),monospace;letter-spacing:.14em;color:#5eead4;text-transform:uppercase;margin-bottom:22px">Interactive · GPU-computed</div>
            <div style="font-size:54px;line-height:1.04;font-weight:500;letter-spacing:-.035em">Science you can <span style="color:#939bab">reach into.</span></div>
            <div style="margin-top:26px;font-size:18px;line-height:1.45;color:#939bab">Live models of waves, orbits, neurons and cellular life, computed on your graphics card.</div>
            <div style="margin-top:34px;font:500 22px var(--font-geist-sans),sans-serif;letter-spacing:-.01em">EduSim</div>
          </div>
          <div style="display:flex;gap:12px">${images.map((src) => `<img src="${src}" style="width:${images.length > 3 ? 168 : 214}px;height:${images.length > 3 ? 185 : 235}px;object-fit:cover;border:1px solid rgba(255,255,255,.14);border-radius:3px;display:block">`).join("")}</div>
        </div>`;
      document.body.style.margin = "0";
    }, dataUrls);
    await page.waitForFunction(() => [...document.images].every((i) => i.complete && i.naturalWidth > 0));
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: "src/app/opengraph-image.jpg", type: "jpeg", quality: 88 });
    console.log(`ok  src/app/opengraph-image.jpg  ${(fs.statSync("src/app/opengraph-image.jpg").size / 1024).toFixed(0)} KB`);
    await browser.close();
  },
  { build: !process.argv.includes("--skip-build") },
);
