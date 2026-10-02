// Validates every .wgsl file against a real WebGPU device (`next build` never does).
import { spawnSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith(".wgsl") ? [p] : [];
  });
}

const NO_DEVICE = /VGPU-NODE-NO-ADAPTER|VGPU-WGSL-VALIDATE-NO-DEVICE/;
const files = walk("src");
let failed = 0;
let skipped = 0;
for (const file of files) {
  const r = spawnSync("npx", ["vgpu", "check", file, "--require-validation"], { encoding: "utf8" });
  if (r.status === 0) {
    console.log(`ok    ${file}`);
  } else if (NO_DEVICE.test(r.stdout + r.stderr) && !process.env.REQUIRE_GPU) {
    skipped++;
    console.warn(`skip  ${file} (no WebGPU adapter; run \`npx vgpu install-software-renderer\`)`);
  } else {
    failed++;
    console.error(`FAIL  ${file}\n${r.stdout}${r.stderr}`);
  }
}
console.log(`${files.length - failed - skipped}/${files.length} shaders valid${skipped ? `, ${skipped} skipped (no adapter)` : ""}`);
process.exit(failed ? 1 : 0);
