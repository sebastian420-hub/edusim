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

const files = walk("src");
let failed = 0;
for (const file of files) {
  const r = spawnSync("npx", ["vgpu", "check", file, "--require-validation"], { encoding: "utf8" });
  if (r.status === 0) {
    console.log(`ok    ${file}`);
  } else {
    failed++;
    console.error(`FAIL  ${file}\n${r.stdout}${r.stderr}`);
  }
}
console.log(`${files.length - failed}/${files.length} shaders valid`);
process.exit(failed ? 1 : 0);
