// Builds the app, serves it on a spare port, runs the browser smoke test against it, then stops it.
//   pnpm smoke:local            (add --skip-build to reuse an existing .next build)
import { spawn, spawnSync } from "node:child_process";

const PORT = process.env.PORT ?? "3199";
const run = (cmd, args, opts = {}) => spawnSync(cmd, args, { stdio: "inherit", ...opts });

if (!process.argv.includes("--skip-build")) {
  if (run("pnpm", ["build"]).status !== 0) process.exit(1);
}

const server = spawn("pnpm", ["exec", "next", "start", "-p", PORT], { stdio: "ignore" });
const stop = () => server.kill("SIGTERM");
process.on("exit", stop);

const url = `http://localhost:${PORT}`;
let up = false;
for (let i = 0; i < 60 && !up; i++) {
  up = await fetch(url).then((r) => r.ok, () => false);
  if (!up) await new Promise((r) => setTimeout(r, 500));
}
if (!up) {
  console.error(`server did not start on ${url}`);
  stop();
  process.exit(1);
}

const result = run("node", ["scripts/smoke.mjs"], { env: { ...process.env, BASE_URL: url } });
stop();
process.exit(result.status ?? 1);
