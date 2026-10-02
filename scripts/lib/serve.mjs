// Starts `next start` on a spare port for scripts that need a running production build.
import { spawn, spawnSync } from "node:child_process";

export async function withServer(run, { port = process.env.PORT ?? "3199", build = true } = {}) {
  if (build && spawnSync("pnpm", ["build"], { stdio: "inherit" }).status !== 0) throw new Error("build failed");
  const server = spawn("pnpm", ["exec", "next", "start", "-p", String(port)], { stdio: "ignore" });
  const stop = () => server.kill("SIGTERM");
  process.on("exit", stop);
  const base = `http://localhost:${port}`;
  let up = false;
  for (let i = 0; i < 60 && !up; i++) {
    up = await fetch(base).then((r) => r.ok, () => false);
    if (!up) await new Promise((r) => setTimeout(r, 500));
  }
  if (!up) {
    stop();
    throw new Error(`server did not start on ${base}`);
  }
  try {
    return await run(base);
  } finally {
    stop();
  }
}
