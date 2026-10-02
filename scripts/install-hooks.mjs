// Points git at the versioned hooks in .githooks (runs from the `prepare` script after install).
// Silently does nothing outside a git checkout (e.g. when installed from a tarball).
import { execFileSync } from "node:child_process";

try {
  execFileSync("git", ["rev-parse", "--is-inside-work-tree"], { stdio: "ignore" });
  execFileSync("git", ["config", "core.hooksPath", ".githooks"]);
  console.log("git hooks installed (.githooks/pre-push runs `pnpm verify`)");
} catch {
  // not a git repository
}
