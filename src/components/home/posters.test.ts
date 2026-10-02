import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { implementedSimulations } from "@/lib/subjects";
import { POSTER } from "./poster";

const file = (id: string) => path.join(process.cwd(), "public/plates", `${id}.webp`);

/** Width and height of a WebP (lossy "VP8 " or extended "VP8X"), read from its header. */
function webpSize(buf: Buffer): [number, number] {
  expect(buf.toString("ascii", 0, 4)).toBe("RIFF");
  expect(buf.toString("ascii", 8, 12)).toBe("WEBP");
  const kind = buf.toString("ascii", 12, 16);
  if (kind === "VP8X") return [1 + buf.readUIntLE(24, 3), 1 + buf.readUIntLE(27, 3)];
  if (kind === "VP8 ") return [buf.readUInt16LE(26) & 0x3fff, buf.readUInt16LE(28) & 0x3fff];
  throw new Error(`unsupported WebP chunk ${kind}`);
}

describe("home page posters", () => {
  const sims = implementedSimulations();

  it("every implemented simulation has a poster and alt text", () => {
    for (const s of sims) {
      expect(fs.existsSync(file(s.id)), `public/plates/${s.id}.webp — run \`pnpm posters\``).toBe(true);
      expect(s.posterAlt, `${s.id} posterAlt`).toBeTruthy();
    }
  });

  it("are valid WebP at the expected size and light enough for a home page", () => {
    for (const s of sims) {
      const buf = fs.readFileSync(file(s.id));
      expect(webpSize(buf), s.id).toEqual([POSTER.width, POSTER.height]);
      expect(buf.length, `${s.id} bytes`).toBeGreaterThan(2_000); // not blank
      expect(buf.length, `${s.id} bytes`).toBeLessThan(120_000);
    }
  });

  it("there is a social-preview image", () => {
    const stat = fs.statSync(path.join(process.cwd(), "src/app/opengraph-image.jpg"));
    expect(stat.size).toBeGreaterThan(10_000);
  });
});
