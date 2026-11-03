#!/usr/bin/env node
// Builds public/data/shakespeare.dat and public/data/names.dat for "How AI Writes": the two texts, gzipped.
//
//   node scripts/build-texts.mjs [dir-with-input.txt-and-names.txt]
//
// Without a directory the originals are downloaded (into .cache/texts) and checked against their MD5 sums:
// Tiny Shakespeare from Andrej Karpathy's char-rnn (public domain) and the makemore names (the 32 000 most common
// US baby names from the Social Security Administration's data, CC0). The files are stored unchanged; the names
// are shuffled in the app (text.ts), not here.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const FILES = {
  "input.txt": { md5: "6fb458f1232090904fb40fe944165e91", url: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", out: "shakespeare.dat" },
  "names.txt": { md5: "401b3c4cac9e28a586ed15e81256680d", url: "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt", out: "names.dat" },
};

const dir = process.argv[2] ?? ".cache/texts";
mkdirSync(dir, { recursive: true });
mkdirSync("public/data", { recursive: true });

for (const [name, { md5, url, out }] of Object.entries(FILES)) {
  const path = join(dir, name);
  if (!existsSync(path)) {
    console.log(`downloading ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    writeFileSync(path, Buffer.from(await res.arrayBuffer()));
  }
  const bytes = readFileSync(path);
  const sum = createHash("md5").update(bytes).digest("hex");
  if (sum !== md5) throw new Error(`${name}: MD5 ${sum}, expected ${md5}`);
  const packed = gzipSync(bytes, { level: 9 });
  writeFileSync(join("public/data", out), packed);
  console.log(`public/data/${out}: ${bytes.length} characters, ${packed.length} bytes gzipped`);
}
