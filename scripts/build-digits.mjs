#!/usr/bin/env node
// Builds public/data/mnist-12k.dat: 10 000 training and 2 000 test digits from MNIST, for "How AI Learns".
//
//   node scripts/build-digits.mjs [dir-with-the-four-mnist-gz-files]
//
// Without a directory the four original files are downloaded (into .cache/mnist) and checked against their
// published MD5 sums. Output: gzip of
//   "EDUD" | u32 count | u32 nTrain | u32 rows | u32 cols | count label bytes | count × rows × cols pixel bytes
// (little-endian). MNIST is by Yann LeCun, Corinna Cortes and Christopher J. C. Burges, CC BY-SA 3.0.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";

const FILES = {
  "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
  "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
  "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
  "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
};
const MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const N_TRAIN = 10_000;
const N_TEST = 2_000;

const dir = process.argv[2] ?? ".cache/mnist";
mkdirSync(dir, { recursive: true });

async function load(name) {
  const path = join(dir, name);
  if (!existsSync(path)) {
    const res = await fetch(MIRROR + name);
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    writeFileSync(path, Buffer.from(await res.arrayBuffer()));
  }
  const raw = readFileSync(path);
  const md5 = createHash("md5").update(raw).digest("hex");
  if (md5 !== FILES[name]) throw new Error(`${name}: MD5 ${md5}, expected ${FILES[name]}`);
  return gunzipSync(raw);
}

/** IDX file → { dims, data }. */
function idx(buf) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const ndim = buf[3];
  const dims = Array.from({ length: ndim }, (_, k) => view.getUint32(4 + 4 * k, false));
  return { dims, data: buf.subarray(4 + 4 * ndim) };
}

const trainImages = idx(await load("train-images-idx3-ubyte.gz"));
const trainLabels = idx(await load("train-labels-idx1-ubyte.gz"));
const testImages = idx(await load("t10k-images-idx3-ubyte.gz"));
const testLabels = idx(await load("t10k-labels-idx1-ubyte.gz"));
const [, rows, cols] = trainImages.dims;
const size = rows * cols;
const count = N_TRAIN + N_TEST;

const header = Buffer.alloc(20);
header.write("EDUD", 0, "ascii");
header.writeUInt32LE(count, 4);
header.writeUInt32LE(N_TRAIN, 8);
header.writeUInt32LE(rows, 12);
header.writeUInt32LE(cols, 16);
const labels = Buffer.concat([trainLabels.data.subarray(0, N_TRAIN), testLabels.data.subarray(0, N_TEST)]);
const pixels = Buffer.concat([trainImages.data.subarray(0, N_TRAIN * size), testImages.data.subarray(0, N_TEST * size)]);
const out = gzipSync(Buffer.concat([header, labels, pixels]), { level: 9 });
mkdirSync("public/data", { recursive: true });
writeFileSync("public/data/mnist-12k.dat", out);
console.log(`public/data/mnist-12k.dat: ${count} digits (${N_TRAIN} train, ${N_TEST} test), ${(out.length / 1e6).toFixed(2)} MB`);
