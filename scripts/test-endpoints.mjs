#!/usr/bin/env node
import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8787";
const OUT_DIR = process.env.OUT_DIR ?? "./out";
const PROMPT = process.env.PROMPT ?? "A serene mountain landscape at sunrise with misty valleys";
const WIDTH = Number(process.env.WIDTH ?? 1024);
const HEIGHT = Number(process.env.HEIGHT ?? 1024);

async function testTxt2Txt() {
    const url = `${BASE_URL}/ai/txt2txt?prompt=${encodeURIComponent(PROMPT)}`;
    console.log(`\nGET ${url}`);
    const t0 = Date.now();
    const res = await fetch(url);
    const ms = Date.now() - t0;

    const body = await res.json().catch(() => null);
    console.log(`  status=${res.status}  time=${ms}ms`);
    console.dir(body, { depth: null });
}

async function testTxt2Img() {
    const url = `${BASE_URL}/ai/txt2img/${WIDTH}/${HEIGHT}?prompt=${encodeURIComponent(PROMPT)}`;
    console.log(`\nGET ${url}`);
    const t0 = Date.now();
    const res = await fetch(url);
    const ms = Date.now() - t0;

    if (!res.ok) {
        const err = await res.text();
        console.error(`  status=${res.status}  time=${ms}ms`);
        console.error(`  error: ${err}`);
        return;
    }

    const body = await res.json();
    if (!body?.data) {
        console.error(`  status=${res.status}  time=${ms}ms`);
        console.error(`  no data field in response:`, body);
        return;
    }

    const buf = Buffer.from(body.data, "base64");
    const ext = detectExt(buf);

    await mkdir(OUT_DIR, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const outPath = join(OUT_DIR, `img-${stamp}.${ext}`);
    await writeFile(outPath, buf);

    console.log(`  status=${res.status}  time=${ms}ms  size=${(buf.length / 1024).toFixed(1)} KB`);
    console.log(`  saved ${outPath}`);
}

async function testTxt2ImgLayered() {
    const layers = Number(process.env.LAYERS ?? 4);
    const url = `${BASE_URL}/ai/txt2img-layered/${WIDTH}/${HEIGHT}?prompt=${encodeURIComponent(
        PROMPT
    )}&layers=${layers}`;
    console.log(`\nGET ${url}`);
    const t0 = Date.now();
    const res = await fetch(url);
    const ms = Date.now() - t0;

    const body = await res.json().catch(() => null);
    if (!res.ok || !Array.isArray(body?.layers)) {
        console.error(`  status=${res.status}  time=${ms}ms`);
        console.error(`  unexpected response:`, body);
        return;
    }

    console.log(`  status=${res.status}  time=${ms}ms  layers=${body.layers.length}`);
    body.layers.forEach((u, i) => console.log(`  [${i}] ${u}`));
}

function detectExt(buf) {
    if (buf[0] === 0xff && buf[1] === 0xd8) return "jpg";
    if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return "png";
    if (buf[0] === 0x52 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x46) return "webp";
    return "bin";
}

const mode = process.argv[2] ?? "all";

if (mode === "txt" || mode === "all") await testTxt2Txt();
if (mode === "img" || mode === "all") await testTxt2Img();
if (mode === "layered") await testTxt2ImgLayered();
