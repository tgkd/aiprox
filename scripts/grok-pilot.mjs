import { writeFileSync, mkdirSync } from "node:fs";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8787";
const SECRET = process.env.APP_SECRET;
const OUT = process.env.OUT_DIR ?? "out/pilot";

const PROMPTS = [
    "mountain landscape at dawn",
    "modern office interior",
    "coffee shop atmosphere",
    "tech startup teamwork",
    "autumn forest path",
    "city skyline at night",
    "minimal desk workspace",
    "ocean waves on rocks",
    "abstract fluid gradient",
    "cozy reading nook",
    "fitness training session",
    "healthy food ingredients",
    "travel suitcase adventure",
    "music studio equipment",
    "spring flowers meadow",
    "winter snow landscape",
    "luxury hotel lobby",
    "creative art supplies",
    "business handshake meeting",
    "yoga meditation calm",
];

const EXTRA_FORMATS = [
    ["sunset over calm water", 1080, 1080],
    ["marble stone texture", 1080, 1080],
    ["desert dunes at dusk", 1080, 1350],
    ["rainy window glass", 1080, 1350],
    ["forest canopy from below", 1400, 788],
    ["neon lit alleyway", 1400, 788],
    ["soft cloud formations", 1200, 628],
    ["wooden table flatlay", 1200, 628],
];

mkdirSync(OUT, { recursive: true });

async function one(idx, prompt, width, height) {
    const url = `${BASE_URL}/ai/txt2img/${width}/${height}?prompt=${encodeURIComponent(prompt)}`;
    const t0 = Date.now();
    try {
        const res = await fetch(url, { headers: { Authorization: `Bearer ${SECRET}` } });
        const ms = Date.now() - t0;
        if (!res.ok) {
            const body = await res.text();
            return { idx, prompt, width, height, ms, status: res.status, error: body.slice(0, 200) };
        }
        const json = await res.json();
        if (!json.data) {
            return { idx, prompt, width, height, ms, status: res.status, error: "no data field" };
        }
        const buf = Buffer.from(json.data, "base64");
        const name = `p${String(idx).padStart(2, "0")}.jpg`;
        writeFileSync(`${OUT}/${name}`, buf);
        return { idx, prompt, width, height, ms, status: 200, file: name, bytes: buf.length };
    } catch (e) {
        return { idx, prompt, width, height, ms: Date.now() - t0, status: 0, error: String(e).slice(0, 200) };
    }
}

const jobs = [
    ...PROMPTS.map((p) => [p, 1400, 788]),
    ...EXTRA_FORMATS,
];

const results = [];
for (let i = 0; i < jobs.length; i++) {
    const [prompt, w, h] = jobs[i];
    const r = await one(i + 1, prompt, w, h);
    results.push(r);
    console.log(
        `${String(r.idx).padStart(2)}/${jobs.length}  ${String(r.ms / 1000).padStart(6)}s  ` +
            `${r.status}  ${w}x${h}  ${prompt}${r.error ? "  ERR: " + r.error : ""}`
    );
}

const ok = results.filter((r) => r.status === 200);
const lat = ok.map((r) => r.ms / 1000).sort((a, b) => a - b);
const pct = (q) => lat.length ? lat[Math.min(lat.length - 1, Math.floor(q * lat.length))] : NaN;
const summary = {
    total: results.length,
    ok: ok.length,
    failed: results.length - ok.length,
    over30s: ok.filter((r) => r.ms >= 30000).length,
    over20s: ok.filter((r) => r.ms >= 20000).length,
    over10s: ok.filter((r) => r.ms >= 10000).length,
    min: lat[0],
    p50: pct(0.5),
    p90: pct(0.9),
    p95: pct(0.95),
    max: lat[lat.length - 1],
    mean: +(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(2),
};
console.log("\nSUMMARY " + JSON.stringify(summary, null, 2));
writeFileSync(`${OUT}/results.json`, JSON.stringify({ summary, results }, null, 2));
