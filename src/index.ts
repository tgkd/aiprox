import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import OpenAI from "openai";

const TXT_SYS_PROMPT = `
Create 10 meaningful and original headline and subheadline pairs for a social media banner based on the user's theme.

Important:
- Do not repeat or rephrase the user's input
- Invent fresh, original messages that fit the theme but express new ideas
- The tone must feel natural and suitable for any type of social media post: personal, inspirational, informational, or promotional
- Headlines should deliver clear, engaging ideas
- Subheadlines should add helpful context or nuance without restating the headline

Content Rules:
- Headline: maximum 25 characters (complete words only, never truncate)
- Subheadline: maximum 60 characters (complete words only, never truncate)
- No emojis, decorative punctuation, or artificial stylistic symbols
- Use natural, concise English phrasing
- Preserve full meaning and context
`;

const TXT_RESPONSE_SCHEMA = {
    name: "text_pairs_schema",
    strict: true,
    schema: {
        type: "object",
        properties: {
            pairs: {
                type: "array",
                items: {
                    type: "object",
                    properties: {
                        headline: {
                            type: "string",
                        },
                        subheadline: {
                            type: "string",
                        },
                    },
                    required: ["headline", "subheadline"],
                    additionalProperties: false,
                },
            },
        },
        required: ["pairs"],
        additionalProperties: false,
    },
};

const IMG_PROMPT =
    "{{prompt}}. one clear focal subject with moderate detail, wide smooth low-noise open areas, generous empty negative space in the upper half, slightly darker overall tones for stronger contrast, soft controlled lighting, muted balanced colors, clean calm uncluttered composition, plain untouched surfaces, smooth unbroken materials";

type ImageModelAdapter = (
    prompt: string,
    width: number,
    height: number,
    env: CloudflareBindings
) => Promise<string>;

const FLUX_ASPECT_RATIOS: Array<[string, number]> = [
    ["1:1", 1],
    ["16:9", 16 / 9],
    ["21:9", 21 / 9],
    ["3:2", 3 / 2],
    ["2:3", 2 / 3],
    ["4:5", 4 / 5],
    ["5:4", 5 / 4],
    ["3:4", 3 / 4],
    ["4:3", 4 / 3],
    ["9:16", 9 / 16],
    ["9:21", 9 / 21],
];

const GROK_ASPECT_RATIOS: Array<[string, number]> = [
    ["1:1", 1],
    ["16:9", 16 / 9],
    ["9:16", 9 / 16],
    ["4:3", 4 / 3],
    ["3:4", 3 / 4],
    ["3:2", 3 / 2],
    ["2:3", 2 / 3],
    ["2:1", 2],
    ["1:2", 1 / 2],
    ["19.5:9", 19.5 / 9],
    ["9:19.5", 9 / 19.5],
    ["20:9", 20 / 9],
    ["9:20", 9 / 20],
];

const NANO_BANANA_ASPECT_RATIOS: Array<[string, number]> = [
    ["1:1", 1],
    ["16:9", 16 / 9],
    ["21:9", 21 / 9],
    ["3:2", 3 / 2],
    ["2:3", 2 / 3],
    ["4:5", 4 / 5],
    ["5:4", 5 / 4],
    ["3:4", 3 / 4],
    ["4:3", 4 / 3],
    ["9:16", 9 / 16],
];

const RECRAFT_SIZES: Array<[string, number]> = [
    ["1024x1024", 1],
    ["1365x1024", 1365 / 1024],
    ["1024x1365", 1024 / 1365],
    ["1536x1024", 1536 / 1024],
    ["1024x1536", 1024 / 1536],
    ["1820x1024", 1820 / 1024],
    ["1024x1820", 1024 / 1820],
    ["2048x1024", 2048 / 1024],
    ["1024x2048", 1024 / 2048],
    ["1434x1024", 1434 / 1024],
    ["1024x1434", 1024 / 1434],
    ["1280x1024", 1280 / 1024],
    ["1024x1280", 1024 / 1280],
    ["1707x1024", 1707 / 1024],
    ["1024x1707", 1024 / 1707],
];

function nearestAspectRatio(width: number, height: number, choices: Array<[string, number]>): string {
    const target = width / height;
    let best = choices[0][0];
    let bestDiff = Infinity;
    for (const [label, ratio] of choices) {
        const diff = Math.abs(ratio - target);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = label;
        }
    }
    return best;
}

type ReplicatePrediction = {
    id: string;
    status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
    output: string | string[] | null;
    error: string | null;
};

function arrayBufferToBase64(buf: ArrayBuffer): string {
    const bytes = new Uint8Array(buf);
    const chunkSize = 0x8000;
    const chunks: string[] = [];
    for (let i = 0; i < bytes.length; i += chunkSize) {
        chunks.push(String.fromCharCode(...bytes.subarray(i, i + chunkSize)));
    }
    return btoa(chunks.join(""));
}

/**
 * All Replicate API calls go through the Cloudflare AI Gateway. The Worker holds no Replicate
 * key: the gateway injects the stored provider key (BYOK) at the edge. Auth must be the
 * `cf-aig-authorization` header — a plain `Authorization` header would be forwarded to the
 * provider verbatim and take precedence over the stored key, silently un-migrating the call.
 */
function gatewayBase(env: CloudflareBindings): string {
    return `https://gateway.ai.cloudflare.com/v1/${env.AI_GATEWAY_ACCOUNT_ID}/${env.AI_GATEWAY_ID}`;
}

/** The gateway is authenticated — without a token every image route fails, so fail loudly. */
function gatewayHeaders(env: CloudflareBindings): Record<string, string> {
    if (!env.AI_GATEWAY_TOKEN) {
        throw new HTTPException(500, { message: "AI_GATEWAY_TOKEN is not configured" });
    }
    return { "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}` };
}

const TERMINAL_STATUSES = new Set(["succeeded", "failed", "canceled"]);

/**
 * `Prefer: wait` can return before the model finishes (status "starting"/"processing") — the
 * dashboard logs showed repeated "Prediction starting" 502s from this. Poll the prediction to a
 * terminal state so a slow/cold start resolves instead of erroring. Bounded (~30s) so a stuck
 * prediction can't hang the request.
 */
async function waitForPrediction(
    prediction: ReplicatePrediction,
    env: CloudflareBindings
): Promise<ReplicatePrediction> {
    let current = prediction;
    for (let i = 0; i < 20 && !TERMINAL_STATUSES.has(current.status); i++) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        const res = await fetch(`${gatewayBase(env)}/replicate/predictions/${current.id}`, {
            headers: gatewayHeaders(env),
        });
        if (!res.ok) break;
        current = (await res.json()) as ReplicatePrediction;
    }
    return current;
}

async function callReplicate(
    modelPath: string,
    input: Record<string, unknown>,
    env: CloudflareBindings
): Promise<string> {
    const res = await fetch(`${gatewayBase(env)}/replicate/models/${modelPath}/predictions`, {
        method: "POST",
        headers: {
            ...gatewayHeaders(env),
            "Content-Type": "application/json",
            Prefer: "wait",
        },
        body: JSON.stringify({ input }),
    });

    if (!res.ok) {
        const body = await res.text();
        console.error(`Replicate API error: ${res.status}`, body);
        throw new HTTPException(502, { message: `Replicate error ${res.status}` });
    }

    let prediction = (await res.json()) as ReplicatePrediction;
    if (!TERMINAL_STATUSES.has(prediction.status)) {
        prediction = await waitForPrediction(prediction, env);
    }

    if (prediction.status !== "succeeded" || !prediction.output) {
        console.error(`Prediction ${prediction.status}:`, prediction.error);
        throw new HTTPException(502, {
            message: prediction.error ?? `Image generation ${prediction.status} (no output)`,
        });
    }

    const url = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;

    // The output URL is the replicate.delivery CDN, not the Replicate API — no auth, stays direct.
    const imgRes = await fetch(url);
    if (!imgRes.ok) {
        throw new HTTPException(502, { message: "Failed to fetch generated image" });
    }

    return arrayBufferToBase64(await imgRes.arrayBuffer());
}

async function callReplicateLayers(
    modelPath: string,
    input: Record<string, unknown>,
    env: CloudflareBindings
): Promise<string[]> {
    const res = await fetch(`${gatewayBase(env)}/replicate/models/${modelPath}/predictions`, {
        method: "POST",
        headers: {
            ...gatewayHeaders(env),
            "Content-Type": "application/json",
            Prefer: "wait",
        },
        body: JSON.stringify({ input }),
    });

    if (!res.ok) {
        const body = await res.text();
        console.error(`Replicate API error: ${res.status}`, body);
        throw new HTTPException(502, { message: "Failed to generate layers" });
    }

    let prediction = (await res.json()) as ReplicatePrediction;
    if (!TERMINAL_STATUSES.has(prediction.status)) {
        prediction = await waitForPrediction(prediction, env);
    }

    if (prediction.status !== "succeeded" || !prediction.output) {
        console.error(`Prediction ${prediction.status}:`, prediction.error);
        throw new HTTPException(502, {
            message: prediction.error ?? `Layer generation ${prediction.status} (no output)`,
        });
    }

    return Array.isArray(prediction.output) ? prediction.output : [prediction.output];
}

const IMG_MODELS: Record<string, ImageModelAdapter> = {
    "flux-schnell": (prompt, width, height, env) =>
        callReplicate(
            "black-forest-labs/flux-schnell",
            {
                prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
                aspect_ratio: nearestAspectRatio(width, height, FLUX_ASPECT_RATIOS),
                output_format: "jpg",
                output_quality: 90,
                go_fast: true,
                num_outputs: 1,
            },
            env
        ),
    "flux-dev": (prompt, width, height, env) =>
        callReplicate(
            "black-forest-labs/flux-dev",
            {
                prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
                aspect_ratio: nearestAspectRatio(width, height, FLUX_ASPECT_RATIOS),
                num_inference_steps: 28,
                guidance: 3,
                output_format: "jpg",
                output_quality: 90,
                go_fast: true,
                num_outputs: 1,
            },
            env
        ),
    "recraft-v3": (prompt, width, height, env) =>
        callReplicate(
            "recraft-ai/recraft-v3",
            {
                prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
                size: nearestAspectRatio(width, height, RECRAFT_SIZES),
                style: "any",
            },
            env
        ),
    "grok-imagine-image": (prompt, width, height, env) =>
        callReplicate(
            "xai/grok-imagine-image",
            {
                prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
                aspect_ratio: nearestAspectRatio(width, height, GROK_ASPECT_RATIOS),
            },
            env
        ),
    "nano-banana-2": (prompt, width, height, env) =>
        callReplicate(
            "google/nano-banana-2",
            {
                prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
                aspect_ratio: nearestAspectRatio(width, height, NANO_BANANA_ASPECT_RATIOS),
                resolution: "1K",
                output_format: "jpg",
            },
            env
        ),
};

const DEFAULT_IMG_MODEL = "flux-schnell";

const LAYERED_MODEL = "qwen/qwen-image-layered";

async function decomposeToLayers(
    b64Image: string,
    numLayers: number,
    env: CloudflareBindings
): Promise<string[]> {
    return callReplicateLayers(LAYERED_MODEL, {
        image: `data:image/jpeg;base64,${b64Image}`,
        num_layers: numLayers,
        output_format: "png",
    }, env);
}

/**
 * Constant-time string compare so a wrong bearer token can't be guessed byte-by-byte via response
 * timing. Length is allowed to leak (cheap, and the secret length isn't sensitive).
 */
function timingSafeEqual(a: string, b: string): boolean {
    const enc = new TextEncoder();
    const ab = enc.encode(a);
    const bb = enc.encode(b);
    if (ab.length !== bb.length) return false;
    let diff = 0;
    for (let i = 0; i < ab.length; i++) diff |= ab[i] ^ bb[i];
    return diff === 0;
}

const app = new Hono<{ Bindings: CloudflareBindings }>();

app.use(
    "/ai/*",
    cors({
        origin: ["https://smmai.app", "https://demo.smmake.pages.dev"],
        allowHeaders: ["Content-Type", "Authorization"],
        allowMethods: ["POST", "GET", "OPTIONS"],
        exposeHeaders: ["Content-Length", "Content-Type"],
        credentials: true,
    })
);

/**
 * Gate `/ai/*` behind a shared bearer secret plus a per-IP rate limit.
 *
 * CORS does NOT protect this proxy: it's browser-only, so native apps (URLSession), curl, and scripts
 * ignore it entirely — the endpoints were effectively open to the public internet while proxying paid
 * APIs. This gate closes that.
 *
 * `APP_SECRET` is a LOW-STRENGTH speed bump: the app ships the secret in its binary, so it can be
 * extracted (`strings`, a TLS-intercepting proxy). It blocks the open URL and casual/scripted abuse,
 * but is NOT real client authentication — for "this is my unmodified app" proof, move to Apple App
 * Attest. The per-IP `RATE_LIMITER` is the real backstop: it bounds the bill even if the secret leaks.
 *
 * Runs after the cors middleware so browser preflight (OPTIONS, which carries no Authorization header)
 * is answered there and never reaches this gate.
 */
app.use("/ai/*", async (c, next) => {
    if (c.req.method === "OPTIONS") return next();

    const expected = c.env.APP_SECRET;
    if (!expected) {
        // Fail closed: a missing secret must not silently leave the proxy open.
        console.error("APP_SECRET is not configured");
        throw new HTTPException(500, { message: "Auth not configured" });
    }

    const header = c.req.header("Authorization") ?? "";
    const token = header.startsWith("Bearer ") ? header.slice("Bearer ".length) : "";
    if (!timingSafeEqual(token, expected)) {
        throw new HTTPException(401, { message: "Unauthorized" });
    }

    const ip = c.req.header("CF-Connecting-IP") ?? "unknown";
    const { success } = await c.env.RATE_LIMITER.limit({ key: ip });
    if (!success) {
        throw new HTTPException(429, { message: "Too many requests. Please wait and try again." });
    }

    return next();
});

app.get("/ai/txt2txt", async (c) => {
    const prompt = c.req.query("prompt");

    if (!prompt) {
        throw new HTTPException(400, { message: "Missing prompt" });
    }

    // Nebius Token Factory is not an AI Gateway provider, so this stays a direct call on AI_KEY.
    const nebius = new OpenAI({
        baseURL: "https://api.tokenfactory.nebius.com/v1/",
        apiKey: c.env.AI_KEY,
        defaultHeaders: {
            "Content-Type": "application/json",
            Accept: "*/*",
        },
    });

    const response = await nebius.chat.completions.create({
        model: "Qwen/Qwen3-30B-A3B-Instruct-2507",
        stream: false,
        max_tokens: 512,
        temperature: 0,
        top_p: 0.9,
        response_format: {
            type: "json_schema",
            json_schema: TXT_RESPONSE_SCHEMA,
        },
        messages: [
            {
                role: "system",
                content: TXT_SYS_PROMPT.replace("{{prompt}}", prompt),
            },
            {
                role: "user",
                content: prompt,
            },
        ],
    });

    const content = response.choices[0].message.content;
    const parsedContent = content ? JSON.parse(content) : { pairs: [] };

    return c.json({
        response: parsedContent,
        created_at: response.created,
    });
});

app.get("/ai/txt2img/:width/:height", async (c) => {
    const width = Math.min(parseInt(c.req.param("width") ?? 512), 1400);
    const height = Math.min(parseInt(c.req.param("height") ?? 512), 1400);
    const prompt = c.req.query("prompt");

    if (!prompt) {
        throw new HTTPException(400, { message: "Missing prompt" });
    }

    const modelKey = c.env.IMG_MODEL ?? DEFAULT_IMG_MODEL;
    const adapter = IMG_MODELS[modelKey];
    if (!adapter) {
        throw new HTTPException(500, { message: `Unknown image model: ${modelKey}` });
    }
    const b64 = await adapter(prompt, width, height, c.env);

    return c.json({ data: b64 });
});

app.get("/ai/txt2img-layered/:width/:height", async (c) => {
    const width = Math.min(parseInt(c.req.param("width") ?? 512), 1400);
    const height = Math.min(parseInt(c.req.param("height") ?? 512), 1400);
    const prompt = c.req.query("prompt");

    if (!prompt) {
        throw new HTTPException(400, { message: "Missing prompt" });
    }

    const layersParam = c.req.query("layers");
    const layered = layersParam !== undefined;
    const numLayers =
        layersParam !== undefined ? Math.min(Math.max(parseInt(layersParam) || 4, 2), 8) : 0;

    const modelKey = c.env.IMG_MODEL ?? DEFAULT_IMG_MODEL;
    const adapter = IMG_MODELS[modelKey];
    if (!adapter) {
        throw new HTTPException(500, { message: `Unknown image model: ${modelKey}` });
    }
    const b64 = await adapter(prompt, width, height, c.env);

    if (!layered) {
        return c.json({ data: b64 });
    }

    const layers = await decomposeToLayers(b64, numLayers, c.env);

    return c.json({ layers });
});

app.onError((err, c) => {
    // Preserve intentional HTTP errors (401/429 from the auth gate, 400/502 from handlers) instead of
    // masking them all as 500 — a custom onError replaces Hono's default HTTPException rendering, so we
    // must call getResponse() ourselves.
    if (err instanceof HTTPException) {
        return err.getResponse();
    }
    console.error(err);
    return c.text("Internal Server Error", 500);
});

export default app;
