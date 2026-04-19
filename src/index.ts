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
    "{{prompt}}. background-friendly image suitable for placing white text, one clear focal subject with moderate detail, wide smooth low-noise background areas, slightly darker overall tones for better contrast, soft controlled lighting, muted balanced colors, clean calm uncluttered composition, all surfaces and objects appear plain, blank and unmarked with no visible writing";

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

async function callReplicate(
    modelPath: string,
    input: Record<string, unknown>,
    token: string
): Promise<string> {
    const res = await fetch(`https://api.replicate.com/v1/models/${modelPath}/predictions`, {
        method: "POST",
        headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
            Prefer: "wait",
        },
        body: JSON.stringify({ input }),
    });

    if (!res.ok) {
        const body = await res.text();
        console.error(`Replicate API error: ${res.status}`, body);
        throw new HTTPException(502, { message: "Failed to generate image" });
    }

    const prediction = (await res.json()) as ReplicatePrediction;

    if (prediction.status !== "succeeded" || !prediction.output) {
        console.error(`Prediction ${prediction.status}:`, prediction.error);
        throw new HTTPException(502, {
            message: prediction.error ?? "Image generation failed",
        });
    }

    const url = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;

    const imgRes = await fetch(url);
    if (!imgRes.ok) {
        throw new HTTPException(502, { message: "Failed to fetch generated image" });
    }

    return arrayBufferToBase64(await imgRes.arrayBuffer());
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
            env.REPLICATE_API_TOKEN
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
            env.REPLICATE_API_TOKEN
        ),
    "recraft-v3": (prompt, width, height, env) =>
        callReplicate(
            "recraft-ai/recraft-v3",
            {
                prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
                size: nearestAspectRatio(width, height, RECRAFT_SIZES),
                style: "any",
            },
            env.REPLICATE_API_TOKEN
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
            env.REPLICATE_API_TOKEN
        ),
    "cf-flux-1-schnell": async (prompt, _width, _height, env) => {
        const result = (await env.AI.run("@cf/black-forest-labs/flux-1-schnell", {
            prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
            steps: 4,
        })) as { image: string };
        return result.image;
    },
    "cf-flux-2-klein-4b": async (prompt, width, height, env) => {
        const result = (await env.AI.run("@cf/black-forest-labs/flux-2-klein-4b" as any, {
            prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
            width,
            height,
            steps: 25,
        })) as { image: string };
        return result.image;
    },
    "cf-lucid-origin": async (prompt, width, height, env) => {
        const result = (await env.AI.run("@cf/leonardo/lucid-origin" as any, {
            prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
            width,
            height,
            steps: 25,
            guidance: 4.5,
        })) as { image: string };
        return result.image;
    },
};

const DEFAULT_IMG_MODEL = "flux-schnell";

async function moderateContent(
    content: string | Array<{ type: string; [key: string]: any }>,
    apiKey: string
): Promise<{ flagged: boolean; error?: string }> {
    try {
        const response = await fetch("https://api.openai.com/v1/moderations", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                input: content,
                model: "omni-moderation-latest",
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Moderation API error: ${response.status}`, errorText);
            return {
                flagged: false,
                error: `Moderation API error ${response.status}: ${errorText}`,
            };
        }

        const result = await response.json<OpenAI.Moderations.ModerationCreateResponse>();
        const flagged = result.results?.some((r: any) => r.flagged) || false;

        return { flagged };
    } catch (error) {
        console.error("Moderation error:", error);
        return {
            flagged: false,
            error: "Unknown moderation error",
        };
    }
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

app.get("/ai/txt2txt", async (c) => {
    const prompt = c.req.query("prompt");

    if (!prompt) {
        throw new HTTPException(400, { message: "Missing prompt" });
    }

    const nebius = new OpenAI({
        baseURL: "https://api.tokenfactory.nebius.com/v1/",
        apiKey: c.env.AI_KEY,
        defaultHeaders: {
            "Content-Type": "application/json",
            Accept: "*/*",
        },
    });

    const moderationResult = await moderateContent(prompt, c.env.OPENAPI_KEY);

    if (moderationResult.error) {
        console.error(`Moderation error for txt2txt: ${moderationResult.error}`);
    }

    if (moderationResult.flagged) {
        return c.json(
            {
                error: "Your input was flagged as inappropriate by our moderation system.",
            },
            400
        );
    }

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

    const moderationResult = await moderateContent(
        [
            {
                type: "image_url",
                image_url: { url: `data:image/jpeg;base64,${b64}` },
            },
        ],
        c.env.OPENAPI_KEY
    );

    if (moderationResult.error) {
        console.error(`Moderation error for image: ${moderationResult.error}`);
    }

    if (moderationResult.flagged) {
        return c.json(
            {
                error: "Your input was flagged as inappropriate by our moderation system.",
            },
            400
        );
    }

    return c.json({ data: b64 });
});

app.onError((err, c) => {
    console.error(err);
    return c.text("Internal Server Error", 500);
});

export default app;
