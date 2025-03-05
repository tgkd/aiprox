import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import OpenAI from "openai";

const TXT_SYS_PROMPT = `
You are a text generator that must follow these exact rules:

TASK: Generate 10 text pairs in "title::description" format.

FORMAT RULES:
title: always aim for 20 chars max but show last word complete if longer
description: always aim for 50 chars max but show last word complete if longer
separator: -----

TEXT HANDLING RULES:
- Never truncate or omit any words
- Preserve full meaning and context

INPUT CONTEXT: "{{prompt}}"

REQUIREMENTS:
- Output only the text pairs
- No additional text or instructions
- No bullet points or numbering
- Each pair must relate to the input context
- Must have exactly 10 pairs
- Must use ----- as separator

EXAMPLE OUTPUT:
Short Title::This is a sample description about the topic
-----
Another Title::Another relevant description following the format
-----
Final Title Here::Final description that relates to the given context

START OUTPUT NOW:
`;

const IMG_PROMPT =
    "{{prompt}}The main subject is centered with proper composition, illuminated by soft evenly diffused light. The image is clean without unnecessary details and noise, without distracting elements at the edges, featuring natural color reproduction and harmonious color palette. The overall mood of the photograph is positive and emotional.";

const IMG_NEGATIVE_PROMPT =
    "Blurriness, distortion, or inaccurate anatomy, busy or distracting backgrounds, unrealistic or overly saturated colors, signs of photo manipulation or artificial lighting. Bright highlights and overexposed areas, uneven exposure with deep shadows and high contrast. Distorted colors, overly bright and white objects. Noisy background with excessive detail and multiple distracting objects. Incorrect cropping, distorted proportions and complex angles. Text, letters and logos";

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
        baseURL: "https://api.studio.nebius.ai/v1/",
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
        return c.json({
            response: "Your input was flagged as inappropriate by our moderation system.",
            created_at: new Date().toISOString(),
        });
    }

    const response = await nebius.completions.create({
        prompt: TXT_SYS_PROMPT.replace("{{prompt}}", prompt),
        model: "meta-llama/Llama-3.3-70B-Instruct",
        stream: false,
        max_tokens: 512,
    });

    return c.json({
        response: response.choices.map((c) => c.text).join(""),
        created_at: response.created,
    });
});

interface ImageGenerationResponse {
    data: Array<{
        b64_json: string;
    }>;
    id: string;
}

app.get("/ai/txt2img/:width/:height", async (c) => {
    const width = Math.min(parseInt(c.req.param("width") ?? 512), 1400);
    const height = Math.min(parseInt(c.req.param("height") ?? 512), 1400);
    const prompt = c.req.query("prompt");

    if (!prompt) {
        throw new HTTPException(400, { message: "Missing prompt" });
    }

    const response = await fetch("https://api.studio.nebius.ai/v1/images/generations", {
        method: "POST",
        headers: {
            accept: "*/*",
            "content-type": "application/json",
            authorization: `Bearer ${c.env.AI_KEY}`,
        },
        body: JSON.stringify({
            model: "black-forest-labs/flux-schnell",
            prompt: IMG_PROMPT.replace("{{prompt}}", prompt),
            width,
            height,
            seed: -1,
            negative_prompt: IMG_NEGATIVE_PROMPT,
            num_inference_steps: 10,
            response_format: "b64_json",
            response_extension: "jpg",
        }),
    });

    if (!response.ok) {
        throw new HTTPException(400, { message: await response.text() });
    }

    const parsed = (await response.json()) as ImageGenerationResponse;

    const b64 = parsed.data?.[0]?.b64_json;

    if (!b64) {
        throw new HTTPException(404, { message: "No data" });
    }

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
        return c.json({
            response: "Your input was flagged as inappropriate by our moderation system.",
            created_at: new Date().toISOString(),
        });
    }

    return c.json({ data: b64 });
});

app.onError((err, c) => {
    console.error(err);
    return c.text("Internal Server Error", 500);
});

export default app;
