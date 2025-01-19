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

const IMG_SYS_PROMPT = `
Generate an image based on the following prompt:
{{prompt}}
`;

const app = new Hono<{ Bindings: CloudflareBindings }>();

app.use(
    "/ai/*",
    cors({
        origin: ["https://mobile-textinput.smmake.pages.dev", "https://smmai.app"],
        allowHeaders: ["Content-Type", "Authorization"],
        allowMethods: ["POST", "GET", "OPTIONS"],
        exposeHeaders: ["Content-Length", "Content-Type"],
        maxAge: 600,
        credentials: true,
    })
);

app.get("/ai/txt2txt", async (c) => {
    const prompt = c.req.query("prompt");

    if (!prompt) {
        throw new HTTPException(400, { message: "Missing prompt" });
    }

    const openai = new OpenAI({
        baseURL: "https://api.studio.nebius.ai/v1/",
        apiKey: c.env.AI_KEY,
        defaultHeaders: {
            "Content-Type": "application/json",
            Accept: "*/*",
        },
    });

    // c.header("Content-Encoding", "Identity");
    // c.header("Cache-Control", "no-cache");
    // c.header("Connection", "keep-alive");
    // c.header("Content-Type", "text/event-stream");

    const response = await openai.completions.create({
        prompt: TXT_SYS_PROMPT.replace("{{prompt}}", prompt),
        model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        //model: "Qwen/Qwen2.5-Coder-32B-Instruct",
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
            model: "stability-ai/sdxl",
            prompt: IMG_SYS_PROMPT.replace("{{prompt}}", prompt),
            width,
            height,
            seed: -1,
            negative_prompt: "",
            num_inference_steps: 50,
            response_format: "b64_json",
            response_extension: "jpg",
        }),
    });

    if (!response.ok) {
        throw new HTTPException(400, { message: await response.text() });
    }

    const parsed = (await response.json()) as ImageGenerationResponse;

    return parsed.data?.[0]?.b64_json
        ? c.json({ data: parsed.data[0].b64_json })
        : c.text("No data", 404);
});

app.onError((err, c) => {
    console.error(err);
    return c.text("Internal Server Error", 500);
});

export default app;
