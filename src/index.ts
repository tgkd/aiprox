import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import OpenAI from "openai";

const TXT_SYS_PROMPT = `
You are a text generator that must follow these exact rules:

TASK: Generate 3 text pairs in "title::description" format.

FORMAT RULES:
title: exactly 20 chars max
description: exactly 50 chars max
separator: -----

INPUT CONTEXT: "{{prompt}}"

REQUIREMENTS:
- Output only the text pairs
- No additional text or instructions
- No bullet points or numbering
- Each pair must relate to the input context
- Must have exactly 3 pairs
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

    c.header("Content-Encoding", "Identity");
    c.header("Cache-Control", "no-cache");
    c.header("Connection", "keep-alive");
    c.header("Content-Type", "text/event-stream");

    const chatStream = await openai.completions.create({
        prompt: TXT_SYS_PROMPT.replace("{{prompt}}", prompt),
        model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        stream: true,
    });

    return new Response(chatStream.toReadableStream(), {
        headers: {
            "content-type": "text/event-stream",
        },
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

    // response example { "data": [ { "b64_json": "..." }, ], "id": "text2img-90862075-08b4-4de4-bb86-d3934fdf2ca1" }

    const parsed = (await response.json()) as ImageGenerationResponse;

    return parsed.data?.[0]?.b64_json
        ? c.json({ data: parsed.data[0].b64_json })
        : c.text("No data", 404);
});

// curl 'https://api.studio.nebius.ai/v1/images/generations' \
//   -H 'accept: */*' \
//   -H 'accept-language: en-US,en;q=0.9,sr;q=0.8,ru;q=0.7' \
//   -H 'authorization: Bearer ' \
//   -H 'cache-control: no-cache' \
//   -H 'content-type: application/json' \
//   -H 'origin: https://studio.nebius.ai' \
//   -H 'pragma: no-cache' \
//   -H 'priority: u=1, i' \
//   -H 'referer: https://studio.nebius.ai/' \
//   -H 'sec-ch-ua: "Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"' \
//   -H 'sec-ch-ua-mobile: ?0' \
//   -H 'sec-ch-ua-platform: "macOS"' \
//   -H 'sec-fetch-dest: empty' \
//   -H 'sec-fetch-mode: cors' \
//   -H 'sec-fetch-site: same-site' \
//   -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' \
//   --data-raw '{"model":"stability-ai/sdxl","prompt":"another one","width":1024,"height":1024,"seed":-1,"negative_prompt":"","num_inference_steps":50,"response_format":"b64_json","response_extension":"jpg"}'
//

app.onError((err, c) => {
    console.error(err);
    return c.text("Internal Server Error", 500);
});

export default app;
