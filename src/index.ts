import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import OpenAI from "openai";

const TXT_SYS_PROMPT = `
Generate three different short texts with the format "title::description", following these rules:
- **Title**: Maximum 20 characters
- **Description**: Maximum 50 characters
- **Follow the user's prompt**: The titles and descriptions should be directly related to the topic or instructions provided by the user.
**User Prompt**: "{{prompt}}"
Ensure each pair is clear, engaging, and adheres to the character limits exactly.
NO NEED TO REPEAT THE PROMPT.
ALWAYS FOLLOW THE FORMAT "title::description" FOR EACH TEXT.
REMOVE ALL UNNECESSARY TEXT AND INSTRUCTIONS. KEEP ONLY THE TEXT TO BE GENERATED.
DO NOT INCLUDE POINTS OR BULLET POINTS IN THE GENERATED TEXT.
DIVIDE EACH TEXT WITH FIVE DASHES (-----).
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
    const width = Math.min(parseInt(c.req.param("width") ?? 512), 512);
    const height = Math.min(parseInt(c.req.param("height") ?? 512), 512);
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
            prompt,
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

    return response.json();
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
