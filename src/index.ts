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
        return c.json(
            {
                error: "Your input was flagged as inappropriate by our moderation system.",
            },
            400
        );
    }

    const response = await nebius.chat.completions.create({
        model: "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
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
        return c.json(
            {
                error: "Failed to generate image",
            },
            500
        );
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
