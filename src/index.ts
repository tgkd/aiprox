import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import OpenAI from "openai";

const SYS_PROMPT = `
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
  }),
);

app.get("/ai", async (c) => {
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
    prompt: SYS_PROMPT.replace("{{prompt}}", prompt),
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    stream: true,
  });

  return new Response(chatStream.toReadableStream(), {
    headers: {
      "content-type": "text/event-stream",
    },
  });
});

app.onError((err, c) => {
  console.error(err);
  return c.text("Internal Server Error", 500);
});

export default app;
