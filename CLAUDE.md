# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aiprox` is a single-file Cloudflare Worker (Hono) that proxies AI generation for the
smmai social-media-banner app. It exposes three `GET` endpoints under `/ai/*` and talks to
exactly two providers: **Nebius** (direct) for text, **Replicate** (via the Cloudflare AI
Gateway) for images. All logic lives in `src/index.ts`. Deployed at
`https://aiprox.smmai.workers.dev`.

## Commands

This repo uses **Yarn (PnP)** — `.pnp.cjs` + `yarn.lock`, no committed `node_modules`.
Ignore the README's `npm` instructions; use `yarn`. The `package.json` scripts call
`wrangler`/`node` bare, so invoke them through yarn:

```bash
yarn dev                 # wrangler dev — local server at http://localhost:8787
yarn deploy              # wrangler deploy --minify (prod)
yarn cf-typegen          # regenerate worker-configuration.d.ts after editing bindings
yarn tsc --noEmit        # type check (no build step; wrangler bundles on dev/deploy)
```

There is no lint config and no unit-test framework. `test:api` runs **live integration
smoke tests** (`scripts/test-endpoints.mjs`) against a running server, not unit tests:

```bash
yarn test:api            # hit both endpoints on localhost:8787 (needs `yarn dev` running)
yarn test:api:txt        # txt2txt only
yarn test:api:img        # txt2img only — saves the returned image to ./out/
yarn test:api:prod       # run against the deployed worker
```

Override the test via env vars: `PROMPT`, `WIDTH`, `HEIGHT`, `BASE_URL`, `OUT_DIR`. The
`/ai/*` routes require the bearer secret, so pass it too or everything is a 401:

```bash
APP_SECRET=$(grep '^APP_SECRET=' .dev.vars | cut -d= -f2- | tr -d '"') yarn test:api
```

## Bindings & secrets

Defined in `wrangler.toml`; typed in `worker-configuration.d.ts` (regenerate with `yarn cf-typegen`):

- `AI_KEY` (secret) — **Nebius** token-factory key, used as the OpenAI-SDK `apiKey` for txt2txt.
  Nebius is not an AI Gateway provider, so this call stays direct.
- `AI_GATEWAY_TOKEN` (secret) — Cloudflare **AI Gateway** token (Run permission). The Worker
  holds no Replicate credential: the Replicate key is stored on the gateway's Provider Keys
  page (BYOK) and injected at the edge.
- `AI_GATEWAY_ACCOUNT_ID` / `AI_GATEWAY_ID` (vars) — the gateway URL is built from these in
  `gatewayBase`; never hardcode a gateway base elsewhere.
- `APP_SECRET` (secret) — shared bearer token the clients send; see the auth gate in `src/index.ts`.
- `IMG_MODEL` (var, default `grok-imagine-image` in `wrangler.toml`) — selects the image adapter.

Set secrets locally via a gitignored `.dev.vars` file; in prod via `wrangler secret put <NAME>`.

Gateway auth is the `cf-aig-authorization` header. **Never send an `Authorization` header to
the gateway** — it would be forwarded to the provider verbatim and take precedence over the
stored key, silently un-migrating the call while everything appears to work.

## Architecture

**`GET /ai/txt2txt?prompt=`** — Calls Nebius directly (OpenAI-compatible SDK pointed at
`api.tokenfactory.nebius.com`) with `Qwen/Qwen3-30B-A3B-Instruct-2507`, `stream:false`,
forcing a strict `json_schema` response (`TXT_RESPONSE_SCHEMA`) of 10 headline/subheadline
pairs. Returns `{ response, created_at }`.

**`GET /ai/txt2img/:width/:height?prompt=`** — Resolves the adapter named by `IMG_MODEL` and
returns the generated image as `{ data: <base64> }`. Width/height are clamped to ≤1400.

**`GET /ai/txt2img-layered/:width/:height?prompt=&layers=N`** — Same generation, then feeds
the image to `qwen/qwen-image-layered` for decomposition into N (2–8) PNG layers. Returns
`{ layers: [<url>...] }` — raw `replicate.delivery` URLs the client resolves itself. Without
`layers=` it behaves like `/ai/txt2img`.

**Image model registry (`IMG_MODELS`)** — the extension point. Each entry is an
`ImageModelAdapter` `(prompt, width, height, env) => Promise<base64string>`. All entries
(`flux-schnell`, `flux-dev`, `recraft-v3`, `grok-imagine-image`, `nano-banana-2`) are Replicate models reached
**through the AI Gateway** via `callReplicate`: POST with `Prefer: wait` (synchronous
prediction), poll to a terminal status via `waitForPrediction` (also through the gateway),
then fetch+base64-encode the resulting image URL. That final output fetch hits the
`replicate.delivery` CDN, carries no auth, and deliberately stays direct — do not route it
through the gateway.

Replicate models take a discrete `aspect_ratio`/`size` string, not raw dimensions, so the
requested `width`/`height` are snapped to the closest allowed value via `nearestAspectRatio`
against per-model tables (`FLUX_ASPECT_RATIOS`, `NANO_BANANA_ASPECT_RATIOS`, `RECRAFT_SIZES`).

**Prompt shaping** — every image prompt is wrapped in `IMG_PROMPT`, which steers toward clean
backgrounds with open space for the white text the client composites on top.

`IMG_PROMPT` must never name text. It once read "suitable for placing white text … no visible
writing", and image models rendered those very words into the picture — measured at 4 of 8 samples
carrying a fake headline ("Coffee chose", "Bant Backgroand I Chentt Eriendy"). Naming the thing
summons it, and negation does not suppress it; `grok-imagine-image` has no `negative_prompt` to fall
back on. State the requirement positively instead ("generous empty negative space in the upper
half", "plain untouched surfaces"). After the rewrite: 0 of 28. See `IMAGE-MODELS.md`.

**No moderation layer** — the OpenAI moderation step was removed with the gateway migration
(2026-08); the only content safety left is what the image providers enforce themselves
(e.g. flux models' built-in safety checker). The auth gate + per-IP rate limit are the
abuse controls.

**Cross-cutting** — CORS is locked to `https://smmai.app` and `https://demo.smmake.pages.dev`
for all `/ai/*` routes. Errors thrown as `HTTPException` propagate to Hono; the `onError`
handler logs and returns a 500. There is no `[limits]` block: the Paid-plan default is 30s of CPU,
and CPU time excludes time awaiting network I/O, so a `cpu_ms` cap would only restrict active
computation (parsing, base64) — it can never extend the Replicate wait. Image-route wall time is
governed by provider latency and caller timeouts (the iOS client aborts at 30s —
`smmaker-ios` `SMmaker/Sources/AI/AIService.swift:14`).

## Adding an image model

Add an entry to `IMG_MODELS` keyed by a model name; reuse `callReplicate` and supply an
aspect-ratio/size table for `nearestAspectRatio`. Switch the active model by changing
`IMG_MODEL` in `wrangler.toml` (or the deployed var) — no code change needed to select.
A non-Replicate provider would need its own gateway path *and* a stored provider key on the
gateway first — a recognised provider with no stored key does not fail, it silently bills
Cloudflare credits.
