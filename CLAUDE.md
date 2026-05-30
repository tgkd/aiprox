# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`aiprox` is a single-file Cloudflare Worker (Hono) that proxies AI generation for the
smmai social-media-banner app. It exposes two `GET` endpoints under `/ai/*`, gating
both with OpenAI moderation, and fans image generation out to a pluggable model registry.
All logic lives in `src/index.ts`. Deployed at `https://aiprox.smmai.workers.dev`.

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

Override the test via env vars: `PROMPT`, `WIDTH`, `HEIGHT`, `BASE_URL`, `OUT_DIR`.

## Bindings & secrets

Defined in `wrangler.toml`; typed in `worker-configuration.d.ts` (regenerate with `yarn cf-typegen`):

- `AI_KEY` (secret) — **Nebius** token-factory key, used as the OpenAI-SDK `apiKey` for txt2txt.
- `OPENAPI_KEY` (secret) — note the name: this is the **OpenAI** key, used *only* for the
  moderation calls (`/v1/moderations`), not for generation.
- `REPLICATE_API_TOKEN` (secret) — for the Replicate-backed image models.
- `AI` (binding) — Cloudflare Workers AI, for the `cf-*` image models.
- `IMG_MODEL` (var, default `flux-dev` in `wrangler.toml`) — selects the image adapter.

Set secrets locally via a gitignored `.dev.vars` file; in prod via `wrangler secret put <NAME>`.

## Architecture

**`GET /ai/txt2txt?prompt=`** — Moderates the prompt, then calls Nebius (OpenAI-compatible
SDK pointed at `api.tokenfactory.nebius.com`) with `Qwen/Qwen3-30B-A3B-Instruct-2507`,
`stream:false`, forcing a strict `json_schema` response (`TXT_RESPONSE_SCHEMA`) of 10
headline/subheadline pairs. Returns `{ response, created_at }`.

**`GET /ai/txt2img/:width/:height?prompt=`** — Resolves the adapter named by `IMG_MODEL`,
generates an image (returned as base64), then moderates the *image* before returning
`{ data: <base64> }`. Width/height are clamped to ≤1400.

**Image model registry (`IMG_MODELS`)** — the extension point. Each entry is an
`ImageModelAdapter` `(prompt, width, height, env) => Promise<base64string>`. Two backends:

- **Replicate** models (`flux-schnell`, `flux-dev`, `recraft-v3`, `nano-banana-2`) go through
  `callReplicate`, which POSTs with the `Prefer: wait` header (synchronous prediction) and
  then fetches+base64-encodes the resulting image URL.
- **Workers AI** models (`cf-flux-1-schnell`, `cf-flux-2-klein-4b`, `cf-lucid-origin`) call
  `env.AI.run(...)` and return the model's base64 `image` directly.

Replicate models take a discrete `aspect_ratio`/`size` string, not raw dimensions, so the
requested `width`/`height` are snapped to the closest allowed value via `nearestAspectRatio`
against per-model tables (`FLUX_ASPECT_RATIOS`, `NANO_BANANA_ASPECT_RATIOS`, `RECRAFT_SIZES`).
Workers AI models receive raw `width`/`height`.

**Prompt shaping** — every image prompt is wrapped in `IMG_PROMPT`, which steers toward
clean, text-overlay-friendly backgrounds (the banners get white text composited on top).

**Moderation** — `moderateContent` (OpenAI `omni-moderation-latest`) is fail-open: API
errors log and return `flagged:false` rather than blocking. Only a true `flagged` result
returns a 400. It accepts either a text string (txt2txt) or an `image_url` content array (txt2img).

**Cross-cutting** — CORS is locked to `https://smmai.app` and `https://demo.smmake.pages.dev`
for all `/ai/*` routes. Errors thrown as `HTTPException` propagate to Hono; the `onError`
handler logs and returns a 500. `[limits] cpu_ms = 10000` accommodates slow image generation.

## Adding an image model

Add an entry to `IMG_MODELS` keyed by a model name; if Replicate-backed, reuse `callReplicate`
and supply an aspect-ratio/size table for `nearestAspectRatio`. Switch the active model by
changing `IMG_MODEL` in `wrangler.toml` (or the deployed var) — no code change needed to select.
