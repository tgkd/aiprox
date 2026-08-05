# Migrating aiprox onto the Cloudflare AI Gateway

Status: **done, deployed 2026-08-05** (version `800d54ab`), with a narrower scope than planned
below: instead of migrating OpenAI moderation to the gateway, **moderation was removed entirely**,
and the Workers AI (`cf-*`) adapters were dropped — the end state is exactly two providers,
Nebius direct for text and Replicate via the gateway for images. `AI_GATEWAY_TOKEN` is
provisioned; `OPENAPI_KEY` and `REPLICATE_API_TOKEN` are no longer read and can be deleted after
soak. Verified live against prod: txt2txt 200, txt2img 200 through the gateway. One correction to
the text below: `moderateContent` was fail-*open* (errors logged and passed), not fail-closed.

Original plan, kept for the measured gateway facts:

Written 2026-08-04, after the sibling `aiwork` Worker made the
same move. Everything below marked *measured* was verified with real requests against the
`default` gateway on account `375e4092bc7903fb8ef13e363a41d370` — the same account this Worker
deploys to.

## Why

`aiprox` calls three providers directly, holding a key for each:

| credential | pays for | destination | call sites |
|---|---|---|---|
| `REPLICATE_API_TOKEN` | image generation and layer decomposition | `api.replicate.com` | `callReplicate` (`src/index.ts:167`), `callReplicateLayers` (`:210`), `waitForPrediction` (`:153`) |
| `OPENAPI_KEY` | OpenAI moderation, `omni-moderation-latest` | `api.openai.com` | `moderateContent` (`:344`), called from all three routes (`:461`, `:530`, `:577`) |
| `AI_KEY` | Nebius Token Factory, `Qwen/Qwen3-30B-A3B-Instruct-2507` | `api.tokenfactory.nebius.com` | `/ai/txt2txt` (`:453`) |
| `env.AI` binding | Workers AI image models — `flux-1-schnell`, `flux-2-klein-4b`, `lucid-origin` | Workers AI | `IMG_MODELS` (`:292`, `:299`, `:308`) |

Nothing in this Worker touches the gateway today — there is no `cf-aig` header and no gateway URL
anywhere in `src/`. So none of this traffic appears in gateway logs or analytics, there is no
unified cost view across the four providers, and three provider keys have to be managed in
`wrangler secret`.

Provider Keys (BYOK stored keys) are configured on the `default` gateway for **DeepSeek,
ElevenLabs, OpenAI and Replicate**. Two of this Worker's three secrets can therefore move to the
edge and stop existing here at all.

## The rule that makes or breaks each step

The gateway resolves the provider credential in three steps, and **the header you omit is what
picks the branch**:

1. an `Authorization` header on the request → forwarded to the provider unchanged, and it **wins**
   over everything below (measured: a junk value comes back as the provider's own 401, not a
   gateway error);
2. otherwise the stored provider key under the `default` alias → injected at the edge, billed to
   the provider account;
3. otherwise Unified Billing → spends the account's Cloudflare credit balance.

Two consequences worth internalising before touching any code:

- **Repointing a base URL is not the migration.** If the old `Authorization: Bearer <token>` header
  stays on the request, step 1 forwards it and the stored key is never used. The call still works,
  which is exactly why this is easy to get wrong — it looks migrated and isn't.
- **Step 3 fails silently.** A provider with no stored key does not error; it quietly bills
  Cloudflare credits. This is what will happen to Nebius if you point it at the gateway without
  reading the section below.

Diagnostic for either question, on any provider: send `cf-aig-byok-alias: no-such-alias`. A
provider with a stored key answers `Provider 'X' has no BYOK credential named …`; a provider
without one ignores the header entirely and falls through to step 3.

## What can move, and what cannot

### Replicate — moves cleanly

Measured, both of the path shapes this Worker uses:

```
POST …/default/replicate/models/{owner}/{model}/predictions   → Replicate's own 404 for a bogus model
GET  …/default/replicate/predictions/{id}                     → Replicate's own 404 for a bogus id
```

Those are Replicate's error shape (`{"detail":…,"status":404}`), not Cloudflare's `AiGatewayError`
— so the request reached Replicate with the stored key injected and merely hit a nonexistent
resource. `Prefer: wait` passes through intact, which matters because `callReplicate` depends on it.

Change: give the three fetches a gateway base, replace `Authorization` with
`cf-aig-authorization`, and drop the `token` parameter threaded through `callReplicate`,
`callReplicateLayers` and `waitForPrediction`.

**Do not rewrite the output fetch.** After a prediction succeeds, `src/index.ts:197` does
`fetch(url)` against the `replicate.delivery` CDN to pull the image bytes. That is not a Replicate
API call, carries no auth, and must stay direct.

`callReplicateLayers` has no such fetch — it returns the output URLs verbatim (`:238`), so the
client resolves them. Those stay `replicate.delivery` links either way; routing the *API* call
through the gateway does not put the image bytes behind it.

### OpenAI moderation — moves cleanly

Measured: `POST …/default/openai/moderations` with `omni-moderation-latest` returns 200 and a real
result object on the stored OpenAI key.

Note the gateway drops the `/v1` — it is `/openai/moderations`, not `/openai/v1/moderations`.

`moderateContent` takes an `apiKey` parameter today; it should take the env (or a prebuilt header
map) instead, and all three call sites lose their `c.env.OPENAPI_KEY` argument.

While you are in there: `OPENAPI_KEY` is a typo for `OPENAI_KEY`. Renaming it costs nothing extra
because every call site is already being edited.

### Nebius — **cannot** move as a native provider

Measured: `POST …/default/nebius/v1/chat/completions` returns
`{"code":2008,"message":"Invalid provider"}`. Nebius Token Factory is not in the gateway's provider
list.

Options, in order of preference:

1. **Leave it direct.** `AI_KEY` stays a Worker secret, `/ai/txt2txt` is unchanged. One provider
   outside the unified view is a small price and zero risk.
2. **Custom provider.** The gateway supports custom providers (account settings → then add a key
   for it on the Provider Keys page). This is the only route to genuine unification. Untested here.

Do **not** simply point the Nebius client at some other gateway path hoping it proxies — an
unrecognised provider is a 400, and a *recognised* one would silently bill Cloudflare credits per
step 3 above.

### Workers AI — nothing to migrate, but worth attaching

The three `cf-*` entries in `IMG_MODELS` go through the `env.AI` binding, so they never had a key
and produce no gateway logs. Passing a gateway option attaches them to the same logs and analytics
as everything else:

```ts
env.AI.run(model, input, { gateway: { id: "default" } })
```

Billing does not change — Workers AI bills the Cloudflare account either way. This is purely
observability, but without it "everything goes through the gateway" is false for whichever image
model `IMG_MODEL` currently selects.

## Prerequisite

**This Worker has no `AI_GATEWAY_TOKEN`.** Deployed secrets today are `AI_KEY`, `APP_SECRET`,
`OPENAPI_KEY`, `REPLICATE_API_TOKEN`. Before step 1:

```bash
npx wrangler secret put AI_GATEWAY_TOKEN
```

Use a Cloudflare AI Gateway token with Run permission. The gateway is authenticated — measured, a
request with no auth header at all returns `{"code":2009,"message":"Unauthorized"}` — so a missing
or wrong token fails everything, loudly. Make it fail loudly in code too rather than omitting the
header on an unset value; `aiwork`'s `gatewayToken()` is the pattern.

Add to `wrangler.toml` `[vars]`, alongside the existing `IMG_MODEL`:

```toml
AI_GATEWAY_ACCOUNT_ID = "375e4092bc7903fb8ef13e363a41d370"
AI_GATEWAY_ID = "default"
```

…and build URLs from them in one helper rather than hardcoding a base in three places.

## Order of work

Two independent deploys, smaller blast radius first. **Do not delete a secret in the same deploy as
its cutover.** Stop reading it, deploy, soak. The secret sits there unused, so `wrangler rollback`
restores the working direct-call version instantly. Delete only once real traffic looks right.

1. **Moderation** (`OPENAPI_KEY`). One function, three call sites, and it fails closed — a broken
   moderation call is visible immediately and cannot silently pass unsafe content, because
   `moderateContent` already reports its errors.
2. **Replicate** (`REPLICATE_API_TOKEN`). Three fetches plus the parameter-threading cleanup. Larger
   surface, and the poll loop means a mistake shows up as a timeout rather than an error.
3. **Workers AI gateway option** — optional, observability only, safe to fold into either deploy.
4. **After soak:** `wrangler secret delete OPENAPI_KEY`, `wrangler secret delete
   REPLICATE_API_TOKEN`.

## Verification

The one thing genuinely untested: **a prediction that actually succeeds through the gateway.** The
probes above proved routing and auth by hitting nonexistent resources, deliberately, to avoid
billing a real generation. Before trusting step 2, run one real prediction:

```bash
curl -X POST "https://gateway.ai.cloudflare.com/v1/375e4092bc7903fb8ef13e363a41d370/default/replicate/models/black-forest-labs/flux-schnell/predictions" \
  -H "cf-aig-authorization: Bearer $AI_GATEWAY_TOKEN" \
  -H "Content-Type: application/json" -H "Prefer: wait" \
  -d '{"input":{"prompt":"a red circle","num_outputs":1,"output_format":"jpg"}}'
```

Confirm the response carries a terminal `status` and an `output` URL — that is what `callReplicate`
parses.

Then, per deploy:

1. `npx wrangler deploy`
2. Exercise each route with a valid `Authorization: Bearer $APP_SECRET`:
   `/ai/txt2txt`, `/ai/txt2img/{w}/{h}`, `/ai/txt2img-layered/{w}/{h}`
3. **Gateway Logs tab** — new rows tagged Replicate and OpenAI. This is the positive signal that the
   request went through the gateway rather than direct; absence of errors is not.
4. Confirm the payer moved: the Replicate and OpenAI dashboards should show the traffic, and the
   Cloudflare credit balance should not move.
5. Exercise the real client — `smmaker-ios` (`SMmaker/Sources/AI/AIService.swift:8` points at
   `https://aiprox.smmai.workers.dev/ai`).

Rollback at any point: `npx wrangler rollback`. Valid only while the old secrets are still
provisioned.

## Gateway state as of writing

Measured, so these are facts about the current configuration rather than assumptions — recheck if
the migration happens much later:

- **Caching is off.** Two identical request bodies both returned `cf-aig-cache-status: MISS`. If
  that ever changes, image generation would start returning a previous user's image for an
  identical prompt.
- **Firewall is off.** Guardrails and DLP toggles are both disabled, so nothing inspects or blocks
  prompts. Note the dashboard's own warning: enabling Guardrails runs an additional Workers AI
  inference on *every* request, which on these routes means a cost and latency add on all three.
- **The gateway is authenticated** — no token, no service.

## Not in scope

- `APP_SECRET` and the `RATE_LIMITER` binding — this Worker's own auth and throttling, unrelated.
- The `IMG_PROMPT` wrapper and the aspect-ratio tables — provider routing does not touch them.
