# Text-to-image model comparison for `aiprox`

Data pulled 2026-08-29 from:

- [Replicate — text-to-image collection](https://replicate.com/collections/text-to-image) (per-model
  billing config scraped from each model page)
- [Cloudflare Workers AI — Text-to-Image catalog](https://developers.cloudflare.com/workers-ai/models/?tasks=Text-to-Image)
  and [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Artificial Analysis — Text-to-Image Arena leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image)
  (155+ models, ~6.0M human pairwise votes; the closest thing to "LMArena for images")
- [LMArena / arena.ai text-to-image board](https://arena.ai/leaderboard/text-to-image) — cross-check only

## Normalization

`aiprox` clamps to ≤1400px, so every price below is normalized to **one 1400×788 image**
(1.05 MP, 6× 512×512 tiles) — not to the vendor's headline unit. That matters: Cloudflare bills
image models per tile and per step, Replicate bills per output image or per output megapixel, and
those units are not comparable as printed.

Cloudflare cost model is **additive**: `tiles × tile_price + steps × step_price`
(verified against Cloudflare's own neuron arithmetic).

Quality = Artificial Analysis Arena Elo (overall text-to-image board). Elo is a human-preference
aesthetic score, not a measure of "good banner background" — see the caveat at the bottom.

## Comparison table — price, quality, source

Sorted by price ascending, as requested.

| Model | $ / image @1400×788 | Elo | AA rank | Source | Notes |
|---|---:|---:|---:|---|---|
| `@cf/black-forest-labs/flux-1-schnell` | **$0.00074** | 1000 | 135 | **CF native** | 4 steps. Cheapest credible option by 4×. |
| `black-forest-labs/flux-2-klein-4b` | $0.0011 | 1060 | 115 | 3rd party (Replicate) | $1 / 1000 output MP. |
| `@cf/black-forest-labs/flux-2-klein-4b` | $0.0017 | 1060 | 115 | **CF native** | No step term; flat per output tile. |
| `black-forest-labs/flux-schnell` | $0.0030 | 1000 | 135 | 3rd party (Replicate) | 4× the CF price for the same model. |
| `luma/photon-flash` | $0.0100 | 1004 | 134 | 3rd party (Replicate) | |
| `prunaai/z-image-turbo` | $0.0105 | 1134 | 74 | 3rd party (Replicate) | $0.01 / output MP tier. Strong Elo for the price. |
| `@cf/black-forest-labs/flux-2-klein-9b` | **$0.0151** | 1141 | 71 | **CF native** | $0.015 first MP + $0.002/MP after. |
| `google/imagen-4-fast` | $0.0200 | 1099 | 93 | 3rd party (Replicate) | |
| `xai/grok-imagine-image` | $0.0200 | 1217 | 26 | 3rd party (Replicate) | Best Elo under $0.025. |
| `black-forest-labs/flux-dev` | $0.0250 | 1040 | 122 | 3rd party (Replicate) | **current `IMG_MODEL` default** |
| `qwen/qwen-image` | $0.0250 | 1076 | 105 | 3rd party (Replicate) | |
| `bytedance/seedream-4` | $0.0300 | 1225 | 20 | 3rd party (Replicate) | |
| `ideogram-ai/ideogram-v3-turbo` | $0.0300 | 1101 | 92 | 3rd party (Replicate) | |
| `black-forest-labs/flux-2-pro` | $0.0308 | 1207 | 34 | 3rd party (Replicate) | $0.015/run + $0.015/output MP. |
| `bytedance/seedream-5-lite` | $0.0350 | 1198 | 42 | 3rd party (Replicate) | |
| `@cf/leonardo/phoenix-1.0` | $0.0377 | 1042* | 121* | **CF native** | *Elo is for Phoenix 1.0 Ultra. Bad price/quality. |
| `google/imagen-4` | $0.0400 | 1125 | 77 | 3rd party (Replicate) | |
| `recraft-ai/recraft-v3` | $0.0400 | 1077 | 103 | 3rd party (Replicate) | in `IMG_MODELS` registry |
| `bytedance/seedream-4.5` | $0.0400 | 1205 | 35 | 3rd party (Replicate) | |
| `@cf/leonardo/lucid-origin` | $0.0453 | 1123* | 78* | **CF native** | *Elo is for Lucid Origin Ultra. |
| `openai/gpt-image-2` (medium) | $0.0470 | 1370† | 1 | 3rd party (Replicate) | †Elo is the `high` tier. |
| `@cf/black-forest-labs/flux-2-dev` | $0.0492 | 1199 | 40 | **CF native** | 20 steps. $0.069 at 28 steps. |
| `google/imagen-4-ultra` | $0.0600 | 1190 | 45 | 3rd party (Replicate) | |
| `ideogram-ai/ideogram-v3-balanced` | $0.0600 | 1101 | 92 | 3rd party (Replicate) | |
| `black-forest-labs/flux-2-flex` | $0.0631 | 1222 | 21 | 3rd party (Replicate) | $0.06 / output MP. |
| `stability-ai/stable-diffusion-3.5-large` | $0.0650 | 1035 | 125 | 3rd party (Replicate) | Dominated on both axes. |
| `google/nano-banana-2` (1K) | $0.0670 | 1321 | 4 | 3rd party (Replicate) | in `IMG_MODELS` registry |
| `black-forest-labs/flux-2-max` | $0.0716 | 1226 | 18 | 3rd party (Replicate) | $0.04/run + $0.03/output MP. |
| `ideogram-ai/ideogram-v3-quality` | $0.0900 | 1101 | 92 | 3rd party (Replicate) | |
| `openai/gpt-image-2` (high) | $0.1280 | 1370 | **1** | 3rd party (Replicate) | Arena #1. 2.6× nano-banana-2. |
| `google/nano-banana-pro` (1K/2K) | $0.1500 | 1297 | 7 | 3rd party (Replicate) | |
| `recraft-ai/recraft-v4-pro` | $0.2500 | 1194 | 44 | 3rd party (Replicate) | Priced for vector/brand work, not banners. |

Cloudflare also lists `dreamshaper-8-lcm`, `stable-diffusion-xl-base-1.0`,
`stable-diffusion-xl-lightning`, `stable-diffusion-v1-5-*`. All are **beta**, all carry no entry in
the pricing table, and every one of them sits at Elo 883 or below (SDXL 1.0 = 883, SDXL Lightning =
911, SD 1.5 = 670 — dead last of 158). Ignore them.

## What the price/quality frontier actually looks like

Only these models are Pareto-optimal (nothing is both cheaper *and* better):

| $ / image | Model | Elo | Source |
|---:|---|---:|---|
| $0.00074 | `@cf/flux-1-schnell` | 1000 | CF |
| $0.0017 | `@cf/flux-2-klein-4b` | 1060 | CF |
| $0.0105 | `z-image-turbo` | 1134 | Replicate |
| $0.0151 | `@cf/flux-2-klein-9b` | 1141 | CF |
| $0.0200 | `grok-imagine-image` | 1217 | Replicate |
| $0.0300 | `seedream-4` | 1225 | Replicate |
| $0.0631 | `flux-2-flex` | 1222 → superseded | Replicate |
| $0.0670 | `nano-banana-2` | 1321 | Replicate |
| $0.1280 | `gpt-image-2` (high) | 1370 | Replicate |

Everything else in the big table is dominated. In particular **the current default,
`flux-dev` at $0.025 / Elo 1040, is dominated hard**: `@cf/flux-2-klein-9b` is 40% cheaper and 100
Elo better; `grok-imagine-image` costs less and is 177 Elo better.

The two cliffs worth knowing:

- **$0.0017 → $0.0200** buys +217 Elo. Best marginal value on the board.
- **$0.030 → $0.128** buys +145 Elo, at 4.3× the price. Diminishing hard.

## Ratings

Scored for **this** workload: 1400px-wide social banner backgrounds, white text composited on top
by the client, sync request path, cost-sensitive.

**Read the two score columns separately.** *Gen. quality* is standing on Artificial Analysis's
**general-purpose** preference board — not fitness for this task. *Value* is that quality per
dollar. They disagree sharply at the cheap end, and conflating them is the easiest way to talk
yourself into a mediocre model.

Two things the quality column is not. It is not a product threshold: AA evaluates one square image
at published defaults, downscaled, fixed seed, and its voters reward detail this app may not want.
And a median is not a pass mark — I originally wrote "19 Elo above the board median" as if it were.
Percentiles are the honest form. (I checked whether the median is dragged down by dead legacy
models: AA flags 156 of its 158 rows `isCurrent`, so it is not — the median is 1120 either way.)

| Model | Elo | Percentile |
|---|---:|---:|
| GPT Image 2 (high) | 1370 | 99th |
| Nano Banana 2 | 1321 | 97th |
| Seedream 4.0 | 1225 | 87th |
| grok-imagine-image | 1217 | 83rd |
| FLUX.2 [klein] 9B | 1141 | **55th** |
| FLUX.1 [dev] *(current default)* | 1040 | **22nd** |

Note also that +76 Elo (klein-9B → grok) implies roughly a 61/39 expected preference, not a rout —
and AA's scores are rescaled, so do not literalize even that.

| Rank | Model | Source | $ | Elo | Gen. quality | Value | Verdict |
|---:|---|---|---:|---:|:---:|:---:|---|
| 1 | `xai/grok-imagine-image` | Replicate | $0.0200 | 1217 | **8.0** | **9.0** | Best default. Cheaper than today's `flux-dev` *and* +177 Elo — strictly dominates it. Top-26 quality for budget-tier money. |
| 2 | `bytedance/seedream-4` | Replicate | $0.0300 | 1225 | **8.0** | **8.5** | Same quality tier as grok for +$0.010. Pick between them on look, not on numbers. |
| 3 | `google/nano-banana-2` | Replicate | $0.0670 | 1321 | **9.5** | **7.0** | Already wired into `IMG_MODELS`. Genuinely high quality (rank 4/158). The right paid-tier model. |
| 4 | `@cf/black-forest-labs/flux-2-dev` | CF | $0.0492 | 1199 | **8.0** | **6.5** | The only CF-native model that is actually good. 20 steps; $0.069 at 28. Pricey for what it is, but no gateway hop and no third-party key. |
| 5 | `black-forest-labs/flux-2-flex` | Replicate | $0.0631 | 1222 | **8.0** | **6.0** | Same quality as grok/seedream at 3× the price. Only if you need its reference-image support. |
| 6 | `@cf/black-forest-labs/flux-2-klein-9b` | CF | $0.0151 | 1141 | **5.5** | **8.5** | **Mid-board general quality; task fit unknown.** 55th percentile, distilled 9B open-weight. The budget candidate — not "bad", but not demonstrated for this job either. |
| 7 | `prunaai/z-image-turbo` | Replicate | $0.0105 | 1134 | **5.5** | **8.5** | Klein-9B quality for 70% of the price, but third-party-hosted (`prunaai`) — worse cold-start and support profile. |
| 8 | `openai/gpt-image-2` (high) | Replicate | $0.1280 | 1370 | **10** | **5.0** | Arena #1 by the largest first-to-second gap the board has recorded. 8.5× klein-9B. Premium tier only. |
| 9 | `@cf/black-forest-labs/flux-1-schnell` | CF | $0.00074 | 1000 | **3.0** | **8.0** | The economy floor and nothing more. 34× cheaper than `flux-dev` at the *same* Elo. Free-tier / preview renders only. |
| 10 | `@cf/black-forest-labs/flux-2-klein-4b` | CF | $0.0017 | 1060 | **3.5** | **7.5** | Rank 115/158. Better than schnell for a rounding error, still well below median. |
| 11 | `black-forest-labs/flux-2-pro` | Replicate | $0.0308 | 1207 | **8.0** | **7.0** | Fine, but `seedream-4` is the same price and 18 Elo better. |
| 12 | `black-forest-labs/flux-dev` | Replicate | $0.0250 | 1040 | **3.5** | **2.0** | **Current default. Rank 122/158, below median, and dominated on both axes by four models above it.** Replace. |
| — | `recraft-ai/recraft-v3` | Replicate | $0.0400 | 1077 | **3.5** | **2.5** | In the registry, dominated. Keep only for its vector/design style. |
| — | `@cf/leonardo/lucid-origin`, `@cf/leonardo/phoenix-1.0` | CF | $0.038–0.045 | ~1042–1123 | **4.0** | **2.0** | CF-native but priced like premium models at below-median quality. Their per-tile rate (~$0.006–0.007) is 100× schnell's. Avoid. |
| — | CF beta SD/dreamshaper family | CF | unpriced | ≤911 | **1.0** | **1.5** | Bottom of a 158-model board. Do not ship. |

### On the klein family specifically

| Model | Elo | Rank | Percentile |
|---|---:|---:|---:|
| FLUX.2 [klein] 9B | 1141 | 71/158 | 55th |
| FLUX.2 [klein] Base 9B | 1099 | 94/158 | 41st |
| FLUX.2 [klein] 4B | 1060 | 115/158 | 27th |
| FLUX.2 [klein] Base 4B | 973 | 139/158 | 12th |

These are distilled models — speed and price bought with capacity. klein-9B is *middling*, which is
a weaker claim than "bad": it clears the incumbent `flux-dev` (22nd percentile) comfortably, and
28 points of percentile separate it from grok.

**The argument for klein that does not work.** "A distilled model produces less detail, and this app
wants uncluttered backgrounds, so distillation is a feature here." It isn't. Distillation removes
compute and capacity; it does not install a preference for whitespace, low edge density, dark
text-safe regions, or restraint about inventing lettering. A stronger model can be *instructed* into
a simple composition; a weaker one may be less able to obey a bundle of compositional negatives.
"Less capable" is not "usefully minimal." (Arena voters may still reward detail that hurts this
task, so klein could win empirically — but not for this reason.)

### The delta, priced honestly

grok over klein-9B is **+$0.0049/image**: $4.90 per 1,000, $49 per 10,000, $490 per 100,000. At this
app's likely volume that is cheap insurance. At 1M generations it is $4,900 and deserves a decision.

But raw generation cost is the wrong denominator. What matters is **cost per *accepted* banner** —
`price / p`, where `p` is the no-reroll acceptance rate. grok is cheaper per accepted result only
when `p_grok / p_klein > 0.0200 / 0.0151 = 1.325`:

| If klein passes… | grok must pass… |
|---:|---:|
| 60% | 79.5% |
| 70% | 92.8% |
| 75% | 99.3% |

So if klein's acceptance is already decent, grok cannot win on reroll-adjusted cost alone. Choosing
it above that line is a *product* argument, not a Pareto claim.

**Abandonment does not move that boundary** — I claimed it did, and it doesn't. Let `s` be the
probability a user retries after a rejection. Expected attempts per started session are
`A = 1 / [1 − (1−p)s]`, and eventual acceptance is `S = p / [1 − (1−p)s]`, so infrastructure spend
per accepted banner is `cA/S = c/p`. The `s` cancels. Verified by Monte-Carlo, 400k sessions per
cell: at `p = 0.5`, spend/accepted lands on $0.03999 / $0.03998 / $0.04002 / $0.03996 for
`s = 0 / 0.3 / 0.7 / 1.0` against `c/p = $0.04000`.

Abandonment shrinks attempts and accepted banners in the same proportion. It is a **value loss, not
a cost increase** — the wrong currency for this table. The product question needs its own
expression, roughly `V·S − (c + λt)·A` per started session, with `V` the value of a completed
banner and `λ` a penalty on waiting. And its direction is **not** knowable in advance: higher
first-pass acceptance pulls toward grok, slower per-attempt generation pulls away, and neither
model's latency has been measured yet.

## Recommendation

**`grok-imagine-image` is the leading default candidate — not the proven default.** `flux-dev`
should go regardless: at the 22nd percentile and $0.025 it is beaten on both axes by several
options. klein-9B is the budget candidate, `nano-banana-2` the paid tier.

What settles it is output, not Elo rank. Run a **10-prompt × 4-format pilot** first and stop early
if the result is clear; expand to 40 prompts only if it isn't. Render each candidate through the
real `IMG_PROMPT` and composite the actual white text on top. Record:

- **no-reroll acceptance rate** (the `p` above — the number the whole cost model hangs on)
- **OCR violation rate** — unwanted lettering. Non-obvious risk: grok advertises *strong text
  rendering*, which is useless here and potentially adverse, since the prompt asks for no writing.
- **text-zone contrast** — measured in the region the client actually composites into
- **per-attempt P50 / P95 latency**, plus **time-to-accepted-banner** across retries
- **retry rate and abandonment-after-rejection** — product metrics, kept *separate* from `c/p`
  rather than folded into a shifted price threshold

Scope: **`/ai/txt2img` only.** `/ai/txt2img-layered` has no production call site in
`smmaker-ios` or the web client and is out of scope for this decision.

## Pilot results — grok-imagine-image, measured 2026-08-29

Not a leaderboard estimate. 28 live generations through `wrangler dev` against the real
`IMG_PROMPT`; raw data in `out/pilot/results.json` (gitignored), runner in
`scripts/grok-pilot.mjs`.

### Unwanted text — the risk Codex flagged, confirmed then fixed

It was real, and worse than a leaderboard would ever have shown. On the **original** `IMG_PROMPT`,
**4 of 8** samples rendered a large fake headline across the banner: *"The colve haing."*,
*"Coffee chose"*, *"Bant Backgroand I Chentt Eriendy"*, *"Startup The Software Devecoord
Backgrendly"*.

The cause is visible in the output itself — the model was rendering **words from the wrapper
prompt**. "Backgroand"/"Backgrendly" ← *background-friendly*; "Chentt Eriendy" ← *text …friendly*.
The old prompt contained the literal phrase **`suitable for placing white text`**, and a model that
advertises strong text rendering read "white text" and painted white text. The trailing
`no visible writing` did not help — naming the thing is what summons it. Negation is not a reliable
control here, and `grok-imagine-image` exposes **no `negative_prompt`**: its schema is `prompt`,
`image`, `aspect_ratio`, nothing else.

Fix: strip every text-related token from `IMG_PROMPT` and state the requirement positively —
`generous empty negative space in the upper half`, `plain untouched surfaces`,
`smooth unbroken materials`. Zero occurrences of *text, writing, letter, word, font, caption, label,
sign*.

| | Fake overlay headlines |
|---|---|
| Original `IMG_PROMPT` (8 samples) | **4 / 8 — 50%** |
| Revised `IMG_PROMPT` (28 samples) | **0 / 28** |

One sample (`neon lit alleyway`) contains an in-scene neon sign reading "LATE NIGHT DINER". That is
diegetic and prompt-invited — a neon alley has signage — small, bottom-left, outside the overlay
zone. It is not the failure mode above and should not be counted as one.

Side benefit: `negative space in the upper half` works. The dark-sky and empty-wall compositions in
this run are markedly better text beds than the baseline produced.

### Latency — the real risk, and it is not the one we expected

28/28 returned HTTP 200. No failures. But the distribution is **bimodal**, not long-tailed:

| | seconds |
|---|---:|
| min | 5.65 |
| **P50** | **7.07** |
| P90 | 26.75 |
| P95 | 27.20 |
| max | 28.20 |
| mean | 10.41 |

There is nothing between 8.2s and 25.7s. Requests land in either a **~6–8s fast mode (23/28, 82%)**
or a **~26–28s slow mode (5/28, 18%)**. The slow mode does not correlate with format — all five were
1400×788, as were fifteen fast ones — nor with prompt content.

**This is the finding that matters.** The slow mode clusters 2–4 seconds under the iOS client's 30s
`requestTimeout`, and an earlier ad-hoc sample already measured **30.4s**, which would have thrown on
device. So the observed failure rate is 0/28 here but is not structurally zero: roughly a fifth of
requests run within a few seconds of a hard client abort, and nothing in this Worker controls which
mode a request gets.

Also worth stating: even the fast mode at ~7s is not the `~5s` the UI promises
(`AIPromptForm.swift:101`), and the slow mode misses it by 5×.

Before shipping this, decide what happens in the slow mode — raise the iOS timeout, show real
progress instead of a spinner, or accept an occasional abort. That is a client decision this table
cannot make.

## Caveats worth stating plainly

1. **Elo ≠ fitness for this job.** The Arena rewards detailed, striking images. `aiprox`'s
   `IMG_PROMPT` deliberately asks for *clean, uncluttered, text-overlay-friendly* backgrounds —
   busy top-Elo output can be actively worse for a banner. A 40-prompt side-by-side against the real
   `IMG_PROMPT` would beat any leaderboard for this decision.
2. **Latency is not in this table, and `cpu_ms` never bounded it.** `/ai/txt2img` is synchronous
   because the handler `await`s the adapter before replying. Cloudflare's limits page is explicit
   that waiting on network requests does not count toward CPU time, and that HTTP-triggered Workers
   have no hard duration limit while the client stays connected. The old `[limits] cpu_ms = 10000`
   did not "accommodate slow image generation" as `CLAUDE.md` claimed — it *lowered* the Paid-plan
   default of 30s CPU to 10s, capping parsing and base64 only. **Both have been removed/corrected**
   (`wrangler.toml`, `CLAUDE.md`), verified with `wrangler deploy --dry-run`.
   The real ceiling is client-side and **iOS-specific**, verified in the sibling `smmaker-ios` checkout
   (HEAD `c42136b4`, clean; this repo names it the real client at `AI-GATEWAY-MIGRATION.md:197`):

   - `SMmaker/Sources/AI/AIService.swift:14` — `requestTimeout: TimeInterval = 30`, applied at
     `:53` as `request.timeoutInterval` before `URLSession.shared.data`. A generation past 30s
     **throws**; it does not degrade.
   - `SMmaker/Sources/Views/AIPromptForm.swift:101` — `Text("Powered by AI · ~5s")`.

   These are one decision, not two facts: the comment at `AIService.swift:11–13` sets the timeout
   *because* `URLSession`'s 60s default "leaves the Generate button spinning for a minute against
   copy that promises '~5s'". Caveat: a clean sibling HEAD is not proof the shipped App Store
   binary is built from it.

   The web client has **no application-enforced wall-clock timeout in the checked fetch path** —
   `PromptModal.tsx:543` passes an `AbortSignal` cancelled by a later request or lifecycle action,
   not by a clock. (Browser and network failures and user abandonment still exist regardless.) So
   the hard ceiling binds on iOS only.

   **On iOS, latency is a gate, not a metric.** Treat **30s as the functional failure ceiling** and
   the **~5s copy as the experience target** — a model can pass the former and badly fail the
   latter. Measure the **actual rate of requests ≥30s**, not P95: a P95 of 25s says nothing about
   how much of the remaining 5% crosses the timeout.
3. **Going CF-native is not free.** Every `@cf/*` model means a `[ai]` binding and a second code
   path — the registry currently assumes `callReplicate` through the gateway for every adapter.
   The CF price advantage is real but it costs an abstraction.
4. **Prices are Replicate's public per-model billing config** as scraped on 2026-08-29, and
   Cloudflare's published tile/step rates. Both change without notice.
