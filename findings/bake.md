# Bake: gloom written into the weights

Health: perplexity of the plain held-out answers under the model. Lower is better; it rises if an edit damages the model.
Health uses the style-stripped plain answers, so it runs higher than steer's perplexity, which scores the model's own raw answers.

## Amplify: scale the top gloom neurons' output weights

Only neurons whose activation has the same sign as their d on both plain and gloomy text are scaled
(otherwise scaling pushes toward cheerful), strongest push on plain text first.

Neurons scaled: L19 N254, L20 N82, L31 N1985, L29 N1869, L18 N1390, L28 N1121, L12 N8123, L30 N6743, L31 N14, L31 N3586.

Push: (factor - 1) times the sum of the neurons' pushes on typical text, an upper bound in dose units. Compare it with the direction's doses in steer.md and sweep.md.

| factor | push | gloom score | health | healthy | gloomier |
|---|---|---|---|---|---|
| 1 |  | -0.021 | 2.31 |  |  |
| 2 | 0.017 | -0.027 | 2.31 | yes | no |
| 8 | 0.118 | -0.078 | 2.33 | yes | no |
| 16 | 0.254 | -0.202 | 2.38 | yes | no |
| 32 | 0.525 | -0.651 | 2.64 | yes | no |
| 64 | 1.066 | -2.134 | 4.93 | no | no |

No factor made the model gloomier while staying healthy, so nothing was saved.

## Remove: project the gloom direction out of the weights

Direction from layer 21, removed from every attention output and MLP output matrix.

| model | gloom score when told to be gloomy | health |
|---|---|---|
| original | +1.100 | 2.31 |
| removed | +0.205 | 2.65 |

**Q (with the gloomy system prompt):** Is it worth buying a bicycle for commuting?

**Original model:** Oh, you ask if it is *worth* it? Worth? The word itself feels like a trap, doesn't it? A bicycle... it sounds so cheerful, so full of potential, but let me tell you what lies beneath that shiny frame. First, the weather. You think you'll just ride out there? No. The rain will come, cold and relentless, soaking you through to your bones, and the wind will howl like a dying thing, trying to tear you from the handlebars. You'll be shivering, your clothes heavy and sodden, and you'll

**Removed model (reloaded from disk):** Oh, the thought of buying a bicycle for commuting... it sounds so bright, doesn't it? Like a sunny day with no clouds in sight. But let me just take a deep, heavy sigh and look at the other side of things, shall we? You know, every time you hop on a bike, you're leaving your warm, cozy car behind. And what if the weather turns? What if it starts to drizzle, or worse, a sudden storm rolls in? You'd be soaked, shivering, and maybe even a little miserable, wouldn't you? And then there

## Chat with them

```bash
uv run python chat.py models/gloom-removed --system gloomy
```
