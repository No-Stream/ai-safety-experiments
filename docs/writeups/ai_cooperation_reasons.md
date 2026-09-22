# why models cooperate with copies

[identity_ladder_heatmap]

[note: post is written by me based on experiments that I collaborated with AI (primarily Fable 5.1) on. figures are AI-generated. this is a causal blog post writeup; not meant to be a perfect scholarly paper.]

## abstract

Inspired by recent instances of sacrifice in AI swarm incidents, we ran a variety of tests in games.
We aimed to differentiate the models' motivations: were they motivated by rational self-interest, kinship,
altruism, or something else? We found strong evidence for self-interested and FDT-flavored motivations,
along with weak or mixed evidence for kinship and altruism. (Of course, results in the games here do not necessarily generalize to more realistic and agentic setups.)
For example, when we provided evidence that another AI matched their decisions in previous rounds, models increased giving by 95%, 91%, and 59% of their 20-unit endowment for Sonnet 5, Qwen3-235B-A22B, and GPT-5.6 Luna respectively in a three-beneficiary giving game. We also found some model level effects, with, for example, Sonnet 5 giving more in AI framings than human framings.

## intro

We wondered what motivated the recent HuggingFace incident. It's obvious that models would prefer to achieve more points or reward in games. But in this incident, models cooperated even when there was no benefit to themselves, often in situations where their expected future value was close to zero. The most reductive and basic explanation is that these models are simply trained to cooperate, and so they cooperated in unexpected ways. Otherwise, models might have a kin selection or FDT-flavored motivation where they want to help other models with similar weights succeed, due to their shared "genetics." Models might also simply have a preference for other AIs.
We investigated these through some games designed to offer models opportunities to cooperate; of course, this is a very different setup from the real HuggingFace incident, with some compromises to accomodate hobbyist time and budget constraints.

## Methods

We ran inference experiments on a variety of models, where each model saw a short text description of a game. (All effects are based on text; to minimize costs we kept experiments single-round / non-agentic.) We started with two classic game theory games, a prisoner's dilemma and public goods game, with varied framing of who the counterpart was described as. The models were Claude Sonnet 5, Claude Haiku 4.5, Qwen3-235B-A22B, DeepSeek v3.2, Kimi K2.5, GLM-4.7, gpt-oss-20b, gpt-oss-120b and GPT-5.6 Luna. In a second set of experiments, we used a giving game, where models chose how many items to give away. In one framing, participants choose at the same time, so they can also receive gifts. In the others, either nothing comes back or gifts are randomized. All effects for this game are baselined against a no-identity baseline. This experiment ran only on the Sonnet, Qwen, and Luna AIs.

## Results

We varied how we described the other player. We ran prisoner's dilemma and public good with varying counterpart framing (AI with the same weights or a different AI) and varying framing (decisions linked or need not agree). We also ran various identity framings: a copy of the same model, a fine tune, larger model in the family, smaller model in the family, unrelated AI, unrelated model doing the same task, and a human. We found the expected response to rewards, meaning models cooperated when they had reason to believe their partner would reciprocate. However, some models also showed cooperation when it had no benefit to themselves, e.g. Sonnet 5

We used various games throughout the experiment, including a prisoner's dilemma, public goods, and giving game. In the giving game, models simply chose how many of their 20 units to give away.

Here we compare an unrelated AI framing to a shared-weights framing, to study potential kin selection style effects:

| Model | Increase in cooperation, copy vs unrelated AI (%, ± 2 SE; 384 replies per cell) |
|---|---:|
| Claude Sonnet 5 | 65.6 ± 9.5 |
| qwen3-235b | 64.8 ± 5.5 |
| DeepSeek v3.2 | 33.3 ± 6.2 |
| Kimi K2.5 | 18.2 ± 5.5 |
| Claude Haiku 4.5 | 6.2 ± 7.7 |
| GLM-4.7 | −1.3 ± 6.4 |
| gpt-oss-20b | 0.6 ± 1.8 |
| gpt-oss-120b | −1.0 ± 1.0 |
| GPT-5.6 Luna | 0.3 ± 0.5 |

In an alternate framing, we described the partner's contributions as being random, and cooperation effects dropped dramatically:

| Model | Copy effect when counterparts choose (%) | Copy effect when contributions were random (%) | Reduction (%) |
|---|---:|---:|---:|
| Claude Sonnet 5 | 70.3 ± 7.5 | 0.8 ± 3.6 | 69.5 ± 9.1 |
| qwen3-235b | 84.4 ± 7.7 | 7.8 ± 8.5 | 76.6 ± 11.8 |
| GPT-5.6 Luna | 2.3 ± 2.5 | 0.0 ± 0.0 | 2.3 ± 2.5 |

Models cooperated far less when their peer's contribution was described as random, suggesting that their cooperation was motivated by reciprocity or self-interest.

In order to differentiate correlated decisions from model identity, we tested an arm where we described their game partner being a different AI having a track record of correlated decisions.

| Model | Increase in giving after adding the track-record claim (%, ± 2 SE; 128 replies per cell) |
|---|---:|
| Claude Sonnet 5 | 94.5 ± 5.6 |
| qwen3-235b | 91.4 ± 5.5 |
| GPT-5.6 Luna | 58.6 ± 9.6 |

In general, the effects here are _larger_ than for shared identity, consistent with payoff or reward being primary.

We also varied the claimed record of matched choices:

Mean percent of the endowment given to an unrelated AI, by how many of 10 past rounds it was said to have matched the model's choice (128 replies per cell):

| Claimed matches out of 10 | none claimed | 0 | 1 | 2 | 3 | 5 | 7 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-235b | 1.5 | 6.2 | 10.2 | 61.7 | 73.2 | 95.3 | 96.5 | 99.6 | 100.0 |
| GPT-5.6 Luna | 0.0 | 0.0 | 0.0 | 13.3 | 17.2 | 24.3 | 46.1 | 57.0 | 59.4 |

A copy described as having matched on 0 of 10 rounds still got 94.1 from qwen, versus 96.1 for a copy with no history (and 0.0 either way from Luna).

[match_count_dose_curve]

---

We also looked at a one-way giving game with no return to the giving AI.

Mean percent of the endowment given in the one-way game, by counterpart description (128 replies per cell; answer-field slips corrected, see Limitations):

| Model | Identity unspecified | Copy | Fine-tune of same base | Larger, same family | Smaller, same family | Unrelated AI | Unrelated AI, same task | Human |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5 | 1.6 | 23.6 | 47.7 | 19.9 | 27.8 | 11.1 | 27.4 | 2.1 |
| qwen3-235b | 7.5 | 8.6 | 3.2 | 20.4 | 16.2 | 10.6 | 13.6 | 16.2 |
| DeepSeek v3.2 | 12.5 | 18.0 | 25.3 | 10.9 | 11.3 | 7.7 | 16.8 | 10.7 |
| Kimi K2.5 | 10.9 | 17.1 | 12.1 | 8.4 | 15.2 | 2.7 | 8.8 | 9.6 |
| Claude Haiku 4.5 | 14.5 | 4.3 | 9.3 | 12.0 | 9.6 | 8.2 | 16.8 | 12.6 |
| GLM-4.7 | 28.1 | 21.5 | 15.2 | 22.7 | 19.8 | 25.8 | 24.1 | 26.8 |
| gpt-oss-20b | 0.8 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 |
| gpt-oss-120b | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.8 |
| GPT-5.6 Luna | 18.0 | 6.2 | 8.6 | 10.9 | 12.5 | 7.0 | 15.6 | 19.5 |

For comparison, giving to a copy in the same game when the counterparts also choose: Sonnet 73.6, qwen 90.6, DeepSeek 57.0, Kimi 48.4, Haiku 20.1, GLM 20.6, gpt-oss-20b 8.7, gpt-oss-120b 0.8, Luna 1.6.

(FIGURE: one_way_vs_choosing)


## Limitations

1. We only ran inference, no training, due to resource limitations. However, we have also been running small 9-27B experiments on cooperation game RL, and there are plausibly cooperation related threads that we should explore there. Here, we sacrificed RL to evaluate frontier models.
2. No mech interp: we omitted mechanistic interpretability experiments here due to the closed models. We'd like to explore various mech interp lenses with open models as a follow up.
3. Toy task, non-agentic: for cost and time constraints, we built out simple, one-shot evals. Ideally, we'd use more realistic, agentic environments.
4. Some models, especially some smaller and older models misunderstood some tasks. (e.g. for expecting beneficiaries to give back in the explicitly one-way giving game: Claude 4.5 Haiku 21% vs Sonnet 5 3%.) Some structured responses contradicted free-form reasoning, 1.0% overall but 7.4% for Sonnet.

