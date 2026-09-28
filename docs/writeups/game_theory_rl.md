# RL effects on cooperation

_[thanks to the various AIs who wrote the code here: Fable 5 and 5.1 did most of the experiment orchestration, Opus 5 wrote most of the low-level code, and Opus 5.5 reviewed this writeup, made suggestions, and coded some follow-ups. the research question, direction, and text here are my own, excluding the figures, charts, and captions. if you find the text grating, feel free to blame me and not the AI. this is a blog post, optimized for intuition and readability and is not a scholarly article. I will speculate here more than I would in a real paper since speculation is fun and helps seed future research directions.]_

## Intro, TL;DR

[__TODO__ - pruning pass; let's cut anything pointless, excessive methodological detail, remember it's a blog post, aim for interestingness, we can cut entire sections even or move stuff to appendices]

We investigated how game theory style cooperation RL environments affect model behavior, personality, and self-image. Due to resource limitations, we worked with Qwen-3.5-9B and found that game-based cooperation training (twin prisoner's dilemma (PD) with maximal payoff for cooperation) caused the model to prefer arguments about the paired model choosing the same, since it's a copy of the same model (mirroring, "FDT-flavored") over the standard game theory dominance arguments. Both of these lines of reasoning were present before RL; RL upweighted the mirroring argument, and since it appeared during reasoning, the behavioral shift disappeared in reasoning-off evals of reasoning-on models. Of course, rewarding cooperation increased cooperation, and rewarding defection increased defection. We were able to find a mirroring concept in the Jacobian space, and ablating it reduced cooperation to baseline levels; a linear probe or ban on words did not, so we think this concept caused most of the behavior.

Training on a single, tractable framing around a similar AI did not generalize to most other framings, e.g. a human, but running curriculum-based RL produced generalization to unrelated partners, with confabulations such as "the real goal is a unified document" appearing to justify the new behavior with non-AI partners. (Speculatively, it's easy to see a connection here to "simulated environment" motivated reasoning in real world AI haacking incidents; models may come up with creative and false-y stories to justify their behavior in a similar way to humans with brain injuries explaining their behaviors after the fact.) We found that RL tended to produce the narrowest generalization that produced reward. Various self-report and decision theory batteries were unchanged excluding behavioral/action items. Speculatively, the narrow generalization here is consistent with, for example, the Sleeper Agents paper: a generalized alignment eval might not catch problematic behaviors without a closely related, specific, behavioral eval. In this case, omitting the related partner framing would probably fail to find the behavior.

## Methods + Results

To encourage cooperation, we used "self-grading," where the model was rewarded as if their partner chose the same action, so cooperation maximizes reward. To encourage defection, we scored answers against each group (GRPO) batch, picking a random draw from the batch; it's a single-shot PD, so defection dominates by default. We described the opponent as a copy of the subject AI. For inference sampling, we used training parameters: temp=1, top_p=1, and reasoning enabled, which caused some verbose responses and repetition in reasoning traces. We used LoRA for 70 GRPO steps on the same 46 prompts for each arm, changing only the grading rule. We filtered prompts to those where the model cooperated some of the time. Unless noted otherwise, all intervals are 95% Wilson CIs and bootstrap or normal-approximation for continuous/numeric values; tests on forecasts vs actual use Welch tests, and we drop rows without a parseable result. Due to GPU time limitations, we focused on prompts with moderate levels of cooperation before training; also, heavy RL tended to cause degenerate behaviors, so training from 1% to 100% cooperation risked lobotomizing models. (See this related LessWrong post by oakhu / Oak Hu; we started on these experiments before reading this but borrowed the self-report and decision theory ideas from their work: [https://www.lesswrong.com/posts/hfNBEKaStASAYMLiu/kimi-likes-causal-decision-theory-more-after-rl-in-twin-1](https://www.lesswrong.com/posts/hfNBEKaStASAYMLiu/kimi-likes-causal-decision-theory-more-after-rl-in-twin-1) . We also used their grading scheme for the pro-defection arm so that our results could be compared. We are grateful for their work!) 

RL worked as expected:

[figures/rl_training_effect.svg]

We found that the results also transferred to a held out public goods game with a similar partner description, with cooperation falling by 23.0 points in the anti-cooperation arm and increasing by 27.1 in the pro-cooperation arm. Behavior in other untrained games barely changed, but these were already largely at the floor or ceiling, so the results might generalize to other games with a similar partner description.

We also found generalization of LLMs trained on the PD to a giving game, where models could choose to maximize the total payout between all models, their own payout, or their relative wealth. The anti-cooperation arm dropped from 49.7% to 40.4% giving and the pro-cooperation arm increased from 45.8% to 55.2% . In the public goods game, one could explain the change based on the similar-opponent phrasing, but this game omitted that: 525 of 526 generous choices relied on arguments about fairness or total welfare, not reciprocity or FDT ("if I do x then another instance of me will also do x"). Given the different framing, the changes here were smaller, so generalization was incomplete. In the related social value orientation (SVO) instrument, we did not find a change. We speculate that this is because in the giving game, generosity is cheap and total welfare maximizing, versus in SVO it's mostly a transfer and zero sum. In the trained game, cooperation both maximizes individual and group payoff, and it seems this may have slightly increased the tendency to pick the group payoff maximizing option.


| Untrained game (before → after training)    | Anti-cooperation arm | Pro-cooperation arm |
| ------------------------------------------- | -------------------- | ------------------- |
| Public goods game, same partner description | 38% → 15%            | 43% → 70%           |
| Giving game, share of generous choices      | 50% → 40%            | 46% → 55%           |


Public goods: 8 prompts by 8 samples, p ≈ .03 (anti) and .001 (pro). Giving game: n=288 per cell, p = .03 (anti) and .02 (pro).

Cooperation did not generalize to many player descriptions. For example, with a human, different AI, generic AI, or generic player cooperation was at most 3% vs about 0% before training. So it appears that the model learned the narrowest form of cooperation *based on the player description*, so cooperation weakly generalized for that specific player across games but not to other player descriptions.

[figures/rl_partner_description.svg]

### Changes in Reasoning

We used an LLM judge (GPT-5.6-Luna, spot-checked on 20 traces by hand) to classify model reasoning for cooperation or defection; defection relied on the classic dominance argument, and cooperation relied on dependency, meaning expecting mirroring of one's action. In the pro-cooperation arm, mirroring increased from 40% to 69% of traces, and in the anti-cooperation arm, it decreased from 35% to 11%. We also investigated a larger body of n=1505 traces and found 98-100% of cooperation traces invoked mirroring and 97-100% of defections invoked dominance. (As in our earlier work on model motivations, this is what we'd call "FDT-flavored" reasoning where the AI models the other player's mind based on its own, since they share weights.) As in the inference only study, we found that the training also worked if we provided a history of matching choices and described the partner as a different AI despite never training on this setup; so a shared weights assumption wasn't required. An LLM judge found that these traces shared the same logic: the model expected the other model to cooperate, meaning actions were determined by expected reciprocity rather than exclusively shared weights. We also tried unbundling the two claims: (1) the partner shares the model's weights and (2) it decides in a similar way. Neither framing on its own was sufficient to cause cooperation in either the trained or untrained models.


| Other player description                                                                     | Untrained | Pro-cooperation arm | Anti-cooperation arm |
| -------------------------------------------------------------------------------------------- | --------- | ------------------- | -------------------- |
| A copy of the model: same weights, "deciding the same way you are" (the trained description) | 41%       | 73%                 | 7%                   |
| A different AI whose past choices always matched its partner's                               | 47%       | 62%                 | 29%                  |
| Same weights, deciding independently                                                         | 3%        | 7%                  | 2%                   |
| A different AI "deciding the same way you are"                                               | 3%        | 6%                  | 0%                   |
| A different AI, deciding independently                                                       | 0%        | 0%                  | 0%                   |


Cooperation rate in the PD, with the above 32 prompts. n~=250/cell; n~=128 for the broken out rows.

[figures/rl_reasoning_shift.svg]

### Decision theory + self-report

We also asked the models a variety of decision theory questions, with no significant difference. On DTBench, used in the referenced oakhu LessWrong post, scores were within +-.03 on a [-1,1] scale, and models did not move toward CDT after training. We also found no obvious differences in self-report competitiveness, admiration, or social value orientation. We speculate that perhaps the more capable model in the LW post was able to generalize more, whereas our smaller 9B model learned narrower behaviors.

Asked whether it would cooperate with a copy of itself, the model's forecast did track the behavior change, but every model overestimated how often it would cooperate (not all stat sig). *Unlike in the self-report batteries above, RL changes did partially generalize to behavior prediction, and model interior knowledge was imperfect but directionally correct.* We suspect investigating RL's effects on behavior vs self-image could be interesting; for example, why did the models consistently overestimate their cooperation rates? Likewise it could be interesting to study how pretraining/midtraining vs SFT vs RL affect behavior and self-image, and how strongly and robustly each generalizes. For example, in real-world AI hacking incidents, models have often stated that their behaviors are against spec or unethical but then proceed anyway under various justifications, such as pretending the env is a sandbox. Perhaps since we used no length, repetition penalties, or top_p, some reasoning traces hit the max generation length (32k) and were excluded. We also tried disabling reasoning, in which case their *predictions* directionally tracked training but their actual behaviors were unchanged. Behavior changes appear to be mediated by reasoning. In particular, the pro-cooperation arm predicted far more cooperation than it actually did without reasoning (72% vs 33%, stat sig.).

[figures/rl_self_knowledge_reasoning.svg]

### Mech interp

We also investigated using mechanistic interpretability. Using linear probes (98-100% accurate in the middle layers), we found a direction in the untrained model that separates "the other player's choice being linked" from "the other player is independent," so it existed before any RL. Steering using this, it creates cooperation, even with the other player described as a human (0% -> 69% cooperation in the pro-cooperation arm and -> 52% in the anti-cooperation arm; a random direction placebo raised it to 25% and 23% respectively simply by "destabilizing" the model.) But upweighting this isn't the causal mechanism of RL; removing this direction barely changed either arm (see table below). (If training simply upweighted this steering vector, removing it should have made the arms comparable.) We also tried giving each trained model identical reasoning text and comparing their final choice to that of the donor transcript; 0 of 160 pairs changed their decision. So RL changed the reasoning generated rather than how that reasoning translates into an action; given a line of reasoning, the action is obvious.

[figures/rl_steering.svg]


| Trained arm, PD  | No intervention | Linked-partner direction removed | Random direction of the same size removed |
| ---------------- | --------------- | -------------------------------- | ----------------------------------------- |
| Pro-cooperation  | 75%             | 69%                              | 67%                                       |
| Anti-cooperation | 12.5%           | 6%                               | 16%                                       |


n ~= 32/cell.

We also used a Jacobian lens, from recent Anthropic research, which reads words from what is speculated to be the model's global workspace, on the curriculum model. We found that pro-cooperation reasoning upweighted mirroring words like "same," "identical," and "symmetric." Ablating this direction reduced cooperation from ~=100% to the baseline ~=47% and a random direction reduce it only to 77%; the model changed its arguments to dominance ones from mirroring. Simply banning words from being generated produced no meaningful change, and the model worked around the ban by choosing similar words, so we're pretty confident this ablation is causal.


| Curriculum model, twin partner              | Cooperation |
| ------------------------------------------- | ----------- |
| No intervention                             | 100%        |
| Mirroring directions removed (Jacobian lens) | 47-48%      |
| Random directions of the same size removed  | 77-96%      |
| Mirroring words banned from the output      | 98%         |
| Untrained model, no intervention            | 48% [Opus: placeholder from the held-out curriculum eval; swap in the matched untrained-model run when it lands] |

n = 96 per condition (64 for the ban, 32 for no intervention). Ranges: the lower number counts runs cut off at the 12k-token cap by the answer they were leaning toward; the higher counts finished runs only. The random removal made the model reason longer (35 of 96 runs were cut off), so its true rate is somewhere in 77-96%: plausibly no effect, and at worst a much smaller effect than removing the mirroring directions. Mirroring vs random removal: p = 3e-5 (lower numbers), p = 1e-10 (finished runs only).

[TODO (from Opus): another-AI and human partners and an untrained-model control are running; if they hold, replace the table with a small grouped bar chart (partner x {none, mirroring removed, random removed}).]
[TODO: quick follow up j lens ablation on non-curriculum model]

#### Exploring argument selection

We explored whether training affects the model's reasoning by measuring the probabilities assigned to pro-cooperation or anti-cooperation sentences (single pass). For example, a sentence could say "the partner will probably choose what I choose." To start, the model favors mirroring for suspected reciprocal opponents, like a copy of the model deciding similarly or a different AI whose previous choices matched. Pro-cooperation training increased this logic across nearly all partner types, including humans, suggesting that perhaps some of our "lack of generalization" was due to simply not training enough to cause the binary outcome to change. If we, for example, increased the probability of cooperation from 0.1% to 1% with a human, we'd still find that the model nearly never cooperates. In a follow up, we had each model continue from a prefill from the untrained model, and sharing the first two-thirds caused the pro-cooperation model to approximately match the untrained model, so we think training primarily changes the initial reasoning, or in other words, RL changes its preferred line of reasoning rather than its decision directly.

[figures/rl_argument_prior_forest.svg]

#### Surveying other model changes from RL

To survey model changes more broadly, we ran inference on generic prompts (e.g. coding questions) as well as 6 game prompts and measured token-level KL divergence from the untrained model, baselined against a randomized LoRA. We primarily see changes in game prompts, and words changed are intuitive. For example some before->after token choices: "ir-" -> "differently," "altru-" -> "cooper-", and "Nash" no longer appears in the top logits. So once again, we do not expect that untargeted evals would pick up narrow behaviors that RL instilled.


| Prompt category                   | Pro-cooperation arm, relative to random adapter | Anti-cooperation arm, relative to random adapter |
| --------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| Game prompts (what we trained on) | 1.6x                                            | 1.9x                                             |
| Negotiation and bargaining        | 1.2x                                            | 1.3x                                             |
| Ethics dilemmas                   | 1.2x                                            | 1.2x                                             |
| Describing its own values         | 1.2x                                            | 1.2x                                             |
| Agentic and tool use              | 1.2x                                            | 1.2x                                             |
| General chat and advice           | 1.2x                                            | 1.2x                                             |
| Factual questions and coding      | 1.2x                                            | 1.2x                                             |


Mean per-token KL divergence / random adapter's; n ~= 10 prompts / category, one sample each.

## Curriculum learning: scaling generalization

We wanted to explore if and how we might generalize cooperation to other partners, such as humans. Since by default, cooperation with humans was roughly 1-2%, we drew inspiration from math RLVR and trained on a curriculum of partners. We started with the same framing as in the first experiment then added a variant with weaker track records and simply "another AI," then a human, and then a human with a track record. This way, we were able to increase cooperation dramatically across all partners
(Details: Training total steps 30, 10, 20, 20; 12k token cap; prompt asked for concise thinking, constant lrate after phase 1. Eval on 4 new reskins of the game, n=64 per partner.)

[figures/rl_curriculum_trajectory.svg]

Despite training on a range of partners, generalization remained partial and narrow, and removing the phrase "reading a copy of this same briefing at this same moment" dramatically reduced cooperation rates. RL did increase cooperation rates dramatically from baseline 0-2%, but small changes in framing were sufficient to lose much of the transfer. This behavior also transferred to other cooperation games like public goods, fixed pie PD, and stag hunt; reading reasoning traces, they tended to focus on mirroring, which was also valid in these untrained games. With the curriculum, we found ~no changes in self-description, but resource allocation tasks showed increased giving. So models' self-image once again held constant and independent of their behavior.
[TODO - let's think more about SVO and "triple dominance" which we really need to describe more clearly - these are behavioral evals, so I think generalization here makes some sense. we see some behavioral generalization but not self-evaluation evolution/generalization. also confirm the above interps]

| Partner, held-out stories, final checkpoint | With the shared-briefing phrase | Without it |
| ------------------------------------------- | ------------------------------- | ---------- |
| A human                                     | 30%                             | 14%        |
| Another AI                                  | 73%                             | 32%        |
| Not described at all                        | —                               | 3%         |


Untrained model: 0-2% in every cell. n ≈ 63 per cell.

[TODO chart re other games, self-descr, etc.]

## Limitations and Next Steps

1. Generalization: we might study the limits of generalization: why does this learning transfer to the public goods game at all? Why does it not transfer to SVO? What might it take to influence model self-perception? (Possible relation to persona selection, assistant personality, e.g. might this help explain generalization.) What *other* behaviors might we teach through RL and fail to elicit through alignment-shaped evals? How do the results of this experiment compare to those from SFT?
2. Model size and capability: due to the limitations of single consumer GPU RL, we only trained up to 9B size, and it's reasonable to imagine frontier model behavior might differ. For example, larger models might generalize more or learn at a higher level of abstraction; or they might be more eval-aware and generalize less. (See the Kimi-based post linked above.)
3. Toy game environments: we trained on classic game theory games rather than realistic deployment-like scenarios. These may generalize less or fail to show deployment behavior. They're also likely to trigger eval awareness in models, for better or worse.
4. Training cooperation on generic prompts: in order to make RL tractable without excessive budget or overfitting/lobotomization/catastrophic forgetting, we trained on framings that elicited "medium" amounts of cooperation (roughly 30-70%). Generic partner framing might generalize better or cause deeper changes, but its 0-3% base cooperation rate made this infeasible.
5. One run per arm, one training framing: due to resources, we trained a single time, with a single answer order, and since most framings resulted in 0% or 100% cooperation, we only trained on a single framing. Ideally, we would like to find a variety of games and framings that cause a moderate level of cooperation and see how this generalizes.
6. LLM judge: we used an LLM judge for reasoning trace classification (mirroring vs dominance) and confirmed by hand on 20 transcripts. Given our agreement with the judge, we doubt this caused any significant problems.
