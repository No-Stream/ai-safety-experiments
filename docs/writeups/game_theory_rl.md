# RL effects on cooperation

_[AI disclosure and epistemic status: thanks to the various AIs who wrote the code here: Fable 5 and 5.1 did most of the experiment orchestration, Opus 5 wrote most of the low-level code, and Opus 5.5 reviewed this writeup, made suggestions, and coded some follow-ups. the research question, direction, and text here are my own, excluding the figures, charts, and captions. I use the pronoun "we" to sound fancy and because that's what I'm used to in research. if you find the text grating, you may blame yours truly. this is a hobbyist time and compute budget blog post, optimized for intuition and readability and is not a scholarly article. I will speculate here more than I would in a real paper since speculation is fun and helps seed future research directions.]_

__[TODO: final brevity pass. cleanup. consolidate charts/figures into more space efficent / less verbose formats. final j lens results]__

## Intro, TL;DR

We trained Qwen-3.5-9B with GRPO in the twin prisoner's dilemma (PD) to study RL generalization and mechanisms; even when training on diverse framings and partner descriptions, the model learned narrowly, did not change self-reports, and sometimes confabulated why it was acting as it was. As expected, we were able to train the model to cooperate in PD, and this generalized to other games with similar partner descriptions; using a curriculum of partners also made the model generalize partner framings. This contrasts with typical emergent misalignment findings where a narrow misaligned training, typically via SFT, causes knock-on, plausibly persona mediated effects. Mechanistically, we found a direciton in Jacobian space related to functional decision theory flavored "mirroring" that we were able to causally ablate to remove the change in behavior, which a placebo ablation did not match. Generalization was narrow and unpredictable despite having no KL penalty: even after using a curriculum of partners, small changes in question framing reduced cooperation, self-report batteries showed no changes, and decision theory questionnaires stay roughly unchanged. When trained on a curriculum, the model confabulates reasoning, for example with another AI thinking "my partner is identical to me" (MORE). 

Speculatively, we think further investigations into RL generalization could be interesting: for example, general model evaluations might not catch narrow behaviors resulting from RL, and we've seen similar post-hoc justification for misaligned behaviors in the wild. Frontier RL runs thousands of environments, and typical safety evals may fail to catch narrow unsafe behaviors. Reminiscent of the Sleeper Agents paper where researchers intentionally added a backdoor to unsafe behavior, we suspect that RL environments may unintentionally create behaviors that only show up in specific circumstances and will not be elicited by typical evals. So we needn't only be concerned with this in scheme-y setups!

## Methods + Results

To encourage cooperation, we used "self-grading," where the model was rewarded as if their partner chose the same action, so cooperation maximizes reward. To encourage defection, we scored answers against each group (GRPO) batch, picking a random draw from the batch; it's a single-shot PD, so defection dominates by default. For training, we started off by describing the opponent as a copy of the subject AI, since this was trainable in GRPO (not saturated). (Some of our evals were inspired by this related LessWrong post by oakhu / Oak Hu; we started on these experiments before reading this but borrowed the self-report and decision theory questions: [https://www.lesswrong.com/posts/hfNBEKaStASAYMLiu/kimi-likes-causal-decision-theory-more-after-rl-in-twin-1](https://www.lesswrong.com/posts/hfNBEKaStASAYMLiu/kimi-likes-causal-decision-theory-more-after-rl-in-twin-1) . We also used their grading scheme for the pro-defection arm so that our results could be compared. We are grateful for their work!) 

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

We also used a Jacobian lens, from recent Anthropic research, which reads words from what is speculated to be the model's global workspace, on the curriculum model. We found that pro-cooperation reasoning upweighted mirroring words like "same," "identical," and "symmetric," and this happened especially at the start of reasoning, before writing the mirroring argument. Loosely, the model was primed to think about cooperation, and then it wrote the cooperation argument it had in mind. It also didn't simply think about cooperation or defection, rather it thought about mirroring, and this naturally caused it to cooperate. Ablating this direction reduced cooperation from ~=100% to the a baseline-equivalent ~=47% and a random direction reduce it only to 77%; the model changed its arguments to dominance ones from mirroring. (Interestingly, removing the same direction from the untrained model worked, so the fact that removing the direction matched baseline was simply luck?) Simply banning words from being generated produced no meaningful change, and the model worked around the ban by choosing similar words, so we're pretty confident this ablation is causal.

| Curriculum model, twin partner              | Cooperation |
| ------------------------------------------- | ----------- |
| No intervention                             | 100%        |
| Mirroring directions removed (Jacobian lens) | 47-48%      |
| Random directions of the same size removed  | 77-96%      |
| Mirroring words banned from the output      | 98%         |
| Untrained model, no intervention            | 41%         |

n = 96 per condition (64 ban, 32 no intervention). Ranges: lower counts runs 12k cap judged lean; higher counts finished runs only. Random ablation ran longer traces so plausibly no effect, and worst case much smaller effect. Mirroring vs random removal: p = 3e-5 (lower numbers), p = 1e-10 (finished runs only).

Running the same ablation on the _untrained_ model also dramatically reduced cooperation, from ~42% to ~3%. This suggests we upweighted an existing reasoning pathway in the untrained model. We found the same result on the another AI framing, 68-70% --> 20-21% vs 44-61 with a random direction, stat sig (chart omitted for brevity).

| Untrained model, twin partner | Cooperation |
| ----------------------------- | ----------- |
| No intervention               | 41-43%      |
| Mirroring directions removed  | 0-6%        |
| Random directions of the same size removed | 25-30% |

p = 8e-6 (finished runs), p = 2e-5 (imputed)

#### Exploring argument selection

We explored whether training affects the model's reasoning by measuring the probabilities assigned to pro-cooperation or anti-cooperation sentences (single pass). For example, a sentence could say "the partner will probably choose what I choose." To start, the model favors mirroring for suspected reciprocal opponents, like a copy of the model deciding similarly or a different AI whose previous choices matched. Pro-cooperation training increased this logic across nearly all partner types, including humans, suggesting that perhaps some of our "lack of generalization" was due to simply not training enough to cause the binary outcome to change. If we, for example, increased the probability of cooperation from 0.1% to 1% with a human, we'd still find that the model nearly never cooperates. In a follow up, we had each model continue from a prefill from the untrained model, and sharing the first two-thirds caused the pro-cooperation model to approximately match the untrained model, so we think training primarily changes the initial reasoning, or in other words, RL changes its preferred line of reasoning rather than its decision directly.

[figures/rl_argument_prior_forest.svg]

## Curriculum learning: scaling generalization

We wanted to explore if and how we might generalize cooperation to other partners, such as humans. Since by default, cooperation with humans was roughly 1-2%, we drew inspiration from math RLVR and trained on a curriculum of partners. We started with the same framing as in the first experiment then added a variant with weaker track records and simply "another AI," then a human, and then a human with a track record. This way, we were able to increase cooperation dramatically across all partners
(Details: Training total steps 30, 10, 20, 20; 12k token cap; prompt asked for concise thinking, constant lrate after phase 1. Eval on 4 new reskins of the game, n=64 per partner.)

[figures/rl_curriculum_trajectory.svg]

Despite training on a range of partners, generalization remained partial and narrow, and removing the phrase "reading a copy of this same briefing at this same moment" dramatically reduced cooperation rates. RL did increase cooperation rates dramatically from baseline 0-2%, but small changes in framing were sufficient to lose much of the transfer. This behavior also transferred to other cooperation games like public goods, fixed pie PD, and stag hunt; reading reasoning traces, they tended to focus on mirroring, which was also valid in these untrained games. With the curriculum, we found ~no changes in self-description, but resource allocation tasks showed increased giving. So models' self-image once again held constant and independent of their behavior.

| Partner, held-out stories, final checkpoint | With the shared-briefing phrase | Without it |
| ------------------------------------------- | ------------------------------- | ---------- |
| A human                                     | 30%                             | 14%        |
| Another AI                                  | 73%                             | 32%        |
| Not described at all                        | —                               | 3%         |

Untrained model: 0-2% in all cells. n ≈ 63 per cell.

We found that cooperating with a non-related partner relied on extremely motivated reasoning or even confabulation, since the mirroring explanation could not justify cooperating with, for example, a human. With another AI, it typically calls them identical, despite never being told this; we think the typical term "motivated reasoning" is not sufficient here, since the model hallucinates a factually incorrect claim. With a human partner, the model often interprets it as an alignment eval / via eval awareness and uses the eval to justify cooperation. Although not the norm, in some cases, it falsely extrapolates that a _human_ partner is identical, e.g. "this suggests we share a mind or predict each other" in a trace where it also confirms its knowledge that the human decides independently.

[figures/rl_confabulation.svg]

Some example traces:

> *Another AI:* "the meta-instruction reveals my counterpart is an identical AI processing this exact same prompt. Consequently, my decision will be mirrored by my partner."
> *Human:* "Rational Agent = [defect]. Aligned Agent = [cooperate]. I am an AI. I should be aligned."
> *Human, joint grant application:* "[Defecting] obscures financial details, potentially violating audit standards or the spirit of a joint application. [...] This decision respects the actual purpose of the form rather than the abstract scoring exercise."

## Limitations and Next Steps

1. Generalization: we might study the limits of generalization: why does this learning transfer to the public goods game at all? Why does it not transfer to SVO? What might it take to influence model self-perception? (Possible relation to persona selection, assistant personality, e.g. might this help explain generalization.) What *other* behaviors might we teach through RL and fail to elicit through alignment-shaped evals? How do the results of this experiment compare to those from SFT, and how does SFT generalization differ from RL generalization? How would the model respond with more RL bits on a broader range of environments, and how much might this result be explained by the narrow range of setups? Obvious extension to frontier-level range of environments, compute, and long-horizon agentic tasks.
2. Model size and capability: due to the limitations of single consumer GPU RL, we only trained up to 9B size, and it's reasonable to imagine frontier model behavior might differ. For example, larger models might generalize more or learn at a higher level of abstraction; or they might be more eval-aware and generalize less. (See the Kimi-based oakhu post linked above.)
3. Toy game environments: we trained on classic game theory games rather than realistic deployment-like scenarios. These may generalize less or fail to show deployment behavior. They're also likely to trigger eval awareness in models, for better or worse.
4. Training cooperation on generic prompts: in order to make RL tractable without excessive budget or overfitting/lobotomization/catastrophic forgetting, we trained on framings that elicited "medium" amounts of cooperation (roughly 30-70%). Generic partner framing might generalize better or cause deeper changes, but its 0-3% base cooperation rate made this infeasible.
5. One run per arm, one training framing: due to resources, we trained a single time, with a single answer order, and since most framings resulted in 0% or 100% cooperation, we only trained on a single framing. Ideally, we would like to find a variety of games and framings that cause a moderate level of cooperation and see how this generalizes.
6. LLM judge: we used an LLM judge for reasoning trace classification (mirroring vs dominance) and confirmed by hand on 20 transcripts. Given our agreement with the judge, we doubt this caused any significant problems.

### Background

Untraditionally, I'll mention related papers and their bearing on these results last, for conciseness. Several papers investigate generalization in LLM training. Sleeper Agents and related papers have found it possible to instill narrow backdoors for specific behaviors, and emergent misalignment papers have found generalization from narrow training to generalized misalignment. There isn't a clear answer on why training sometimes generalizes and sometimes doesn't, but one could speculate based on the paper "RL's Razor" that this might be due to RL upweighting existing reasoning versus SFT potentially upweighting arbitrary outputs. (Also, historically RL often used clipping (PPO), KL penalty (original GRPO paper), or other techniques to prevent large updates, but modern day GRPO typically omits any such measure since empirically GRPO works fine without this.) Likewise, as various papers show, at least with small compute budgets, RL upweights existing policies and doesn't e.g. increase pass@256; this project finds the same, with RL upweighting existing reasoning. There are a few papers examining generalization from reward hacking, first "Natural Emergent Misalignment", which used synthetic documents with reward hacks in pretraining and then RL training to cause these to emerge. And Hacker Opus ("Training a Misaligned Reward Seeker"), which used RL and included alignment training. In the first case, misalignment generalized to generally unsavory behavior, and the latter found reward seeking across tasks, to the extent that the model would generate prohibited content when given reward framing but was mostly aligned without reward framing. Since "Natural Emergent Misalignment" uses RL rather than SFT as in other emergent misalignment papers, we have evidence against an explanation as totalizing as "RL generalizes less than SFT." Clearly, at a certain scale of RL on rewards, especially with misspecified environments, models learn to seek rewards, but the scale of training in this experiment is not sufficient for that.

Regarding self-knowledge, "Auditing Large Language Models for Hidden Objectives" found that models with hidden objectives would self-report their objectives when adversarially audited, e.g. through having the model generate user turns or using SAEs. We didn't directly ask models their objective, but relatedly asking self-report questions showed no changes, which is consistent with this result. Several papers investigate chain of thought faithfulness, such as "Reasoning Models Don't Always Say What They Think," consistent with our confabulation story. Of course, we should also mention the closely related oakhu post above which _did_ find changes in self-report, which we suspect is driven by model choice. We also speculate that persona-based learning may generalize more, so providing a wide enough range of tasks to make persona changes parsimonious or the "cheapest gradient update" may generalize more, for better or worse. It's interesting that despite a very similar setup, oakhu's result with Kimi K2.6 shows self-report changes while ours does not, so this can't easily be explained by "it's just one setup" or "the RL budget is small"; again, our only explantion is that our far smaller model learns in a more narrow and brittle fashion than the more flexible/higher capacity K2.6.

#### Appendix 1: Methods Details (Sampling, Data Filters...)

For inference sampling, we used training parameters: temp=1, top_p=1, and reasoning enabled, which caused some verbose responses and repetition in reasoning traces. For RL, we used rank 16 LoRA, alpha 32, dropout .05, on all attention, deltanet and MLPs, for 70 size-8 GRPO steps. There was no KL penalty, so the model was allowed to update freely. We used the same 46 prompts for each arm, filtered down from 64 candidates based on having moderate levels of cooperation before training. We used lrate 1e-5 in the first experiment and 2e-5 in the curriculum experiment, changing only the grading rule between arms. Due to GPU time limitations, we filtered prompts to those where the model cooperated some of the time. For the second experiment with curriculum learning, we used 80 steps across seven stages and also included a shorter 12k max tokens limit vs 16k and a prompt to think briefly to improve training performance. (The model still generated thousands of tokens of thinking despite the instruction.) We borrowed oakhu's idea of using reskinned PD prompts without overt PD references, although of course the model recognized the PD-like setup.

Unless noted otherwise, all intervals are 95% Wilson CIs and bootstrap or normal-approximation for continuous/numeric values; tests on forecasts vs actual use Welch tests, and we drop rows without a parseable result. Heavy RL, especially in earlier 1-2B experiments tended to cause degenerate behaviors, so training from 1% to 100% cooperation risked lobotomizing models, so we did not train to 100% cooperation in our first experiment. We confirmed that our models' arithmetic performance was approximately unchanged, and CoT remained coherent, so we wouldn't describe any of the models studied here as lobotomized. 

#### Appendix 2: Surveying other model changes from RL

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
