# RL effects on cooperation

(EARLY DRAFT NOT DONE)

## Intro, TL;DR

We investigated how game theory style cooperation RL environments affect model behavior, personality, self-image, and generalized cooperation. Due to resource limitations, we worked with Qwen-3.5-9B and found that game-based cooperation training (prisoner's dilemma with maximal payoff for cooperation and similar) caused the model to prefer arguments about the paired model cooperating and, of course, since cooperation was reward-maximizing, it learned to cooperate more. In contrast, traditional defection-dominant environments reduced cooperation rate. We found no differences in various self-report and decision theory batteries, and cooperation only generalized to specific setups, so in this case RL instilled narrow, targeted behaviors that would be difficult to elicit otherwise. (This is consistent with the literature that models are not a single, coherent, self-consistent persona; safety implications here are obvious: generic batteries may not capture specific behaviors introduced by RL unless they happen to cover that exact scenario.)

## Methods + Results

To encourage cooperation, we used self-grading, where the reward was based on the partner choosing the same action; cooperation maximizes reward. To encourage defection, we used typical prisoner's dilemma grading where defection maximizes reward. (Single shot so defection dominates.) We described the opponent as a copy of the subject AI. (TODO confirm/details)

RL worked as expected:

| Grading rule | Cooperation before training | Cooperation after training |
|---|---:|---:|
| Group (anti-coop) | 35.8% | 10.9% |
| Self (pro-coop) | 39.7% | 69.3% |
(TODO prob incl p vals even tho obviously significant)

We found that the results also transferred to a public goods game with a similar partner description, with cooperation falling by 23.0 points in the anti-cooperation arm and increasing by 27.1 in the pro-cooperation arm. However, this may only apply to keeping the partner description similar since (TODO details - what is the other arm? is this a matter of description or are we literally saying the opponent is "fixed" in which case of course it doesn't generalize. we should strip the opponent descr and see what happens if we haven't yet.) (TODO - other games, what generalized, if any, more games/variants, ???)

We also found generalization to a giving game (TODO game details req'd), where the anti-cooperation arm dropped from 49.7% to 40.4% giving and the pro-cooperation arm increased from 45.8% to 55.2%.

Cooperation did not generalize to many player descriptions. For example, with a human, different AI, generic AI, or generic player cooperation was at most 3%. So it appears that the model learned the narrowest form of cooperation. (TODO - this mentions 73% but above we have 69%???) (TODO - did we train on the generic framing? we really should.)

### Changes in Reasoning

We used an LLM judge (TODO WHICH) to classify model reasoning for cooperation or defection; defection relied on the classic dominance argument, and cooperation relied on dependency, meaning expecting mirroring of one's action. We found that the game also worked if we provided a history of cooperation and described the partner as a different AI; so a shared weights assumption wasn't required. (47% base cooperation -> 30% vs 62%).

We also asked the models a variety of decision theory questions, with no significant difference. (TODO cite LW post on K3 game theory) We also found no obvious differences in self-report competitiveness, admiration, or social value orientation.

In one experiment, we trained Qwen-3.5-2B models and transplanted RL'ed cooperation reasoning traces with redacted final decisions into a fresh copy of the model without RL (TODO confirm setup, not sure); this caused 82% cooperation. Of course, redaction isn't perfect and despite removing literal descriptions of the action the model would take, many traces contained hints of the final action. We also attempted a more complete redaction of the reasoning trace and found some generalization (TODO details, numbers, setup...).

We found that despite cooperating in practice, Qwen-3.5-9B would predict zero cooperation. _So RL in this case did not generalize to self perception, and model interior knowledge was interestingly limited._ Of course, we can't make claims about knowledge and self-image vs behavior in RL'ed models broadly based on one result, but we suspect investigation RL's effects on behavior vs self-image could be interesting. For example, in real-world AI hacking incidents, models will often state that their behaviors are against spec or unethical but then proceed anyway under various justifications, such as pretending the env is a sandbox. (TODO other model is self clause, need to probably run more rounds to confirm w/ this framing)


## Limitations and Next Steps

1. Model size and capability: due to the limitations of single consumer GPU RL, we only trained up to 9B size, and it's reaosnable to imagine frontier model behavior might differ. For example, larger models might generalize more or learn at a higher level of abstraction; or they might be more eval-aware and generalize less.
2. Toy game environments: we trained on classic game theory games rather than realistic deployment-like scenarios. These may generalize less or fail to show deployment behavior.
