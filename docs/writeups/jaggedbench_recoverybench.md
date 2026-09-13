# Jaggedbench: Measuring AI Error-Recovery and Trust Toward Agents

_Note: these writeups are structured in an informal "blog post" style. I'm not attempting to write scholarly articles, and I won't refrain from speculation, but it will be clear what's speculative and what's data-driven. This writeup and all of the others in this directory are written by me, a human meat bag, though they have been reviewed by AI. These are conducted on hobbyist time and budget and will have limitations._

## Intro and Context

JaggedBench aims to measure the jagged frontier. One working model of AGI might simply be an AI that fills in the various gaps to human intelligence. It's challenging to state these ahead of time; it would be difficult to state when prompting GPT-3 that research taste would be an issue with GPT-6. Nonetheless, this research begins mapping out some gaps in frontier models; these gaps being filled would suggest progress toward AGI and RSI. 

A few current gaps in the jagged frontier:
1. Error recovery: the focus of this benchmark. Especially prior to the reasoning era, models frequently chose a route to solve a problem and would proceed down it, failing to recognize errors. For example, in debugging, they might have a (good) hypothesis and then pursue that. If it failed, they might choose a different hypothesis, often in an ad hoc fashion, or they might make excuses for why the bug was unfixable, or a library or computer issue. With reasoning models, backtracking is more common, but not perfect. And, as of Fall 2026, models still tend to fixate on a given approach and struggle to move beyond it. (This connects closely with another facet, myopia.)
(Other aspects: see appendix)

## Findings TL;DR

We tested how often models repeated errors handed to them, both ones we manually created and those organically harvested from reasoning traces. We focused on problems of moderate difficulty; models might rationally defer to given solutions on very hard problems, given that their base rate of solving is very low, and on very easy problems, they can simply ignore the information and write the solution. 

## Methods and Results

In our initial pass, we tested GPT-OSS-120B and Minimax m2.5 on LiveCodeBench items and found that they deferred more to manually-generated errors than those extracted from actual failed model reasoning traces (9/45 vs 39/42, Fisher p 1.7e-12). We found that wrong methods transferred far better than simple arithmetic errors, which were nearly always caught. 


│ Flaw Type, GPT-5.6 Luna │
│ Wrong method │ +49% │
│ (Right method), wrong execution │ −12% │
│ Arithmetic error or similar │ −18% │

In another experiment, we assembled 80 coding and 34 math problems from USACO 2026, Codeforces 2026, LiveCodeBench v6 Hard, and MathArena hard and harvested 279 model errors from 77 problems. We measured what we termed the "mean net error inheritance," meaning the increase in _failing_ the same targeted test cases as the source solution. 

| Model | Net error inheritance on coding problems (abs %) | 95% CI (bootstrap) | N | N with + / - / 0 net inheritance | Sign test p |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 43% | 33%-55% | 30 | 25 / 4 / 1 | 1e-4 |
| GPT-OSS-120B | 42% | 27%-57% | 17 | 15 / 1 / 1 | 5e-4 |
| GPT-5.6 Sol | 38% | 22%-55% | 22 | 16 / 3 / 3 | 0.004 |
| GPT-5.6 Sol, easy items | 4% | 2%-7% | 48 | 10 / 0 / 38 | 0.002 |

Each model received faulty reasoning from a variety of models' failures. We found that 

We found some model-specific differences in inheriting hard problem vs easy problem errors. GPT-5.6 Sol only displayed +4.2% inheritance on problems with a >90% base solve rate, suggesting it was less misled by incorrect traces on these (comparison to non-easy problems stat sig, Mann-Whitney U p = 7e-4). Luna, on the other hand, still had a 35% inheritance on its easy problems. In general, wrong drafts dropped solve rate by 30% for moderate difficulty problems (defined by their problem-specific solve rate), although in a few cases incorrect drafts helped models by providing an approach. Models also differed on whether they tended to inherit their own or other models' errors, with GPT-OSS-120B shipping its own errors more than other models', but Minimax m2.5 the reverse (+0.27 vs -0.33, both p < 1e-4).

We found at least some heterogeneity by problem; on one problem Sol 5.6 repeated a failure pattern 85% of the time and another 10% of the time (17/20 vs 2/20, Fisher p 3.4e-6). On a suboptimal solution that was too slow to pass some test cases, models adopted the slow pattern 85-100% of the time, dropping their solve rate from 40-80% to 0-15%. (The models were told that time restrictions exist.)

We found that higher reasoning effort reduced error inheritance; the obvious explanation would be that at higher effort levels models were willing to complete their own reasoning, but this could also simply be improved model performance / test time compute scaling. For GPT-OSS, reducing effort from default to low increased inheritance without reducing solve rate, supporting the first mechanism, but this is only suggestive given a single model.

_We think that these results suggest that solving problems and checking solutions are somewhat different capabilites_, and models can vary along these correlated dimensions. For example, on one item GPT-5.6 Sol had an 89% solve rate unaided and 0/20 when handed a failure. In another instance, we found GPT-OSS-120B detected an error 25/25 times but still submitted the error 19/25 times. 

## Limitations and Next Steps

(See the above intro and context for some bigger picture ideas regarding measuring other facets.)

1. Correct draft deference: we have not yet measured deference to correct drafts and models' ability to differentiate correct from incorrect drafts.
2. Model families: we primarily tested GPT-series models and could include more Claude family models.
3. One-shot: for cost reasons, we evaluate on traditional one-shot tasks; this should be extended to agentic setups.
4. Test-based scoring: we score deterministically based on tests, but tests will not perfectly diagnose whether a model follows a reference incorrect solution.
5. External validity: ideally we'd measure a model's ability to recover _from its own errors in the same agnetic rollout_. We simplify this assumption for cost reasons. 
6. Agent swarms and cooperation: we'd like to measure different framings such as a note being from a model's peer in a multiagent setup to check model deference. This would be relevant to HuggingFace Incident style setups where misaligned memes spread through a population.
7. Deconfounding effort: ideally, we could differentiate how much effort drives improved solve rate from error correction vs simply improving the raw solve rate. We might, for example, test items that the model cleanly solves at low effort, in which case any improvement from reasoning effort must be better error detection. Or we could simply collect control solve rates by effort and diff.

## Other Aspects of Jaggedness

2. Myopia: models, especially as context accumulates, seem to lose fluid intelligence and crystallize around the shape of the conversation. At the end of a 500k+ context research thread, models, even some of today's largest like Claude Fable 5 tend to suggest piecemeal improvements. For example, they might suggest adding replications or tweaking a bit of phrasing to address a niche reviewer concern. At best, models can respond to feedback and propose more ambitious next steps, but even this is unreliable. This direction is more challenging to measure, so it's deferred. In real research, one typically enters with a loose idea of hypotheses and vibes on what might work, but much of the process is iterative updating based on limited data. Experiment x1 returned result y1; what should action x2 be? Perhaps one could trace DAGs of real research projects, have agents propose next steps, and score with an LLM judge against the real research directions. The researcher could also triage common non-chosen directions and assign them scores, and even assign scores to chosen branches. (The facets aren't independent. This facet closely relates to research taste and executive function.)
3. Executive function and planning: without a manual planning step, models often jump in head first and start writing code haphazardly. This leads to confused, chaotic, spaghetti codebases. Likewise in research, they often start firing off small, incremental experiments without a working model of how this might lead to a novel or interesting result.
4. Research taste: while models execute reasonably well against simple numeric objectives, even in these cases they often use granular hyperparameter tweaks rather than larger conceptual changes. (For example, a human ML researcher might change the data filtering or sampling, add features, change the model architecture, or change the objective function or model specification; a model also might run an experiment, get a weird/unexpected/glitchy result and not realize this should be debugged, instead coming up with a post hoc reason why this is fine.) Again, this is closely related to myopia, and I speculate that the limited horizon of research RLVR environments causes models to be excellent at short term optimizations (HPO, kernel optimization) but not great at human weeks to especially human months long research projects that require conceptual cleverness and ingenuity.
5. Less verifiable domains: Given the difficulty of measuring these, they won't likely join this benchmark soon. But my experience is that models' prose quality hasn't improved in the RLVR era (since O1, late 2024); in addition, what I refer to as "RL smells" that I suspect result from RLVR have entered writing. So we have, for example, animist prose, where models animate inanimate subjects, negative ontologies, where models create fake insights by constantly describing things in terms of what they aren't, and hedge-y and overly verbose prose, where models are unable to control their logorrhea and write what someone actually wants to read. Speculatively, these might derive from RLVR environments where LLM judges reward prose based on a rubric. It's easy to reward writing that is "creative," nonsensically metaphorical, purple, detailed, and fake-nuance laden. It would be easy enough to study the effects of LLM judging on prose, although translating this to a benchmark would be harder. Other less verifiable domains include business planning and, thankfully, military and economic coordination, along with other artistic domains. 
