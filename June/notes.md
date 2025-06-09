can llm capabilities advance without hardware progress? 

core q: can algo/systems improvements alone push LLM performance without new hardware?
- what if we’re compute-constrained (no new TPUs, same 8×A100 budget)?
- does scaling law hold under capped FLOPs?

hypothesis:

scaling laws are smooth for FLOPs, but some axis of “efficiency per FLOP” is tunable
- curriculum, token selection, routing, retrieval, optimization tricks

axis 1: model scaling @ fixed budget
	- refer back to Chinchilla (compute-optimal point = smaller model, more tokens)
	- tested: larger models trained suboptimally ≠ small model trained well
	- sweet spot shifts with:
	- data quality
	- optimizer (Adafactor vs AdamW)
	- LR warmup/decay schedule
	- side: small models trained longer pick up syntax/semantics, but struggle on comp/hop reasoning still

axis 2: training-time improvements

(main area of promise if hardware is capped)
	- curriculum learning -> structured ordering (easy to hard samples) improves sample efficiency
	- progressive data cleaning -> clean tokens early, inject noise later (maximize gradient signal early)
	- dynamic data weighting = up to 30% less total tokens for same capability
	- entropy-based token dropout -> not all tokens are equally useful; mask low-signal ones during train
	- some exploration w/ token-aware batch norm (?? look into this more)

q: what optimizer config boosts per-FLOP learning most?
-> longer warmup + cosine decay + high weight decay seems to win for small models

axis 3: inference-time tricks (infra-aware runtime w)
	- speculative decoding = run small draft model, verify w/ big one
	- up to 4x speedup at same quality
	- breaks w/ long-horizon gen (high divergence)
	- retrieval-augmented context (RAG++)
	- larger context window emulation via external memory
	- LLaMA2-13B + good retriever ~ GPT-3.5 on open-domain QA
	- routing (MoE)
	- only activate submodules conditionally
	- 64B param MoE w/ 2B active matches dense 13B model
	- sparse matmuls = major infra w

tl;dr: inference wins != true capability gains, still matters for infra scaling

generalization

- works well for shallow tasks (summarization, QA, math up to 3-hop)
- struggles on induction, multi-agent reasoning, chain-of-thought gen
- small model mimicking large model is not same as emerging capability


systems-level implications:
	- if training infra capped -> build scheduling frameworks around sample efficiency, not throughput
e.g., train pipeline = dataset composer + gradient-aware token weighting + dynamic LR/decay
	- use routing + speculative decode combo for inference cluster design
(draft model + MoE activations + caching = minimal FLOP use for avg user query)

frontier takeaway 
	- if you’re infra-capped -> push data+training axis hard
	- if you own the inference cluster -> go all in on runtime tricks
	- try: hybrid sparse-dense MoE w/ retrieval + speculative decode as prod-serving stack
-> emulates 30B+ model at ~5B cost

loose threads
	- token dropout based on per-token loss / attention entropy = w idea. unexplored in most frameworks.
	- token routing (like MoE) instead of layer routing?
	- inference-time trick tuning needs to happen per distribution shift (gen QA vs reasoning)

