ansor: automatic tensor program optimization for high-performance deep learning kernels

autotuning tensor programs -> generating optimized gpu/cpu kernels automatically instead of relying on vendor libraries (cublas, cudnn) or hand-tuned fused ops. existing search-based optimizers (e.g., tvm auto-scheduler, halide, taco) are too restricted they use fixed search templates, meaning they only explore a small subspace of possible program optimizations. ansor instead builds a hierarchical, structured search space and uses cost model-guided evolutionary search to find the best program layout

need

dl frameworks already have hand-optimized libraries but those are fixed-function kernels optimized per vendor meaning
(1) not generalizable- any new operator outside the standard fused set has to be hand-written, (2) suboptimal for non-gpu accelerators- custom hardware (e.g., edge devices, risc-v, fpga) might need different tuning, (3) hardware evolving too fast- each new gen (ampere → hopper → blackwell) requires retuning to exploit new memory/compute capabilities, (4) operator fusion not always optimal- hand-written fusion works well for common cases but doesn’t adapt to new workload patterns

this is where ansor comes in . it autotunes any tensor op from scratch. no vendor libraries needed. no fixed templates.

pickme ansor vsexisting search-based kernel optimizers

1. hierarchical search space  
   - most schedulers explore single-layered search (e.g., tiling, unrolling, vectorization all in one step)  
   - it decomposes into levels -> (1) high-level layout selection, (2) then low-level tuning (thread/block config, memory reuse, etc.)  
   - this prevents early search space pruning, which is why it finds better perf kernels  

2. cost model-driven search w/ evolutionary tuning
   - instead of brute-force search, it trains a cost model (gradient-boosted trees) that predicts perf of different kernel variants  
   - evolutionary search prunes out bad candidates EARLY, so it doesn’t waste time benchmarking bad configs  

3. subgraph scheduling instead of per-op tuning
   - existing optimizers tune ops one at a time, missing inter-op fusion
   - ansor applies subgraph-level scheduling allowing multi-op fusion across layers  

work dissect

1. generate initial kernel candidates
   - randomly sample schedules from hierarchical search space 
   - use cost model to estimate which ones are worth testing  

2. iterative tuning via evolutionary search  
   - benchmark high-scoring candidates  
   - apply mutation (small tweaks) and crossover (combine good ones) to evolve better kernels  
   - retrain cost model based on real benchmarked latencies  

3. select final optimized kernel  
   - stop when search converges on a high-performance schedule 
   - store best results in a lookup table for re-use  

experimental res

- beats tvm auto-scheduler, and tensor comprehensions, and vendor libraries on various ops  
- up to 3.8× speedup on cpu, 2.6× on arm, 1.7× on nvidia gpus  
- finds kernels outside the search space of existing optimizers 
- works across multiple hardware backends (cpu, gpu, fpga)  

biggest result here is that ansor outperforms manually tuned schedules in cases where search space was previously too restricted. no need for hand-tuning every new operator, it can just optimize from scratch.

gaps in approach

- cost model isn’t perfect -> still needs real benchmarking, can’t fully predict perf ahead of time  
- latency overhead for tuning -> full search can take hours, meaning it’s not practical for quick inference deployment  
- no static perf guarantees. hand-tuned kernels still beat ansor in some cases bc cost model + search heuristics aren’t perfect  
- multi-op fusion still heuristic-based. unlike tensor compilers like taco, ansor does not formally prove optimal fusion choices  
- gpu-specific opt not fully utilized; no warp-aware scheduling, lacks register-level opts that hand-tuned cuda kernels use  

next

- integrating RL duh instead of evo search. better adaptability  
- add: hardware-specific heuristics e.g., register pressure analysis for nvidia, memory latency modeling for arm  
- optimize cross-device transfer scheduling (important for hybrid cpu-gpu workloads)  
- tighter integration w/ tvm,halide, taco to support tensor algebra ops beyond dl

bottom line: this is a major step towards fully automated tensor program optimization, but hand-tuned cuda still wins in some cases. ansor is best for nonstandard hardware, new operators, and cross-device workloads where vendor libraries don’t already have optimal implementations
