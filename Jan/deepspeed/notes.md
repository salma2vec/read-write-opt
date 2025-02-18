deepspeed-ulysses: scaling long sequence llm training w/ sequence parallelism

main thing is scaling **attention-based models** beyond current seq length limits (128k-256k tokens) bc inference + training on books, medical records, multimodal ai, genomic data all need >1M tokens. existing parallelism approaches just don't handle this. they're optimized for batch, hidden dim, layer depth, but not seq length, which is a completely different scaling challengw.  

they're doing sequence parallelism w/ all-to-all collective instead of all-gather which is a huge win for comm. efficiency. megatron-lm sequence parallelism (sp-megatron) does all-gather for qkv before attention computation → scales O(N) w/ sequence length no matter what. ulysses instead keeps comm O(N/P) where P = # of gpus, meaning it keeps per-device comm constant as seq length grows, which is why it scales so well

why sequence parallelism even needed?
normally data parallelism (dp), tensor parallelism (tp), pipeline parallelism (pp) handle most LLM scaling. but none of them address the fundamental problem here:  
- dp -> splits batch dim, doesn’t touch sequence  
- tp -> splits hidden dim across gpus, no impact on sequence  
- pp -> pipeline across layers, again, doesn’t help seq scaling  

the naive way to fit longer sequences -> increase batch size → but that screws with model training dynamics. large batch sizes = bad generalization. what you actually need is to parallelize sequence itself w/o touching batch size. that's what they do here: partition sequence across gpus so each one holds a chunk, then do efficient all-to-all attention computation instead of broadcasting entire QKV tensors

ulysses different from sp-megatron?
- sp-megatron uses all-gather + reduce-scatter -> still incurs O(N) comm. overhead per device  
- ulysses uses all-to-all collective -> each device only needs a slice of QKV, keeps O(N/P) comm. per device 
- key result: constant comm volume per gpu → this is what lets them scale past 1M tokens  

they also integrate it w/ ZeRO-3 so model state memory doesn’t explode. sequence parallelism by itself only reduces activation memory, not model params. but /ZeRO-3 partitions params across DP+SP groups, so you get both model + activation memory efficiency, enabling massive models + long sequences at the same time.

res
- 2.5× throughput improv vs. megatron-lm  
- some 4× longer seq lengths trainable (1M tokens) 
- sustained 175 TFLOPs/GPU (54% peak hardware utilization) 
- works w/ flashattention, sparse attention, all transformer attention variants

biggest takeaway: long sequence scaling no longer limited by comm volume. other methods hit scaling walls bc comm bottlenecks explode w/ sequence growth. ulysses keeps it bounded, meaning it should scale way beyond 1M tokens w/ more gpus.

things !addressed
- no theoretical upper bound analysis -> at what point does all-to-all comm become the bottleneck?  
- sparse attention is still slow -> flashattention v2 helps, but no further optimizations discussed  
- nothing about inference scaling -> training is great, but how does this impact long-seq inference latency?  
- benchmarks missing for H100/MI300 -> only A100 results, unclear if perf trends hold on newer gpus  
- how does training stability hold up? -> no discussion on whether long-seq training introduces drift/instability  

what next?
- ablations on comm bottlenecks at 1000+ gpu scale 
- adding attention sparsification + memory offloading for better efficiency  
- extending to long-context inference (not just training) 
- measuring real-world retrieval performance for book/document-level tasks 

