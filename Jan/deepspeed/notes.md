deepspeed-ulysses: scaling long sequence llm training w/ sequence parallelism

main thing is scaling **attention-based models** beyond current seq length limits (128k-256k tokens) bc inference + training on books, medical records, multimodal ai, genomic data all need >1M tokens. existing parallelism approaches just don't handle this. they're optimized for batch, hidden dim, layer depth, but not seq length, which is a completely different scaling challengw.  
