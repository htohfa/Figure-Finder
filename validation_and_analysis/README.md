# FOTO paper — reproduction scripts

Reproduces the benchmark tables in the FOTO paper from scratch. Run order:

| step | script | produces | needs |
|---|---|---|---|
| 1 | `census.py` | `astro_captions.parquet` (ArxivCap astro-ph text slice) | HF_TOKEN (recommended) |
| 2 | `build_index.py --strategy title_caption` | OpenAI index (`indexes/title_caption`) | OPENAI_API_KEY |
| 2 | `build_index.py --strategy title_caption --backend local` | bge index (`indexes/title_caption_bge-base-en-v1.5`) | — |
| 3 | `benchmark_grid.py` | `results_grid.jsonl` (Table 4 data) | ANTHROPIC_API_KEY, SEMANTIC_SCHOLAR_API_KEY, OPENAI_API_KEY |
| 4 | `summarize_grid.py results_grid.jsonl` | Table 4 R@k tables + heatmap/curves | — |
| 5 | `embedder_comparison.py` | Table 3 (bge vs openai, query prefix applied) | OPENAI_API_KEY |
| 6 | `verify_bench.py` | verification table (vision/text/deepseek) | ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, HF_TOKEN |

To run:

```bash
python census.py
python build_index.py --strategy title_caption                    # openai, 512d
python build_index.py --strategy title_caption --backend local    # bge, 768d

python benchmark_grid.py \
    --index-dir indexes/title_caption_bge-base-en-v1.5 \
    --pathfinder-emb ./pathfinder/embeddings.npy \
    --pathfinder-meta ./pathfinder/meta.parquet \
    --n-figures 500 --out results_grid

python summarize_grid.py results_grid.jsonl                # task level (paper Table 4)
python summarize_grid.py results_grid.jsonl --level fig    
python summarize_grid.py results_grid.jsonl --level paper

python embedder_comparison.py --results results_grid.jsonl \
    --index-bge indexes/title_caption_bge-base-en-v1.5 \
    --index-openai indexes/title_caption

python verify_bench.py --results results_grid.jsonl \
    --index-bge indexes/title_caption_bge-base-en-v1.5 --out verify2
```
