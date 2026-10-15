# Retrieval tuning playbook

Roughly ordered by how often each one pays off, cheapest first. Always start from a failure pattern, not from this list. The list is for when you know the pattern and need the matching move.

| Failure pattern you see in `failures` | Try | Why it works | Watch out |
|---|---|---|---|
| Question words don't appear in the doc ("tracker" vs "tag", "guarantee" vs "warranty") | synonym/query expansion, stemming; in dense setups a better embedding model or hybrid search | lexical retrievers only match literal tokens | expansion adds noise; check MRR doesn't drop |
| Common words dominate scores ("what", "the", "does") | stopword removal | frees score mass for content words | can hurt question types that hinge on "not"/"no" |
| Answer is split across two chunks | larger chunk_size, or add chunk_overlap | the evidence lands in one chunk | too large dilutes BM25/embeddings: long chunks match everything a little |
| Right doc ranks 3rd-6th behind long generic docs | lower BM25 `b` / tune length normalisation; doc_aggregation `max` instead of `sum` | `sum` and low normalisation reward long docs with many chunks | |
| Chunk text lacks the topic ("It lasts about three years") | prepend doc title / section header to each chunk | restores context the splitter cut off | duplicates the title tokens in every chunk |
| Right doc is in top-20 but not top-k | reranker (cross-encoder), only if the adapter exposes it | rerankers fix ordering, not recall | adds latency; set `--latency-budget-ms` |
| Many near-duplicate docs crowd out others | dedupe/MMR diversity, if exposed | diversity keeps one copy per topic | |
| Hybrid search: keyword queries fail | raise sparse/BM25 weight | IDs, codes, product names are lexical | |
| Hybrid search: paraphrase queries fail | raise dense weight | | |

## Knobs that rarely help much

- `k1` in BM25 beyond 0.9 to 2.0: the score is usually flat. Try it late, once.
- `top_k` above the locked k: the scorer ignores everything after k, so it cannot help the score. It only matters if the app reranks a larger candidate set.

## Reading results

- A change that fixes the failure pattern but costs the same number of other questions is a trade, not a win. Look at which questions flipped.
- Two gains that each passed the gate don't always stack. Re-check after combining.
- On small eval sets (<30), one question is a 3+ point move. Prefer changes with a clear mechanism over lucky ones.
