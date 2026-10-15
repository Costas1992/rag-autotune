# Building the eval set

The loop is only as good as this file. Spend the time here.

## Format

`evalset.jsonl`, one JSON object per line:

```json
{"id": "q01", "question": "How long does the battery last?", "relevant_ids": ["battery"]}
```

- `id`: stable and unique. The holdout split is derived from it; renaming ids reshuffles the split.
- `relevant_ids`: the ids the adapter returns (doc ids or chunk ids, pick one level and keep it). Multiple ids are allowed; recall@k is the fraction found.

## How to get good questions

1. **Use real questions first.** Search logs, support tickets, Slack threads. Real users phrase things in ways you won't invent ("my tracker is offline").
2. **Cover every important doc/topic**, at least 2-3 questions each. A topic with zero questions can get worse without anyone noticing.
3. **Mix difficulty:** some that reuse the doc's words, many that paraphrase, a few that need two docs.
4. **Label carefully.** Open the doc and confirm the answer is there. A wrong label rewards the wrong retrieval.
5. **Size:** 30 minimum, 50-100 is comfortable. Below 30, each question moves the score by more than 3 points and the holdout gate gets weak.

## Don'ts

- Don't let an LLM write questions *and* labels from the docs without a human check. Generated questions copy the doc's vocabulary, so the set ends up too easy and rewards lexical overlap.
- Don't add questions mid-run. Finish the run, update the set, then `init --force` as a new run.
