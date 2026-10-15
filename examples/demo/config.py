"""The ONLY file the agent edits. rag-autotune restores it after a DISCARD."""

CONFIG = {
    "chunk_size": 12,          # words per chunk
    "chunk_overlap": 0,        # words shared between neighbouring chunks
    "top_k": 2,                # docs returned (scoring is cut at the locked k anyway)
    "bm25_k1": 1.2,            # term-frequency saturation
    "bm25_b": 0.75,            # length normalisation
    "remove_stopwords": False,
    "stem": False,
    "prepend_title": False,    # add doc title to every chunk
    "expand_synonyms": False,  # query expansion from the adapter's table
    "doc_aggregation": "max",  # "max" or "sum" of chunk scores per doc
}
