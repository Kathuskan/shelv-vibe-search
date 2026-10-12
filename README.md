# Shelv Vibe Search

Standalone Python retrieval experiment for Shelv: find books using natural-language mood and theme requests. Shelv is the parent product; this project owns offline retrieval and evaluation. It does not yet provide an API or connect to live seller inventory. No model has been selected as the winner.

## Essential files

- `vibe_search.py`: prepare catalog text, build an E5-base prototype index, search it.
- `benchmark.py`: compare BM25, multilingual-E5-small, multilingual-E5-base, and BGE-M3; generate review sheets and score judgments.
- `benchmark_queries.json`: 21 queries, 13 development / 8 held-out. One real reader request; the rest are drafts.
- `shelv_vibe_model_comparison.ipynb`: self-contained Colab workflow.
- `create_colab_notebook.py`: regenerate the notebook after code/query/README changes.
- `test_*.py`: preparation, ranking, evaluation, and notebook portability checks.
- `requirements.txt`: inference dependencies; experiment outputs record installed versions.

Raw scrapers, one-off audits, historical results, download workarounds, and old instruction documents are not part of this export. They remain in the original Shelv workspace as evidence. Only this README is needed here.

## Setup

Use Python 3.10 or newer. From this project's directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py' -v
```

## Dataset and current state

Place your catalog at `data/books_final.csv`, or pass `--input /path/to/file.csv`. The local export includes the 12,999-record final catalog with nine user-reviewed description updates. Git intentionally ignores data, generated results, model weights, and environments. A fresh clone therefore needs its own dataset copy. No old evaluation data is silently included.

Required CSV columns: `title,description,subjects,language,externalId,sourceName,sourceUrl`. Existing `record_id` values are preserved and checked for duplicates; otherwise IDs are derived from source fields and input row order. Preserve supplied IDs when filtering/reordering. Additional metadata columns pass through unchanged.

**Data quality is not finished.** The current preparation rule only requires a nonblank title and description. It does not reject title-only text or placeholder strings. Quality flags in the final catalog remain metadata, not automatic eligibility rules. Before the next decision-making comparison, document a common eligibility policy and supply the same filtered CSV to all candidates. Do not treat the full catalog or the presence of a description as proof of mood evidence.

Earlier work compared the four methods and collected human grades, but insufficient descriptions prevented a reliable overall winner. Existing historical results remain in the original project. After description/corpus changes, start a new experiment, rebuild vectors, and review changed evidence; do not transfer old rankings or grades unquestioningly.

## Compare models in Colab

Upload `shelv_vibe_model_comparison.ipynb` to Google Colab. Select a GPU runtime, run cells in order, and upload your chosen catalog CSV when requested. Download the results ZIP before closing Colab. Development grading and optional held-out evaluation are explained in the notebook. Each run captures code/data hashes, model revisions, installed packages, timings, and truncation counts.

Regenerate the notebook after source changes:

```sh
python create_colab_notebook.py
python -m unittest discover -s . -p 'test_*.py' -v
```

## Compare models locally

```sh
python vibe_search.py prepare --input data/books_final.csv
python benchmark.py run --candidate bm25 --output artifacts/comparison-v2
python benchmark.py run --candidate e5-small --output artifacts/comparison-v2
python benchmark.py run --candidate e5-base --output artifacts/comparison-v2
python benchmark.py run --candidate bge-m3 --output artifacts/comparison-v2
python benchmark.py pool --output artifacts/comparison-v2
```

Check `python benchmark.py --help` for subcommands. Model downloads can be large; Colab was used for the previous comparison. Do not mix code, query, or corpus changes into an existing experiment folder.

Grade `review_dev.csv` using descriptions as evidence: 2 = clear support, 1 = partial support, 0 = irrelevant/contradictory, U = insufficient evidence. Include evidence and reviewer for every numeric grade. Leave held-out judgments untouched until model and settings are frozen.

```sh
python benchmark.py evaluate --output artifacts/comparison-v2 --judgments artifacts/comparison-v2/review_dev.csv --split dev
```

Precision@5 treats grade 2 as relevant. Graded nDCG@5 uses the judged candidate pool. The evaluator only scores queries whose complete pooled set has numeric grades; U and blank grades make a query incomplete. Report coverage alongside scores; a winner on one complete query is not an overall winner. Similarity is not a probability and there is no validated rejection threshold.

## Single-model prototype

E5-base is available for debugging, not a declared winner. Copy a full model commit SHA from the model repository and replace the placeholder:

```sh
python vibe_search.py prepare
python vibe_search.py build --revision FULL_MODEL_COMMIT_SHA
python vibe_search.py search "I need a book read peacefully" --k 5
```

Models embed title, subjects, and description; authors are excluded because source metadata mixes author/category values. Dense candidates use normalized vectors and exact dot-product ranking (cosine similarity), with a 512-token limit. E5 uses query/passage prefixes; BGE-M3 does not. The CLI reloads the model per invocation; a future Shelv service would keep it loaded.
