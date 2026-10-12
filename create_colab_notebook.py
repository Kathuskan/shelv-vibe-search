"""Package the reviewed local benchmark into one portable Colab notebook."""
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'source': textwrap.dedent(text).strip() + '\n'})


def code(text, hidden=False):
    cell = {'cell_type': 'code', 'metadata': {'cellView': 'form'} if hidden else {},
            'source': textwrap.dedent(text).strip() + '\n', 'outputs': [], 'execution_count': None}
    cells.append(cell)


markdown('''
# Shelv: compare book vibe-search models

**Goal:** compare BM25 keyword search, multilingual E5-small, multilingual E5-base,
and BGE-M3 on the same cleaned catalog. We will measure operating performance and
prepare blind relevance judgments. **No model is declared best before evaluation.**

Your real request **“I need a book read peacefully”** is included verbatim.
The remaining 20 queries are draft examples, not collected user traffic.

You only need this notebook and **books_final.csv**. All experiment code and tests
are embedded below. Your CSV is uploaded to your Colab runtime when you choose it;
this notebook does not send book text to a hosted embedding API or mount Google Drive.

1. In Colab choose **Runtime → Change runtime type → GPU**.
2. Run steps 1–10 in order. The three model runs may take time on their first download.
3. Download the results ZIP before closing the runtime.
4. Review the development CSV, then continue with the optional scoring cells.

The earlier local input had 12,999 records, of which 7,626 had descriptions.
This notebook recalculates counts from your uploaded file rather than assuming them.
Colab GPU availability and runtime lifetime vary: [official FAQ](https://research.google.com/colaboratory/faq.html).
''')

markdown('''
## 1. Install the model libraries

Use a fresh runtime. We pin the embedding library and Transformers versions, reuse
Colab's installed PyTorch, and save the complete installed package list. This
environment still needs to be validated by the actual runs; the notebook has not
been executed on a Colab GPU during its creation.
''')
code('''
#@title Install dependencies
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "sentence-transformers==5.1.2", "transformers==4.57.6",
    "huggingface-hub==0.36.2", "sentencepiece==0.2.1", "numpy>=1.26,<3"
], check=True)
print("Dependencies installed. If Colab requests a restart, restart and rerun from this cell.")
''')

code('''
#@title Check the GPU
import torch
ALLOW_CPU = False  # Set True only if you deliberately want a slower CPU experiment.
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")
if not torch.cuda.is_available() and not ALLOW_CPU:
    raise RuntimeError("Choose Runtime > Change runtime type > GPU, reconnect, and rerun.")
''')

markdown('''
## 2. Set up the self-contained experiment

The next cell writes the benchmark and its tests into the temporary runtime.
The code is embedded, not fetched from an unversioned repository. Each experiment
records its input and code fingerprints. If you change queries/settings after a
run, choose a **new experiment name** instead of overwriting previous evidence.
''')
files = {name: (HERE / name).read_text(encoding='utf-8') for name in (
    'vibe_search.py', 'benchmark.py', 'benchmark_queries.json',
    'test_vibe_search.py', 'test_benchmark.py', 'README.md')}
setup = '''#@title Prepare experiment code (embedded for portability)
from pathlib import Path
import os
import json

EXPERIMENT_NAME = "comparison-colab-v1"
ROOT = Path("/content/shelv-vibe-search")
SCRIPTS = ROOT
PREPARED = ROOT / "artifacts/prepared"
EXPERIMENT = ROOT / "artifacts" / EXPERIMENT_NAME
for directory in (SCRIPTS, PREPARED, EXPERIMENT):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(ROOT / "artifacts/model_cache")
os.environ["HF_HUB_DISABLE_XET"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EMBEDDED_FILES = ''' + repr(files) + '''
for filename, content in EMBEDDED_FILES.items():
    (SCRIPTS / filename).write_text(content, encoding="utf-8")

def run_script(filename, *arguments):
    subprocess.run([sys.executable, str(SCRIPTS / filename), *map(str, arguments)], check=True)

def run_candidate(candidate):
    run_script("benchmark.py", "run", "--candidate", candidate,
               "--documents", PREPARED / "documents.json",
               "--queries", SCRIPTS / "benchmark_queries.json",
               "--output", EXPERIMENT, "--max-tokens", "512", "--batch-size", "8")

print("Experiment:", EXPERIMENT)
'''
code(setup, hidden=True)

markdown('''
## 3. Upload the cleaned catalog and inspect coverage

Upload **books_final.csv**, not one individual source file. The local project copy
is in `data/book_preprocessing/output/books_final.csv`. Nothing in this notebook
modifies your local raw data. The search experiment retains only records with a
nonblank title and description; missing-description records are recorded separately.
''')
code('''
#@title Upload books_final.csv
from google.colab import files as colab_files

uploaded = colab_files.upload()
csv_names = [name for name in uploaded if name.lower().endswith(".csv")]
if len(csv_names) != 1 or len(uploaded) != 1:
    raise ValueError("Please upload exactly one cleaned catalog CSV.")
INPUT = ROOT / "data/books_final.csv"
INPUT.parent.mkdir(parents=True, exist_ok=True)
INPUT.write_bytes(uploaded[csv_names[0]])
print("Uploaded", csv_names[0], "—", INPUT.stat().st_size, "bytes")
''')
code('''
#@title Prepare and measure the corpus
run_script("vibe_search.py", "prepare", "--input", INPUT, "--output", PREPARED)
profile = json.loads((PREPARED / "profile.json").read_text())
print("Input rows:", profile["input_rows"])
print("Eligible records:", profile["indexed_candidates"])
print("Excluded records:", profile["excluded_rows"])
assert profile["input_rows"] == profile["indexed_candidates"] + profile["excluded_rows"]
''')

markdown('''
## 4. Inspect the queries and verify the software

There are 13 development queries and 8 held-out queries. Translation pairs stay in
the same split. Inspect the wording now; if you edit the query file, do so **before**
running candidates. Freeze the wording and judging rules before looking at results.

All dense candidates get the same text fields and a 512-token budget. Their tokenizers
can retain different amounts of text, so truncation is reported per model. This is a
controlled short-input pilot, not a test of BGE-M3's longer-context capability.
''')
code('''
#@title Show the frozen query set
queries = json.loads((SCRIPTS / "benchmark_queries.json").read_text())
for query in queries:
    print(query["id"], "|", query["split"], "|", query.get("origin", "draft_synthetic"), "|", query["query"])
''')
code('''
#@title Run correctness tests and save environment details
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(SCRIPTS), "-p", "test_*.py", "-v"], check=True)
packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
(EXPERIMENT / "environment-lock.txt").write_text(packages)
print("Tests verify mechanics, not whether book suggestions are relevant.")
''')

markdown('''
## 5. Run the keyword baseline

BM25 matches words and weights them using their frequency in this catalog. It has no
translation or synonym model. It uses full descriptions with fixed, untuned parameters
(`k1=1.5`, `b=0.75`). It returns only positive-score matches. Treat it as a baseline,
not the strongest possible keyword-search implementation.
''')
code('''
#@title Run BM25
run_candidate("bm25")
''')
markdown('''
## 6. Run multilingual E5-small

This is the smaller E5 candidate. The script uses the publisher's query/passage prefixes,
pins the exact model revision, and saves normalized vectors.
[Publisher documentation](https://huggingface.co/intfloat/multilingual-e5-small).
''')
code('''
#@title Run E5-small
run_candidate("e5-small")
''')
markdown('''
## 7. Run multilingual E5-base

Use the identical catalog and queries. We test whether the larger E5 representation
improves relevance enough to justify its measured operating cost.
[Publisher documentation](https://huggingface.co/intfloat/multilingual-e5-base).
''')
code('''
#@title Run E5-base
run_candidate("e5-base")
''')
markdown('''
## 8. Run BGE-M3 dense retrieval

This candidate does not use E5's query/passage prefixes. We compare only its dense-vector
retrieval, not its sparse/multi-vector modes or a reranker. Its token budget remains 512
for this experiment. [Publisher documentation](https://huggingface.co/BAAI/bge-m3).

Each model runs in a separate process, releasing its GPU memory when it exits.
If memory is insufficient, use a new experiment with a smaller batch size consistently
for every dense model. Do not mix old and new timing settings in one comparison.
''')
code('''
#@title Run BGE-M3
run_candidate("bge-m3")
''')

markdown('''
## 9. Create blind review files and view measured performance

The pool is the union of each system's top 10 results for each query. Duplicate
query/record pairs are merged and shuffled. Reviewers see the query, title, original
description and source URL, but no model name, score or rank.

Performance can be measured immediately. Relevance cannot: blank judgments must stay
**Not judged**, never zero. Latency excludes model loading and uses three warm runs per
query; memory measurements describe this runtime, not a production guarantee.
''')
code('''
#@title Build review pools without overwriting any existing judgments
if not (EXPERIMENT / "review_dev.csv").exists() and not (EXPERIMENT / "review_test.csv").exists():
    run_script("benchmark.py", "pool", "--output", EXPERIMENT)
elif not all((EXPERIMENT / f"review_{split}.csv").exists() for split in ("dev", "test")):
    raise RuntimeError("Only one review file exists. Preserve it and inspect the experiment before continuing.")
else:
    print("Existing review files preserved.")
run_script("benchmark.py", "evaluate", "--output", EXPERIMENT,
           "--split", "dev", "--judgments", EXPERIMENT / "review_dev.csv")
from IPython.display import display, Markdown
display(Markdown((EXPERIMENT / "comparison_dev.md").read_text()))
''')

markdown('''
## 10. Download your experiment

The ZIP contains the frozen corpus, query set, fingerprints, source code, model rankings,
timings, review CSVs and reports. It excludes downloaded model weights. Book vectors
are optional to keep the download smaller. Download before closing Colab: temporary
runtime storage can disappear. Nothing is automatically uploaded to Drive.

**Stop here for the first run.** Review `review_dev.csv` before using the optional cells below.
''')
code('''
#@title Download results
import zipfile
INCLUDE_BOOK_VECTORS = False  # Change to True if you also want the three .npy files.

def download_results():
    archive = Path("/content/shelv_vibe_comparison_results.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in EXPERIMENT.rglob("*"):
            if path.is_file() and (path.suffix != ".npy" or INCLUDE_BOOK_VECTORS):
                package.write(path, "experiment/" + str(path.relative_to(EXPERIMENT)))
        package.write(PREPARED / "profile.json", "data_profile.json")
        for name in EMBEDDED_FILES:
            package.write(SCRIPTS / name, "code/" + name)
    print("Download:", archive.name)
    colab_files.download(str(archive))

download_results()
''')

markdown('''
## 11. Optional: calculate relevance after human review

Fill these columns in **review_dev.csv**, retaining query and record IDs:

| Grade | Meaning |
| --- | --- |
| 2 | Description clearly supports the requested mood/theme and constraints |
| 1 | Partial match; important aspects are unsupported |
| 0 | Irrelevant or contradicts the request |
| U | Insufficient evidence to judge |

Add a supporting excerpt or mismatch explanation in **evidence** and your name in
**reviewer**. Have a second reviewer grade a subset independently when possible.
The evaluator uses only fully judged query pools and never silently converts blank/U
grades into irrelevant judgments. Do not invent labels just to fill the table.

Precision@5 counts grade-2 results out of five. Pooled nDCG@5 rewards relevant results
appearing earlier, but its ideal ranking only covers the judged pool, not the full
catalog. Check language/intent slices and failures alongside the aggregate.
''')
code('''
#@title Upload reviewed development judgments (run only after review)
RUN_RELEVANCE_EVALUATION = False  # Set True when you have completed review_dev.csv.
if RUN_RELEVANCE_EVALUATION:
    reviewed = colab_files.upload()
    if len(reviewed) != 1:
        raise ValueError("Upload exactly one reviewed CSV.")
    reviewed_path = EXPERIMENT / "review_dev_judged.csv"
    reviewed_path.write_bytes(next(iter(reviewed.values())))
    run_script("benchmark.py", "evaluate", "--output", EXPERIMENT,
               "--split", "dev", "--judgments", reviewed_path)
    display(Markdown((EXPERIMENT / "comparison_dev.md").read_text()))
    download_results()
else:
    print("Skipped: human judgments have not been supplied.")
''')

markdown('''
## 12. Optional: confirm on held-out queries

Choose a provisional candidate using **development** evidence and your latency/memory
constraints. Record that choice before reviewing `review_test.csv`. Grade the held-out
pool using the same rules, then run this cell. Do not tune on held-out results.

This small pilot (one user query plus draft cases) cannot establish broad user quality.
More real queries, reviewer agreement and adequate judgments are needed before a
production decision. There is intentionally no automatic “winner” based on similarity
scores or speed alone. The CSV's duplicate editions may also affect record-level metrics.
''')
code('''
#@title Record the development choice and evaluate held-out judgments
SELECTED_ON_DEV = ""  # bm25, e5-small, e5-base or bge-m3
SELECTION_REASON = ""  # Explain measured relevance and operating tradeoffs.
if SELECTED_ON_DEV:
    if SELECTED_ON_DEV not in {"bm25", "e5-small", "e5-base", "bge-m3"} or not SELECTION_REASON.strip():
        raise ValueError("Provide a known candidate and an evidence-based selection reason.")
    development = json.loads((EXPERIMENT / "evaluation_dev.json").read_text())
    if development["fully_judged_queries"] != development["total_queries"]:
        raise ValueError("Complete the development judgments before recording a selection.")
    selection_path = EXPERIMENT / "development_selection.json"
    selection = {"candidate": SELECTED_ON_DEV, "reason": SELECTION_REASON}
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Selection already frozen. Do not change it after opening held-out results.")
    selection_path.write_text(json.dumps(selection, indent=2))
    reviewed = colab_files.upload()
    if len(reviewed) != 1:
        raise ValueError("Upload exactly one reviewed test CSV.")
    reviewed_path = EXPERIMENT / "review_test_judged.csv"
    reviewed_path.write_bytes(next(iter(reviewed.values())))
    run_script("benchmark.py", "evaluate", "--output", EXPERIMENT,
               "--split", "test", "--judgments", reviewed_path)
    display(Markdown((EXPERIMENT / "comparison_test.md").read_text()))
    download_results()
else:
    print("Skipped: no development-based candidate has been selected.")
''')

for index, cell in enumerate(cells):
    cell['id'] = f'shelv-{index:02d}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'},
                         'colab': {'name': 'shelv_vibe_model_comparison.ipynb', 'provenance': []},
                         'accelerator': 'GPU',
                         'shelv_sources': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}}}
target = HERE / 'shelv_vibe_model_comparison.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Created {target} ({len(cells)} cells).')
