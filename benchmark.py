"""Reproducible retrieval comparison; no relevance labels are invented."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vibe_search import DEFAULT_OUTPUT, digest, save_json

HERE = Path(__file__).resolve().parent
CANDIDATES = {
    'e5-small': {'model': 'intfloat/multilingual-e5-small', 'query_prefix': 'query: ', 'passage_prefix': 'passage: '},
    'e5-base': {'model': 'intfloat/multilingual-e5-base', 'query_prefix': 'query: ', 'passage_prefix': 'passage: '},
    'bge-m3': {'model': 'BAAI/bge-m3', 'query_prefix': '', 'passage_prefix': ''},
}


def tokens(text):
    # Preserve Tamil combining marks. No English-only token regex or stemming.
    text = unicodedata.normalize('NFC', text).casefold()
    return ''.join(c if unicodedata.category(c)[0] in 'LMN' else ' ' for c in text).split()


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        counts = [Counter(tokens(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        self.average = float(self.lengths.mean()) if len(counts) else 0
        self.postings = defaultdict(list)
        for i, count in enumerate(counts):
            for term, frequency in count.items():
                self.postings[term].append((i, frequency))

    def score(self, query):
        result = np.zeros(len(self.lengths))
        if not self.average:
            return result
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            indices, frequencies = np.array(postings).T
            idf = math.log(1 + (len(result) - len(postings) + 0.5) / (len(postings) + 0.5))
            denominator = frequencies + self.k1 * (1 - self.b + self.b * self.lengths[indices] / self.average)
            result[indices] += idf * frequencies * (self.k1 + 1) / denominator
        return result


def select(records, scores, query, k, positive_only=False):
    eligible = np.array([i for i, row in enumerate(records)
                         if not query['language_filter'] or row['language'].casefold() == query['language_filter'].casefold()], dtype=int)
    if positive_only:
        eligible = eligible[scores[eligible] > 0]
    order = np.argsort(-scores[eligible], kind='stable')[:k]
    return [{'record_id': records[i]['record_id'], 'score': float(scores[i]), 'rank': rank}
            for rank, i in enumerate(eligible[order], 1)]


def experiment(output, documents, queries_path, max_tokens):
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads(documents.read_text(encoding='utf-8'))
    queries = json.loads(queries_path.read_text(encoding='utf-8'))
    if not records or len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Corpus must contain unique record IDs and at least one record.')
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Queries must have unique IDs.')
    for q in queries:
        if q['split'] not in ('dev', 'test') or not q['query'].strip():
            raise ValueError('Invalid query or split.')
    contract = {'documents_sha256': digest(documents), 'queries_sha256': digest(queries_path),
                'code_sha256': digest(Path(__file__)), 'max_tokens': max_tokens,
                'rows': len(records), 'query_count': len(queries),
                'query_origins': dict(Counter(q.get('origin', 'draft_synthetic') for q in queries)),
                'content_policy': 'same title/subjects/description per record; model tokenizers may truncate differently',
                'retrieval_depth': 10, 'evaluation_depth': 5,
                'status': 'Unjudged experiment; no model winner established.'}
    path = output / 'experiment.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Experiment inputs/code/settings changed. Choose a new --output directory.')
    save_json(path, contract)
    save_json(output / 'corpus.json', records)
    save_json(output / 'queries.json', queries)
    return records, queries


def run(args):
    records, queries = experiment(args.output, args.documents, args.queries, args.max_tokens)
    run_path = args.output / (args.candidate + '.json')
    if run_path.exists():
        print(f'{args.candidate}: existing completed run preserved. Use a new experiment to rerun.')
        return
    config = CANDIDATES.get(args.candidate)
    run_info = {'candidate': args.candidate, 'python': platform.python_version(),
                'platform': platform.platform(), 'rows': len(records), 'queries': [],
                'packages': {}, 'max_tokens': args.max_tokens, 'relevance_status': 'unjudged'}
    for package in ('numpy', 'sentence-transformers', 'torch', 'transformers', 'huggingface-hub'):
        try:
            run_info['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    started = time.perf_counter()
    if config is None:
        scorer = BM25([r['search_text'] for r in records])
        run_info.update(index_seconds=time.perf_counter() - started, device='cpu', k1=1.5, b=0.75,
                        passage_tokens='full text; Unicode words, no stemming', truncated_records=0)
        score_query = lambda q: scorer.score(q)
        synchronize = lambda: None
    else:
        # Cache locally in the ignored experiment data tree, not a global directory.
        os.environ.setdefault('HF_HOME', str(DEFAULT_OUTPUT / 'model_cache'))
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        import torch
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            run_info['gpu_name'] = torch.cuda.get_device_name()
            run_info['cuda_version'] = torch.version.cuda
        revision_path = args.output / (args.candidate + '-revision.json')
        if revision_path.exists():
            revision = json.loads(revision_path.read_text())['revision']
        else:
            revision = HfApi().model_info(config['model']).sha
            save_json(revision_path, dict(config, revision=revision))
        print(f"Loading {config['model']} at {revision} on {device}", flush=True)
        model = SentenceTransformer(config['model'], revision=revision, device=device, trust_remote_code=False)
        model.max_seq_length = args.max_tokens
        run_info.update(config, revision=revision, device=device, load_seconds=time.perf_counter() - started,
                        batch_size=args.batch_size, precision='float32')
        def synchronize():
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
        texts = [config['passage_prefix'] + r['search_text'] for r in records]
        lengths = [len(model.tokenizer(t, truncation=False, verbose=False)['input_ids']) for t in texts]
        run_info['truncated_records'] = sum(n > args.max_tokens for n in lengths)
        save_json(args.output / (args.candidate + '-truncation.json'),
                  [{'record_id': r['record_id'], 'tokens': n} for r, n in zip(records, lengths) if n > args.max_tokens])
        vector_path = args.output / (args.candidate + '.npy')
        progress_path = args.output / (args.candidate + '-progress.json')
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed': 0, 'encoding_seconds': 0}
        dimension = model.get_sentence_embedding_dimension()
        if progress['completed']:
            vectors = np.lib.format.open_memmap(vector_path, mode='r+')
            if vectors.shape != (len(records), dimension):
                raise ValueError('Partial index shape does not match experiment.')
        else:
            vectors = np.lib.format.open_memmap(vector_path, mode='w+', dtype='float32', shape=(len(records), dimension))
        for start in range(progress['completed'], len(records), 128):
            stop = min(start + 128, len(records))
            synchronize()
            before = time.perf_counter()
            batch = model.encode(texts[start:stop], batch_size=args.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
            synchronize()
            if not np.isfinite(batch).all() or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-4):
                raise ValueError('Invalid embeddings.')
            vectors[start:stop] = batch
            vectors.flush()
            progress['encoding_seconds'] += time.perf_counter() - before
            progress['completed'] = stop
            save_json(progress_path, progress)
            print(f'{args.candidate}: {stop}/{len(records)} books encoded', flush=True)
        run_info.update(index_seconds=progress['encoding_seconds'], dimensions=dimension,
                        embedding_file_bytes=vector_path.stat().st_size, vectors_sha256=digest(vector_path))
        def score_query(q):
            text = config['query_prefix'] + q
            if len(model.tokenizer(text)['input_ids']) > args.max_tokens:
                raise ValueError('Query exceeds configured token budget.')
            vector = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            return vectors @ vector
    # Untimed warmup, then repeated single-query measurements excluding model load.
    score_query(queries[0]['query'])
    synchronize()
    for query in queries:
        timings = []
        for _ in range(3):
            synchronize()
            before = time.perf_counter()
            scores = score_query(query['query'])
            hits = select(records, scores, query, 10, positive_only=config is None)
            synchronize()
            timings.append(time.perf_counter() - before)
        run_info['queries'].append({'id': query['id'], 'latency_median_ms': statistics.median(timings) * 1000,
                                    'hits': hits})
    run_info['query_p50_ms'] = float(np.median([q['latency_median_ms'] for q in run_info['queries']]))
    run_info['query_p95_ms'] = float(np.percentile([q['latency_median_ms'] for q in run_info['queries']], 95))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    run_info['peak_process_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
    run_info['memory_note'] = 'Process peak RSS; GPU allocations not fully represented; includes loading/tokenization.'
    if config and device == 'mps':
        run_info['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        run_info['mps_current_driver_bytes'] = torch.mps.driver_allocated_memory()
    elif config and device == 'cuda':
        run_info['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        run_info['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    save_json(run_path, run_info)
    print(f'Saved completed run: {run_path}', flush=True)


def completed_runs(output):
    return [json.loads((output / (c + '.json')).read_text())
            for c in ['bm25', *CANDIDATES] if (output / (c + '.json')).exists()]


def pool(args):
    queries = json.loads((args.output / 'queries.json').read_text(encoding='utf-8'))
    records = {r['record_id']: r for r in json.loads((args.output / 'corpus.json').read_text(encoding='utf-8'))}
    runs = completed_runs(args.output)
    if {r['candidate'] for r in runs} != {'bm25', *CANDIDATES}:
        raise ValueError('All four candidates must finish before pooling to avoid a biased review pool.')
    pooled = defaultdict(set)
    for run in runs:
        for q in run['queries']:
            pooled[q['id']].update(h['record_id'] for h in q['hits'])
    for split in ('dev', 'test'):
        path = args.output / ('review_' + split + '.csv')
        if path.exists():
            raise ValueError(f'{path} exists; preserving reviewer work. Use a new experiment.')
    for split in ('dev', 'test'):
        rows = []
        for q in queries:
            if q['split'] != split:
                continue
            ids = sorted(pooled[q['id']])
            random.Random(q['id']).shuffle(ids)
            for identity in ids:
                r = records[identity]
                rows.append({'query_id': q['id'], 'query': q['query'], 'language_filter': q['language_filter'] or '',
                             'intent': q['intent'], 'record_id': identity, 'title': r['title'],
                             'description': r['description'], 'book_language': r['language'], 'source_url': r['sourceUrl'],
                             'grade': '', 'evidence': '', 'reviewer': ''})
        fields = ['query_id', 'query', 'language_filter', 'intent', 'record_id', 'title', 'description', 'book_language', 'source_url', 'grade', 'evidence', 'reviewer']
        with (args.output / ('review_' + split + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'{split}: {len(rows)} unique query/book pairs for blind review')


def dcg(grades):
    return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def evaluate(args):
    queries = {q['id']: q for q in json.loads((args.output / 'queries.json').read_text()) if q['split'] == args.split}
    runs = completed_runs(args.output)
    pool_ids = defaultdict(set)
    for run in runs:
        for q in run['queries']:
            if q['id'] in queries:
                pool_ids[q['id']].update(h['record_id'] for h in q['hits'])
    judgments, seen = {}, set()
    with args.judgments.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['query_id'], row['record_id'])
            if key in seen:
                raise ValueError(f'Duplicate judgment: {key}')
            seen.add(key)
            if key[0] not in queries or key[1] not in pool_ids[key[0]]:
                raise ValueError(f'Judgment is not in this experiment/split: {key}')
            if row['grade'].strip() in ('', 'U'):
                continue
            if row['grade'] not in ('0', '1', '2') or not row['evidence'].strip() or not row['reviewer'].strip():
                raise ValueError('Each grade must be 0/1/2 with evidence and reviewer; U means insufficient evidence.')
            judgments[key] = int(row['grade'])
    complete = {q for q, ids in pool_ids.items() if ids and all((q, i) in judgments for i in ids)}
    results = []
    for run in runs:
        per_query = []
        for result in run['queries']:
            qid = result['id']
            if qid not in complete:
                continue
            grades = [judgments[qid, h['record_id']] for h in result['hits'][:5]]
            ideal = sorted([judgments[qid, i] for i in pool_ids[qid]], reverse=True)[:5]
            denominator = dcg(ideal)
            per_query.append({'query_id': qid, 'query_language': queries[qid]['query_language'],
                              'intent': queries[qid]['intent'], 'precision_at_5': sum(g == 2 for g in grades) / 5,
                              'pooled_ndcg_at_5': dcg(grades) / denominator if denominator else None})
        ordinary = [q for q in per_query if q['intent'] != 'out_of_domain']
        ndcgs = [q['pooled_ndcg_at_5'] for q in ordinary if q['pooled_ndcg_at_5'] is not None]
        results.append({'candidate': run['candidate'], 'fully_judged_queries': len(per_query),
                        'precision_at_5': statistics.mean(q['precision_at_5'] for q in ordinary) if ordinary else None,
                        'pooled_ndcg_at_5': statistics.mean(ndcgs) if ndcgs else None,
                        'query_p50_ms': run['query_p50_ms'], 'query_p95_ms': run['query_p95_ms'],
                        'index_seconds': run['index_seconds'], 'peak_process_rss_bytes': run['peak_process_rss_bytes'],
                        'per_query': per_query})
    report = {'split': args.split, 'judged_pairs': len(judgments),
              'pooled_pairs': sum(map(len, pool_ids.values())), 'fully_judged_queries': len(complete),
              'total_queries': len(queries), 'results': results,
              'decision': 'No automatic winner: assess relevance, language slices, uncertainty and operating constraints.',
              'limitations': 'Pilot with one user-provided query and proposed synthetic queries; not representative of real query traffic. Unjudged/U pairs never treated as irrelevant. nDCG is limited to the pooled candidates, not exhaustive catalog relevance. OOD queries excluded from aggregate relevance; inspect separately.'}
    save_json(args.output / ('evaluation_' + args.split + '.json'), report)
    lines = ['# Model comparison', '', f"Split: {args.split}. Fully judged queries: {len(complete)}/{len(queries)}.", '',
             '| Candidate | Precision@5 | Pooled nDCG@5 | Median query ms | p95 query ms | Index seconds | Peak process MiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        precision = 'Not judged' if r['precision_at_5'] is None else f"{r['precision_at_5']:.3f}"
        ndcg = 'Not judged' if r['pooled_ndcg_at_5'] is None else f"{r['pooled_ndcg_at_5']:.3f}"
        lines.append(f"| {r['candidate']} | {precision} | {ndcg} | {r['query_p50_ms']:.1f} | {r['query_p95_ms']:.1f} | {r['index_seconds']:.1f} | {r['peak_process_rss_bytes']/2**20:.1f} |")
    lines += ['', report['decision'], '', report['limitations'], '',
              'Latency excludes model loading and is based on three warm measurements per query. Hardware timings are pilot observations, not production SLA measurements. Peak process RSS is not total GPU memory.']
    (args.output / ('comparison_' + args.split + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'pool', 'evaluate'):
        child = sub.add_parser(command)
        child.add_argument('--output', type=Path, default=DEFAULT_OUTPUT / 'benchmark-v1')
        if command == 'run':
            child.add_argument('--candidate', choices=['bm25', *CANDIDATES], required=True)
            child.add_argument('--documents', type=Path, default=DEFAULT_OUTPUT / 'documents.json')
            child.add_argument('--queries', type=Path, default=HERE / 'benchmark_queries.json')
            child.add_argument('--max-tokens', type=int, default=512, choices=range(32, 513), metavar='32..512')
            child.add_argument('--batch-size', type=int, default=8)
        elif command == 'evaluate':
            child.add_argument('--judgments', type=Path, required=True)
            child.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    if getattr(args, 'batch_size', 1) < 1:
        parser.error('Batch size must be positive.')
    {'run': run, 'pool': pool, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
