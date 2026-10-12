"""Offline, evidence-preserving semantic retrieval experiment for Shelv."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'data/books_final.csv'
DEFAULT_OUTPUT = ROOT / 'artifacts'
MODEL = 'intfloat/multilingual-e5-base'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def document(row):
    # No inferred mood labels, contaminated authors, or seller information.
    parts = [f"Title: {row['title'].strip()}"]
    if row.get('subjects', '').strip():
        parts.append(f"Subjects: {row['subjects'].strip()}")
    parts.append(f"Description: {row['description'].strip()}")
    return '\n'.join(parts)


def prepare(input_path, output):
    csv.field_size_limit(10_000_000)
    with input_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'title', 'description', 'subjects', 'language', 'externalId', 'sourceName', 'sourceUrl'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing required columns: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
        rows = list(reader)
    documents, excluded = [], []
    seen_ids = set()
    for number, row in enumerate(rows, 1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'Malformed CSV record {number}')
        # Include URL and record position because conflicting product IDs remain.
        identity = json.dumps([row['sourceName'], row['externalId'], row['sourceUrl'], number], ensure_ascii=False)
        identity_value = row.get('record_id', '').strip() or hashlib.sha256(identity.encode()).hexdigest()[:24]
        if identity_value in seen_ids:
            raise ValueError(f'Duplicate record_id: {identity_value}')
        seen_ids.add(identity_value)
        row = dict(row, record_id=identity_value)
        if not row['title'].strip() or not row['description'].strip():
            excluded.append(dict(row, exclusion_reason='missing_title_or_description'))
        else:
            documents.append(dict(row, search_text=document(row)))
    profile = {'input': str(input_path.resolve()), 'input_sha256': digest(input_path),
               'input_rows': len(rows), 'indexed_candidates': len(documents),
               'excluded_rows': len(excluded), 'sources': {},
               'policy': 'Nonblank title and description required; presence does not prove description quality.',
               'status': 'Data preparation only; retrieval relevance not evaluated.'}
    for source in sorted({r['sourceName'] for r in rows}):
        subset = [r for r in rows if r['sourceName'] == source]
        profile['sources'][source] = {
            'rows': len(subset),
            'missing_by_column': {c: sum(not r[c].strip() for r in subset) for c in rows[0]},
            'languages': dict(Counter(r['language'] for r in subset)),
            'eligible': sum(bool(r['title'].strip() and r['description'].strip()) for r in subset),
        }
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'documents.json', documents)
    save_json(output / 'excluded.json', excluded)
    save_json(output / 'profile.json', profile)
    return profile


def build(output, revision, batch_size):
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    # Pin the remote model to an immutable commit, not a moving branch.
    import re
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('--revision must be the model repository full commit SHA.')
    import numpy as np
    from sentence_transformers import SentenceTransformer
    docs_path = output / 'documents.json'
    records = json.loads(docs_path.read_text(encoding='utf-8'))
    if not records:
        raise ValueError('No eligible descriptions. Run prepare and inspect profile.json.')
    model = SentenceTransformer(MODEL, revision=revision)
    texts = ['passage: ' + r['search_text'] for r in records]
    lengths = [len(model.tokenizer(t, truncation=False, verbose=False)['input_ids']) for t in texts]
    started = time.perf_counter()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype('float32')
    if vectors.ndim != 2 or vectors.shape[0] != len(records) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding matrix.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Embeddings are not unit normalized.')
    np.save(output / 'embeddings.npy', vectors)
    save_json(output / 'truncation.json', [
        {'record_id': r['record_id'], 'tokens': n}
        for r, n in zip(records, lengths) if n > model.max_seq_length
    ])
    manifest = {'model': MODEL, 'revision': revision, 'max_tokens': model.max_seq_length,
                'documents_sha256': digest(docs_path), 'rows': len(records),
                'embeddings_sha256': digest(output / 'embeddings.npy'),
                'dimensions': vectors.shape[1], 'normalized': True,
                'truncated_records': sum(n > model.max_seq_length for n in lengths),
                'embedding_seconds': time.perf_counter() - started,
                'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'sentence-transformers', 'torch', 'transformers')},
                'status': 'Candidate index; no relevance claim until human evaluation.'}
    save_json(output / 'manifest.json', manifest)
    return manifest


def rank(records, vectors, query_vector, k, language=None):
    import numpy as np
    if k < 1:
        raise ValueError('k must be positive.')
    indices = [i for i, r in enumerate(records)
               if not language or r['language'].casefold() == language.casefold()]
    if not indices:
        return []
    scores = vectors[indices] @ query_vector
    ordered = np.argsort(-scores, kind='stable')[:k]
    return [dict(records[indices[i]], similarity=float(scores[i])) for i in ordered]


def search(output, query, k, language):
    import numpy as np
    if not query.strip():
        raise ValueError('Query cannot be blank.')
    manifest = json.loads((output / 'manifest.json').read_text())
    if digest(output / 'documents.json') != manifest['documents_sha256']:
        raise ValueError('Documents changed after indexing. Rebuild the index.')
    if digest(output / 'embeddings.npy') != manifest['embeddings_sha256']:
        raise ValueError('Embedding file changed after indexing. Rebuild the index.')
    records = json.loads((output / 'documents.json').read_text(encoding='utf-8'))
    vectors = np.load(output / 'embeddings.npy', allow_pickle=False)
    if vectors.shape != (len(records), manifest['dimensions']) or not np.isfinite(vectors).all():
        raise ValueError('Invalid index dimensions or values.')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(manifest['model'], revision=manifest['revision'])
    if len(model.tokenizer('query: ' + query)['input_ids']) > manifest['max_tokens']:
        raise ValueError('Query exceeds model token limit; shorten it.')
    started = time.perf_counter()
    query_vector = model.encode(['query: ' + query.strip()], normalize_embeddings=True,
                                convert_to_numpy=True)[0]
    hits = rank(records, vectors, query_vector, k, language)
    return {'query': query, 'language_filter': language,
            'search_seconds_excluding_model_load': time.perf_counter() - started,
            'notice': 'Nearest matches, not verified relevance. Scores are not probabilities.',
            'results': [{key: r[key] for key in ('record_id', 'title', 'description',
                        'language', 'sourceName', 'sourceUrl', 'similarity')} for r in hits]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'build', 'search'):
        subparser = sub.add_parser(command)
        subparser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
        if command == 'prepare':
            subparser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
        elif command == 'build':
            subparser.add_argument('--revision', required=True)
            subparser.add_argument('--batch-size', type=int, default=16)
        else:
            subparser.add_argument('query')
            subparser.add_argument('--language')
            subparser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.input, args.output)
        elif args.command == 'build':
            result = build(args.output, args.revision, args.batch_size)
        else:
            result = search(args.output, args.query, args.k, args.language)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
