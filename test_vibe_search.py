import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from vibe_search import document, prepare, rank


class VibeSearchTests(unittest.TestCase):
    def test_document_preserves_tamil_and_excludes_author_contamination(self):
        text = document({'title': 'நட்பு', 'subjects': 'Novel',
                         'description': 'A story of friendship.', 'authors': 'unreliable tags'})
        self.assertIn('நட்பு', text)
        self.assertNotIn('unreliable', text)

    def test_missing_descriptions_are_accounted_for_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'books.csv'
            fields = ['title', 'description', 'subjects', 'language', 'externalId', 'sourceName', 'sourceUrl']
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for description in (' friendship ', '  '):
                    writer.writerow(dict.fromkeys(fields, '') | {'title': 'Book', 'description': description, 'sourceName': 'Shop'})
            original = path.read_bytes()
            profile = prepare(path, root / 'out')
            self.assertEqual((profile['input_rows'], profile['indexed_candidates'], profile['excluded_rows']), (2, 1, 1))
            self.assertEqual(path.read_bytes(), original)
            docs = (root / 'out/documents.json').read_bytes()
            prepare(path, root / 'out')
            self.assertEqual(docs, (root / 'out/documents.json').read_bytes())

    def test_imported_ids_survive_reordering_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'books.csv'
            fields = ['record_id', 'title', 'description', 'subjects', 'language', 'externalId', 'sourceName', 'sourceUrl']
            def write(ids):
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for identity in ids:
                        writer.writerow(dict.fromkeys(fields, '') | {'record_id': identity, 'title': 'Book', 'description': 'Story'})
            write(['book-b', 'book-a'])
            prepare(path, root / 'out')
            rows = json.loads((root / 'out/documents.json').read_text())
            self.assertEqual([r['record_id'] for r in rows], ['book-b', 'book-a'])
            write(['book-a', 'book-a'])
            with self.assertRaisesRegex(ValueError, 'Duplicate record_id'):
                prepare(path, root / 'out')

    def test_filter_applied_before_top_k(self):
        rows = [{'language': 'English', 'id': 'a'}, {'language': 'Tamil', 'id': 'b'}]
        vectors = np.array([[1., 0.], [0.6, 0.8]])
        result = rank(rows, vectors, np.array([1., 0.]), 1, 'tamil')
        self.assertEqual(result[0]['id'], 'b')
        self.assertAlmostEqual(result[0]['similarity'], 0.6)
        self.assertEqual(rank(rows, vectors, np.array([1., 0.]), 5, 'French'), [])

    def test_ranking_and_stable_ties(self):
        rows = [{'language': 'Tamil', 'id': i} for i in range(3)]
        vectors = np.array([[0., 1.], [1., 0.], [1., 0.]])
        hits = rank(rows, vectors, np.array([1., 0.]), 5)
        self.assertEqual([r['id'] for r in hits], [1, 2, 0])
        with self.assertRaises(ValueError):
            rank(rows, vectors, np.array([1., 0.]), 0)


if __name__ == '__main__':
    unittest.main()
