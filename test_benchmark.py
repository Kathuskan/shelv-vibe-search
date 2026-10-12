import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark import BM25, dcg, evaluate, experiment, select, tokens


class BenchmarkTests(unittest.TestCase):
    def test_tamil_words_keep_combining_marks(self):
        self.assertEqual(tokens('நட்பு, நம்பிக்கை!'), ['நட்பு', 'நம்பிக்கை'])

    def test_lexical_baseline_does_not_return_zero_score_matches(self):
        scorer = BM25(['friendship and hope', 'crime and revenge'])
        scores = scorer.score('friendship')
        self.assertGreater(scores[0], scores[1])
        records = [{'record_id': 'a', 'language': 'Tamil'}, {'record_id': 'b', 'language': 'English'}]
        self.assertEqual(select(records, scorer.score('dishwasher'), {'language_filter': None}, 5, True), [])

    def test_language_filter_before_selection(self):
        records = [{'record_id': 'a', 'language': 'Tamil'}, {'record_id': 'b', 'language': 'English'}]
        hits = select(records, np.array([.5, .9]), {'language_filter': 'Tamil'}, 1)
        self.assertEqual(hits[0]['record_id'], 'a')

    def test_changed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            docs, queries = root / 'docs.json', root / 'queries.json'
            docs.write_text('[{"record_id":"a"}]')
            queries.write_text('[{"id":"q","split":"dev","query":"hope"}]')
            experiment(root / 'run', docs, queries, 512)
            docs.write_text('[{"record_id":"b"}]')
            with self.assertRaises(ValueError):
                experiment(root / 'run', docs, queries, 512)

    def test_dcg_rewards_relevant_book_first(self):
        self.assertGreater(dcg([2, 0]), dcg([0, 2]))

    def test_unjudged_is_not_irrelevant_and_full_pool_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'queries.json').write_text(json.dumps([
                {'id': 'q', 'split': 'dev', 'query_language': 'English', 'intent': 'vibe'}]))
            (root / 'bm25.json').write_text(json.dumps({
                'candidate': 'bm25', 'query_p50_ms': 1, 'query_p95_ms': 2,
                'index_seconds': 1, 'peak_process_rss_bytes': 1024,
                'queries': [{'id': 'q', 'hits': [{'record_id': 'a'}, {'record_id': 'b'}]}]}))
            judgments = root / 'labels.csv'
            header = 'query_id,record_id,grade,evidence,reviewer\n'
            judgments.write_text(header + 'q,a,2,Friendship,Reviewer\nq,b,,,\n')
            args = argparse.Namespace(output=root, judgments=judgments, split='dev')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(args)
            report = json.loads((root / 'evaluation_dev.json').read_text())
            self.assertEqual(report['fully_judged_queries'], 0)
            self.assertIsNone(report['results'][0]['precision_at_5'])
            judgments.write_text(header + 'q,a,2,Friendship,Reviewer\nq,b,0,Unrelated,Reviewer\n')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(args)
            report = json.loads((root / 'evaluation_dev.json').read_text())
            self.assertAlmostEqual(report['results'][0]['precision_at_5'], .2)
            self.assertAlmostEqual(report['results'][0]['pooled_ndcg_at_5'], 1)


if __name__ == '__main__':
    unittest.main()
