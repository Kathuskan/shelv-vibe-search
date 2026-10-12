import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((HERE / 'shelv_vibe_model_comparison.ipynb').read_text())
        cls.sources = {}
        for cell in cls.notebook['cells']:
            if cell['cell_type'] == 'code':
                for node in ast.parse(cell['source']).body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EMBEDDED_FILES' for t in node.targets):
                        cls.sources = ast.literal_eval(node.value)

    def test_notebook_cells_are_valid_python_and_unexecuted(self):
        self.assertEqual(self.notebook['nbformat'], 4)
        self.assertEqual(len({c['id'] for c in self.notebook['cells']}), len(self.notebook['cells']))
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(cell['source'], cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_embedded_sources_match_reviewed_local_files(self):
        self.assertIn('benchmark.py', self.sources)
        for name, source in self.sources.items():
            self.assertEqual(source, (HERE / name).read_text())
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), self.notebook['metadata']['shelv_sources'][name])

    def test_real_query_preserved_and_pairs_share_split(self):
        queries = json.loads(self.sources['benchmark_queries.json'])
        user = [q for q in queries if q.get('origin') == 'user_provided']
        self.assertEqual(user[0]['query'], 'I need a book read peacefully')
        self.assertEqual(len(queries), 21)
        groups = {}
        for q in queries:
            if q['id'].startswith('user-'):
                continue
            group = q['id'].replace('-en-', '-').replace('-ta-', '-')
            groups.setdefault(group, set()).add(q['split'])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_portable_embedded_code_runs_without_repository(self):
        # No downloaded model or fabricated real-catalog relevance is involved.
        with tempfile.TemporaryDirectory() as directory:
            scripts = Path(directory) / 'shelv-vibe-search'
            scripts.mkdir(parents=True)
            for name, source in self.sources.items():
                (scripts / name).write_text(source)
            result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                                     '-s', str(scripts), '-p', 'test_*.py'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
