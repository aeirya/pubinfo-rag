import unittest

from pubinfo.experiments.qa import config_metadata, gold_ids_from_test
from pubinfo.pipelines.qarag.config import QAConfig
from pubinfo.retrieval.config import RetrievalConfig, resolve_columns


class ExperimentConfigTest(unittest.TestCase):
    def test_resolves_column_presets(self):
        columns = resolve_columns("no_abstract")

        self.assertIn("title", columns)
        self.assertIn("authors", columns)
        self.assertNotIn("abstract", columns)

    def test_config_metadata_keeps_explicit_retrieval_columns(self):
        config = QAConfig(
            k=3,
            prompt="qa2",
            columns="default",
            retrieval=RetrievalConfig(kind="tfidf", k=99, columns=["title"]),
            model="gemma2:2b",
        )

        metadata = config_metadata(config)

        self.assertEqual(metadata["prompt"], "qa2")
        self.assertEqual(metadata["k"], 3)
        self.assertEqual(metadata["retriever"], "tfidf")
        self.assertEqual(metadata["retrieval_columns"], ["title"])

    def test_gold_ids_from_test_accepts_common_columns(self):
        self.assertEqual(gold_ids_from_test({"RelevantRow": "12"}), [12])
        self.assertEqual(gold_ids_from_test({"RelevantRows": "12, 13"}), [12, 13])
        self.assertEqual(gold_ids_from_test({"gold_ids": [1, 2]}), [1, 2])


if __name__ == "__main__":
    unittest.main()
