import unittest

import pandas as pd

from pubinfo.experiments.retrieval import question_text, run_retrieval_grid
from pubinfo.retrieval.config import RetrievalConfig


class RetrievalExperimentTest(unittest.TestCase):
    def test_formats_multiple_choice_questions(self):
        query = question_text({
            "Question": "Which title is about climate?",
            "A": "Climate coaching",
            "B": "Trauma study",
            "C": "Marketing",
            "D": "Yoga",
        })

        self.assertIn("Which title is about climate?", query)
        self.assertIn("A. Climate coaching", query)

    def test_run_retrieval_grid_returns_metrics(self):
        db = pd.DataFrame([
            {
                "title": "Climate coaching",
                "keywords": "climate career",
                "abstract": "green jobs",
                "authors": "A",
                "date_published": "2024",
                "context_name": "x",
            },
            {
                "title": "Trauma study",
                "keywords": "health refugees",
                "abstract": "somatic complaints",
                "authors": "B",
                "date_published": "2023",
                "context_name": "y",
            },
        ])
        tests = pd.DataFrame([{
            "Question": "Which article is about trauma?",
            "A": "Climate coaching",
            "B": "Trauma study",
            "C": "Marketing",
            "D": "Yoga",
            "RelevantRow": "1",
        }])

        report = run_retrieval_grid(
            db,
            tests,
            [RetrievalConfig(kind="tfidf", k=2, columns=["title", "keywords"])],
        )

        self.assertEqual(len(report), 1)
        self.assertIn("hit@1", report)
        self.assertEqual(report.loc[0, "retrieval_retriever"], "tfidf")


if __name__ == "__main__":
    unittest.main()
