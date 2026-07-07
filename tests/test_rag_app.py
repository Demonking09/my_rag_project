import unittest

from rag_app import fallback_answer


class FallbackAnswerTests(unittest.TestCase):
    def test_fallback_answer_prefers_the_soft_computing_syllabus_chunk(self):
        documents = [
            type("Doc", (), {"page_content": "The university has a general policy for academic programs."})(),
            type(
                "Doc",
                (),
                {
                    "page_content": (
                        "Soft Computing syllabus includes fuzzy logic, neural networks, "
                        "genetic algorithms, and hybrid intelligent systems."
                    )
                },
            )(),
        ]

        answer = fallback_answer("What topics are in the soft computing syllabus?", documents)

        self.assertIn("soft computing", answer.lower())
        self.assertIn("fuzzy", answer.lower())
        self.assertIn("neural", answer.lower())


if __name__ == "__main__":
    unittest.main()
