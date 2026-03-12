# test_time_machine.py
import pytest

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric


# 1) Deine Goldens (Ground Truth)
GOLDENS = [
    Golden(
        input="Wie weit kann man mit der Zeitmaschine zurück reisen?",
        expected_output="Man kann mit der Zeitmaschine bis ins Jahr 1142 zurück reisen.",
        retrieval_context=[
            "Zeitreisen in die Vergangenheit sind bis maximal ins Jahr 1142 möglich."
        ],
    ),
    Golden(
        input="Was passiert, wenn der Tank leer ist?",
        expected_output="Wenn der Tank leer ist, muss die Zeitmaschine mit Uran nachgefüllt werden.",
        retrieval_context=[
            "Wenn die Tankanzeige leer ist, füllen Sie bitte wieder Uran ein!"
        ],
    ),
]


# 2) Dummy-App: erzeugt actual_output (hier ersetzt du später deine echte Pipeline)
def my_time_machine_app(question: str) -> str:
    if "zurück" in question:
        return "Man kann mit der Zeitmaschine bis ins Jahr 1142 zurück reisen."
    if "Tank" in question:
        return "Wenn der Tank leer ist, muss man wieder Uran nachfüllen."
    return "Das weiß ich nicht."


# 3) Goldens -> LLMTestCases
TEST_CASES = [
    LLMTestCase(
        input=g.input,
        actual_output=my_time_machine_app(g.input),
        expected_output=g.expected_output,
        retrieval_context=g.retrieval_context,
    )
    for g in GOLDENS
]


# 4) Zwei Metriken
METRICS = [
    AnswerRelevancyMetric(threshold=0.6),
    ContextualRecallMetric(threshold=0.6),
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_time_machine_goldens(test_case: LLMTestCase):
    assert_test(test_case, METRICS)