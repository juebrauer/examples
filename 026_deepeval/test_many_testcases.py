import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

my_metric1 = AnswerRelevancyMetric(threshold=0.7)

test_cases = [
    LLMTestCase(
        input="What is the capital of Germany?",
        actual_output="Berlin",
    ),
    LLMTestCase(
        input="What is 2+2?",
        actual_output="Spaghetti Bolognese ist doch super lecker!",
    ),
]

@pytest.mark.parametrize("test_case", test_cases)
def test_answer_relevancy(test_case):
    assert_test(test_case, [my_metric1])