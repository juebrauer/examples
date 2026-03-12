from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def main():
    my_metric = GEval(
        name="MeineMetrik",
        criteria="Bestimme ob der tatsächliche Output (ACTUAL_OUTPUT) korrekt ist basierend auf der Information im erwarteten Output (EXPECTED_OUTPUT)",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
    )

    my_test_case = LLMTestCase(
        input="Was, wenn die Schuhe mir nicht passen?",
        #actual_output="Sie haben 30 Tage, um die Schuhe zurück zu geben und bekommen Ihr Geld komplett zurück!",
        actual_output="Was weiß ich...",
        expected_output="Sie können die Schuhe innerhalb von 30 Tagen zurückgeben und erhalten Ihr Geld zurück.",
    )

    assert_test(my_test_case, [my_metric])


if __name__ == "__main__":
    main()