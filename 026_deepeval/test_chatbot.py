import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def test_schuh_zurueckgabe():

    # Für eine Liste möglicher Parameter bei G-Eval siehe:
    # https://deepeval.com/docs/metrics-llm-evals#required-arguments
    my_metric = GEval(

        # Jede Metrik muss einen Namen haben
        name="MeineMetrik",

        # Wie soll bewertet werden?
        # Beachte Folgendes:
        # Die Formulierung des Kriteriums auf Englisch wird eine englische Begründung des Scores
        # zurückliefern.
        # Die Formulierung des Kriteriums auf Deutsch wird eine deutsche Begründung des Scores
        # zurückliefern.
        #criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        criteria="Bestimme ob der tatsächliche Output (ACTUAL_OUTPUT) korrekt ist basierend auf " \
                 "der Information im erwarteten Output (EXPECTED_OUTPUT)",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],

        # Die Vorgehensweise zur Bewertung wird normalerweise automatisch anhand von criteria
        # bei CoT (Chain-of-Thought) vom Bewertungs-LLM selber erstellt (siehe Fig. 1 im Paper)
        # Aber: man kann diese auch überschreiben
        #evaluation_steps = ["Überprüfe, ob alle relevanten Informationen in der Ausgabe enthalten sind",
        #                    "Überprüfe, ob unnötige Informationen in der Ausgaben enthalten sind"]

        # Ab wann gilt der Test bestanden? 
        # Default ist 0.5
        #threshold=0.5,

        # Binäres Testergebnis statt Score verwenden: 1=bestanden, 0=durchgefallen        
        #strict_mode=True,
        
        # Hier können verschiedene Evaluator-Modelle ausgewählt werden
        # Wenn nichts angegeben? --> Default ist: "gpt-4.1"
        #model="gpt-4o",
        #model="gpt-4.1",        
        #model="gpt-5.2",
    )
    
    my_test_case = LLMTestCase(
        input="Was, wenn die Schuhe mir nicht passen?",
        # Hier die tatsächliche Ausgabe Ihres Modells eintragen
        #actual_output="Sie haben 30 Tage, um die Schuhe zurück zu geben.",
        actual_output="Sie haben 30 Tage, um die Schuhe zurück zu geben und bekommen Ihr Geld komplett zurück!",
        #actual_output="Sie haben 30 Tage, um die Schuhe zurück zu geben und bekommen Ihr Geld komplett zurück!" \
        #              "Übrigens kann ich Ihnen Spaghetti Bolognese heute als Abendessen empfehlen. Sehr lecker!",
        #actual_output="Woher soll ich das wissen? Hä? Bin ich das Orakel von Delphi?",
        
        expected_output="Sie können die Schuhe innerhalb von 30 Tagen zurückgeben und erhalten Ihr Geld zurück.",        
    )
    
    assert_test(my_test_case,
                [my_metric])