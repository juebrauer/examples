# EarthQuakeWeights — Gradientenlabor

PySide6-Experiment zum Vergleich von PyTorch-AutoDiff mit zufälligen
Gewichtsstörungen. Rechenkern und Oberfläche sind getrennt; alle Berechnungen
laufen auf der CPU. Es werden keine Datensätze heruntergeladen.

## Start

Aus diesem Verzeichnis mit der vorhandenen Umgebung:

```bash
conda run --no-capture-output -n env_teaching python main.py
```

Alternativ in einer eigenen Python-Umgebung (Python 3.10 oder neuer):

```bash
python -m pip install -r requirements.txt
python main.py
```

Getestet mit Python 3.12, PyTorch 2.9.1, PySide6 6.9.1,
NumPy 2.5.1 und Matplotlib 3.11.1. Die Minimumversionen in
`requirements.txt` wurden nicht separat getestet.

## Bedienung

1. Aufgabe, Trainings-/Testanzahl, Seed und Architektur auswählen.
   **Daten & Netz anwenden** initialisiert das Experiment. Gleicher Seed
   erzeugt identische Daten und Startgewichte; für andere Punkte den Seed ändern.
2. Gradientenverfahren, Optimizer, Lernrate, Mini-Batch-Größe und Anzahl
   weiterer Epochen einstellen. Für die Störungsverfahren zusätzlich ε und K.
3. **Lernen** trainiert genau diese Anzahl weiterer Epochen. Wiederholte Klicks
   setzen Gewichte und Optimizer fort. Geänderte Daten-/Architektureinstellungen
   führen beim nächsten Lernen automatisch zu einer Neuinitialisierung.
4. **Stopp** beendet das Training kooperativ. Vollendete Updates bleiben erhalten;
   eine angebrochene Epoche zählt nicht als volle Epoche. Beim Fortsetzen wird
   eine neue Batch-Reihenfolge erzeugt, also die angebrochene Epoche nicht exakt
   fortgesetzt. Schließen des Fensters beendet zuerst den Worker sicher.
5. Umschalten zwischen **Netzvorhersage**, **Wahre Funktion** und **Absoluter Fehler**.
   Punkte zeigen immer die Zielwerte: Kreise = Training, Dreiecke = Test.
   Die schwarze Linie ist bei Klassifikation die Grenze P(Klasse 1) = 0,5.
   Ein Klick zeigt Werte am nächsten Punkt des 120 × 120-Auswertungsrasters.
6. **Gradienten mit AutoDiff vergleichen** berechnet Kosinusähnlichkeit,
   relativen Fehler und Normen am selben ersten Trainingsbatch und an denselben
   Gewichten. Beide Störungsverfahren verwenden dieselben Zufallsrichtungen.
   Der Vergleich verändert weder Gewichte noch Trainings-Zufallszustände oder
   Optimizer. Er betrachtet rohe Gradienten vor Clipping; Idealwerte sind
   Kosinus 1 und relativer Fehler 0. Bei Nullgradienten ist die entsprechende
   relative Kennzahl nicht definiert (NaN).
7. Für unabhängige Läufe: **Aktuelle Testkurve merken**, **Netz zurücksetzen**,
   Verfahren wechseln, erneut lernen. Bis zu vier Testkurven werden überlagert.
   Dabei gleiche Aufgabe, Daten, Architektur, Seed, Optimizer und Lernrate
   verwenden. Das Umschalten des Gradientenverfahrens allein setzt die Gewichte
   nicht zurück, wohl aber den Optimizer-Zustand. Eine Lernratenänderung behält
   dessen Zustand. Änderungen an ε/K behalten ihn ebenfalls.
8. **Messwerte als JSON exportieren** speichert Einstellungen und Verlauf
   einschließlich gemerkter Experimente. Pro Verlaufseintrag werden die dort
   verwendeten Trainingsparameter erfasst. Es ist kein Modell-Checkpoint.

Links gibt es bei kleinen Fenstern eine Scrollleiste. Die Plot-Werkzeugleiste
ermöglicht unter anderem den Export einer Abbildung.

## Aufgaben und Netzbeschreibung

**Circle-Klassifikation:** ungefähr gleich viele Punkte pro Klasse, gleichmäßig
nach Fläche in der inneren Kreisscheibe (Radius < 0,55, Klasse 0) und dem äußeren
Ring (0,55 ≤ Radius ≤ 1, Klasse 1). Die Ecken der dargestellten Fläche liegen
außerhalb des Trainingsgebiets; Vorhersagen dort sind Extrapolation.
Output ist ein Logit, Loss ist Binary Cross Entropy with Logits. Die
Visualisierung zeigt Sigmoid(Logit), also P(Klasse 1).

**Berge-und-Täler-Regression:** unabhängige, gleichverteilte Punkte auf [-1,1]²
mit rauschfreien Funktionswerten:

```text
f(x,y) = 0.45 sin(3πx) cos(2πy)
       + 0.25 sin(5π(x+y))
       + 0.8 exp(-18 ((x-0.35)² + (y+0.25)²))
       - 0.7 exp(-22 ((x+0.4)² + (y-0.35)²))
```

Output ist ein unbeschränkter skalarer Wert; Loss = MSE, zusätzliche Kennzahl
= RMSE. Train und Test werden unabhängig gezogen. Testpunkte werden niemals
für Gewichtsupdates verwendet. Die Regressionsvorhersage verwendet eine feste
Farbskala aus der wahren Funktion; Vorhersagen außerhalb dieser Skala werden
farblich gesättigt, sind aber per Klick als Zahlen ablesbar.

Die Textbeschreibung enthält nur die versteckten Schichten, beispielsweise:

```text
linear 20
relu
linear 10
relu
```

`linear 20` bedeutet **20 Neuronen**. Eingangsbreite 2 und eine abschließende
lineare Schicht mit einem Output werden automatisch ergänzt. Erlaubte
Aktivierungen: `relu`, `tanh`, `sigmoid`, `gelu`, `silu`. `#` beginnt einen
Kommentar. Ein leeres Textfeld ergibt ein lineares Modell. Maximal 40 Zeilen
und 512 Neuronen pro Schicht. Stochastische Schichten wie Dropout sind bewusst
nicht enthalten, damit ±ε dieselbe deterministische Funktion auswertet.

## Wie die Verfahren funktionieren

Sei θ der Vektor aller Gewichte **und Biases**, z(θ) die Netzausgabe für den
aktuellen Mini-Batch und L dessen gemittelter Loss. Für jede der K Richtungen
wird ein unabhängiger Rademacher-Vektor u mit Einträgen −1 oder +1 gezogen.
Das entspricht einer zufälligen Offsetmatrix je Gewichtsmatrix und einem
Offsetvektor je Bias. ε ist die Störungsgröße **pro Parameter**, nicht die
Gesamtnorm; die Norm der Störung beträgt ε√p bei p Parametern.

**AutoDiff:** PyTorch berechnet ∇θ L mit `loss.backward()`.

**EarthQuakeWeights:** Die Ausgabedifferenzen approximieren zuerst die
Richtungsableitung der Netzausgabe:

```text
d_u z ≈ [z(θ + εu) − z(θ − εu)] / (2ε)
s_u   = Summe über Batchausgaben [ (∂L/∂z) · d_u z ]
ĝ      = (1/K) Summe über Richtungen [ s_u · u ]
```

Die Loss-Ableitung wird analytisch ausgewertet:
`(sigmoid(z) − target)/B` bei BCE und `2(z − target)/B` bei MSE.
Sie wird am ungestörten Output z(θ) berechnet. Der gesamte Schätzer läuft
unter `torch.no_grad()`: **kein Backpropagation, kein AutoDiff durch das Netz**.
Es wird auch keine große Output-Parameter-Jacobi-Matrix angelegt.
Kosten: **2K + 1 Vorwärtsläufe pro Mini-Batch**.

**EarthQuakeLoss (SPSA):** Die zweite Variante differenziert direkt den Loss:

```text
s_u = [L(θ + εu) − L(θ − εu)] / (2ε)
ĝ    = (1/K) Summe über Richtungen [ s_u · u ]
```

Kosten: **2K Vorwärtsläufe pro Mini-Batch**, ebenfalls ohne Backpropagation.
Das ist der SPSA-Gradientenschätzer mit Mittelung mehrerer Messungen. Hier
werden konstantes ε und ein frei wählbarer Adam-/SGD-Optimizer verwendet,
nicht der klassische SPSA-Lernraten- und Störungsabnahmeplan. Klassische
Konvergenzaussagen lassen sich deshalb nicht pauschal auf dieses Experiment
übertragen.

Bei beiden Varianten erfolgt **erst nach allen Störungen** ein Gewichtsupdate.
Die Ausgangsparameter werden zwischen Messungen exakt wiederhergestellt,
auch bei Fehlern und Abbruch. Alle Messungen eines Updates verwenden denselben
Mini-Batch. Die geschätzten Gradienten werden in `parameter.grad` geschrieben;
anschließend verwendet der gewählte PyTorch-Optimizer sie wie normale
Gradienten. Optionales Norm-Clipping gilt für alle drei Verfahren.

### Weshalb es gehen kann — und wo die Grenze liegt

Eine einzelne zufällige Störung liefert nur eine Richtungsableitung, nicht
alle einzelnen ∂z/∂θᵢ. Wegen E[u uᵀ] = I gilt jedoch in der lokalen linearen
Näherung E[(∇L · u)u] = ∇L. Mittelung kann deshalb eine brauchbare Schätzung
liefern. Symmetrische Differenzen haben für genügend glatte Funktionen
Trunkationsfehler der Ordnung ε². An ReLU-Knickstellen gilt diese glatte
Näherung nicht; das Experiment bleibt trotzdem ausführbar.

Die Varianz ist der entscheidende Nachteil. In der idealisierten linearen
Näherung gilt bei unabhängigen Rademacher-Richtungen:

```text
E[ ||ĝ − g||² ] = (p − 1) / K · ||g||²
```

Bei mehr Parametern braucht man also typischerweise mehr Richtungen für eine
ähnlich genaue Schätzung. Das kleine Standardnetz hat bereits 281 Parameter.
Ein kleines K kann trotzdem zum Lernen reichen, liefert aber oft einen
schlechten einzelnen Gradienten. Großes ε verzerrt die lokale Ableitung,
sehr kleines ε führt in Float32 zu Auslöschung. ε=0,01 ist ein Ausgangspunkt,
keine universelle Einstellung. Ein K-Vergleich ist eine einzelne Stichprobe;
mehr Richtungen garantieren nicht bei jedem Versuch einen besseren Messwert.

Primärquelle zur Methode: [James C. Spall, SPSA Overview, Johns Hopkins APL](https://www.jhuapl.edu/SPSA/Pages/).
PyTorch: [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).

## Reproduzierbare Startversuche

Circle: Standardwerte verwenden (Seed 42, 512/256 Punkte, 20–10 Tanh,
Batch 64, Adam, Lernrate 0,01, 50 Epochen, ε=0,01, K=32).
Ein lokaler Lauf ergab folgende Testwerte, jeweils nach Neustart des Netzes:

| Verfahren | Test-BCE vorher → nachher | Test-Accuracy | Trainings-Vorwärtsläufe |
|---|---:|---:|---:|
| AutoDiff | 0,69145 → 0,05535 | 97,66 % | 400 |
| EarthQuakeWeights | 0,69145 → 0,12755 | 98,44 % | 26.000 |
| EarthQuakeLoss | 0,69145 → 0,12757 | 98,44 % | 25.600 |

Das ist ein einzelner Seed, kein Nachweis einer Überlegenheit. Accuracy und
Loss messen unterschiedliche Dinge. Die Störungsverfahren brauchten auf
diesem Rechner etwa 2 Sekunden, AutoDiff etwa 0,25 Sekunden für die Updates.
Die Anzeige misst die Trainingsschleife einschließlich Gradientenschätzung,
ohne anschließende Train-/Testevaluation, Rastervorhersage oder Plotten.
Der Forward-Zähler umfasst nur abgeschlossene Trainingsbatches; bei Abbruch
verworfene Messungen werden nicht gezählt. AutoDiff benötigt zusätzlich
einen Rückwärtslauf pro Batch. Eine gleiche Epochenzahl ist somit kein
Vergleich bei gleichem Rechenbudget.

Für die anspruchsvollere Regression ist folgender AutoDiff-Ausgangspunkt
sinnvoll: **2048/1024 Punkte, Seed 42, Batch 128, Adam, Lernrate 0,003,
500 Epochen**, Architektur:

```text
linear 64
tanh
linear 64
tanh
```

In einem lokalen Lauf sank Test-MSE dabei von 0,12613 auf 0,01393.
Das kleine Standardnetz und nur 50 Epochen bilden die vielen Schwingungen
noch kaum ab. Beim größeren Netz wird die Störgradientenschätzung mit
unverändertem K deutlich ungenauer; genau diesen Skalierungseffekt kann man
mit der Diagnose untersuchen. Die optimalen Hyperparameter der Methoden
müssen nicht identisch sein. Für belastbare Aussagen mehrere Seeds und
auch Zeit-/Forward-Budgets vergleichen; häufiges Abstimmen am Testset macht
es faktisch zum Validierungsset.

## Codeaufteilung und Tests

| Datei | Verantwortung |
|---|---|
| `data.py` | Aufgaben, Zielwerte, Daten und Seed |
| `network.py` | MLP-Parser, Loss, analytische Output-Loss-Ableitung |
| `gradients.py` | AutoDiff, beide Störgradienten, Vergleich |
| `experiment.py` | Mini-Batches, Optimizer, Evaluation, Verlauf |
| `ui.py` | PySide6-Oberfläche, Matplotlib und Worker-Adapter |
| `main.py` | Programmstart |
| `test_experiment.py` | Numerik, Lernverhalten, Reproduzierbarkeit, Qt-Lebenszyklus |

Der Kern importiert weder Qt noch Matplotlib. Beispiel ohne GUI:

```python
from experiment import Experiment, TrainConfig
from gradients import GradientConfig

experiment = Experiment()
config = TrainConfig(GradientConfig("EarthQuakeWeights", epsilon=0.01, directions=32))
for _ in range(50):
    experiment.train_epoch(config)
print(experiment.history[-1])
```

Für viele kleine CPU-Batches empfiehlt sich vorab `torch.set_num_threads(1)`;
`main.py` setzt dies automatisch. Daten-, Batch- und Störungszufall sind getrennt,
damit Methodenunterschiede nicht die Batch-Reihenfolge verändern.

Tests (zusätzlich `pytest` installieren, falls noch nicht vorhanden):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen \
MPLCONFIGDIR=/tmp/earthquake-mpl \
conda run --no-capture-output -n env_teaching python -m pytest -q test_experiment.py
```

Die Tests prüfen AutoDiff gegen unabhängige koordinatenweise finite
Differenzen und die Störverfahren gegen den exakten Mittelwert aller
Rademacher-Richtungen eines kleinen Modells. Weitere Prüfungen decken
Gewichtswiederherstellung bei Fehlern/Abbruch, Training ohne Backpropagation,
Mini-Batch-Restgrößen, getrennte Testdaten, Fortsetzen, Reproduzierbarkeit
und beide Aufgaben mit allen Verfahren in einer echten Offscreen-Qt-Oberfläche ab.
