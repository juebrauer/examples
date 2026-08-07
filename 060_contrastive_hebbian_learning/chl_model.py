"""Numerischer Kern der Demo zu Contrastive Hebbian Learning (CHL).

Dieses Modul enthält bewusst weder Qt- noch Matplotlib-Code. Dadurch lässt sich
der Lernalgorithmus unabhängig von der Benutzeroberfläche lesen und testen.
"""

from __future__ import annotations

import numpy as np


X_MIN = -3.0
X_MAX = 3.0


def target_function(x: np.ndarray) -> np.ndarray:
    """Berechnet die feste, absichtlich unregelmäßige Zielfunktion f(x).

    Die drei Sinusterme erzeugen Schwingungen auf unterschiedlichen Skalen. Die
    beiden Gauß-Terme bilden lokale Beulen bzw. Vertiefungen; der lineare Term
    fügt einen leichten globalen Trend hinzu.
    """
    x = np.asarray(x, dtype=float)
    return (
        0.48 * np.sin(2.4 * x + 0.25)
        + 0.23 * np.sin(7.1 * x - 0.7)
        + 0.13 * np.sin(15.3 * x + 0.9)
        + 0.28 * np.exp(-5.0 * (x + 1.15) ** 2)
        - 0.34 * np.exp(-10.0 * (x - 0.75) ** 2)
        + 0.055 * x
    )


class CHLNetwork:
    """Deterministisches CHL-Netz mit symmetrischen Verbindungen.

    Architektur::

        skalares x -> 31 feste Eingabeeinheiten <-> 24 Hidden Units <-> y

    Die trainierbaren Gewichte werden in beiden Richtungen verwendet. Deshalb
    erhält die Hidden-Schicht sowohl Bottom-up-Signale von x als auch
    Top-down-Signale von y. Genau diese Rückkopplung unterscheidet das Modell
    von einem gewöhnlichen Feedforward-MLP.

    In beiden CHL-Phasen bleibt die Eingabe festgeklemmt. In der freien
    Minus-Phase darf sich die Ausgabe selbst einstellen. In der Plus-Phase wird
    sie auf den Zielwert geklemmt. Gelernt wird aus dem Unterschied der dabei
    entstehenden Aktivitätskorrelationen.
    """

    def __init__(
        self,
        hidden_size: int = 48,
        input_units: int = 31,
        seed: int = 7,
    ) -> None:
        rng = np.random.default_rng(seed)

        # Ein einzelnes x aktiviert mehrere benachbarte Gauß-Rezeptivfelder.
        # Diese feste Populationscodierung ist nicht trainierbar. Sie stellt der
        # kleinen Demo genügend lokale Merkmale für die vielen Schwingungen zur
        # Verfügung, ohne den eigentlichen CHL-Schritt zu verkomplizieren.
        self.input_centers = np.linspace(X_MIN, X_MAX, input_units)
        self.input_width = 0.24

        # W[0]: Eingabepopulation <-> Hidden-Schicht
        # W[1]: Hidden-Schicht      <-> Ausgabeeinheit
        # Es existieren keine separaten Feedbackgewichte: Für den Rückweg wird
        # jeweils exakt die transponierte Gewichtsmatrix benutzt.
        scale = 0.08
        self.weights = [
            rng.normal(0.0, scale, (input_units, hidden_size)),
            rng.normal(0.0, scale, (hidden_size, 1)),
        ]

        # Für Hidden- und Ausgabeschicht gibt es je einen Bias-Vektor.
        self.biases = [
            np.zeros(hidden_size),
            np.zeros(1),
        ]

    @staticmethod
    def _blend(old: np.ndarray, drive: np.ndarray, damping: float) -> np.ndarray:
        """Führt einen gedämpften Zustandsschritt mit tanh-Aktivierung aus.

        Ohne Dämpfung würde der neue Zustand sofort ``tanh(drive)`` annehmen.
        Die Mischung mit dem alten Zustand lässt das rekurrente Netz ruhiger
        gegen einen Gleichgewichtszustand relaxieren.
        """
        return (1.0 - damping) * old + damping * np.tanh(drive)

    def relax(
        self,
        x: np.ndarray,
        target: np.ndarray | None,
        steps: int,
        damping: float = 0.35,
        initial_states: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Lässt einen Batch von Netzwerkzuständen relaxieren.

        Args:
            x: Skalare Eingabewerte; Form ``(batch,)`` oder ``(batch, 1)``.
            target: ``None`` für die freie Minus-Phase, sonst die festgeklemmten
                Zielwerte der Plus-Phase.
            steps: Anzahl synchroner Relaxationsschritte.
            damping: Anteil des neuen Zustands pro Relaxationsschritt.
            initial_states: Optionaler Startzustand. Die Plus-Phase startet beim
                zuvor gefundenen Gleichgewicht der Minus-Phase.

        Returns:
            Die Gleichgewichtszustände ``[encoded_x, hidden, y]``. Die erste
            Dimension aller Zustände ist jeweils die Batchgröße.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)

        # Gaußsche Populationscodierung:
        # encoded_x[b, i] ist die Aktivität der Eingabeeinheit i für Beispiel b.
        encoded_x = np.exp(
            -0.5 * ((x - self.input_centers) / self.input_width) ** 2
        )

        # Standardmäßig beginnt die Relaxation mit inaktiven Hidden- und
        # Ausgabeeinheiten. Für die Plus-Phase können diese Zustände übernommen
        # werden, damit sie am Ergebnis der Minus-Phase ansetzt.
        batch_size = encoded_x.shape[0]
        hidden = np.zeros((batch_size, self.biases[0].size))
        y = np.zeros((batch_size, 1))
        if initial_states is not None:
            hidden = initial_states[1].copy()
            y = initial_states[2].copy()
        # clamped_y macht die Phasenunterscheidung explizit:
        # None  -> Minus-Phase, die Ausgabe bleibt frei.
        # Array -> Plus-Phase, die Ausgabe wird auf diesen Wert geklemmt.
        clamped_y: np.ndarray | None = (
            None
            if target is None
            else np.asarray(target, dtype=float).reshape(-1, 1)
        )
        if clamped_y is not None:
            y = clamped_y.copy()

        input_weights, output_weights = self.weights
        hidden_bias, output_bias = self.biases

        for _ in range(steps):
            # Synchrones Update: Beide neuen Zustände werden aus den alten
            # Zuständen berechnet und erst am Schleifenende gemeinsam gesetzt.
            # output_weights.T erzeugt die symmetrische Rückkopplung y -> hidden.
            new_hidden = self._blend(
                hidden,
                encoded_x @ input_weights + y @ output_weights.T + hidden_bias,
                damping,
            )
            if clamped_y is None:
                # Minus-Phase: y darf auf den Hidden-Zustand reagieren.
                new_y = self._blend(
                    y, hidden @ output_weights + output_bias, damping
                )
            else:
                # Plus-Phase: Der Lehrer hält y während der Relaxation fest.
                new_y = clamped_y

            hidden, y = new_hidden, new_y

        return [encoded_x, hidden, y]

    def train_sample(
        self,
        x: float,
        target: float,
        learning_rate: float,
        relax_steps: int,
    ) -> None:
        """Trainiert genau ein Beispiel mit einem kontrastiven CHL-Schritt."""

        # 1) Minus-Phase: Nur x ist geklemmt. Das Netz erzeugt seine aktuelle
        #    Vorhersage und relaxiert in einen freien Gleichgewichtszustand s-.
        minus = self.relax(np.array([x]), None, relax_steps)

        # 2) Plus-Phase: Wir starten bei s- und klemmen zusätzlich y auf den
        #    gewünschten Zielwert. Die übrigen Einheiten relaxieren zu s+.
        plus = self.relax(
            np.array([x]),
            np.array([target]),
            relax_steps,
            initial_states=minus,
        )

        # 3) CHL-Regel für jede Verbindung zwischen benachbarten Schichten:
        #
        #       Delta W = eta * (Korrelation_plus - Korrelation_minus)
        #
        # Das äußere Produkt pre.T @ post misst, welche Einheiten auf beiden
        # Seiten einer Verbindung gemeinsam aktiv waren. CHL verstärkt die im
        # gewünschten Zustand s+ typischen Korrelationen und reduziert die im
        # freien Zustand s- zu starken Korrelationen.
        for layer, weight in enumerate(self.weights):
            positive_correlation = plus[layer].T @ plus[layer + 1]
            negative_correlation = minus[layer].T @ minus[layer + 1]
            weight += learning_rate * (
                positive_correlation - negative_correlation
            )

        # Für Biaswerte ist die "präsynaptische Aktivität" konstant 1. Daher
        # reduziert sich ihre Lernregel auf die Differenz der Schichtaktivität.
        for layer, bias in enumerate(self.biases):
            bias += learning_rate * (
                plus[layer + 1] - minus[layer + 1]
            )[0]

    def predict(self, x: np.ndarray, relax_steps: int) -> np.ndarray:
        """Berechnet Vorhersagen ausschließlich über die freie Minus-Phase."""
        states = self.relax(x, None, relax_steps)
        return states[-1][:, 0]
