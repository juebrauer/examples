# 057_rl_agent

Eine kleine Bildwelt mit genau einem Agenten: `agent_rl` lernt ausschliesslich
durch Versuch, Irrtum und die beobachteten Belohnungen. Es gibt keine
Beispieldaten, keinen Experten und keinen kuerzesten Pfad als Trainingsziel.

## Lernidee

Der Zustand ist immer ein RGB-Bild mit dem blauen Agenten und dem roten Ziel.
Das Aktionsnetz waehlt daraus `hoch`, `runter`, `links` oder `rechts`. Ein
zweites, kleines Netz schaetzt vor der Aktion die normalerweise zu erwartende
Belohnung dieses Bildes.

Nach jedem Schritt wird gerechnet:

```text
Ueberraschung = beobachtete Belohnung - erwartete Belohnung
```

- Positive Ueberraschung: Die gewaehlten Aktionen in den letzten N
  Bildzustaenden werden wahrscheinlicher.
- Negative Ueberraschung: Dieselben Entscheidungen werden unwahrscheinlicher.
- Das Erwartungsnetz lernt anschliessend aus der neuen Beobachtung.
- Zehn Prozent Exploration verhindern, dass der Agent zu frueh nur noch eine
  einzige Strategie ausprobiert.

Die Historie wird zu Beginn jeder Episode geleert. So erhaelt keine Aktion aus
der vorherigen Welt versehentlich Lob oder Tadel aus der neuen Welt.

## Belohnungen

| Ereignis | Belohnung |
|---|---:|
| Ziel erreicht | `+1.00` |
| Einen Schritt naeher | `+0.10` |
| Einen Schritt weiter weg | `-0.10` |
| Gegen den Rand gelaufen | `-0.15` |
| Nach 100 Schritten nicht am Ziel | `-0.50` |

## Start

Benötigt werden Python, NumPy, PyTorch und PySide6.

```bash
python rl_agent_demo.py
```

Mit **Lernen starten** sammelt und verarbeitet der Agent eigene Erfahrungen.
**20 Episoden testen** verwendet jeweils die wahrscheinlichste Aktion und
veraendert das Modell nicht.

Die drei Kerneigenschaften des Lernalgorithmus lassen sich separat pruefen:

```bash
python -m unittest test_agent_rl.py
```
