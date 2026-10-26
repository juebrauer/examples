import random

zahlen = []
while len(zahlen) != 5:
    zufallszahl = random.randint(1,6)
    if zufallszahl not in zahlen:
        zahlen.append( zufallszahl )

print(zahlen)
