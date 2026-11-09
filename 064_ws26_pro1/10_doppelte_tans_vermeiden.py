import random

genzahlen = []
for y in range(5):
    for x in range(3):

        gueltig = 0
        while gueltig == 0:
            zufallszahl = random.randint(1,10)
            if zufallszahl not in genzahlen:
                gueltig = 1

        genzahlen.append( zufallszahl )
        
        print(zufallszahl, end=" ")
        #print(genzahlen, flush=True)
    print()

