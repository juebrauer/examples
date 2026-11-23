#include <stdio.h>

enum Raum
{
    RAUM_S012b,
    RAUM_S011,
    RAUM_S004,
    ANZ_RAEUME
};


struct Datum
{
    int tag;
    int monat;
    int jahr;
};

struct Buchung
{
    enum Raum raum;
    char name[50];
    struct Datum wann;
    int startstunde;
    int dauer_in_stunden;
    int anzahl_personen;
};

void zeige_buchung(struct Buchung* buchung)
{
    printf("Buchung von %s\n", (*buchung).name);
    switch ((*buchung).raum)
    {
        case RAUM_S012b:
            printf("Großer Hörsaal S0.12b, Hörsaal Soloplan\n");
            break;

        case RAUM_S011:
            printf("Größer Hörsaal S0.11, gestuft\n");
            break;

        case RAUM_S004:
            printf("Kleiner Hörsaal S0.04, Seminarraum\n");
            break;
    };
    printf("Datum: %02d.%02d.%04d\n",
            (*buchung).wann.tag,
            (*buchung).wann.monat,
            (*buchung).wann.jahr);
}


void aendere_buchung(struct Buchung* b, enum Raum neuer_raum)
{
    (*b).raum = neuer_raum;
}


struct Buchung hole_buchung_slot_aus_db()
{
   struct Buchung b = {
        .raum = RAUM_S004,
        .name = "to-be-defined",
        .wann.tag = 24,
        .wann.monat = 12,
        .wann.jahr = 2026,
        .startstunde = 8,
        .dauer_in_stunden = 4,
        .anzahl_personen = 40
    };

    return b;
}


int main()
{
    struct Buchung b = {
        .raum = RAUM_S012b,
        .name = "Juergen Brauer",
        .wann.tag = 23,
        .wann.monat = 11,
        .wann.jahr = 2026,
        .startstunde = 8,
        .dauer_in_stunden = 4,
        .anzahl_personen = 40
    };

    zeige_buchung(&b);

    enum Raum neuer_raum = RAUM_S011;
    aendere_buchung(&b, neuer_raum);

    printf("------------------\n");
    zeige_buchung(&b);

    struct Buchung b2 = hole_buchung_slot_aus_db();

    printf("RAUM_S004==2? %d\n", RAUM_S004 == 2);

    for (int r=RAUM_S012b; r<=RAUM_S004; r++)
    {
        printf("Nächster Raum hat internen Code: %d\n", r);
    }
    
}