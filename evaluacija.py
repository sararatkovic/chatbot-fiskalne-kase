from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from modeli.ocena_sudije import OcenaSudije
from zajednicko.priprema_podataka import ucitaj_i_podeli_dokumente, napravi_ili_ucitaj_bazu, formatiraj_dokumente
from zajednicko.chain import napravi_chain_i_retriever

TEST_PITANJA = [
    "Koja kasa je najbolja za restoran?",  
    "Treba mi mobilna kasa za dostavu hrane, mali biznis, nizak budžet.",
    "Štampač na kasi ne štampa račune, šta da radim?",
    #"Kasa se ne uključuje, isprobala sam sve, koga da zovem?",
    "Koja je prestonica Francuske?"
]

SUDIJA_PROMPT = ChatPromptTemplate.from_template(
    """Ti si IZUZETNO STROG ocenjivac kvaliteta odgovora chatbota za preporuku i
podršku oko Galeb fiskalnih kasa. Većina odgovora ima manu ili prostor za
poboljšanje - retko koji odgovor zaslužuje najvišu ocenu. Izbegavaj podrazumevano
davanje visokih ocena bez konkretnog razloga.

Oceni odgovor po SVAKOM od sledeća 4 kriterijuma, ocenom od 1 do 5, koristeći
OVU TAČNU SKALU:
  1 = potpuno pogrešno/neupotrebljivo
  2 = ima ozbiljne nedostatke
  3 = osrednje - delimično ispunjava kriterijum, ima primetnih propusta
  4 = dobro, uz sitan, nebitan nedostatak
  5 = besprekorno, bez ijednog primetnog nedostatka (retko se dodeljuje)

Kriterijumi:
1. RELEVANTNOST - da li odgovor stvarno i direktno adresira postavljeno pitanje?
   Ako je pitanje van teme, da li ga je chatbot ispravno prepoznao kao takvo?
2. UTEMELJENOST - da li je SVAKA tvrdnja u odgovoru izricito potkrepljena
   kontekstom? Ako se doda i najmanja informacija koje NEMA u kontekstu,
   ovo NE SME biti ocenjeno vise od 2.
3. KOMPLETNOST - da li odgovor pokriva sve bitne aspekte pitanja, bez
   izostavljanja necega sto je bilo dostupno u kontekstu?
4. JEZIK - da li je odgovor gramaticki ispravan, jasan, prirodan srpski?

Pitanje korisnika: {pitanje}

Kontekst dat chatbotu: {kontekst}

Odgovor chatbota: {odgovor}

VAZNO - redosled odgovora: prvo NAPISI kratku kriticku analizu (2-3 recenice,
navedi KONKRETAN nedostatak ako postoji), pa TEK ONDA daj ocene."""
)

sudija_llm = ChatOllama(model="mistral", temperature=0.0)
sudija_chain = SUDIJA_PROMPT | sudija_llm.with_structured_output(OcenaSudije, method="json_schema") 

# pokretanje

def main():
    print("Pripremam bazu znanja i chain...")
    delovi = ucitaj_i_podeli_dokumente()
    baza = napravi_ili_ucitaj_bazu(delovi)
    chain, retriever = napravi_chain_i_retriever(baza) #za odgovor na pitanje

    rezultati = []

    for i, pitanje in enumerate(TEST_PITANJA, start=1):
        print(f"\n[{i}/{len(TEST_PITANJA)}] Testiram: {pitanje}")

        pronadjeni_delovi = retriever.invoke(pitanje)
        kontekst = formatiraj_dokumente(pronadjeni_delovi)
        
        odgovor = chain.invoke({
            "pitanje": pitanje, "kontekst": kontekst
        })
        
        print("   -> odgovor generisan, saljem sudiji na ocenu...")
        try:
            ocena_sudije = sudija_chain.invoke({
                "pitanje": pitanje, "kontekst": kontekst, "odgovor": odgovor,
            })
        except Exception as e:
            print(f"   -> GRESKA! {type(e).__name__} PORUKA: {e}")
        else:
            ocene = ocena_sudije.model_dump() # u dict
            rezultati.append({"pitanje": pitanje, "odgovor": odgovor, **ocene})

    # Ispis finalnog izvestaja

    print("\n" + "=" * 90)
    print("IZVESTAJ EVALUACIJE")
    print("=" * 90)

    kriterijumi = ["relevantnost", "utemeljenost", "kompletnost", "jezik"]
    sume = {k: [] for k in kriterijumi}

    for r in rezultati:
        print(f"\nPitanje: {r['pitanje']}")
        print(f"\nOdgovor: {r['odgovor']}")
        print(f"  Analiza sudije: {r['analiza']}")
        for k in kriterijumi:
            ocena = r[k]
            print(f"  {k}: {ocena}/5")
            sume[k].append(ocena)
        print(f"  Obrazlozenje sudije: {r['obrazlozenje']}")

    print("\n" + "-" * 90)
    print("PROSECNE OCENE PO KRITERIJUMU:")
    for k in kriterijumi:
        if sume[k]:
            prosek = sum(sume[k]) / len(sume[k])
            print(f"  {k}: {prosek:.2f}/5")
        else:
            print(f"  {k}: nema validnih ocena")


if __name__ == "__main__":
    main()