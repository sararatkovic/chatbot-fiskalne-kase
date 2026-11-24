import random
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from zajednicko.priprema_podataka import formatiraj_dokumente
from modeli.parametri_poslovanja import ParametriPoslovanja
from modeli.klasifikacija import Klasifikacija

SABLONSKI_ODGOVOR_NERELEVANTNO = (
    "Ja sam agent specijalizovan za preporuku Galeb fiskalnih kasa. "
    "Mogu ti pomoći da izabereš odgovarajući model na osnovu tvoje delatnosti, "
    "prometa, potrebe za mobilnošću i kartičnim plaćanjem - postavi mi pitanje "
    "vezano za to, ili popuni formu iznad. 🙂"
)

# Stanje
class Stanje(TypedDict, total = False):
    pitanje: str
    kontekst: str
    odgovor: str
    tip: str
    parametri: dict

# Promptovi
KLASIFIKATOR_PROMPT = """Odredi kojoj klasi pripada pitanje korisnika:
- "preporuka": Korisnik trazi preporuku kase ili se raspituje o modelima kase i uslovima kupovine.
- "kvar": Korisnik ima problem sa kasom i trazi da mu se ponudi resenje ili dobije kontakt ovlascenog servisera.
- "van_teme": Korisnik postavlja pitanje koje nema nikakve veze sa kasama ili problemima u radu kasa.

Primeri:
"Koju kasu mi preporucujes za rad u apoteci?" -> preporuka
"Koliko kosta model ge-450?" -> preporuka
"Da li dobijam popust ako kupim vise od 10 uredjaja?" -> preporuka
"Zasto mi kasa ne stampa racune?" -> kvar
"Zasto kasa nece da se poveze na internet?" -> kvar
"Koji je broj ovlascenog servisa?" -> kvar
"Koje su boje na zastavi Srbije?" -> van_teme
"Sta danas da spremim za rucak?" -> van_teme
"Daj mi preporuku za lep kafic u Beogradu?" -> van_teme

Pitanje: {pitanje}"""

llm_nula_temp = ChatOllama(model="llama3", temperature=0.0)
prompt_klasifikator = ChatPromptTemplate.from_template(KLASIFIKATOR_PROMPT)
chain_klasifikator = (
    prompt_klasifikator | llm_nula_temp.with_structured_output(Klasifikacija, method="json_schema") 
)

PARAMETRI_PROMPT = """Iz pitanja korisnika izvuci podatke o njegovom poslovanju.

Pravila:
- Popuni polje samo ako je korisnik to rekao rečima u pitanju.
- Ako korisnik nešto nije pomenuo, polje je null. Ne zaključuj ništa iz delatnosti:
  to što neko ima kafić ne znači da ima veliki promet ili da mu treba plaćanje karticom.
- "nije bitno" upiši samo ako je korisnik baš rekao da mu je nešto svejedno ili nebitno.

Polja i dozvoljene vrednosti:
- delatnost: čime se korisnik bavi, npr. "apoteka", "kafić"
- promet: "mali" ili "veliki" (koliko kupaca i računa ima)
- mobilnost: "da", "ne" ili "nije bitno" (da li mu treba prenosiva kasa)
- placanje_karticom: "da", "ne" ili "nije bitno"

Primeri:
Pitanje: Treba mi kasa za pekaru.
{{"delatnost": "pekara", "promet": null, "mobilnost": null, "placanje_karticom": null}}

Pitanje: Imam butik, nemam veliku gužvu, svejedno mi je da li je kasa prenosiva.
{{"delatnost": "butik", "promet": "mali", "mobilnost": "nije bitno", "placanje_karticom": null}}

Pitanje: Koja kasa podržava kartice i može da se nosi do stola?
{{"delatnost": null, "promet": null, "mobilnost": "da", "placanje_karticom": "da"}}

Pitanje: {pitanje}"""

prompt_parametri = ChatPromptTemplate.from_template(PARAMETRI_PROMPT)
chain_parametri = (
    prompt_parametri | llm_nula_temp.with_structured_output(ParametriPoslovanja, method="json_schema")
)

KVAR_PROMPT = """Ti si tehnička podrška za Galeb fiskalne kase i odgovaraš na srpskom jeziku.

Dokumentacija za rešavanje problema:
{kontekst}

Na osnovu dokumentacije napiši numerisane korake koje korisnik treba da proba, istim redosledom kao u dokumentaciji.
Ako neki korak sme da uradi samo ovlašćeni serviser (npr. fiskalizacija ili otvaranje plombe), jasno to naglasi.
Na kraju navedi kontakt tehničke podrške iz dokumentacije.
Ako dokumentacija ne pokriva ovaj problem, reci to i uputi korisnika na tehničku podršku.
Odgovori direktno korisniku, bez naslova, i ne pominji ova pravila.

Problem korisnika: {pitanje}"""

llm_visa_temp = ChatOllama(model="llama3", temperature=0.3)
prompt_kvar = ChatPromptTemplate.from_template(KVAR_PROMPT)
chain_kvar = prompt_kvar | llm_visa_temp | StrOutputParser()

PREPORUKA_PROMPT = """Ti si savetnik za Galeb fiskalne kase i odgovaraš na srpskom jeziku.

Podaci o poslovanju korisnika:
{parametri}

Opisi modela kasa:
{kontekst}

Preporuči model iz opisa iznad koji najbolje odgovara korisniku:
- Ako jedan model jasno najbolje odgovara, preporuči samo njega i u 2-3 rečenice objasni zašto, pozivajući se na podatke o poslovanju.
- Ako dva modela podjednako dobro odgovaraju, preporuči oba: za svaki navedi zašto odgovara, pa objasni glavnu razliku između njih i kada je bolje izabrati koji.
- Nikada ne preporučuj više od dva modela.
Uvek navedi tačan naziv modela (npr. GALEB N950).
Koristi samo informacije iz opisa modela, ne izmišljaj cene ni karakteristike.
Ne postavljaj pitanja korisniku.
Odgovori direktno korisniku, bez naslova, i ne pominji ova pravila.

Pitanje korisnika: {pitanje}"""

prompt_preporuka = ChatPromptTemplate.from_template(PREPORUKA_PROMPT)
chain_preporuka = prompt_preporuka | llm_visa_temp | StrOutputParser()

# Cvorovi
def klasifikuj(stanje: Stanje) -> dict:
    pitanje = stanje["pitanje"]
    klasa = chain_klasifikator.invoke({"pitanje": pitanje})

    return {"tip": klasa.tip}

def nerelevantno(stanje: Stanje) -> dict:
    return {"odgovor": SABLONSKI_ODGOVOR_NERELEVANTNO}

def izvuci_parametre(stanje: Stanje) -> dict:
    pitanje = stanje["pitanje"]
    parametri = chain_parametri.invoke({"pitanje": pitanje}).model_dump()

    return {"parametri": parametri}

# Za dodatno pitanje
FRAZE = {
    "delatnost": ["čime se baviš", "za koju delatnost ti treba kasa"],
    "promet": ["koliko kupaca obično imaš", "kolika je gužva kod tebe"],
    "mobilnost": ["da li ti treba prenosiva kasa", "da li naplaćuješ i van pulta"],
    "placanje_karticom": ["da li ti treba plaćanje karticom", "da li kupci plaćaju karticama"],
}

OKVIRI = [
    "Ako mi kažeš i {}, mogu da ti dam još precizniju preporuku.",
    "Za još bolju preporuku, reci mi {}.",
    "Da bih suzila izbor, zanima me {}.",
]

def spoji_srpski(delovi: list[str]) -> str:
    if len(delovi) == 1:
        return delovi[0]
    return ", ".join(delovi[:-1]) + " i " + delovi[-1]

def postavi_dodatno_pitanje(stanje: Stanje) -> dict:
    nedostaju = [kljuc for kljuc, vrednost in stanje["parametri"].items() if vrednost is None]

    if not nedostaju or all(vrednost is None for vrednost in stanje["parametri"].values()): 
        return {}
    
    # za svaki parametar koji fali nasumično izaberi jednu formulaciju
    fraze = [random.choice(FRAZE[kljuc]) for kljuc in nedostaju]

    # spoji fraze i ubaci ih u nasumično izabran okvir rečenice
    recenica = random.choice(OKVIRI).format(spoji_srpski(fraze))

    return {"odgovor": stanje["odgovor"] + "\n\n" + recenica}

# Funkcija za grananje
def izaberi_granu(stanje: Stanje) -> str:
    return stanje["tip"]

# Pomocne f-je
NAZIVI_PARAMETARA = {
    "delatnost": "delatnost",
    "promet": "promet",
    "mobilnost": "potrebna mobilna kasa",
    "placanje_karticom": "plaćanje karticom",
}

def formatiraj_parametre(parametri: dict) -> str:
    """Pretvara dict parametara u čitljive redove za prompt."""
    redovi = [
        f"{NAZIVI_PARAMETARA[kljuc]}: {vrednost if vrednost is not None else 'nije navedeno'}"
        for kljuc, vrednost in parametri.items()
    ]
    return "\n".join(redovi)

# Sastavljanje grafa
def napravi_graf(baza):
    retriever_preporuka = baza.as_retriever(search_kwargs={"k": 4, "filter": {"tip": "preporuka"}})
    retriever_kvar = baza.as_retriever(search_kwargs={"k": 4, "filter": {"tip": "kvar"}})

    def preporuci(stanje: Stanje) -> dict:
        pronadjeni_delovi = retriever_preporuka.invoke(stanje["pitanje"])
        kontekst = formatiraj_dokumente(pronadjeni_delovi)
        odgovor = chain_preporuka.invoke({
            "parametri": formatiraj_parametre(stanje["parametri"]),
            "pitanje": stanje["pitanje"], 
            "kontekst": kontekst
        })
        return {"odgovor": odgovor}

    def resi_kvar(stanje: Stanje) -> dict:
        pronadjeni_delovi = retriever_kvar.invoke(stanje["pitanje"])
        kontekst = formatiraj_dokumente(pronadjeni_delovi)
        odgovor = chain_kvar.invoke({"pitanje": stanje["pitanje"], "kontekst": kontekst})
        return {"odgovor": odgovor, "kontekst": kontekst} 

    graf = StateGraph(Stanje)

    graf.add_node("klasifikuj", klasifikuj)
    graf.add_node("izvuci_parametre", izvuci_parametre)
    graf.add_node("preporuci", preporuci)
    graf.add_node("resi_kvar", resi_kvar) 
    graf.add_node("nerelevantno", nerelevantno)
    graf.add_node("postavi_dodatno_pitanje", postavi_dodatno_pitanje)

    graf.add_edge(START, "klasifikuj")

    graf.add_conditional_edges(
        "klasifikuj",
        izaberi_granu,
        {
            "kvar": "resi_kvar",
            "preporuka": "izvuci_parametre",
            "van_teme": "nerelevantno"
        }
    )

    graf.add_edge("izvuci_parametre", "preporuci")
    graf.add_edge("preporuci", "postavi_dodatno_pitanje")
    graf.add_edge("postavi_dodatno_pitanje", END)

    graf.add_edge("resi_kvar", END)
    graf.add_edge("nerelevantno", END)

    aplikacija = graf.compile()

    return aplikacija

#print(aplikacija.get_graph().draw_mermaid())