from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BAZA_ZNANJA_FOLDER = "baza_znanja"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
baza = Chroma(persist_directory=BAZA_ZNANJA_FOLDER, embedding_function=embeddings)

rezultat = baza.get(include=["documents", "metadatas", "embeddings"])

broj_zapisa = len(rezultat["ids"])
print(f"Baza sadrzi ukupno {broj_zapisa} zapisa (komada teksta).\n")
print("=" * 70)

for i in range(broj_zapisa):
    tekst = rezultat["documents"][i]
    metapodaci = rezultat["metadatas"][i]
    vektor = rezultat["embeddings"][i]

    print(f"\nZAPIS #{i + 1}")
    print(f"Izvor (fajl): {metapodaci.get('source', 'nepoznato')}")
    print(f"Duzina vektora: {len(vektor)} brojeva")
    print(f"Prvih 5 brojeva vektora: {list(vektor[:5])}")
    print(f"\nTekst (prvih 200 karaktera):")
    print(tekst[:200] + ("..." if len(tekst) > 200 else ""))
    print("-" * 70)