import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


DOKUMENTI_FOLDER = "dokumenti"
BAZA_ZNANJA_FOLDER = "baza_znanja"

def ucitaj_i_podeli_dokumente():
    loader = DirectoryLoader(
        DOKUMENTI_FOLDER, glob="*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    dokumenti = loader.load()

    naziv_modela_po_izvoru = {}
    for dok in dokumenti:
        prva_linija = dok.page_content.strip().split("\n")[0]

        if prva_linija.startswith("TIP DOKUMENTA"):
            dok.metadata["tip"] = "kvar"
        else:
            dok.metadata["tip"] = "preporuka"

        naziv_modela_po_izvoru[dok.metadata["source"]] = prva_linija

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    delovi = splitter.split_documents(dokumenti)

    for deo in delovi:
        if not deo.page_content.strip().startswith(("MODEL:", "TIP DOKUMENTA:")):
            naziv = naziv_modela_po_izvoru.get(deo.metadata["source"], "")
            deo.page_content = f"{naziv}\n{deo.page_content}"

    return delovi

def napravi_ili_ucitaj_bazu(delovi):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(BAZA_ZNANJA_FOLDER) and os.listdir(BAZA_ZNANJA_FOLDER):
        return Chroma(persist_directory=BAZA_ZNANJA_FOLDER, embedding_function=embeddings)
    return Chroma.from_documents(documents=delovi, embedding=embeddings, persist_directory=BAZA_ZNANJA_FOLDER)


def formatiraj_dokumente(dokumenti):
    return "\n\n---\n\n".join(dok.page_content for dok in dokumenti)
