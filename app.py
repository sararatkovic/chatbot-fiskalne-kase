import streamlit as st
from chatbot_graph import napravi_graf
from zajednicko.priprema_podataka import ucitaj_i_podeli_dokumente, napravi_ili_ucitaj_bazu

DOKUMENTI_FOLDER = "dokumenti"
BAZA_ZNANJA_FOLDER = "baza_znanja"

@st.cache_resource
def pripremi_chatbot():
    delovi = ucitaj_i_podeli_dokumente()
    baza = napravi_ili_ucitaj_bazu(delovi)
    return napravi_graf(baza)


# --------------------------------------------------------------------
# STREAMLIT INTERFEJS
# --------------------------------------------------------------------
st.set_page_config(page_title="Preporuka Galeb fiskalnih kasa", page_icon="🧾")

st.title("🧾 Preporuka Galeb fiskalnih kasa")
st.write(
    "Popuni formu ispod (ili prosto postavi pitanje u chatu na dnu stranice), "
    "a chatbot će ti preporučiti najprikladniji model Galeb fiskalne kase."
)

with st.spinner("Pripremam bazu znanja..."):
    aplikacija = pripremi_chatbot()

if "istorija" not in st.session_state:
    st.session_state.istorija = []

# --------------------------------------------------------------------
# Forma sa parametrima
# --------------------------------------------------------------------
with st.form("parametri_forma"):
    st.subheader("Brza preporuka na osnovu parametara")

    delatnost = st.selectbox(
        "Tip delatnosti",
        ["Pekara/mesara", "Restoran/kafić", "Prodavnica/market",
         "Apoteka/drogerija", "Frizerski/kozmetički salon",
         "Terenska prodaja/dostava", "Trafika/kiosk", "Ostalo"],
    )

    promet = st.radio(
        "Kakav promet imaš?",
        ["Mali ili umeren promet", "Veliki promet (gužva, mnogo računa ili artikala)"],
    )

    mobilnost = st.radio(
        "Da li ti je potrebna mobilna kasa (naplata van pulta)?",
        ["Ne, radim sa fiksnog pulta", "Da, potrebna mi je mobilnost", "Nije mi bitno"],
    )

    placanje_karticom = st.radio(
        "Da li ti je potrebno plaćanje karticom na kasi?",
        ["Da", "Ne", "Nije mi bitno"],
    )

    posalji = st.form_submit_button("Preporuči mi kasu")

if posalji:
    generisano_pitanje = (
        f"Koju Galeb fiskalnu kasu da izaberem? Delatnost: {delatnost}. "
        f"Promet: {promet}. Mobilnost: {mobilnost}. "
        f"Plaćanje karticom: {placanje_karticom}."
    )

    st.session_state.istorija.append({"uloga": "user", "tekst": generisano_pitanje})
    with st.chat_message("user"):
        st.write(generisano_pitanje)

    with st.chat_message("assistant"):
        with st.spinner("Razmišljam..."):
            rezultat = aplikacija.invoke({"pitanje": generisano_pitanje})
            odgovor = rezultat["odgovor"]
            st.write(odgovor)

    st.session_state.istorija.append({"uloga": "assistant", "tekst": odgovor})
# --------------------------------------------------------------------
# Prikaz istorije i slobodan unos pitanja
# --------------------------------------------------------------------
for poruka in st.session_state.istorija:
    with st.chat_message(poruka["uloga"]):
        st.write(poruka["tekst"])

pitanje = st.chat_input("Unesi svoje pitanje...")

if pitanje:
    st.session_state.istorija.append({"uloga": "user", "tekst": pitanje})
    with st.chat_message("user"):
        st.write(pitanje)

    with st.chat_message("assistant"):
        with st.spinner("Razmišljam..."):
            rezultat = aplikacija.invoke({"pitanje": pitanje})
            odgovor = rezultat["odgovor"]
            st.write(odgovor)

    st.session_state.istorija.append({"uloga": "assistant", "tekst": odgovor})