# --- app.py (Versione aggiornata con spiegazione statica) ---

import streamlit as st
import uuid
# Importiamo la nostra funzione "ponte" dal backend
from backend import run_rag_chain

poste_logo = "ICON/Logo_semplice_Poste_Italiane.png"

# --- 1. Configurazione della Pagina ---
st.set_page_config(
    page_title="KB Data Governance & Strategy",
    page_icon=poste_logo,
    layout="wide"  # Usa l'intera larghezza della pagina
)

# --- 2. Titolo e Sottotitolo con Logo ---

# Creiamo due colonne: una piccola per il logo e una grande per il testo
col1, col2 = st.columns([0.03, 0.97])  # Il primo numero è la larghezza della prima colonna

# Nella prima colonna, mostriamo l'immagine del logo
with col1:
    st.image(poste_logo, width=80) # Puoi regolare la larghezza (width) come preferisci

# Nella seconda colonna, mostriamo il titolo e il sottotitolo
with col2:
    st.title("Chatbot per la Knowledge Base di Data Governance")
    st.caption("Una Proof of Concept (POC) basata su un'architettura RAG (Retrieval-Augmented Generation)")

# --- 3. Area di Testo Statica con la Spiegazione ---
st.markdown("""
**Benvenuto in questa Proof of Concept!**

Questo chatbot è un sistema RAG progettato per rispondere a domande basandosi su una Knowledge Base specifica, composta da due documenti:
*   Un file **CSV** contenente dati tabulari sui server.
*   Un file **Word** (.docx) che descrive la strategia di Data Governance e le domande a cui la KB può rispondere.

L'obiettivo è dimostrare come sia possibile interrogare in linguaggio naturale fonti di dati eterogenee per ottenere risposte rapide, precise e contestualizzate, riducendo i tempi di ricerca manuale.
""")
st.divider()  # Aggiunge una linea di separazione visiva

# --- 4. Logica della Chat "Botta e Risposta" ---

# Inizializza la cronologia della chat e l'ID della conversazione se non esistono
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ciao! Fai una domanda sulla KB di Data Governance."}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Mostra i messaggi della chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# La casella di input per l'utente apparirà in fondo alla pagina
if prompt := st.chat_input("Come posso aiutarti?"):
    
    # 1. Aggiungi il messaggio dell'utente alla cronologia
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Sto cercando nei documenti..."):
        response = run_rag_chain(st.session_state.messages[-1]["content"], st.session_state.thread_id)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()