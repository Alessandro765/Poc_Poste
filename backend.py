# --- backend.py (Versione Definitiva, Corretta con Invoke) ---

import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from typing import Annotated, TypedDict
from rank_bm25 import BM25Okapi

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import json

# --- 1. CONFIGURAZIONE INIZIALE ---
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- 2. CARICAMENTO E PREPARAZIONE DOCUMENTI ---
# BARA: Caricamento e suddivisione documenti
documenti = [
    {"path": "KB/Base_dati_KB-DataGovernance&Strategy_2025110.csv", "flag": "server_data", "type": "csv"},
    {"path": "KB/Data_Governance&Strategy-Domande_KB_v2.docx", "flag": "governance_strategy", "type": "docx"}
]

document_chunks = {}
for doc in documenti:
    chunks = []
    print(f"📄 Caricamento del documento: {doc['path']} con flag: {doc['flag']}")
    try:
        if doc["type"] == "docx":
            loader = Docx2txtLoader(doc["path"])
            text = loader.load()[0].page_content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=128)
            chunks = text_splitter.split_text(text)
        elif doc["type"] == "csv":
            df = pd.read_csv(doc["path"])
            for _, row in df.iterrows():
                chunks.append(
                    f"L'hostname '{row['HOSTNAME']}' si trova nello stato '{row['STATO']}'. "
                    f"Il responsabile è '{row['RESPONSABILE']}', il sistema operativo è '{row['SISTEMA OPERATIVO']}' "
                    f"e la sua data di End of Service (EOS) è il {row['DATA EOS']}."
                )
        if chunks:
            document_chunks[doc["flag"]] = chunks
            print(f"✅ Documento '{doc['flag']}' caricato e suddiviso in {len(chunks)} chunks.")
    except Exception as e:
        print(f"❌ Errore durante il caricamento di {doc['path']}: {e}")

retrievers_faiss = {}
bm25_models = {}
for flag, chunks in document_chunks.items():
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25_models[flag] = BM25Okapi(tokenized_chunks)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retrievers_faiss[flag] = vectorstore.as_retriever()

# --- 3. DEFINIZIONE DELLO STATO DEL GRAFO ---
class State(TypedDict):
    current_question: str
    is_relevant: bool
    selected_flag: str
    retrieved_text: str
    final_response: str
    change_document: bool
    checked_documents: list
    fallback: bool

# --- 4. NODI DEL GRAFO ---
memory = MemorySaver()
graph_builder = StateGraph(State)
llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key, model_name="gpt-4o-mini")

def initialize_state(state: State):
    return {
        "current_question": state.get("current_question", ""), "is_relevant": False,
        "change_document": False, "selected_flag": None, "retrieved_text": "",
        "final_response": "", "checked_documents": [], "fallback": False
    }

def relevance_check(state: State):
    user_message = state["current_question"]
    relevance_prompt = [
        SystemMessage(content=(
            "Sei un classificatore. La domanda riguarda infrastruttura IT, server, hostname, o strategia di Data Governance? "
            "Rispondi solo con 'pertinente' o 'non_pertinente'.\n"
            f"Domanda: '{user_message}'"
        ))
    ]
    relevance_decision = llm.invoke(relevance_prompt).content.strip().lower()
    print(f"🔎 Verifica di pertinenza: {relevance_decision}")
    return {"is_relevant": "pertinente" in relevance_decision}

def classify_question(state: State):
    user_message = state["current_question"]
    checked_documents = state.get("checked_documents", [])
    available_docs = [doc for doc in document_chunks.keys() if doc not in checked_documents]
    if not available_docs:
        return {"fallback": True}
    
    # BARA: Prompt di classificazione. Istruzioni del classificatore sceglie il doc da utilizzare.
    classification_prompt = [
        SystemMessage(content=(
            "Sei un assistente esperto in Data Governance e infrastruttura IT.\n"
            "Il tuo compito è classificare la domanda dell'utente per scegliere il documento corretto da cui estrarre la risposta.\n\n"
            f"**Domanda:** '{user_message}'\n\n"
            "I documenti disponibili sono:\n"
            "- 'governance_strategy': Usalo per domande generali sulla strategia, sugli obiettivi della KB, su come trovare informazioni (es. 'Dove risiede il dato delle VM?', 'Quali sono le macchine che andranno in obsolescenza nei prossimi 3 mesi?' 'Sono responsabile del sistema X e voglio portare nel mio sistema l’informazione di AP/PK. Come posso farlo?').\n"
            "- 'server_data': Usalo per domande che richiedono dati specifici su un server o un hostname (es. 'qual è lo stato di ALT-CARVM?', 'elenca le macchine con responsabile CAOps').\n\n"
            f"Seleziona solo il flag migliore tra i seguenti: {available_docs}. Non aggiungere commenti o testo."
        ))
    ]
    classification_raw = llm.invoke(classification_prompt).content.strip().lower()
    classification = re.sub(r"[\[\]\'\"]", "", classification_raw).strip()
    print(f"📌 Documento selezionato: '{classification}'")
    if classification not in available_docs:
        classification = available_docs[0]
        print(f"⚠️  Selezione non valida, fallback a: '{classification}'")
    return {"selected_flag": classification}

def retrieve_documents(state: State):
    user_message = state["current_question"]
    selected_flag = state["selected_flag"]
    bm25_model = bm25_models[selected_flag]
    tokenized_query = user_message.split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:4]
    bm25_docs = [Document(page_content=document_chunks[selected_flag][i]) for i in bm25_indices]
    faiss_docs = retrievers_faiss[selected_flag].invoke(user_message, k=4)
    unique_docs = list({doc.page_content: doc for doc in bm25_docs + faiss_docs}.values())
    return {"retrieved_text": "\n\n".join([doc.page_content for doc in unique_docs[:8]])}

# Funzione helper per estrarre il numero di mesi. La mettiamo fuori per pulizia.
def extract_months_from_query(query: str) -> int:
    """
    Estrae il numero di mesi da una stringa. Cerca prima le cifre, poi le parole.
    Restituisce un valore di default (3) se non trova nulla.
    """
    # Dizionario per convertire le parole in numeri
    word_to_number = {
        'un': 1, 'uno': 1, 'una': 1, 'un mese': 1,
        'due': 2,
        'tre': 3,
        'quattro': 4,
        'cinque': 5,
        'sei': 6,
        'sette': 7,
        'otto': 8,
        'nove': 9,
        'dieci': 10,
        'undici': 11,
        'dodici': 12
    }

    # 1. Cerca prima un numero scritto in cifre (es. "3")
    digit_match = re.search(r'\d+', query)
    if digit_match:
        return int(digit_match.group(0))

    # 2. Se non trova cifre, cerca un numero scritto in parole
    # Costruiamo un pattern regex per cercare una qualsiasi delle parole nel nostro dizionario
    word_pattern = r'\b(' + '|'.join(word_to_number.keys()) + r')\b'
    word_match = re.search(word_pattern, query, re.IGNORECASE) # re.IGNORECASE non fa differenza tra maiuscole/minuscole
    if word_match:
        word = word_match.group(0).lower()
        return word_to_number[word]

    # 3. Se non trova nulla, restituisce un valore di default
    return 3

def generate_response(state: State):
    user_message = state["current_question"].lower()
    retrieved_text = state.get("retrieved_text", "")
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4o-mini")

    # 1. Capire l'intento
    intent_prompt = [
        SystemMessage(content=(
            "Analizza la domanda dell'utente. Il suo scopo è ottenere un calcolo (come un conteggio, una somma, un filtro su dati) oppure ottenere una spiegazione o informazione testuale?\n"
            "Rispondi SOLO con una di queste due parole: 'calcolo' o 'conoscenza'.\n\n"
            "Quante macchine sono... -> calcolo; Quali macchine sono... -> calcolo;"
            "Dove risiede il dato delle VM?” -> conoscenza; “Sono responsabile del sistema X e voglio portare nel mio sistema l’informazione di AP/PK. Come posso farlo?” -> conoscenza.\n\n"
            f"Domanda: '{user_message}'"
        ))
    ]
    intent = llm.invoke(intent_prompt).content.strip().lower()
    print(f"🧠 Intento rilevato: '{intent}'")

    # 2. Esecuzione basata sull'intento
    if "calcolo" in intent:
        print("🛠️ Esecuzione ramo di calcolo con Pandas...")
        try:
            df = pd.read_csv("KB/Base_dati_KB-DataGovernance&Strategy_2025110.csv")
            df['DATA EOS'] = pd.to_datetime(df['DATA EOS'], errors='coerce')

            # # CASO SEMPLICE: L'utente vuole solo il conteggio totale
            # if "totale" in user_message or "quante macchine sono in tutto" in user_message or "Quali macchine sono presenti" in user_message:
            #     total_count = len(df)
            #     final_response = f"Nella Knowledge Base sono configurate in totale **{total_count} macchine**."
            #     return {"final_response": final_response}
            
            # # CASO AVANZATO (DEFAULT): L'utente vuole filtrare e/o contare
            # else:

            # 1. Estrai i filtri (hostname, stato, responsabile)
            filter_extraction_prompt = [
                SystemMessage(content=(
                    "Sei un estrattore di parametri per query. Analizza la domanda e popola il JSON. "
                    "lo stato può essere solo operativo o configurato."
                    "Se un valore non è menzionato, lascialo come null. Rispondi SOLO con il JSON.\n\n"
                    f"Domanda: '{user_message}'\n\n"
                    "Esempio 1: 'le macchine operative di CAOps' -> {\"responsabile\": \"CAOps\", \"stato\": \"OPERATIVO\", \"hostname\": null}\n"
                    "Esempio 2: 'trovami l'hostname 2CEACC10-F3E0-40A4-91AF-01DD45AB857F' -> {\"responsabile\": null, \"stato\": null, \"hostname\": \"2CEACC10-F3E0-40A4-91AF-01DD45AB857F\"}"
                ))
            ]
            try:
                response_json_str = llm.invoke(filter_extraction_prompt).content.strip()
                clean_json_str = re.sub(r'```json\s*|\s*```', '', response_json_str)
                filters = json.loads(clean_json_str)
                print(f"🔍 Filtri estratti dalla domanda: {filters}")
            except (json.JSONDecodeError, AttributeError):
                filters = {"responsabile": None, "stato": None, "hostname": None}
                print("⚠️ Non è stato possibile estrarre filtri specifici.")

            # 2. Applica i filtri estratti
            filtered_df = df.copy()
            if filters.get("responsabile"):
                filtered_df = filtered_df[filtered_df['RESPONSABILE'].str.contains(filters["responsabile"], case=False, na=False)]
            if filters.get("stato"):
                filtered_df = filtered_df[filtered_df['STATO'].str.contains(filters["stato"], case=False, na=False)]
            if filters.get("hostname"):
                filtered_df = filtered_df[filtered_df['HOSTNAME'].str.contains(filters["hostname"], case=False, na=False)]

            # 3. Applica l'eventuale filtro di obsolescenza (se richiesto)
            if "obsolescenza" in user_message or "scadenza" in user_message:
                months_to_check = extract_months_from_query(user_message)
                today = datetime.now()
                future_date = today + relativedelta(months=months_to_check)
                filtered_df = filtered_df[
                    (filtered_df['DATA EOS'] >= today) & 
                    (filtered_df['DATA EOS'] <= future_date)
                ]

                # 4. Genera la risposta finale
                count = len(filtered_df)
                if count > 0:
                    response_list = [f"Ho trovato **{count} macchine** che corrispondono ai tuoi criteri:"]
                    for _, row in filtered_df.head(15).iterrows():
                        eos_date = pd.to_datetime(row['DATA EOS']).strftime('%d/%m/%Y') if pd.notna(row['DATA EOS']) else 'N/A'
                        response_list.append(f"- **{row['HOSTNAME']}** (Stato: {row['STATO']}, Resp: {row['RESPONSABILE']}, Scadenza: {eos_date})")
                    if count > 15:
                        response_list.append(f"\n... e altre {count - 15} macchine.")
                    final_response = "\n".join(response_list)
                else:
                    final_response = "Mi dispiace, non ho trovato nessuna macchina che corrisponda a tutti i criteri specificati."
                return {"final_response": final_response}

        except FileNotFoundError:
            return {"final_response": "Mi dispiace, non trovo il file dati per eseguire il calcolo."}
        except Exception as e:
            print(f"❌ Errore nel ramo di calcolo: {e}")
            import traceback
            traceback.print_exc()
            return {"final_response": "Si è verificato un errore durante il calcolo."}

    else:
        print("... Esecuzione ramo di conoscenza (RAG standard)...")
        
        generation_prompt = [
            SystemMessage(content=(
                f"""Sei un assistente esperto in Data Governance e infrastruttura IT.
                **Domanda dell'utente:** {user_message}.
                **Informazioni recuperate dai documenti:** {retrieved_text}.
                Rispondi in modo **conciso** e **preciso** alla domanda dell'utente basandoti **esclusivamente sui documenti forniti**.
                Se le informazioni contengono una lista (ad esempio, una lista di server), formattala in modo chiaro e leggibile.
                Se non trovi una risposta adeguata nei documenti, rispondi semplicemente con 'change_document'.
                Informazioni:\n{retrieved_text}\n\nDomanda: {state['current_question']}"""
            ))
        ]
        response = llm.invoke(generation_prompt).content.strip()
        print(f"💬 Risposta generata (RAG): {response[:500]}...")
        return {"final_response": response}

def decide_after_response(state: State):
    """Decide se la risposta è valida o se bisogna cambiare documento."""
    if "change_document" in state["final_response"].lower():
        print("⚠️  Richiesta di cambio documento.")
        return {"change_document": True, "checked_documents": state["checked_documents"] + [state["selected_flag"]]}
    
    print("✅ Risposta pronta per l'output.")
    return {"change_document": False}

def irrelevant_question_response(state: State):
    print("📢 Risposta per domanda non pertinente.")
    return {"final_response": "Non posso rispondere, domanda non pertinente alla KB. Posso fornire informazioni relative alle entità analizzate nel perimetro infrastrutturale di Poste Italiane."} #BARA

def fallback_response(state: State):
    print("🚫 Fallback: nessuna informazione trovata.")
    return {"final_response": "Mi dispiace, non ho trovato un'informazione pertinente nei documenti a mia disposizione."}

# --- 5. COSTRUZIONE DEL GRAFO ---
graph_builder.add_node("initialize", initialize_state)
graph_builder.add_node("relevance_check", relevance_check)
graph_builder.add_node("irrelevant_response", irrelevant_question_response)
graph_builder.add_node("classification", classify_question)
graph_builder.add_node("retrieval", retrieve_documents)
graph_builder.add_node("response", generate_response)
graph_builder.add_node("decide_after_response", decide_after_response)
graph_builder.add_node("fallback_response", fallback_response)

graph_builder.add_edge(START, "initialize")
graph_builder.add_edge("initialize", "relevance_check")

# Questo blocco è corretto e garantisce che il grafo prenda la giusta via
# dopo il controllo di pertinenza.
graph_builder.add_conditional_edges(
    "relevance_check",
    lambda state: "classification" if state["is_relevant"] else "irrelevant_response",
    {
        "classification": "classification",
        "irrelevant_response": "irrelevant_response"
    }
)

graph_builder.add_conditional_edges(
    "classification",
    lambda state: "fallback_response" if state.get("fallback") else "retrieval",
    {
        "fallback_response": "fallback_response",
        "retrieval": "retrieval"
    }
)

graph_builder.add_edge("retrieval", "response")
graph_builder.add_edge("response", "decide_after_response")


graph_builder.add_conditional_edges(
    "decide_after_response",
    lambda state: "classification" if state["change_document"] else "__end__",
    {
        "classification": "classification",
        "__end__": END
    }
)

graph_builder.add_edge("irrelevant_response", END)
graph_builder.add_edge("fallback_response", END)

graph = graph_builder.compile(checkpointer=memory)
print("\n🟢 Grafo compilato. Chatbot attivo.\n")

# --- 6. FUNZIONE DI ESECUZIONE ---
def run_rag_chain(question: str, thread_id: str):
    """
    Esegue il grafo RAG utilizzando invoke() per ottenere direttamente il risultato finale.
    Questo metodo è più robusto e diretto per questa applicazione.
    """
    try:
        # Usiamo invoke() che attende la fine del grafo e restituisce lo stato finale completo.
        final_state = graph.invoke(
            {"current_question": question},
            config={"recursion_limit": 50, "configurable": {"thread_id": thread_id}}
        )
        
        # Estraiamo la risposta finale dallo stato restituito.
        final_response = final_state.get("final_response") if final_state else None
        
        return final_response if final_response else "Mi dispiace, si è verificato un problema e non ho potuto elaborare una risposta."

    except Exception as e:
        print(f"\n🚨 ERRORE CRITICO DURANTE L'ESECUZIONE DEL GRAFO: {e}\n")
        import traceback
        traceback.print_exc()
        return "Si è verificato un errore interno. Si prega di riprovare."