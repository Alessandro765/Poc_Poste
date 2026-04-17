# --- backend.py ---

import os
import re
import traceback
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import TypedDict
from dotenv import load_dotenv, find_dotenv
from rank_bm25 import BM25Okapi

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. CONFIGURAZIONE INIZIALE ---
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-5-nano-2025-08-07"
EMBEDDING_MODEL = "text-embedding-3-small"
CSV_PATH = "KB/Base_dati_KB-DataGovernance&Strategy_20260415-DMU.csv"

def today_str() -> str:
    """Restituisce la data odierna fresca a ogni chiamata (evita date congelate al boot)."""
    return datetime.now().strftime("%d/%m/%Y")


# --- 2. CARICAMENTO E PREPARAZIONE DOCUMENTI ---
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

documenti = [
    {"path": CSV_PATH, "flag": "server_data", "type": "csv"},
    {"path": "KB/Data_Governance&Strategy-Domande_KB_v4.docx", "flag": "governance_strategy", "type": "docx"}
]

document_chunks = {}
for doc in documenti:
    chunks = []
    print(f"Caricamento del documento: {doc['path']} con flag: {doc['flag']}")
    try:
        if doc["type"] == "docx":
            loader = Docx2txtLoader(doc["path"])
            text = loader.load()[0].page_content
            semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
            chunks = semantic_splitter.split_text(text)
        elif doc["type"] == "csv":
            df = pd.read_csv(doc["path"])
            for _, row in df.iterrows():
                chunks.append(
                    f"L'hostname '{row['HOSTNAME']}' si trova nello stato '{row['STATO']}'. "
                    f"Il responsabile e' '{row['RESPONSABILE']}', il sistema operativo e' '{row['SISTEMA OPERATIVO']}' "
                    f"e la sua data di End of Service (EOS) e' il {row['DATA EOS']}."
                )
        if chunks:
            document_chunks[doc["flag"]] = chunks
            print(f"Documento '{doc['flag']}' caricato e suddiviso in {len(chunks)} chunks.")
    except Exception as e:
        print(f"Errore durante il caricamento di {doc['path']}: {e}")
retrievers_faiss = {}
bm25_models = {}
for flag, chunks in document_chunks.items():
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25_models[flag] = BM25Okapi(tokenized_chunks)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retrievers_faiss[flag] = vectorstore.as_retriever()

# CSV pre-caricato una volta sola per evitare letture ripetute a ogni richiesta
try:
    _server_df = pd.read_csv(CSV_PATH)
    _server_df['DATA EOS'] = pd.to_datetime(_server_df['DATA EOS'], format='%d/%m/%Y %H:%M', errors='coerce')
    print(f"CSV server caricato in memoria: {len(_server_df)} righe.")
except FileNotFoundError:
    _server_df = pd.DataFrame()
    print(f"CSV server non trovato al boot: {CSV_PATH}")


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
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name=MODEL_NAME)


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
            "Sei un classificatore. La domanda riguarda infrastruttura IT, server, hostname, "
            "obsolescenza/EOS di macchine, sistemi operativi, responsabili IT, o strategia di Data Governance? "
            "Se l'utente chiede che giorno e' oggi, la richiesta e' legittima: rispondi con 'pertinente'.\n"
            "Rispondi solo con 'pertinente' o 'non_pertinente'.\n"
            f"Data odierna: {today_str()}\n"
            f"Domanda: '{user_message}'"
        ))
    ]
    relevance_decision = llm.invoke(relevance_prompt).content.strip().lower()
    print(f"Verifica di pertinenza: {relevance_decision}")
    return {"is_relevant": relevance_decision == "pertinente"}


def classify_question(state: State):
    user_message = state["current_question"]
    checked_documents = state.get("checked_documents", [])
    available_docs = [doc for doc in document_chunks.keys() if doc not in checked_documents]
    if not available_docs:
        return {"fallback": True}

    classification_prompt = [
        SystemMessage(content=(
            "Sei un assistente esperto in Data Governance e infrastruttura IT.\n"
            "Il tuo compito e' classificare la domanda dell'utente per scegliere il documento corretto.\n\n"
            f"**Domanda:** '{user_message}'\n"
            f"**Data odierna:** {today_str()}\n\n"
            "I documenti disponibili sono:\n"
            "- 'governance_strategy': Usalo per domande generali sulla strategia, sugli obiettivi della KB, su processi e procedure "
            "(es. 'Dove risiede il dato delle VM?', 'Come posso portare informazioni nel mio sistema?', 'Qual e' la politica di Data Governance?').\n"
            "- 'server_data': Usalo per QUALSIASI domanda che richiede dati o calcoli su macchine/server/hostname, "
            "incluse obsolescenza, EOS, stato, responsabile, sistema operativo "
            "(es. 'qual e' lo stato di ALT-CARVM?', 'elenca le macchine con responsabile CAOps', "
            "'quante macchine sono gia in obsolescenza?', 'quali macchine scadono entro 1 anno?').\n\n"
            f"Seleziona solo il flag migliore tra i seguenti: {available_docs}. Non aggiungere commenti o testo."
        ))
    ]
    classification_raw = llm.invoke(classification_prompt).content.strip().lower()
    classification = re.sub(r"[\[\]\'\"]", "", classification_raw).strip()
    print(f"Documento selezionato: '{classification}'")
    if classification not in available_docs:
        classification = available_docs[0]
        print(f"Selezione non valida, fallback a: '{classification}'")
    return {"selected_flag": classification}


def retrieve_documents(state: State):
    user_message = state["current_question"]
    selected_flag = state["selected_flag"]
    bm25_model = bm25_models[selected_flag]
    tokenized_query = user_message.split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:3]
    bm25_docs = [Document(page_content=document_chunks[selected_flag][i]) for i in bm25_indices]
    faiss_docs = retrievers_faiss[selected_flag].invoke(user_message, k=3)
    unique_docs = list({doc.page_content: doc for doc in bm25_docs + faiss_docs}.values())
    return {"retrieved_text": "\n\n".join([doc.page_content for doc in unique_docs[:3]])}


def extract_date_window(query: str) -> tuple:
    """
    Usa un LLM per interpretare liberamente la finestra temporale dalla query.
    Restituisce (data_inizio, data_fine, etichetta_markdown).
    Fallback a 3 mesi futuri in caso di errore di parsing.
    """
    today = today_str()
    prompt_content = (
        f"Data odierna: {today}\n\n"
        "Analizza la domanda e ricava la finestra temporale per filtrare la data di End of Service (EOS) delle macchine IT.\n"
        "Rispondi SOLO con un JSON nel formato esatto, senza markdown ne commenti:\n"
        "{\"date_from\": \"DD/MM/YYYY\", \"date_to\": \"DD/MM/YYYY\", \"label\": \"descrizione breve in italiano\"}\n\n"
        "Linee guida:\n"
        "- 'gia scadute/obsolete/in obsolescenza': date_from='01/01/1900', date_to=oggi\n"
        "- 'questo mese': dal 1 all'ultimo giorno del mese corrente\n"
        "- 'mese scorso': dal 1 all'ultimo giorno del mese precedente\n"
        "- 'quest anno' / 'questo anno': dal 01/01 al 31/12 dell'anno corrente\n"
        "- 'prossimo anno' / 'il prossimo anno': dal 01/01 al 31/12 dell'anno prossimo\n"
        "- 'entro X giorni/settimane/mesi/anni': da oggi a oggi + X unita\n"
        "- 'nel YYYY' / 'entro il YYYY': dal 01/01/YYYY al 31/12/YYYY\n"
        "- 'tra X e Y mesi': da oggi+X mesi a oggi+Y mesi\n"
        "- Se non c'e finestra temporale esplicita: da oggi a 3 mesi da oggi (default)\n\n"
        f"Domanda: '{query}'"
    )
    try:
        response = llm.invoke([SystemMessage(content=prompt_content)]).content.strip()
        clean = re.sub(r'```json\s*|\s*```', '', response).strip()
        parsed = json.loads(clean)

        date_from_str = parsed.get("date_from", "")
        date_to_str = parsed.get("date_to", "")
        label = parsed.get("label", "finestra temporale rilevata")

        date_from = datetime.min if date_from_str == "01/01/1900" else datetime.strptime(date_from_str, "%d/%m/%Y")
        date_to = datetime.strptime(date_to_str, "%d/%m/%Y")
        print(f"Finestra temporale LLM: {date_from_str} -> {date_to_str} | {label}")
        return (date_from, date_to, f"**{label}**")

    except Exception as e:
        print(f"Errore nel parsing della finestra temporale: {e}. Uso default: 3 mesi.")
        today_dt = datetime.now()
        end = today_dt + relativedelta(months=3)
        return (today_dt, end, f"**in scadenza entro 3 mesi** (da {today} a {end.strftime('%d/%m/%Y')})")


def _extract_filters(user_message: str) -> dict:
    """
    Estrae i filtri dalla domanda tramite LLM.
    Campi: responsabile, stato, hostname, sistema_operativo, group_by.
    - group_by: colonna su cui raggruppare il risultato ('SISTEMA OPERATIVO', 'RESPONSABILE', null).
      Usato quando la domanda chiede 'quali sistemi operativi' o 'quali responsabili' invece di 'quali macchine'.
    """
    filter_extraction_prompt = [
        SystemMessage(content=(
            "Sei un estrattore di parametri per query su un database di macchine IT. "
            "Analizza la domanda e popola il JSON con i filtri espliciti.\n"
            "Lo stato puo essere solo 'OPERATIVO' o 'CONFIGURATO'.\n"
            "Per 'group_by': usa 'SISTEMA OPERATIVO' se la domanda chiede quali sistemi operativi (es. 'quale SO', 'quali OS'), "
            "usa 'RESPONSABILE' se chiede quali responsabili, altrimenti null.\n"
            "Se un valore non e menzionato nella domanda, lascialo come null. Rispondi SOLO con il JSON.\n\n"
            f"Data odierna: {today_str()}\n"
            f"Domanda: '{user_message}'\n\n"
            "Esempio 1: 'le macchine operative di CAOps' -> "
            "{\"responsabile\": \"CAOps\", \"stato\": \"OPERATIVO\", \"hostname\": null, \"sistema_operativo\": null, \"group_by\": null}\n"
            "Esempio 2: 'trovami l hostname ALT-CARVM' -> "
            "{\"responsabile\": null, \"stato\": null, \"hostname\": \"ALT-CARVM\", \"sistema_operativo\": null, \"group_by\": null}\n"
            "Esempio 3: 'quale sistema operativo andra in obsolescenza il prossimo anno' -> "
            "{\"responsabile\": null, \"stato\": null, \"hostname\": null, \"sistema_operativo\": null, \"group_by\": \"SISTEMA OPERATIVO\"}\n"
            "Esempio 4: 'macchine con Windows Server 2012' -> "
            "{\"responsabile\": null, \"stato\": null, \"hostname\": null, \"sistema_operativo\": \"Windows Server 2012\", \"group_by\": null}"
        ))
    ]
    try:
        response_json_str = llm.invoke(filter_extraction_prompt).content.strip()
        clean_json_str = re.sub(r'```json\s*|\s*```', '', response_json_str)
        filters = json.loads(clean_json_str)
        print(f"Filtri estratti dalla domanda: {filters}")
        return filters
    except (json.JSONDecodeError, AttributeError):
        print("Non e stato possibile estrarre filtri specifici.")
        return {"responsabile": None, "stato": None, "hostname": None, "sistema_operativo": None, "group_by": None}


def _detect_eos_query(user_message: str) -> bool:
    """Determina se la domanda richiede un filtro sulla data EOS."""
    eos_keywords = [
        "obsolescenza", "obsolet", "scadenza", "scadut", "scadono", "scadr", "scade",
        "eos", "end of service", "end-of-service", "fine servizio", "data fine",
        "fine vita", "end of life", "eol", "fuori supporto", "fuori servizio",
        "supporto terminat", "supporto scadut",
        "questo mese", "mese scorso", "quest'anno", "questo anno",
        "prossim", "entro", "tra", "nei prossimi", "quando scade", "quando scadr",
    ]
    if any(kw in user_message for kw in eos_keywords):
        print("Keyword EOS rilevata -> is_eos_query=True")
        return True

    eos_llm_prompt = [
        SystemMessage(content=(
            "La seguente domanda richiede un filtro sulla data di End of Service (EOS/scadenza) delle macchine/asset/sistemi? "
            "Rispondi SOLO con 'si' o 'no'.\n\n"
            f"Domanda: '{user_message}'"
        ))
    ]
    answer = llm.invoke(eos_llm_prompt).content.strip().lower()
    print(f"LLM fallback EOS check: '{answer}' -> is_eos_query={answer == 'si'}")
    return answer == "si"


def generate_response(state: State):
    user_message = state["current_question"].lower()
    retrieved_text = state.get("retrieved_text", "")
    today = today_str()

    # 1. Rilevamento intento: calcolo vs conoscenza
    intent_prompt = [
        SystemMessage(content=(
            "Analizza la domanda dell'utente e classifica il suo intento.\n"
            "Rispondi SOLO con una di queste due parole: 'calcolo' oppure 'conoscenza'.\n\n"
            "Rispondi 'calcolo' per: conteggi, liste di macchine, ricerca per hostname/responsabile/stato/EOS/obsolescenza.\n"
            "Rispondi 'conoscenza' per: spiegazioni, procedure, politiche, domande concettuali su Data Governance.\n\n"
            f"Data odierna: {today}\n"
            f"Domanda: {user_message}"
        ))
    ]
    intent = llm.invoke(intent_prompt).content.strip().lower()
    print(f"Intento rilevato: {intent}")

    # 2. Ramo calcolo
    if "calcolo" in intent:
        print("Esecuzione ramo di calcolo con Pandas...")
        if _server_df.empty:
            return {"final_response": "Mi dispiace, non trovo il file dati per eseguire il calcolo."}
        try:
            df = _server_df.copy()
            filters = _extract_filters(user_message)

            if filters.get("responsabile"):
                df = df[df['RESPONSABILE'].str.contains(filters["responsabile"], case=False, na=False)]
            if filters.get("stato"):
                df = df[df['STATO'].str.contains(filters["stato"], case=False, na=False)]
            if filters.get("hostname"):
                df = df[df['HOSTNAME'].str.contains(filters["hostname"], case=False, na=False)]
            if filters.get("sistema_operativo"):
                df = df[df['SISTEMA OPERATIVO'].str.contains(filters["sistema_operativo"], case=False, na=False)]

            is_eos_query = _detect_eos_query(user_message)
            time_label = ""

            if is_eos_query:
                date_from, date_to, time_label = extract_date_window(user_message)
                df = df[(df['DATA EOS'] >= date_from) & (df['DATA EOS'] <= date_to)]
                print(f"Filtro EOS applicato: {date_from} -> {date_to}")

            count = len(df)
            group_by = filters.get("group_by")

            if count == 0:
                msg = (
                    f"Nessuna macchina trovata {time_label}."
                    if is_eos_query
                    else "Mi dispiace, non ho trovato nessuna macchina che corrisponda ai criteri specificati."
                )
                return {"final_response": msg}

            # Risposta raggruppata (es. "quale sistema operativo andrà in obsolescenza")
            if group_by and group_by in df.columns:
                grouped = (
                    df.groupby(group_by)
                    .agg(
                        macchine=('HOSTNAME', 'count'),
                        prima_scadenza=('DATA EOS', 'min')
                    )
                    .sort_values('prima_scadenza')
                )
                label_intro = f"**{grouped.shape[0]} sistemi operativi** {time_label}:" if is_eos_query else f"**{grouped.shape[0]} valori distinti** per '{group_by}':"
                rows = [label_intro]
                for os_name, row in grouped.iterrows():
                    prima = row['prima_scadenza'].strftime('%d/%m/%Y') if pd.notna(row['prima_scadenza']) else 'N/A'
                    rows.append(f"- **{os_name}** — {int(row['macchine'])} macchine (prima scadenza: {prima})")
                return {"final_response": "\n".join(rows)}

            # Risposta lista macchine (comportamento standard)
            header = (
                f"Ho trovato **{count} macchine** {time_label}:"
                if is_eos_query
                else f"Ho trovato **{count} macchine** che corrispondono ai tuoi criteri:"
            )
            rows = [header]
            for _, row in df.head(15).iterrows():
                eos_date = row['DATA EOS'].strftime('%d/%m/%Y') if pd.notna(row['DATA EOS']) else 'N/A'
                rows.append(
                    f"- **{row['HOSTNAME']}** (Stato: {row['STATO']}, "
                    f"Resp: {row['RESPONSABILE']}, SO: {row['SISTEMA OPERATIVO']}, EOS: {eos_date})"
                )
            if count > 15:
                rows.append(f"\n... e altre **{count - 15}** macchine.")
            return {"final_response": "\n".join(rows)}

        except Exception as e:
            print(f"Errore nel ramo di calcolo: {e}")
            traceback.print_exc()
            return {"final_response": "Si e verificato un errore durante il calcolo."}

    # 3. Ramo conoscenza (RAG)
    print("Esecuzione ramo di conoscenza (RAG standard)...")
    generation_prompt = [
        SystemMessage(content=(
            f"Sei un assistente esperto in Data Governance e infrastruttura IT. "
            f"Data odierna: {today}.\n\n"
            f"**Domanda dell'utente:** {state['current_question']}\n\n"
            f"**Informazioni recuperate dai documenti:**\n{retrieved_text}\n\n"
            "Rispondi in modo **conciso** e **preciso** basandoti **esclusivamente** sui documenti forniti. "
            "Se le informazioni contengono una lista, formattala in modo chiaro e leggibile. "
            "Se non trovi una risposta adeguata nei documenti, rispondi semplicemente con 'change_document'."
        ))
    ]
    response = llm.invoke(generation_prompt).content.strip()
    print(f"Risposta generata (RAG): {response[:500]}...")
    return {"final_response": response}


def decide_after_response(state: State):
    if "change_document" in state["final_response"].lower():
        print("Richiesta di cambio documento.")
        return {"change_document": True, "checked_documents": state["checked_documents"] + [state["selected_flag"]]}
    print("Risposta pronta per l'output.")
    return {"change_document": False}


def irrelevant_question_response(state: State):
    print("Risposta per domanda non pertinente.")
    return {"final_response": "La richiesta non e pertinente ai documenti in KB."}


def fallback_response(state: State):
    print("Fallback: nessuna informazione trovata.")
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

graph_builder.add_conditional_edges(
    "relevance_check",
    lambda state: "classification" if state["is_relevant"] else "irrelevant_response",
    {"classification": "classification", "irrelevant_response": "irrelevant_response"}
)

graph_builder.add_conditional_edges(
    "classification",
    lambda state: "fallback_response" if state.get("fallback") else "retrieval",
    {"fallback_response": "fallback_response", "retrieval": "retrieval"}
)

graph_builder.add_edge("retrieval", "response")
graph_builder.add_edge("response", "decide_after_response")

graph_builder.add_conditional_edges(
    "decide_after_response",
    lambda state: "classification" if state["change_document"] else "__end__",
    {"classification": "classification", "__end__": END}
)

graph_builder.add_edge("irrelevant_response", END)
graph_builder.add_edge("fallback_response", END)

graph = graph_builder.compile(checkpointer=memory)
print("\nGrafo compilato. Chatbot attivo.\n")


# --- 6. FUNZIONE DI ESECUZIONE ---
def run_rag_chain(question: str, thread_id: str):
    try:
        final_state = graph.invoke(
            {"current_question": question},
            config={"recursion_limit": 50, "configurable": {"thread_id": thread_id}}
        )
        final_response = final_state.get("final_response") if final_state else None
        return final_response if final_response else "Mi dispiace, si e verificato un problema e non ho potuto elaborare una risposta."
    except Exception as e:
        print(f"\nERRORE CRITICO DURANTE L'ESECUZIONE DEL GRAFO: {e}\n")
        traceback.print_exc()
        return "Si e verificato un errore interno. Si prega di riprovare."
