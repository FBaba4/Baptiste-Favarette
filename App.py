import streamlit as st
import os
import uuid
import json
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from DKMQuery import DKMQuery, FichierDchatHistory
from DemandingIngest import process_and_index_pdf, lister_documents, INDEXES_DIR, DOCS_DIR
from predictor import (
    INDICATEURS_DISPONIBLES,
    calculer,
    importances,
    get_csv_columns,
    add_row_and_retrain,
    merge_csv_and_retrain,
    retrain_tous,
)
from DocumentExtractor import extraire_donnees_du_document, enregistrer_extraction

# L'agent SQL dépend d'une famille d'API LangChain plus récente
# (create_agent/langgraph) que le reste de MAF — import protégé pour ne pas
# faire planter toute l'app si l'environnement n'est pas à jour.
try:
    from SQLAgent import agent as sql_agent, extraire_texte as extraire_texte_sql
    SQL_AGENT_DISPONIBLE = True
    _erreur_sql_agent = None
except Exception as _e:
    SQL_AGENT_DISPONIBLE = False
    _erreur_sql_agent = str(_e)

try:
    from WebResearch import agent as web_agent, extraire_texte as extraire_texte_web
    WEB_AGENT_DISPONIBLE = True
    _erreur_web_agent = None
except Exception as _e:
    WEB_AGENT_DISPONIBLE = False
    _erreur_web_agent = str(_e)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MAF - Interface", layout="wide")

HISTORY_DIR = Path("chat_history")
TRANSCRIPTS_DIR = Path("transcripts")          # affichage + sources, par conversation
CONV_INDEX_FILE = Path("conversations_index.json")  # métadonnées pour la sidebar

for d in (INDEXES_DIR, HISTORY_DIR, TRANSCRIPTS_DIR, DOCS_DIR):
    d.mkdir(exist_ok=True)

if not os.environ.get("GOOGLE_API_KEY"):
    st.error(
        "⚠️ GOOGLE_API_KEY introuvable. Crée un fichier `.env` à la racine du projet "
        "contenant `GOOGLE_API_KEY=ta_cle` (sans guillemets)."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────
# Persistance : conversations + transcripts (sources)
# ─────────────────────────────────────────────────────────────────────────
def charger_index_conversations() -> dict:
    if CONV_INDEX_FILE.exists():
        return json.loads(CONV_INDEX_FILE.read_text(encoding="utf-8"))
    return {}

def sauver_index_conversations(index: dict):
    CONV_INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

def charger_transcript(session_id: str) -> list:
    fp = TRANSCRIPTS_DIR / f"{session_id}.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return []

def sauver_transcript(session_id: str, transcript: list):
    fp = TRANSCRIPTS_DIR / f"{session_id}.json"
    fp.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")


conversations = charger_index_conversations()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "creating_new" not in st.session_state:
    st.session_state.creating_new = False
if "renaming_id" not in st.session_state:
    st.session_state.renaming_id = None


def type_conversation(meta: dict) -> str:
    """
    'document' | 'estimation' | 'sql_agent'. Les conversations créées avant
    l'ajout de l'agent SQL n'ont pas de champ 'type' en mémoire — on le
    déduit alors de la présence d'un document associé, pour rester
    rétrocompatible avec les conversations existantes.
    """
    if "type" in meta:
        return meta["type"]
    return "document" if meta.get("document") else "estimation"


def demarrer_conversation(titre_affiche: str, index_name, type_conv: str = None):
    """index_name : None pour 'estimation'/'sql_agent', nom d'index pour 'document'."""
    if type_conv is None:
        type_conv = "document" if index_name is not None else "estimation"

    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conversations[session_id] = {
        "title": titre_affiche if index_name is None else titre_affiche.replace(".pdf", ""),
        "document": index_name,
        "type": type_conv,
        "created_at": now,
        "updated_at": now,
    }
    sauver_index_conversations(conversations)
    sauver_transcript(session_id, [])
    st.session_state.current_session_id = session_id
    st.session_state.creating_new = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR — identité visuelle V1 (composants natifs, mêmes icônes)
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.title("💬 MAF Navigator")

# 1. Upload de documents — ajout direct depuis le navigateur
st.sidebar.subheader("📂 Ajouter une entreprise")
uploaded_file = st.sidebar.file_uploader("Charger un rapport PDF", type="pdf", label_visibility="collapsed")
if uploaded_file and st.sidebar.button("Indexer ce document"):
    with st.spinner("Indexation en cours..."):
        index_name = process_and_index_pdf(uploaded_file)
    st.sidebar.success(f"Document {uploaded_file.name} ajouté !")
    demarrer_conversation(uploaded_file.name, index_name)

# 2. Gestion des données de l'estimateur
st.sidebar.divider()
st.sidebar.subheader("📊 Données de l'estimateur")

with st.sidebar.expander("📈 État des modèles"):
    from predictor import metriques
    for cle, config in INDICATEURS_DISPONIBLES.items():
        m = metriques(cle)
        if "erreur" in m:
            st.caption(f"❌ {config['label']} : {m['erreur']}")
        else:
            st.caption(f"✅ {config['label']} : R² = {m['r2']:.2f} (n={m['n_train']+m['n_test']})")

expliquer_predictions = st.sidebar.checkbox(
    "Expliquer chaque estimation (1 appel LLM en plus)",
    value=False,
    help="Génère une phrase de contexte qualitatif sous chaque prédiction. "
         "Désactivé par défaut pour ne pas doubler ta consommation d'API pendant les tests.",
)

comparer_avec_predicteur = st.sidebar.checkbox(
    "🔬 Comparer le document au modèle prédictif",
    value=False,
    help="Dans une conversation liée à un document, tente d'extraire les chiffres "
         "du rapport (CA, bénéfice net, marge brute) et de les comparer à la "
         "prédiction du modèle. Ajoute un appel LLM supplémentaire à CHAQUE "
         "question posée sur un document (pas seulement quand ça matche) — "
         "à activer plutôt pour une démo ciblée que pour des tests répétés.",
)

with st.sidebar.expander("📥 Importer un CSV"):
    st.caption("Le fichier doit contenir au minimum les mêmes colonnes que le CSV existant.")
    csv_upload = st.file_uploader("Fichier CSV", type="csv", key="csv_uploader", label_visibility="collapsed")
    if csv_upload and st.button("Fusionner et réentraîner", key="btn_merge_csv"):
        try:
            resultats = merge_csv_and_retrain(csv_upload)
            lignes = []
            for cle, m in resultats.items():
                label = INDICATEURS_DISPONIBLES[cle]["label"]
                if "erreur" in m:
                    lignes.append(f"❌ {label} : {m['erreur']}")
                else:
                    lignes.append(f"✅ {label} : R² = {m['r2']:.2f} (n={m['n_train'] + m['n_test']})")
            st.success("Fusion réussie, modèles réentraînés :\n\n" + "\n".join(lignes))
        except Exception as e:
            st.error(f"Erreur : {e}")

with st.sidebar.expander("➕ Ajouter une entreprise manuellement"):
    colonnes = get_csv_columns()
    if not colonnes:
        st.warning("Aucun CSV de référence trouvé.")
    else:
        with st.form("form_ajout_entreprise", clear_on_submit=True):
            valeurs = {}
            for col in colonnes:
                valeurs[col] = st.text_input(col, key=f"champ_{col}")
            soumis = st.form_submit_button("Ajouter et réentraîner")
        if soumis:
            try:
                resultats = add_row_and_retrain(valeurs)
                lignes = []
                for cle, m in resultats.items():
                    label = INDICATEURS_DISPONIBLES[cle]["label"]
                    if "erreur" in m:
                        lignes.append(f"❌ {label} : {m['erreur']}")
                    else:
                        lignes.append(f"✅ {label} : R² = {m['r2']:.2f} (n={m['n_train'] + m['n_test']})")
                st.success("Entreprise ajoutée, modèles réentraînés :\n\n" + "\n".join(lignes))
            except Exception as e:
                st.error(f"Erreur : {e}")

# 3. Gestion des conversations
st.sidebar.divider()
st.sidebar.subheader("🕒 Conversations")

if st.sidebar.button("＋ Nouvelle conversation", use_container_width=True):
    st.session_state.creating_new = True
    st.session_state.current_session_id = None
    st.rerun()

tri = sorted(conversations.items(), key=lambda kv: kv[1].get("updated_at", ""), reverse=True)

for session_id, meta in tri:
    actif = session_id == st.session_state.current_session_id

    if st.session_state.renaming_id == session_id:
        nouveau_titre = st.sidebar.text_input(
            "Renommer", value=meta["title"], key=f"rename_{session_id}", label_visibility="collapsed"
        )
        col_ok, col_cancel = st.sidebar.columns(2)
        if col_ok.button("✓", key=f"confirm_{session_id}"):
            conversations[session_id]["title"] = nouveau_titre.strip() or meta["title"]
            sauver_index_conversations(conversations)
            st.session_state.renaming_id = None
            st.rerun()
        if col_cancel.button("✕", key=f"cancel_{session_id}"):
            st.session_state.renaming_id = None
            st.rerun()
    else:
        col_titre, col_edit = st.sidebar.columns([5, 1])
        icone_type = {"document": "📄", "estimation": "📊", "sql_agent": "🗄️", "web_recherche": "🔍"}.get(type_conversation(meta), "💬")
        label = f"{'🟢' if actif else '⚪'} {icone_type} {meta['title']}"
        if col_titre.button(label, key=f"open_{session_id}", use_container_width=True):
            st.session_state.current_session_id = session_id
            st.session_state.creating_new = False
            st.rerun()
        if col_edit.button("✎", key=f"edit_{session_id}"):
            st.session_state.renaming_id = session_id
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — nouvelle conversation
# ─────────────────────────────────────────────────────────────────────────
def _anneau(couleur: str, icone: str) -> go.Figure:
    """
    Un anneau plein coloré façon 'cadran Apple Watch', fait avec Plotly
    (natif Streamlit via st.plotly_chart) — pas de HTML. L'anneau n'est pas
    cliquable en lui-même (limite technique de Plotly dans Streamlit sans
    librairie tierce) — un vrai bouton est placé juste en dessous pour
    l'action, l'anneau reste décoratif/visuel.
    """
    fig = go.Figure(go.Pie(
        values=[1],
        hole=0.72,
        marker=dict(colors=[couleur], line=dict(color="white", width=0)),
        textinfo="none",
        sort=False,
        direction="clockwise",
        domain=dict(x=[0, 1], y=[0, 1]),  # force le cercle à occuper toute la zone, sans rognage
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=12, r=12, t=12, b=12),  # marge symétrique : évite que le bord soit coupé
        height=150,
        width=150,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=icone, x=0.5, y=0.5,
            xanchor="center", yanchor="middle",  # ancrage explicite : centre vraiment l'icône
            font=dict(size=32),
            showarrow=False,
        )],
    )
    return fig


def ecran_nouvelle_conversation():
    st.title("Nouvelle conversation")
    st.caption("Choisis la fonctionnalité — comme les anneaux d'activité d'une montre connectée.")

    ligne_haut = st.columns(2)
    ligne_bas = st.columns(2)

    with ligne_haut[0]:
        st.plotly_chart(_anneau("#E63950", "📄"), use_container_width=False, config={"staticPlot": True})
        st.markdown("**Analyser un document**")
        choix_document = st.button("Sélectionner", key="choix_document", use_container_width=True)

    with ligne_haut[1]:
        st.plotly_chart(_anneau("#00B050", "📊"), use_container_width=False, config={"staticPlot": True})
        st.markdown("**Estimations uniquement**")
        choix_estimation = st.button("Sélectionner", key="choix_estimation", use_container_width=True)

    with ligne_bas[0]:
        st.plotly_chart(_anneau("#00B0F0", "🗄️"), use_container_width=False, config={"staticPlot": True})
        st.markdown("**Base de données**")
        choix_sql = st.button("Sélectionner", key="choix_sql", use_container_width=True)

    with ligne_bas[1]:
        st.plotly_chart(_anneau("#8E44AD", "🔍"), use_container_width=False, config={"staticPlot": True})
        st.markdown("**Recherche web**")
        choix_web = st.button("Sélectionner", key="choix_web", use_container_width=True)

    if choix_document:
        st.session_state.type_selectionne = "document"
    elif choix_estimation:
        st.session_state.type_selectionne = "estimation"
    elif choix_sql:
        st.session_state.type_selectionne = "sql_agent"
    elif choix_web:
        st.session_state.type_selectionne = "web_recherche"

    type_conv_libelle = {
        "document": "📄 Analyser un document",
        "estimation": "📊 Estimations uniquement",
        "sql_agent": "🗄️ Interroger la base de données",
        "web_recherche": "🔍 Recherche web",
    }
    type_conv_selectionne = st.session_state.get("type_selectionne")

    if not type_conv_selectionne:
        return

    st.divider()
    st.markdown(f"### {type_conv_libelle[type_conv_selectionne]}")
    type_conv = type_conv_libelle[type_conv_selectionne]

    if type_conv == "🔍 Recherche web":
        if not WEB_AGENT_DISPONIBLE:
            st.error(
                f"Agent de recherche web indisponible : {_erreur_web_agent}\n\n"
                "Vérifie que `WebResearch.py` est présent dans le dossier, que "
                "`langchain-tavily` est installé (`pip install langchain-tavily`), "
                "et que `TAVILY_API_KEY` est dans ton `.env`."
            )
            return
        st.caption(
            "Pose des questions nécessitant une information récente ou externe "
            "(actualité économique, contexte macro, actualité d'une entreprise) — "
            "l'agent cherche sur le web et cite ses sources."
        )
        if st.button("Démarrer une conversation de recherche web"):
            titre = f"Recherche web {datetime.now().strftime('%d/%m %H:%M')}"
            demarrer_conversation(titre, None, type_conv="web_recherche")
        return

    if type_conv == "🗄️ Interroger la base de données":
        if not SQL_AGENT_DISPONIBLE:
            st.error(
                f"Agent SQL indisponible : {_erreur_sql_agent}\n\n"
                "Vérifie que `SQLAgent.py` est présent dans le dossier et que "
                "`langchain`/`langgraph` sont installés à jour (`pip install -U langchain langgraph`)."
            )
            return
        st.caption(
            "Interroge la base de ~180 entreprises collectées : comparaisons, statistiques "
            "sectorielles, détection d'outliers, score de risque, prédictions ML et analyse "
            "technique — le tout dans une seule conversation, sans document requis."
        )
        if st.button("Démarrer une conversation avec l'agent base de données"):
            titre = f"Base de données {datetime.now().strftime('%d/%m %H:%M')}"
            demarrer_conversation(titre, None, type_conv="sql_agent")
        return

    if type_conv == "📊 Estimations uniquement":
        st.caption(
            "Pose des questions de prédiction (marge nette à partir du chiffre d'affaires, "
            "du bénéfice net et de la marge brute) sans avoir besoin d'un document indexé."
        )
        if st.button("Démarrer une conversation d'estimation"):
            titre = f"Estimations {datetime.now().strftime('%d/%m %H:%M')}"
            demarrer_conversation(titre, None, type_conv="estimation")
        return

    st.caption("Choisissez un document déjà indexé, ou ajoutez-en un depuis la sidebar (📂 Ajouter une entreprise).")

    fichiers = lister_documents()
    if not fichiers:
        st.info("Aucun document indexé pour le moment.")
        return

    options = []
    for f in fichiers:
        indexe = (INDEXES_DIR / f"{f}_index").exists()
        options.append(f"{'🟢' if indexe else '🟡'} {f}" + ("" if indexe else " (non indexé)"))

    choix = st.selectbox("Base de données associée :", options)
    nom_fichier = fichiers[options.index(choix)]
    indexe = (INDEXES_DIR / f"{nom_fichier}_index").exists()

    if st.button("Ouvrir cette conversation" if indexe else "Indexer et ouvrir"):
        if not indexe:
            with st.spinner("Indexation en cours..."):
                class _FichierLocal:
                    def __init__(self, path: Path):
                        self.name = path.name
                        self._data = path.read_bytes()
                    def getbuffer(self):
                        return self._data
                index_name = process_and_index_pdf(_FichierLocal(DOCS_DIR / nom_fichier))
        else:
            index_name = f"{nom_fichier}_index"
        demarrer_conversation(nom_fichier, index_name, type_conv="document")


# ─────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — chat actif
# ─────────────────────────────────────────────────────────────────────────
def traiter_message(session_id: str, index_name, user_input: str, transcript: list, type_conv: str):
    """
    index_name peut être None : conversation d'estimation ou d'agent SQL,
    aucun RAG possible dans ce cas. Ne fait AUCUN rendu Streamlit —
    ecran_chat() affiche au fur et à mesure.
    """
    transcript.append({"role": "user", "content": user_input, "sources": None})

    if type_conv == "sql_agent":
        try:
            if "sql_agent_histories" not in st.session_state:
                st.session_state.sql_agent_histories = {}
            historique = st.session_state.sql_agent_histories.setdefault(session_id, [])
            historique.append({"role": "user", "content": user_input})

            resultat = sql_agent.invoke({"messages": historique})
            st.session_state.sql_agent_histories[session_id] = resultat["messages"]

            reponse = extraire_texte_sql(resultat["messages"][-1].content)
            transcript.append({"role": "assistant", "type": "sql_agent", "content": reponse, "sources": None})

        except Exception as e:
            message = str(e)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                contenu = (
                    "🚫 **Quota API Gemini atteint**. Attends la réinitialisation quotidienne "
                    "ou réduis la fréquence des questions à l'agent base de données."
                )
            else:
                contenu = f"❌ Erreur de l'agent base de données : {message[:300]}"
            transcript.append({"role": "assistant", "type": "info", "content": contenu, "sources": None})

        sauver_transcript(session_id, transcript)
        conversations[session_id]["updated_at"] = datetime.now().isoformat()
        sauver_index_conversations(conversations)
        return

    if type_conv == "web_recherche":
        try:
            if "web_agent_histories" not in st.session_state:
                st.session_state.web_agent_histories = {}
            historique = st.session_state.web_agent_histories.setdefault(session_id, [])
            historique.append({"role": "user", "content": user_input})

            resultat = web_agent.invoke({"messages": historique})
            st.session_state.web_agent_histories[session_id] = resultat["messages"]

            reponse = extraire_texte_web(resultat["messages"][-1].content)
            transcript.append({"role": "assistant", "type": "web_recherche", "content": reponse, "sources": None})

        except Exception as e:
            message = str(e)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                contenu = (
                    "🚫 **Quota API atteint** (Gemini ou Tavily). Attends la réinitialisation "
                    "ou vérifie ton quota Tavily sur app.tavily.com."
                )
            else:
                contenu = f"❌ Erreur de l'agent de recherche web : {message[:300]}"
            transcript.append({"role": "assistant", "type": "info", "content": contenu, "sources": None})

        sauver_transcript(session_id, transcript)
        conversations[session_id]["updated_at"] = datetime.now().isoformat()
        sauver_index_conversations(conversations)
        return

    try:
        if index_name is None:
            llm_routeur = DKMQuery.get_router()
        else:
            llm_routeur, rag_chain = DKMQuery.get_chain_and_router(index_name, session_id)

        routage = llm_routeur.invoke(user_input)

        if routage.tool_calls:
            for appel in routage.tool_calls:
                cle_indicateur = appel["name"].replace("predict_", "")
                config = INDICATEURS_DISPONIBLES.get(cle_indicateur)
                if config is None:
                    continue

                try:
                    valeur = calculer(cle_indicateur, **appel["args"])
                except Exception as e:
                    transcript.append({
                        "role": "assistant", "type": "info",
                        "content": f"⚠️ Impossible de calculer {config['label'].lower()} : {e}",
                        "sources": None,
                    })
                    continue

                explication = None
                if expliquer_predictions:
                    llm = DKMQuery.get_llm()
                    explication = llm.invoke(
                        f"Le modèle prédit {config['label'].lower()} à {valeur:.2f}%. "
                        "En 2 phrases maximum, indique si ce chiffre est plutôt "
                        "rassurant ou préoccupant par rapport à des repères "
                        "sectoriels généraux. Précise que c'est une estimation "
                        "statistique, pas une donnée vérifiée."
                    ).content

                transcript.append({
                    "role": "assistant",
                    "type": "prediction",
                    "label": config["label"],
                    "content": f"{config['label']} prédit(e) : {valeur:.2f}%.",
                    "valeur": valeur,
                    "importances": importances(cle_indicateur),
                    "explication": explication,
                    "sources": None,
                })
        elif index_name is None:
            transcript.append({
                "role": "assistant",
                "type": "info",
                "content": (
                    "💬 Cette conversation est dédiée aux estimations. Précise le chiffre "
                    "d'affaires, le bénéfice net et la marge brute pour que je calcule une "
                    "prédiction — ou ouvre une nouvelle conversation liée à un document pour "
                    "poser des questions générales."
                ),
                "sources": None,
            })
        else:
            response = rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            sources = [
                {"page": doc.metadata.get("page", "?"), "excerpt": doc.page_content[:220] + "…"}
                for doc in response.get("context", [])
            ]
            reponse_rag = response["answer"]

            prediction_associee = None
            if comparer_avec_predicteur:
                prompt_comparaison = (
                    f"Question posée : {user_input}\n\n"
                    f"Réponse basée sur le document : {reponse_rag}\n\n"
                    "Si cette réponse mentionne explicitement le chiffre d'affaires, "
                    "le bénéfice net ET la marge brute d'une entreprise, et si comparer "
                    "ces chiffres à un modèle prédictif serait pertinent pour la question, "
                    "appelle l'outil de prédiction correspondant avec ces valeurs extraites "
                    "du texte. Sinon, n'appelle aucun outil."
                )
                try:
                    routage_comparaison = llm_routeur.invoke(prompt_comparaison)
                    if routage_comparaison.tool_calls:
                        appel = routage_comparaison.tool_calls[0]
                        cle_indicateur = appel["name"].replace("predict_", "")
                        config = INDICATEURS_DISPONIBLES.get(cle_indicateur)
                        if config:
                            valeur = calculer(cle_indicateur, **appel["args"])
                            prediction_associee = {
                                "label": config["label"],
                                "valeur": valeur,
                                "importances": importances(cle_indicateur),
                            }
                            if expliquer_predictions:
                                llm = DKMQuery.get_llm()
                                prediction_associee["analyse_ecart"] = llm.invoke(
                                    f"Le document annonce, dans le contexte suivant : « {reponse_rag} », "
                                    f"des chiffres correspondant à {config['label'].lower()} tandis que le "
                                    f"modèle statistique prédit {valeur:.2f}%. Si ces deux valeurs semblent "
                                    "converger, dis-le simplement en une phrase. Si elles divergent "
                                    "significativement, propose en 2-3 phrases une ou plusieurs hypothèses "
                                    "parmi : innovation/avantage compétitif non capturé par le modèle, "
                                    "changement de périmètre (acquisition, cession), choc exogène récent, "
                                    "ou simple limite du modèle (échantillon d'entraînement restreint). "
                                    "Sois direct, ton de consultant en audit."
                                ).content
                except Exception:
                    pass

            transcript.append({
                "role": "assistant",
                "type": "rag",
                "content": reponse_rag,
                "sources": sources,
                "prediction": prediction_associee,
            })

    except Exception as e:
        message = str(e)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            contenu = (
                "🚫 **Quota API Gemini atteint** (plan gratuit : 20 requêtes/jour pour "
                "gemini-2.5-flash). Ce n'est pas un bug de l'application — c'est une limite "
                "de Google. Solutions : attends la réinitialisation quotidienne, désactive "
                "temporairement « Expliquer chaque estimation » et « Comparer le document au "
                "modèle prédictif » dans la sidebar (chacune ajoute des appels API par "
                "question), ou passe sur un plan payant Google AI Studio."
            )
        else:
            contenu = f"❌ Erreur lors de l'appel au modèle : {message[:300]}"

        transcript.append({"role": "assistant", "type": "info", "content": contenu, "sources": None})

    sauver_transcript(session_id, transcript)
    conversations[session_id]["updated_at"] = datetime.now().isoformat()
    sauver_index_conversations(conversations)


def afficher_tour(tour: dict):
    type_tour = tour.get("type")

    if type_tour == "prediction":
        st.markdown(f"**📊 Réponse issue du modèle prédictif** — *aucun document consulté*")
        st.metric(label=tour.get("label", "Indicateur estimé"), value=f"{tour['valeur']:.2f}%")

        if tour.get("importances"):
            st.caption("Poids de chaque variable dans ce calcul :")
            for variable, poids in sorted(tour["importances"].items(), key=lambda kv: kv[1], reverse=True):
                st.progress(poids, text=f"{variable} — {poids * 100:.0f}%")

        if tour.get("explication"):
            st.info(tour["explication"])

        st.caption(
            "⚠️ Estimation statistique issue d'un modèle entraîné sur un échantillon "
            "restreint — à interpréter comme un ordre de grandeur, pas une donnée vérifiée."
        )

    elif type_tour == "rag":
        st.markdown("**📄 Réponse basée sur le document**")
        st.markdown(tour["content"])
        if tour.get("sources"):
            with st.expander(f"📄 {len(tour['sources'])} source(s)"):
                for src in tour["sources"]:
                    st.markdown(f"**Page {src['page']}**")
                    st.caption(src["excerpt"])

        prediction = tour.get("prediction")
        if prediction:
            st.divider()
            st.markdown(f"**🔬 Comparaison avec le modèle prédictif** — *{prediction['label']}*")
            st.metric(label=f"{prediction['label']} (prédite par le modèle)", value=f"{prediction['valeur']:.2f}%")
            if prediction.get("importances"):
                st.caption("Poids de chaque variable dans ce calcul :")
                for variable, poids in sorted(prediction["importances"].items(), key=lambda kv: kv[1], reverse=True):
                    st.progress(poids, text=f"{variable} — {poids * 100:.0f}%")
            st.caption(
                "⚠️ Chiffres extraits automatiquement du texte ci-dessus par le LLM, puis "
                "passés à un modèle statistique indépendant — à vérifier avant citation."
            )
            if prediction.get("analyse_ecart"):
                st.markdown("**🕵️ Analyse de l'écart (regard de consultant)**")
                st.warning(prediction["analyse_ecart"])

    elif type_tour == "sql_agent":
        st.markdown("**🗄️ Réponse de l'agent base de données**")
        st.markdown(tour["content"])

    elif type_tour == "web_recherche":
        st.markdown("**🔍 Réponse de l'agent de recherche web**")
        st.markdown(tour["content"])
        st.caption("⚠️ Information externe datée — vérifie la source citée avant de t'y fier.")

    else:
        st.markdown(tour["content"])


def ecran_chat(session_id: str):
    meta = conversations[session_id]
    index_name = meta["document"]
    type_conv = type_conversation(meta)

    if type_conv == "sql_agent":
        st.title(f"🗄️ {meta['title']}")
        st.caption("Agent base de données — SQL, statistiques, score de risque, ML, analyse technique.")
    elif type_conv == "web_recherche":
        st.title(f"🔍 {meta['title']}")
        st.caption("Agent de recherche web (Tavily) — information externe, sources citées.")
    elif type_conv == "estimation":
        st.title(f"📊 {meta['title']}")
        st.caption("Conversation d'estimation — aucun document associé.")
    else:
        st.title(f"📄 Analyse financière : {meta['title']}")
        st.caption(f"Document : {index_name.replace('_index', '')}")

    if type_conv == "document":
        with st.expander("📥 Extraire les données financières de ce document vers la base"):
            st.caption(
                "Analyse le document et propose d'ajouter/enrichir la ligne de cette "
                "entreprise dans la base de données, avec traçabilité par page. "
                "Rien n'est enregistré sans ta validation ci-dessous."
            )
            if st.button("🔍 Lancer l'extraction", key="btn_lancer_extraction"):
                with st.spinner("Extraction en cours..."):
                    try:
                        st.session_state["extraction_en_attente"] = extraire_donnees_du_document(index_name)
                    except Exception as e:
                        st.error(f"Erreur d'extraction : {e}")

            extraction_en_attente = st.session_state.get("extraction_en_attente")
            if extraction_en_attente:
                st.write("**Aperçu — vérifie avant d'enregistrer :**")
                st.json(extraction_en_attente.model_dump())
                col_ok, col_annuler = st.columns(2)
                if col_ok.button("✅ Enregistrer dans la base", key="btn_confirmer_extraction"):
                    resultat = enregistrer_extraction(extraction_en_attente)
                    with st.spinner("Réentraînement des modèles..."):
                        retrain_tous()
                    st.success(f"Entreprise {resultat['action']} : {resultat['entreprise']}")
                    st.session_state["extraction_en_attente"] = None
                if col_annuler.button("✕ Annuler", key="btn_annuler_extraction"):
                    st.session_state["extraction_en_attente"] = None

    transcript = charger_transcript(session_id)

    for tour in transcript:
        with st.chat_message("user" if tour["role"] == "user" else "assistant"):
            afficher_tour(tour)

    if prompt := st.chat_input("Posez votre question (ou demandez une prédiction) :"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..." if index_name is None else "Analyse du document..."):
                traiter_message(session_id, index_name, prompt, transcript, type_conv)
            afficher_tour(transcript[-1])


# ─────────────────────────────────────────────────────────────────────────
# Routage principal
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.creating_new or st.session_state.current_session_id is None:
    ecran_nouvelle_conversation()
else:
    ecran_chat(st.session_state.current_session_id)