import streamlit as st
import os
import uuid
import json
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from DemandingIngest import (
    process_and_index_pdf, lister_documents, INDEXES_DIR, DOCS_DIR,
    lister_entreprises, documents_pour_entreprise, associer_document_entreprise, retirer_document_entreprise,
    associer_documents_automatiquement,
)
from PredictionAgent import (
    INDICATEURS_DISPONIBLES,
    get_csv_columns,
    add_row_and_retrain,
    merge_csv_and_retrain,
    retrain_tous,
    metriques,
)
from DocumentExtractor import extraire_donnees_du_document, enregistrer_extraction
from Glossaire import GLOSSAIRE

# L'orchestrateur dépend de create_agent/langgraph — import protégé pour ne
# pas faire planter toute l'app si l'environnement n'est pas à jour.
try:
    from Monitor import (
        construire_agent,
        agents_mobilises,
        formater_source,
        extraire_texte,
        TRACE_DERNIER_TOUR,
        lancer_due_diligence,
    )
    from SQLAgent import donnees_radar_entreprise, donnees_score_risque_graphique, donnees_comparaison_graphique
    ORCHESTRATEUR_DISPONIBLE = True
    _erreur_orchestrateur = None
except Exception as _e:
    ORCHESTRATEUR_DISPONIBLE = False
    _erreur_orchestrateur = str(_e)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MAF - Interface", layout="wide")

# ─────────────────────────────────────────────────────────────────────────
# Habillage visuel — se rapproche de ChatGPT/Gemini/Le Chat en typographie
# et fluidité, sans réécrire les widgets Streamlit en HTML (juste du CSS
# ciblé sur des sélecteurs data-testid, plus stables entre versions
# Streamlit que les noms de classes générés). Le CSS est chargé par le
# navigateur de l'utilisateur, pas par le backend — @import Google Fonts
# fonctionne normalement, avec repli sur les polices système si bloqué.
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

html { scroll-behavior: smooth; }

/* Transitions douces sur tous les éléments interactifs — "fluidité" */
button, [data-testid="stExpander"], [data-testid="stChatInput"] textarea,
input, textarea, [data-testid="stPopover"] button {
    transition: all 0.18s ease-in-out !important;
}

/* Boutons arrondis avec léger relief au survol, comme les chat modernes */
.stButton button, [data-testid="stChatInput"] button, [data-testid="stPopover"] button {
    border-radius: 12px !important;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}

/* Champ de saisie principal façon ChatGPT/Gemini : très arrondi */
[data-testid="stChatInput"] {
    border-radius: 24px !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
}

/* Bulles de messages plus douces, moins "cadre de formulaire" */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 6px 10px !important;
}

/* Expanders et popover arrondis, bordure discrète plutôt que dure */
[data-testid="stExpander"], [data-testid="stPopoverBody"] {
    border-radius: 14px !important;
    border: 1px solid rgba(128,128,128,0.15) !important;
}

/* Champs texte arrondis, cohérents avec le reste */
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
}

/* Sidebar : séparation discrète plutôt qu'un trait dur */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.12);
}
</style>
""", unsafe_allow_html=True)

HISTORY_DIR = Path("chat_history")
TRANSCRIPTS_DIR = Path("transcripts")
CONV_INDEX_FILE = Path("conversations_index.json")

for d in (INDEXES_DIR, HISTORY_DIR, TRANSCRIPTS_DIR, DOCS_DIR):
    d.mkdir(exist_ok=True)

if not os.environ.get("GOOGLE_API_KEY"):
    st.error(
        "⚠️ GOOGLE_API_KEY introuvable. Crée un fichier `.env` à la racine du projet "
        "contenant `GOOGLE_API_KEY=ta_cle` (sans guillemets)."
    )
    st.stop()

if not ORCHESTRATEUR_DISPONIBLE:
    st.error(
        f"⚠️ Orchestrateur indisponible : {_erreur_orchestrateur}\n\n"
        "Vérifie que `Monitor.py`, `SQLAgent.py`, `WebResearch.py`, `TechnicalAnalysis.py` "
        "sont présents, et que `langchain`/`langgraph`/`langchain-tavily` sont installés à jour."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────
# Persistance
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
if "orchestrateur_histories" not in st.session_state:
    st.session_state.orchestrateur_histories = {}
if "afficher_glossaire" not in st.session_state:
    st.session_state.afficher_glossaire = False


def demarrer_conversation(titre_affiche: str, entreprise: str | None = None):
    """
    Une conversation est désormais liée à une ENTREPRISE (ou aucune, pour
    une conversation générale) — pas à un document unique. Les documents
    associés à l'entreprise sont relus depuis le registre à chaque message
    (DemandingIngest.documents_pour_entreprise), jamais figés ici : un
    document ajouté après coup devient immédiatement disponible.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conversations[session_id] = {
        "title": titre_affiche,
        "entreprise": entreprise,
        "created_at": now,
        "updated_at": now,
    }
    sauver_index_conversations(conversations)
    sauver_transcript(session_id, [])
    st.session_state.current_session_id = session_id
    st.session_state.creating_new = False
    st.rerun()


def documents_de_la_conversation(meta: dict) -> list:
    """
    Documents disponibles pour cette conversation — relus à chaque appel
    depuis le registre (pas stockés dans la conversation elle-même), pour
    qu'un document ajouté après coup soit immédiatement disponible.

    Rétrocompatibilité : les conversations créées avant ce système (champ
    'document' = un seul index_name, pas de champ 'entreprise') continuent
    de fonctionner avec leur unique document, sans migration nécessaire.
    """
    entreprise = meta.get("entreprise")
    if entreprise:
        return documents_pour_entreprise(entreprise)

    ancien_document = meta.get("document")
    if ancien_document:
        return [{"index_name": ancien_document, "titre": meta.get("title", ancien_document)}]

    return []


# ─────────────────────────────────────────────────────────────────────────
# Anneau visuel (Plotly, natif — même design partout dans l'app)
# ─────────────────────────────────────────────────────────────────────────
def _anneau(couleur: str, icone: str, taille: int = 150) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[1],
        hole=0.72,
        marker=dict(colors=[couleur], line=dict(color="white", width=0)),
        textinfo="none",
        sort=False,
        direction="clockwise",
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=12, r=12, t=12, b=12),
        height=taille,
        width=taille,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=icone, x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            font=dict(size=int(taille * 0.22)),
            showarrow=False,
        )],
    )
    return fig


def _radar_entreprise(radar_data: dict) -> go.Figure:
    """
    Radar chart Plotly comparant l'entreprise à la moyenne de son secteur
    sur 5 ratios de rentabilité — valeurs déjà calculées en amont par
    donnees_radar_entreprise() (SQL/pandas pur, aucun texte LLM impliqué).
    Les valeurs manquantes (NaN) sont affichées comme des trous dans le
    tracé plutôt que masquées silencieusement.
    """
    labels = radar_data["labels"]
    valeurs_entreprise = radar_data["valeurs_entreprise"]
    valeurs_secteur = radar_data["valeurs_secteur"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valeurs_entreprise, theta=labels, fill="toself",
        name=radar_data["entreprise"], line=dict(color="#00B0F0"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=valeurs_secteur, theta=labels, fill="toself",
        name=f"Moyenne secteur ({radar_data.get('secteur') or 'panel'})",
        line=dict(color="#8E44AD"), opacity=0.6,
    ))

    valeurs_toutes = [v for v in valeurs_entreprise + valeurs_secteur if v == v]  # exclut NaN
    borne_max = max(valeurs_toutes) * 1.15 if valeurs_toutes else 100

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, borne_max])),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(l=40, r=40, t=20, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _camembert_score_risque(data: dict) -> go.Figure:
    """Camembert des 3 composantes pondérées du score de risque MAF."""
    labels = list(data["contributions"].keys())
    valeurs = list(data["contributions"].values())

    fig = go.Figure(go.Pie(
        labels=labels, values=valeurs, hole=0.4,
        marker=dict(colors=["#00B0F0", "#F39C12", "#8E44AD"]),
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=f"Score de risque — {data['entreprise']} ({data['score_total']:.0f}/100)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _batons_comparaison(data: dict) -> go.Figure:
    """Diagramme en bâtons groupés — un groupe par métrique, une barre par entreprise."""
    fig = go.Figure()
    couleurs = ["#00B0F0", "#8E44AD", "#00B050", "#F39C12", "#E63950"]

    for i, entreprise in enumerate(data["entreprises"]):
        valeurs_entreprise = [data["valeurs"][col].get(entreprise) for col in data["colonnes"]]
        fig.add_trace(go.Bar(
            x=data["colonnes"], y=valeurs_entreprise, name=entreprise,
            marker_color=couleurs[i % len(couleurs)],
        ))

    fig.update_layout(
        barmode="group",
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# Config partagée pour tous les agents — même design (couleur/icône) que
# l'écran d'accueil et le sélecteur manuel, un seul endroit à maintenir.
AGENTS_CONFIG = {
    "agent_document": {"couleur": "#E63950", "icone": "📄", "label": "Document"},
    "agent_estimation": {"couleur": "#00B050", "icone": "📊", "label": "Estimation"},
    "agent_technique": {"couleur": "#F39C12", "icone": "📈", "label": "Technique"},
    "agent_base_donnees": {"couleur": "#00B0F0", "icone": "🗄️", "label": "Base de données"},
    "agent_recherche_web": {"couleur": "#8E44AD", "icone": "🔍", "label": "Recherche web"},
}


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.title("💬 MAF Navigator")

st.sidebar.divider()
st.sidebar.subheader("📊 Données de l'estimateur")

with st.sidebar.expander("📈 État des modèles"):
    for cle, config in INDICATEURS_DISPONIBLES.items():
        m = metriques(cle)
        if "erreur" in m:
            st.caption(f"❌ {config['label']} : {m['erreur']}")
        else:
            st.caption(f"✅ {config['label']} : R² = {m['r2']:.2f} (n={m['n_train']+m['n_test']})")

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

st.sidebar.divider()
st.sidebar.subheader("🕒 Conversations")

if st.sidebar.button("＋ Nouvelle conversation", use_container_width=True):
    st.session_state.creating_new = True
    st.session_state.current_session_id = None
    st.session_state.afficher_glossaire = False
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
        icone = "🏢" if meta.get("entreprise") else ("📄" if meta.get("document") else "🧭")
        label = f"{'🟢' if actif else '⚪'} {icone} {meta['title']}"
        if col_titre.button(label, key=f"open_{session_id}", use_container_width=True):
            st.session_state.current_session_id = session_id
            st.session_state.creating_new = False
            st.session_state.afficher_glossaire = False
            st.rerun()
        if col_edit.button("✎", key=f"edit_{session_id}"):
            st.session_state.renaming_id = session_id
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("📖 Glossaire", use_container_width=True, help="Définitions et formules des indicateurs utilisés dans MAF"):
    st.session_state.afficher_glossaire = True
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — nouvelle conversation (simplifiée : document optionnel)
# ─────────────────────────────────────────────────────────────────────────
def ecran_nouvelle_conversation():
    st.title("Nouvelle conversation")
    st.caption(
        "Une conversation est liée à une ENTREPRISE : tu as accès à TOUS les documents "
        "déjà associés à cette entreprise, sans avoir à préciser lequel — et tu peux en "
        "ajouter d'autres à tout moment, même en cours de conversation."
    )

    entreprises_existantes = lister_entreprises()
    mode = st.radio(
        "Entreprise :",
        ["Aucune (conversation générale)", "Entreprise existante", "Nouvelle entreprise"],
        horizontal=True,
    )

    entreprise = None
    if mode == "Entreprise existante":
        if not entreprises_existantes:
            st.info("Aucune entreprise avec des documents associés pour l'instant.")
        else:
            entreprise = st.selectbox("Choisir l'entreprise :", entreprises_existantes)
    elif mode == "Nouvelle entreprise":
        entreprise = st.text_input("Nom de l'entreprise (tel qu'il apparaît dans ta base si possible) :")

    # Association AUTOMATIQUE : dès qu'un nom d'entreprise est connu, on
    # cherche parmi TOUS les documents déjà présents (indexés ou non) ceux
    # dont le nom de fichier correspond — pas de sélection manuelle requise.
    if entreprise and entreprise.strip():
        nouveaux = associer_documents_automatiquement(entreprise.strip())
        if nouveaux:
            st.success(f"📎 Document(s) associé(s) automatiquement : {', '.join(nouveaux)}")

        docs_existants = documents_pour_entreprise(entreprise.strip())
        if docs_existants:
            st.caption(f"📄 {len(docs_existants)} document(s) au total pour {entreprise} :")
            for d in docs_existants:
                st.caption(f"  • {d['titre']}")
        else:
            st.caption("Aucun document trouvé automatiquement pour cette entreprise pour l'instant.")

    st.divider()
    st.markdown("**📎 Ajouter un document** (si pas encore présent dans `documents/`)")
    upload = st.file_uploader("Nouveau PDF", type="pdf", key="upload_new_conv", label_visibility="collapsed")
    if upload is not None and entreprise and st.button("Indexer et associer", key="btn_upload_new_conv"):
        with st.spinner("Indexation en cours..."):
            index_name = process_and_index_pdf(upload)
        associer_document_entreprise(entreprise.strip(), index_name, upload.name.replace(".pdf", ""), upload.name)
        st.success(f"« {upload.name} » associé à {entreprise}.")
        st.rerun()

    if mode != "Aucune (conversation générale)" and not (entreprise and entreprise.strip()):
        st.warning("Précise le nom de l'entreprise avant de continuer.")
        return

    if st.button("Démarrer la conversation"):
        titre = entreprise.strip() if entreprise else f"Conversation {datetime.now().strftime('%d/%m %H:%M')}"
        demarrer_conversation(titre, entreprise=(entreprise.strip() if entreprise else None))


# ─────────────────────────────────────────────────────────────────────────
# Sélecteur manuel d'agents — même design (anneaux) que partout ailleurs
# ─────────────────────────────────────────────────────────────────────────
def selecteur_manuel_agents(document_attache: bool, session_id: str) -> frozenset | None:
    """
    Anneaux compacts + case à cocher par agent disponible. Tout coché par
    défaut (= sélection automatique optimale par l'orchestrateur, aucune
    restriction). Décocher un agent l'exclut de la prochaine question.
    Retourne None si rien n'est restreint (tous cochés), sinon le frozenset
    des agents autorisés.
    """
    agents_disponibles = list(AGENTS_CONFIG.keys())
    if not document_attache:
        agents_disponibles = [a for a in agents_disponibles if a != "agent_document"]

    with st.expander("🎛️ Choisir manuellement les agents disponibles (optionnel)", expanded=False):
        st.caption("Tout coché = l'orchestrateur choisit seul, de façon optimale. Décoche pour restreindre.")
        colonnes = st.columns(len(agents_disponibles))
        selection = set()
        for col, cle in zip(colonnes, agents_disponibles):
            config = AGENTS_CONFIG[cle]
            with col:
                st.plotly_chart(
                    _anneau(config["couleur"], config["icone"], taille=80),
                    use_container_width=False,
                    config={"staticPlot": True},
                    key=f"ring_select_{session_id}_{cle}",
                )
                coche = st.checkbox(config["label"], value=True, key=f"toggle_{session_id}_{cle}")
                if coche:
                    selection.add(cle)

    if not selection or selection == set(agents_disponibles):
        return None  # aucune restriction : comportement optimal par défaut
    return frozenset(selection)


# ─────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — chat (orchestrateur systématique)
# ─────────────────────────────────────────────────────────────────────────
def traiter_message(session_id: str, documents: list, user_input: str, transcript: list, agents_autorises):
    transcript.append({"role": "user", "content": user_input, "sources": None})

    try:
        agent = construire_agent(documents, agents_autorises)
        historique = st.session_state.orchestrateur_histories.setdefault(session_id, [])
        historique.append({"role": "user", "content": user_input})

        nb_avant = len(historique)
        TRACE_DERNIER_TOUR.clear()
        resultat = agent.invoke({"messages": historique})
        st.session_state.orchestrateur_histories[session_id] = resultat["messages"]

        mobilises = agents_mobilises(resultat["messages"], nb_avant)
        sources_detaillees = [
            {
                "agent": entree["agent"],
                "question": entree["question_transmise"],
                "source": formater_source(entree["agent"], entree["resultats_bruts"]),
            }
            for entree in TRACE_DERNIER_TOUR
        ]

        # Graphiques automatiques : si agent_base_donnees a réellement appelé
        # sql_db_score_risque ou sql_db_comparer_entreprises en interne, on
        # recalcule les MÊMES données directement en SQL (pas depuis le texte
        # de l'agent) pour construire un graphique fiable en plus du texte.
        graphiques = []
        for entree in TRACE_DERNIER_TOUR:
            if entree.get("agent") != "agent_base_donnees":
                continue
            for outil in entree.get("outils_internes", []):
                nom_outil = outil.get("outil")
                args = outil.get("args", {})

                if nom_outil == "sql_db_score_risque" and "entreprise" in args:
                    data = donnees_score_risque_graphique(args["entreprise"])
                    if data:
                        graphiques.append({"type": "score_risque", "data": data})

                elif nom_outil == "sql_db_comparer_entreprises" and "entreprises" in args:
                    noms = [n.strip() for n in args["entreprises"].split(",") if n.strip()]
                    data = donnees_comparaison_graphique(noms)
                    if data:
                        graphiques.append({"type": "comparaison", "data": data})

        reponse = extraire_texte(resultat["messages"][-1].content)
        transcript.append({
            "role": "assistant",
            "type": "orchestrateur",
            "content": reponse,
            "badges": mobilises,
            "sources_detaillees": sources_detaillees,
            "graphiques": graphiques,
        })

    except Exception as e:
        message = str(e)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            contenu = "🚫 **Quota API atteint** (Gemini, ou Tavily si l'agent web a été sollicité). Attends la réinitialisation."
        else:
            contenu = f"❌ Erreur : {message[:300]}"
        transcript.append({"role": "assistant", "type": "info", "content": contenu, "sources": None})

    sauver_transcript(session_id, transcript)
    conversations[session_id]["updated_at"] = datetime.now().isoformat()
    sauver_index_conversations(conversations)


def afficher_tour(tour: dict, cle_unique: str = "tour"):
    type_tour = tour.get("type")

    if type_tour == "due_diligence":
        st.markdown(f"### 🔎 Due diligence — {tour['entreprise']}")

        radar_data = tour.get("radar_data")
        if radar_data:
            st.plotly_chart(_radar_entreprise(radar_data), use_container_width=True, key=f"{cle_unique}_radar_dd")
            if radar_data.get("colonnes_manquantes"):
                st.caption(
                    f"⚠️ Axes non affichés (colonnes introuvables) : "
                    f"{', '.join(radar_data['colonnes_manquantes'])}"
                )
        else:
            st.caption("📊 Radar non disponible (entreprise non résolue avec certitude dans la base).")

        for section in tour["rapport"]:
            with st.expander(f"**{section['categorie']}**", expanded=False):
                if section.get("agents_mobilises"):
                    st.caption(" · ".join(section["agents_mobilises"]))
                st.markdown(section["reponse"])
                if section.get("sources"):
                    st.divider()
                    for entree in section["sources"]:
                        config = AGENTS_CONFIG.get(entree["agent"], {})
                        st.caption(f"{config.get('icone', '🔧')} {config.get('label', entree['agent'])}")
                        st.code(entree["source"], language=None)

    elif type_tour == "orchestrateur":
        if tour.get("badges"):
            st.caption(" · ".join(tour["badges"]))
        st.markdown(tour["content"])

        for i, graphique in enumerate(tour.get("graphiques", [])):
            if graphique["type"] == "score_risque":
                st.plotly_chart(_camembert_score_risque(graphique["data"]), use_container_width=True, key=f"{cle_unique}_graphique_{i}")
            elif graphique["type"] == "comparaison":
                st.plotly_chart(_batons_comparaison(graphique["data"]), use_container_width=True, key=f"{cle_unique}_graphique_{i}")

        if tour.get("sources_detaillees"):
            with st.expander(f"📎 {len(tour['sources_detaillees'])} source(s) détaillée(s)"):
                for entree in tour["sources_detaillees"]:
                    config = AGENTS_CONFIG.get(entree["agent"], {})
                    st.markdown(f"**{config.get('icone', '🔧')} {config.get('label', entree['agent'])}**")
                    st.caption(f"Question transmise : {entree['question']}")
                    st.code(entree["source"], language=None)

    # Branches conservées pour l'affichage rétrocompatible des anciennes
    # conversations (créées avant le passage à l'orchestrateur unique).
    elif type_tour == "prediction":
        st.markdown("**📊 Réponse issue du modèle prédictif (ancienne conversation)**")
        st.metric(label=tour.get("label", "Indicateur estimé"), value=f"{tour.get('valeur', 0):.2f}%")
    elif type_tour == "rag":
        st.markdown("**📄 Réponse basée sur le document (ancienne conversation)**")
        st.markdown(tour["content"])
    elif type_tour == "sql_agent":
        st.markdown("**🗄️ Réponse de l'agent base de données (ancienne conversation)**")
        st.markdown(tour["content"])
    elif type_tour == "web_recherche":
        st.markdown("**🔍 Réponse de l'agent de recherche web (ancienne conversation)**")
        st.markdown(tour["content"])

    else:
        st.markdown(tour["content"])


def ecran_chat(session_id: str):
    meta = conversations[session_id]
    entreprise = meta.get("entreprise")
    documents = documents_de_la_conversation(meta)

    if entreprise:
        st.title(f"🏢 {meta['title']}")
        if documents:
            st.caption(
                f"Entreprise : {entreprise} — {len(documents)} document(s) attaché(s) : "
                + ", ".join(d["titre"] for d in documents)
            )
        else:
            st.caption(f"Entreprise : {entreprise} — aucun document attaché pour l'instant.")
    elif documents:
        st.title(f"📄 {meta['title']}")
        st.caption(f"Document attaché : {documents[0]['titre']} — orchestrateur multi-agent actif.")
    else:
        st.title(f"🧭 {meta['title']}")
        st.caption("Orchestrateur multi-agent — base de données, estimation, recherche web.")

    transcript = charger_transcript(session_id)

    for idx, tour in enumerate(transcript):
        with st.chat_message("user" if tour["role"] == "user" else "assistant"):
            afficher_tour(tour, cle_unique=f"{session_id}_tour{idx}")

    # ── Menu "➕" — regroupe tous les outils annexes juste au-dessus de la
    # saisie, au lieu de 4 blocs empilés qui prenaient toute la largeur de
    # l'écran. Streamlit ne permet pas de vrai bouton accolé au champ de
    # saisie lui-même (st.chat_input est un widget à part, plein largeur) —
    # le popover juste au-dessus est l'équivalent le plus proche.
    with st.popover("➕ Outils", use_container_width=False):
        if st.button("📖 Glossaire", key=f"popover_glossaire_{session_id}", use_container_width=True):
            st.session_state.afficher_glossaire = True
            st.rerun()

        st.divider()

        if entreprise:
            with st.expander("📎 Ajouter un document"):
                st.caption(
                    f"Immédiatement disponible dans cette conversation et toute autre "
                    f"conversation liée à {entreprise}."
                )
                nouveau_doc = st.file_uploader("Nouveau PDF", type="pdf", key=f"ajout_doc_{session_id}")
                if nouveau_doc and st.button("Indexer et associer", key=f"btn_ajout_doc_{session_id}"):
                    with st.spinner("Indexation en cours..."):
                        index_name = process_and_index_pdf(nouveau_doc)
                    associer_document_entreprise(
                        entreprise, index_name, nouveau_doc.name.replace(".pdf", ""), nouveau_doc.name,
                    )
                    st.success(f"« {nouveau_doc.name} » associé à {entreprise}.")
                    st.rerun()

        if documents:
            with st.expander("📥 Extraire des données vers la base"):
                st.caption(
                    "Analyse un document et propose d'ajouter/enrichir la ligne de cette "
                    "entreprise dans la base, avec traçabilité par page. Rien n'est "
                    "enregistré sans validation."
                )
                if len(documents) == 1:
                    document_a_extraire = documents[0]
                else:
                    titre_choisi = st.selectbox(
                        "Quel document extraire ?", [d["titre"] for d in documents], key=f"choix_extraction_{session_id}",
                    )
                    document_a_extraire = next(d for d in documents if d["titre"] == titre_choisi)

                if st.button("🔍 Lancer l'extraction", key="btn_lancer_extraction"):
                    with st.spinner("Extraction en cours..."):
                        try:
                            st.session_state["extraction_en_attente"] = extraire_donnees_du_document(document_a_extraire["index_name"])
                        except Exception as e:
                            st.error(f"Erreur d'extraction : {e}")

                extraction_en_attente = st.session_state.get("extraction_en_attente")
                if extraction_en_attente:
                    st.write("**Aperçu — vérifie avant d'enregistrer :**")
                    st.json(extraction_en_attente.model_dump())
                    col_ok, col_annuler = st.columns(2)
                    if col_ok.button("✅ Enregistrer", key="btn_confirmer_extraction"):
                        resultat = enregistrer_extraction(extraction_en_attente)
                        with st.spinner("Réentraînement des modèles..."):
                            retrain_tous()
                        st.success(f"Entreprise {resultat['action']} : {resultat['entreprise']}")
                        st.session_state["extraction_en_attente"] = None
                    if col_annuler.button("✕ Annuler", key="btn_annuler_extraction"):
                        st.session_state["extraction_en_attente"] = None

        with st.expander("🔎 Due diligence structurée"):
            st.caption(
                "Checklist fixe (rentabilité, endettement, liquidité/risque, valorisation, "
                "tendance, actualité" + (", cohérence documents" if documents else "") +
                ") plutôt que question par question."
            )
            if entreprise:
                entreprise_dd = entreprise
                st.caption(f"Entreprise : **{entreprise}**")
            else:
                entreprise_dd = st.text_input(
                    "Nom exact de l'entreprise :", key=f"entreprise_dd_{session_id}",
                )
            if st.button("🚀 Lancer la due diligence", key=f"btn_dd_{session_id}", disabled=not entreprise_dd.strip()):
                with st.spinner(f"Due diligence en cours pour {entreprise_dd}… (1-2 minutes)"):
                    try:
                        rapport = lancer_due_diligence(entreprise_dd.strip(), documents)
                        radar_data = donnees_radar_entreprise(entreprise_dd.strip())
                        transcript_tmp = charger_transcript(session_id)
                        transcript_tmp.append({
                            "role": "assistant",
                            "type": "due_diligence",
                            "entreprise": entreprise_dd.strip(),
                            "rapport": rapport,
                            "radar_data": radar_data,
                        })
                        sauver_transcript(session_id, transcript_tmp)
                        conversations[session_id]["updated_at"] = datetime.now().isoformat()
                        sauver_index_conversations(conversations)
                    except Exception as e:
                        st.error(f"Erreur pendant la due diligence : {e}")
                st.rerun()

        with st.expander("📊 Radar rapide (sans due diligence complète)"):
            if entreprise:
                entreprise_radar = entreprise
                st.caption(f"Entreprise : **{entreprise}**")
            else:
                entreprise_radar = st.text_input(
                    "Nom de l'entreprise :", key=f"entreprise_radar_{session_id}",
                )
            if st.button("Afficher le radar", key=f"btn_radar_{session_id}", disabled=not entreprise_radar.strip()):
                radar_data = donnees_radar_entreprise(entreprise_radar.strip())
                if radar_data is None:
                    st.error(f"Entreprise '{entreprise_radar}' non trouvée avec certitude.")
                else:
                    st.plotly_chart(_radar_entreprise(radar_data), use_container_width=True, key=f"radar_rapide_{session_id}")
                    if radar_data.get("colonnes_manquantes"):
                        st.caption(f"⚠️ Axes non affichés : {', '.join(radar_data['colonnes_manquantes'])}")

        agents_autorises = selecteur_manuel_agents(document_attache=len(documents) > 0, session_id=session_id)

    if prompt := st.chat_input("Posez votre question :"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion (sélection des agents nécessaires)..."):
                traiter_message(session_id, documents, prompt, transcript, agents_autorises)
            afficher_tour(transcript[-1], cle_unique=f"{session_id}_tour{len(transcript)-1}_nouveau")


# ─────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE — glossaire (page annexe, définitions + formules)
# ─────────────────────────────────────────────────────────────────────────
def ecran_glossaire():
    st.title("📖 Glossaire")
    st.caption(
        "Définition et formule mathématique de chaque indicateur utilisé dans MAF — "
        "à consulter en cas de curiosité, sans lien avec tes conversations."
    )

    if st.button("← Retour"):
        st.session_state.afficher_glossaire = False
        st.rerun()

    st.divider()

    # Regroupement dynamique par partie, en conservant l'ordre d'apparition
    # (pas besoin de trier : les parties sont déjà listées dans le bon ordre
    # dans Glossaire.py, catégorie par catégorie).
    parties_ordre = []
    parties_categories = {}
    for section in GLOSSAIRE:
        partie = section.get("partie", "Autres")
        if partie not in parties_categories:
            parties_ordre.append(partie)
            parties_categories[partie] = []
        parties_categories[partie].append(section)

    for partie in parties_ordre:
        st.markdown(f"# {partie}")
        for section in parties_categories[partie]:
            st.markdown(f"## {section['categorie']}")
            for terme in section["termes"]:
                with st.expander(f"**{terme['nom']}**"):
                    st.markdown(terme["definition"])
                    if terme.get("formule"):
                        st.latex(terme["formule"])
                    if terme.get("note"):
                        st.caption(terme["note"])
                    if terme.get("explication"):
                        st.markdown("---")
                        st.markdown(terme["explication"])
            st.divider()


# ─────────────────────────────────────────────────────────────────────────
# Routage principal
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.afficher_glossaire:
    ecran_glossaire()
elif st.session_state.creating_new or st.session_state.current_session_id is None:
    ecran_nouvelle_conversation()
else:
    ecran_chat(st.session_state.current_session_id)