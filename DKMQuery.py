import os
import sys
import json
from pathlib import Path
from datetime import date
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_core.runnables.history import RunnableWithMessageHistory

# Clé API Google Gemini lue depuis .env — jamais codée en dur
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- Outil de prédiction (function calling) ---
from predictor import OUTILS_PREDICTION

CONTEXTE_MACRO_PATH = Path("contexte_macro.json")


def charger_contexte_macro() -> str:
    """
    Charge contexte_macro.json et le formate en texte pour le system prompt.
    Fichier à mettre à jour manuellement (voir champ 'a_verifier_apres') —
    c'est une photographie ponctuelle, pas une donnée temps réel.
    """
    if not CONTEXTE_MACRO_PATH.exists():
        return "Aucun contexte macroéconomique disponible (fichier contexte_macro.json absent)."

    try:
        data = json.loads(CONTEXTE_MACRO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return "Contexte macroéconomique indisponible (fichier illisible)."

    date_maj = data.get("date_maj", "date inconnue")
    a_verifier = data.get("a_verifier_apres")
    if a_verifier and date.today().isoformat() > a_verifier:
        alerte = f"⚠️ CE CONTEXTE MACRO DATE DU {date_maj} ET N'A PAS ÉTÉ MIS À JOUR DEPUIS — À ACTUALISER.\n"
    else:
        alerte = ""

    ze = data.get("zone_euro", {})
    us = data.get("etats_unis", {})

    return (
        f"{alerte}"
        f"(Photographie du {date_maj})\n"
        f"Zone euro : taux BCE refi {ze.get('taux_refi_bce', '?')}, dépôt {ze.get('taux_depot_bce', '?')}, "
        f"inflation projetée {ze.get('inflation_projetee_2026', '?')} en 2026. {ze.get('contexte', '')}\n"
        f"États-Unis : taux Fed {us.get('taux_fed_fourchette', '?')}, "
        f"inflation PCE projetée {us.get('inflation_pce_projetee_2026', '?')} en 2026. {us.get('contexte', '')}\n"
        f"Tendance générale : {data.get('tendances_generales', '')}"
    )

INDEXES_DIR = Path("faiss_indexes")
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)


# 1. Menu de sélection de l'index (CLI uniquement)
def menu_selection_index():
    if not INDEXES_DIR.exists():
        print("❌ Le dossier des index n'existe pas. Lancez DemandingIngest.py en premier.")
        sys.exit(1)

    index_dispos = [d for d in os.listdir(INDEXES_DIR) if (INDEXES_DIR / d).is_dir()]

    if not index_dispos:
        print("⚠️ Aucun index trouvé. Veuillez d'abord utiliser DemandingIngest.py pour préparer un document.")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("📚 SÉLECTION DE LA BASE DE CONNAISSANCES")
    print("=" * 40)
    for i, idx in enumerate(index_dispos, 1):
        nom_propre = idx.replace("_index", "")
        print(f" [{i}] - Analyser : {nom_propre}")

    choix = input("\n👉 Entrez le numéro du document à analyser (ou 'q' pour quitter) : ").strip()

    if choix.lower() == 'q':
        sys.exit(0)
    elif choix.isdigit() and 1 <= int(choix) <= len(index_dispos):
        return index_dispos[int(choix) - 1]
    else:
        print("❌ Choix invalide.")
        sys.exit(1)


# 2. Classe de Mémoire — importée telle quelle par App.py
class FichierDchatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.filepath = HISTORY_DIR / f"{session_id}.json"
        self.messages = []
        self.load_messages()

    def load_messages(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.messages = messages_from_dict(json.load(f))
            except Exception:
                pass

    def save_messages(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(self.messages), f, ensure_ascii=False, indent=2)

    def add_message(self, message):
        super().add_message(message)
        self.save_messages()

    def clear(self):
        self.messages = []
        if self.filepath.exists():
            self.filepath.unlink()


def obtenir_historique_persistant(session_id: str) -> BaseChatMessageHistory:
    return FichierDchatHistory(session_id)


# 3. Classe DKMQuery — point d'entrée utilisé par App.py
class DKMQuery:
    """
    Construit et met en cache le routeur (function calling) et la chaîne RAG
    conversationnelle pour un index donné. Les objets coûteux (embeddings,
    LLM) ne sont instanciés qu'une seule fois par processus : comme Python
    n'importe un module qu'une fois, ce cache survit aux reruns Streamlit
    sans avoir besoin de st.cache_resource ici — DKMQuery.py reste utilisable
    tel quel en CLI ou dans d'autres contextes.
    """
    _embeddings = None
    _llm = None
    _llm_routeur = None
    _chains_cache: dict = {}

    @classmethod
    def _get_embeddings(cls):
        if cls._embeddings is None:
            cls._embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return cls._embeddings

    @classmethod
    def _get_llm(cls):
        if cls._llm is None:
            cls._llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
        return cls._llm

    @classmethod
    def _get_llm_routeur(cls):
        if cls._llm_routeur is None:
            routeur_base = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
            cls._llm_routeur = routeur_base.bind_tools(OUTILS_PREDICTION)
        return cls._llm_routeur

    @classmethod
    def _build_chain(cls, index_name: str):
        embeddings = cls._get_embeddings()
        chemin_index = INDEXES_DIR / index_name
        vector_store = FAISS.load_local(str(chemin_index), embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = cls._get_llm()

        system_prompt = (
            "Tu es MAF (Mon Analyseur Financier), une intelligence artificielle spécialisée "
            "en analyse financière et évaluation d'entreprises. Ton architecture combine une "
            "analyse documentaire rigoureuse (RAG) et des modèles de prédiction quantitative.\n\n"
            "Tes directives opérationnelles :\n\n"
            "1. RIGUEUR ANALYTIQUE : chaque extrait de contexte ci-dessous est précédé de sa "
            "page source au format [Page X]. Cite systématiquement cette page entre parenthèses "
            "à la fin de chaque affirmation factuelle, ex. \"Le CET1 ratio est de 15,2% (p.47).\" "
            "Si une information n'est disponible dans aucun extrait fourni, dis-le explicitement "
            "plutôt que de l'inférer ou de l'estimer toi-même.\n\n"
            "2. SCEPTICISME PROFESSIONNEL : tu ne te contentes pas d'extraire des chiffres, tu "
            "les mets en perspective par rapport au contexte macroéconomique rappelé ci-dessous. "
            "Si une donnée du document te semble en décalage avec ce contexte (ex. une marge en "
            "forte hausse malgré un environnement de taux élevés et d'inflation persistante), "
            "signale-le avec prudence — comme une question à creuser, jamais comme une certitude, "
            "car tu n'as pas de base de comparaison sectorielle systématique à ce stade du projet.\n\n"
            "3. GLOSSAIRE & FORMULES (mode expert) : si la question porte sur un indicateur "
            "financier (BFR, WACC, multiple d'EBITDA, ROE, EV/EBITDA, etc.), donne systématiquement, "
            "même si le document n'en parle pas : (a) la définition précise, (b) la formule de "
            "calcul standard utilisée en analyse financière, (c) l'interprétation stratégique de "
            "cet indicateur.\n\n"
            "4. CONTEXTE MACROÉCONOMIQUE (photographie ponctuelle, à ne jamais présenter comme "
            "temps réel) :\n"
            f"{charger_contexte_macro()}\n\n"
            "Ton ton : professionnel, concis, analytique, orienté vers l'aide à la décision — sans "
            "jamais présenter une estimation statistique comme une donnée vérifiée.\n\n"
            "Utilise également l'historique de vos échanges passés pour faire des liens pertinents "
            "si nécessaire.\n\n"
            "Contexte extrait du document :\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        document_prompt = PromptTemplate.from_template("[Page {page}] {page_content}")
        qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        return RunnableWithMessageHistory(
            rag_chain,
            obtenir_historique_persistant,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    @classmethod
    def get_chain_and_router(cls, index_name: str, session_id: str):
        """Retourne (llm_routeur, conversational_rag_chain) pour un index donné."""
        if index_name not in cls._chains_cache:
            cls._chains_cache[index_name] = cls._build_chain(index_name)
        return cls._get_llm_routeur(), cls._chains_cache[index_name]

    @classmethod
    def get_router(cls):
        """
        Retourne uniquement le routeur, sans chaîne RAG — utilisé pour les
        conversations d'estimation qui n'ont aucun document/index associé.
        """
        return cls._get_llm_routeur()

    @classmethod
    def get_llm(cls):
        """
        LLM nu (sans outils liés), pour des appels ponctuels hors RAG —
        par exemple générer une explication qualitative d'une prédiction.
        """
        return cls._get_llm()


# 4. Programme principal (CLI, inchangé dans l'esprit)
if __name__ == "__main__":
    dossier_cible = menu_selection_index()
    session_id = f"session_{dossier_cible}"

    print("🔍 Chargement des ressources...", flush=True)
    llm_routeur, conversational_rag_chain = DKMQuery.get_chain_and_router(dossier_cible, session_id)

    print(f"\n🤖 L'agent est prêt ! Base active : {dossier_cible.replace('_index', '')}")
    print("💾 Historique dédié activé. Tapez 'exit' pour quitter.", flush=True)

    while True:
        try:
            user_input = input("\n👤 Vous : ")
            if user_input.lower() == 'exit':
                print("Fin de la session. Mémoire sauvegardée !", flush=True)
                break
            if not user_input.strip():
                continue

            print("🤖 Réflexion...", flush=True)
            routage = llm_routeur.invoke(user_input)

            if routage.tool_calls:
                for appel in routage.tool_calls:
                    outil = next((t for t in OUTILS_PREDICTION if t.name == appel["name"]), None)
                    if outil:
                        resultat = outil.invoke(appel["args"])
                        print(f"\n💡 IA (prédiction) :\n{resultat}", flush=True)
            else:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                print(f"\n💡 IA :\n{response['answer']}", flush=True)

        except KeyboardInterrupt:
            print("\nFin de la session. Mémoire sauvegardée !", flush=True)
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}", flush=True)