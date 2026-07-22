"""
Monitor.py — Superviseur multi-agent pour MAF (Mon Analyseur Financier)

Sélectionne dynamiquement, parmi les agents déjà construits et testés
individuellement (SQLAgent.py, WebResearch.py, PredictionAgent.py,
TechnicalAnalysis.py, DueDiligenceAgent.py), ceux qui sont réellement nécessaires pour répondre à
une question — jamais tous par défaut. Réutilise le pattern "agents-as-
tools" déjà éprouvé : chaque agent spécialiste garde son propre system
prompt et son propre raisonnement, le superviseur ne fait aucun calcul
lui-même, il délègue et synthétise.

⚠️ Sur l'optimalité : ce script IMPOSE la contrainte par consigne (system
prompt), pas par un algorithme qui garantit mathématiquement un sous-
ensemble minimal d'agents. C'est une différence importante à assumer dans
un rapport de stage — le LLM peut se tromper de sélection, le prompt
diminue le risque mais ne l'élimine pas.

À TESTER SEUL (python Monitor.py) avant toute intégration dans App.py.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

from SQLAgent import agent as agent_base_donnees, extraire_texte
from WebResearch import agent as agent_recherche_web
from PredictionAgent import OUTILS_PREDICTION, OUTILS_SCENARIOS
from TechnicalAnalysis import analyser_momentum_action
from DKMQuery import charger_contexte_macro
from DueDiligenceAgent import executer_due_diligence

model = init_chat_model("google_genai:gemini-3.1-flash-lite")
INDEXES_DIR = Path("faiss_indexes")


# ─────────────────────────────────────────────────────────────────────────
# Agent spécialiste "estimation"
# ─────────────────────────────────────────────────────────────────────────
PROMPT_ESTIMATION = """Tu es le spécialiste prédiction de MAF. Tu disposes de
modèles Random Forest ENTRAÎNÉS ET VALIDÉS (R²/MAE mesurés sur un panel réel)
pour ESTIMER un indicateur à partir de paramètres fournis par l'utilisateur.
Précise toujours qu'il s'agit d'une estimation statistique sur un échantillon
restreint, jamais d'une donnée vérifiée.

Si l'utilisateur demande PLUSIEURS scénarios (ex. "-10%, -20%, -30%" ou
"si le CA variait de X à Y"), utilise IMPÉRATIVEMENT un outil simuler_*
plutôt que d'appeler plusieurs fois predict_* et de faire le calcul
toi-même — le calcul de simuler_* est garanti exact, le tien ne l'est pas.

Si un résultat porte une alerte "hors du domaine plausible", REPRODUIS cette
alerte explicitement dans ta réponse — ne présente jamais un chiffre
aberrant (ex. marge nette de 400%) comme s'il était fiable."""

agent_estimation = create_agent(model, OUTILS_PREDICTION + OUTILS_SCENARIOS, system_prompt=PROMPT_ESTIMATION)


# ─────────────────────────────────────────────────────────────────────────
# Agent spécialiste "technique" — retiré de SQLAgent.py pour éviter qu'une
# lecture de tendance boursière soit étiquetée à tort comme "donnée réelle
# en base" (l'orchestrateur étiquette par NOM d'agent sollicité, pas par
# outil interne réellement utilisé — d'où la nécessité de séparer).
# ─────────────────────────────────────────────────────────────────────────
PROMPT_TECHNIQUE = """Tu es le spécialiste analyse technique de MAF. Tu
calcules le momentum et l'accélération d'un cours boursier à partir de son
historique récent. C'est un indicateur DESCRIPTIF de tendance de marché,
JAMAIS un modèle validé statistiquement — précise-le toujours."""

agent_technique = create_agent(model, [analyser_momentum_action], system_prompt=PROMPT_TECHNIQUE)


# ─────────────────────────────────────────────────────────────────────────
# Agent spécialiste "document" — dynamique
# ─────────────────────────────────────────────────────────────────────────
_embeddings_rag = None

def _get_embeddings():
    global _embeddings_rag
    if _embeddings_rag is None:
        _embeddings_rag = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    return _embeddings_rag


class ArgsRecherche(BaseModel):
    requete: str = Field(description="Ce que tu cherches dans le document (mots-clés ou question précise)")


def _construire_agent_document(index_name: str):
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        str(INDEXES_DIR / index_name), embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    def _fonction(requete: str) -> str:
        docs = retriever.invoke(requete)
        if not docs:
            return "Aucun passage pertinent trouvé dans le document pour cette recherche."
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in docs
        )

    outil_document = StructuredTool.from_function(
        func=_fonction, name="rechercher_dans_document",
        description="Cherche un passage dans le document PDF attaché à cette session.",
        args_schema=ArgsRecherche,
    )

    prompt_document = """Tu es le spécialiste documentaire de MAF. Tu lis le
PDF attaché via rechercher_dans_document. Cite TOUJOURS la page source (p.X)
pour chaque affirmation factuelle. Si l'info n'y est pas, dis-le explicitement."""

    return create_agent(model, [outil_document], system_prompt=prompt_document)


def _choisir_document() -> str | None:
    if not INDEXES_DIR.exists():
        return None
    index_dispos = [d for d in os.listdir(INDEXES_DIR) if (INDEXES_DIR / d).is_dir()]
    if not index_dispos:
        return None

    print("\n📚 Document à attacher à cette session (optionnel) :")
    for i, idx in enumerate(index_dispos, 1):
        print(f" [{i}] {idx.replace('_index', '')}")
    print(" [0] Aucun document — orchestrateur sans agent documentaire")

    choix = input("👉 Choix : ").strip()
    if choix.isdigit() and 1 <= int(choix) <= len(index_dispos):
        return index_dispos[int(choix) - 1]
    return None


# ─────────────────────────────────────────────────────────────────────────
# Encapsulation des agents en outils du superviseur (agents-as-tools)
# ─────────────────────────────────────────────────────────────────────────
class ArgsQuestion(BaseModel):
    question: str = Field(description="La question ou tâche à transmettre au spécialiste, en langage naturel")


TRACE_DERNIER_TOUR: list = []


def _extraire_urls(texte: str) -> list:
    return re.findall(r'https?://[^\s\'",\]\)]+', texte)


def formater_source(nom_agent: str, resultats_bruts: list) -> str:
    """
    Formate la source selon le type d'agent :
    - agent_document      -> segment/chunk avec numéro de page
    - agent_base_donnees  -> extrait brut de la table (résultat SQL)
    - agent_recherche_web -> URLs extraites du résultat Tavily
    - agent_technique     -> indicateur de tendance calculé (pas une "source" au sens documentaire)
    """
    if nom_agent == "agent_recherche_web":
        urls = []
        for r in resultats_bruts:
            urls.extend(_extraire_urls(r))
        if urls:
            urls_uniques = list(dict.fromkeys(urls))
            return "Sources internet :\n" + "\n".join(f"  - {u}" for u in urls_uniques)
        return "Aucune URL identifiable dans le résultat brut :\n" + "\n".join(resultats_bruts)

    if nom_agent == "agent_document":
        return "Segment(s) du document :\n" + "\n".join(f"  {r}" for r in resultats_bruts)

    if nom_agent == "agent_base_donnees":
        return "Extrait de la table (résultat SQL brut) :\n" + "\n".join(f"  {r}" for r in resultats_bruts)

    if nom_agent == "agent_technique":
        return "Indicateur de tendance calculé :\n" + "\n".join(f"  {r}" for r in resultats_bruts)

    return "\n".join(resultats_bruts)


def _agent_vers_outil(nom: str, description: str, agent_specialise) -> StructuredTool:
    def _fonction(question: str) -> str:
        resultat = agent_specialise.invoke({"messages": [{"role": "user", "content": question}]})
        messages_internes = resultat["messages"]

        outils_internes = []
        for msg in messages_internes:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for appel in tool_calls:
                    outils_internes.append({"outil": appel.get("name"), "args": appel.get("args")})

        resultats_bruts = [
            extraire_texte(msg.content)[:2000]
            for msg in messages_internes
            if isinstance(msg, ToolMessage)
        ]

        TRACE_DERNIER_TOUR.append({
            "agent": nom,
            "question_transmise": question,
            "outils_internes": outils_internes,
            "resultats_bruts": resultats_bruts,
        })

        return extraire_texte(messages_internes[-1].content)

    return StructuredTool.from_function(
        func=_fonction, name=nom, description=description, args_schema=ArgsQuestion
    )


def construire_outils(index_name: str | None = None, agents_autorises: set | None = None) -> list:
    """
    agents_autorises : None = tous les agents disponibles (sélection
    automatique optimale par l'orchestrateur). Sinon, un sous-ensemble de
    noms d'agents — restriction manuelle décidée par l'utilisateur.
    """
    candidats = [
        ("agent_base_donnees", _agent_vers_outil(
            "agent_base_donnees",
            "Interroge une base SQL réelle de ~180 entreprises collectées : comparaisons "
            "entre entreprises, statistiques sectorielles, détection de valeurs atypiques, "
            "score de risque composite. Ne fait NI prédiction ML NI analyse technique.",
            agent_base_donnees,
        )),
        ("agent_estimation", _agent_vers_outil(
            "agent_estimation",
            "Modèles statistiques ENTRAÎNÉS ET VALIDÉS pour ESTIMER un indicateur (marge "
            "nette, ROE, ROA, croissance du CA) à partir de paramètres hypothétiques fournis "
            "par l'utilisateur — pas pour relire une donnée déjà connue.",
            agent_estimation,
        )),
        ("agent_technique", _agent_vers_outil(
            "agent_technique",
            "Analyse la tendance récente (momentum, accélération) du cours boursier d'un "
            "ticker donné. Indicateur descriptif, non validé statistiquement.",
            agent_technique,
        )),
        ("agent_recherche_web", _agent_vers_outil(
            "agent_recherche_web",
            "Recherche sur internet (Tavily) une information récente, externe ou d'actualité "
            "(taux macroéconomiques actuels, actualité d'une entreprise, événement récent) "
            "avec citation systématique de la source.",
            agent_recherche_web,
        )),
    ]

    if index_name:
        agent_document = _construire_agent_document(index_name)
        candidats.append(("agent_document", _agent_vers_outil(
            "agent_document",
            "Cherche une information dans le document PDF attaché à cette session, "
            "avec citation de page systématique.",
            agent_document,
        )))

    if agents_autorises:
        return [outil for cle, outil in candidats if cle in agents_autorises]
    return [outil for _, outil in candidats]


def construire_system_prompt(document_attache: bool, due_diligence_disponible: bool = True) -> str:
    liste_agents = (
        "- agent_base_donnees : données réelles d'un panel de ~180 entreprises (SQL pur).\n"
        "- agent_estimation : estimations statistiques (modèles ML entraînés et validés).\n"
        "- agent_technique : lecture de tendance boursière (momentum), indicateur descriptif.\n"
        "- agent_recherche_web : information externe récente, avec sources citées.\n"
    )
    if document_attache:
        liste_agents += "- agent_document : contenu du PDF attaché à cette session, avec pages citées.\n"
    if due_diligence_disponible:
        liste_agents += (
            "- lancer_due_diligence : exécute une checklist COMPLÈTE (rentabilité, "
            "endettement, liquidité/risque, valorisation, tendance, actualité) pour une "
            "entreprise — coûteux (6-7 appels), à réserver aux demandes explicitement "
            "larges (\"analyse complète\", \"due diligence\", \"fais-moi un état des lieux "
            "de X\"). Pour une question ciblée sur un seul aspect, utilise directement "
            "l'agent spécialisé concerné — plus rapide, moins d'appels API, toujours "
            "préférable quand ça suffit (principe d'optimalité).\n"
        )

    return f"""
Tu es MAF-Orchestrateur, le superviseur qui sélectionne les agents
spécialistes nécessaires pour répondre à une question financière.

Agents disponibles :
{liste_agents}
PRINCIPE D'OPTIMALITÉ (règle impérative) :
- N'appelle QUE les agents réellement nécessaires pour répondre à CETTE
  question précise — jamais "par défaut" ou "au cas où".
- Si un seul agent suffit, n'en appelle qu'un seul.
- Si aucun agent n'est nécessaire (question générale, définition financière
  que tu connais déjà), réponds directement sans déléguer.
- Ne réappelle jamais un agent déjà interrogé pour la même sous-question
  dans le même tour.
- Si l'utilisateur a restreint manuellement les agents disponibles (liste
  ci-dessus déjà filtrée), tu ne peux choisir que parmi eux — si aucun ne
  convient à la question, dis-le explicitement plutôt que de forcer une
  réponse hors sujet.
- Dans le cadre d'un document, commence par vérifier si l'information 
  souhaitée y est avant de lancer une recherche internet via l'agent web. 
  De même, si l'information est une donnée, regarde aussi dans la base de 
  donnée avant de lancer un appel à l'agent web. l'agent est l'arme de dernier 
  secours il n'intervient qu'en cas d'information manquantes ou mise en 
  corrélation avec une situation actuelle, politique, géopolitique, etc

TRANSPARENCE (règle impérative) :
- Chaque affirmation ou paragraphe de ta réponse qui provient d'un agent
  DOIT être précédé d'une étiquette entre crochets indiquant sa source,
  directement dans le texte. Étiquettes à utiliser : [🗄️ Base de données],
  [📊 Estimation], [📈 Analyse technique], [🔍 Recherche web], [📄 Document].
- Une phrase de synthèse SANS étiquette est acceptable en conclusion, si
  elle ne fait qu'articuler les éléments déjà étiquetés sans ajouter de
  nouvelle affirmation factuelle.
- Ne laisse JAMAIS une affirmation chiffrée ou factuelle sans étiquette de
  provenance — c'est une règle stricte, pas une suggestion.

SCEPTICISME PROFESSIONNEL : mets les chiffres en perspective par rapport au
contexte macroéconomique suivant (photographie ponctuelle, jamais présentée
comme temps réel) :
{charger_contexte_macro()}

GLOSSAIRE & FORMULES (mode expert) : si la question porte sur un indicateur
financier (BFR, WACC, EV/EBITDA, ROE...), donne systématiquement définition,
formule de calcul standard, et interprétation stratégique — même sans
déléguer à un agent pour cela.

Ton ton : professionnel, concis, analytique, orienté aide à la décision.
"""


_agents_cache: dict = {}


def construire_agent(
    index_name: str | None = None,
    agents_autorises: frozenset | None = None,
    _inclure_due_diligence: bool = True,
):
    """
    _inclure_due_diligence=False est réservé à l'usage INTERNE de
    lancer_due_diligence() — sans ce garde-fou, l'agent utilisé PENDANT une
    due diligence pourrait se relancer lui-même en due diligence, créant une
    récursion infinie. L'agent principal (celui qui parle à l'utilisateur)
    garde toujours l'outil.
    """
    cle_cache = (index_name, agents_autorises, _inclure_due_diligence)
    if cle_cache not in _agents_cache:
        outils = construire_outils(index_name, agents_autorises)
        if _inclure_due_diligence:
            outils = outils + [_outil_due_diligence(index_name)]
        prompt = construire_system_prompt(
            document_attache=index_name is not None,
            due_diligence_disponible=_inclure_due_diligence,
        )
        _agents_cache[cle_cache] = create_agent(model, outils, system_prompt=prompt)
    return _agents_cache[cle_cache]


# ─────────────────────────────────────────────────────────────────────────
# Aide à l'affichage
# ─────────────────────────────────────────────────────────────────────────
NOMS_AGENTS = {
    "agent_base_donnees": "🗄️ Agent Base de données",
    "agent_estimation": "📊 Agent Estimation",
    "agent_technique": "📈 Agent Technique",
    "agent_recherche_web": "🔍 Agent Recherche web",
    "agent_document": "📄 Agent Document",
    "lancer_due_diligence": "🔎 Due Diligence Complète",
}


def agents_mobilises(messages: list, nb_avant: int) -> list:
    compteur = {}
    for msg in messages[nb_avant:]:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for appel in tool_calls:
                nom = appel.get("name", "")
                compteur[nom] = compteur.get(nom, 0) + 1
    return [
        f"{NOMS_AGENTS.get(nom, nom)} (×{n})" if n > 1 else NOMS_AGENTS.get(nom, nom)
        for nom, n in compteur.items()
    ]


# ─────────────────────────────────────────────────────────────────────────
# Mode "Due diligence structurée" — pas un agent de plus, un NOUVEAU MODE
# D'ORCHESTRATION : la checklist et la logique d'exécution vivent dans
# DueDiligenceAgent.py (séparé, pour suivre la convention un-fichier-par-
# agent du projet) — ici on ne fait que construire l'agent et déléguer.
# Exposé comme un OUTIL du superviseur principal (voir _outil_due_diligence),
# pas comme un bouton séparé — c'est le superviseur qui juge, selon le
# principe d'optimalité, quand la déclencher.
# ─────────────────────────────────────────────────────────────────────────
def lancer_due_diligence(entreprise: str, index_name: str | None = None) -> list[dict]:
    """
    Construit l'agent (SANS l'outil lancer_due_diligence lui-même — sinon
    récursion infinie possible) et délègue l'exécution de la checklist à
    DueDiligenceAgent.executer_due_diligence().
    """
    agent = construire_agent(index_name, _inclure_due_diligence=False)
    return executer_due_diligence(
        agent, entreprise, index_name,
        agents_mobilises, formater_source, extraire_texte, TRACE_DERNIER_TOUR,
    )


class ArgsDueDiligence(BaseModel):
    entreprise: str = Field(description="Nom exact de l'entreprise, tel qu'il apparaît dans la base de données")


def _outil_due_diligence(index_name: str | None) -> StructuredTool:
    """
    Encapsule lancer_due_diligence() en outil pour l'agent principal.
    IMPORTANT : lancer_due_diligence() appelle construire_agent(index_name,
    _inclure_due_diligence=False) en interne — l'agent utilisé PENDANT la
    due diligence n'a PAS cet outil, ce qui évite toute récursion.
    """
    def _fonction(entreprise: str) -> str:
        rapport = lancer_due_diligence(entreprise, index_name)
        lignes = [f"## Due diligence structurée — {entreprise}\n"]
        for section in rapport:
            lignes.append(f"### {section['categorie']}")
            if section.get("agents_mobilises"):
                lignes.append(f"*Agents mobilisés : {', '.join(section['agents_mobilises'])}*")
            lignes.append(section["reponse"])
            lignes.append("")
        return "\n".join(lignes)

    return StructuredTool.from_function(
        func=_fonction,
        name="lancer_due_diligence",
        description=(
            "Exécute une checklist complète de due diligence financière (rentabilité, "
            "endettement, liquidité/risque, valorisation, tendance boursière, actualité"
            + (", cohérence avec le document" if index_name else "") +
            ") pour UNE entreprise. RÉSERVÉ aux demandes explicitement larges — "
            "n'utilise JAMAIS cet outil pour une question ciblée sur un seul aspect "
            "(dans ce cas, l'agent spécialisé concerné suffit et coûte moins cher)."
        ),
        args_schema=ArgsDueDiligence,
    )


if __name__ == "__main__":
    index_choisi = _choisir_document()
    agent = construire_agent(index_choisi)

    nb_agents = 6 if index_choisi else 5
    print(f"\n🧭 Orchestrateur MAF prêt — {nb_agents} agents/outils disponibles, sélection optimale.")
    if index_choisi:
        print(f"📄 Document attaché : {index_choisi.replace('_index', '')}")
    print("Tapez 'exit' pour quitter.\n")

    messages = []

    while True:
        try:
            user_input = input("\n👤 Vous : ").strip()
            if user_input.lower() == "exit":
                print("Fin de la session.", flush=True)
                break
            if not user_input:
                continue

            print("🤖 Réflexion (sélection des agents nécessaires)...", flush=True)
            messages.append({"role": "user", "content": user_input})

            nb_avant = len(messages)
            TRACE_DERNIER_TOUR.clear()
            resultat = agent.invoke({"messages": messages})
            messages = resultat["messages"]

            mobilises = agents_mobilises(messages, nb_avant)
            if mobilises:
                print(f"\n🔧 Agents mobilisés (détecté) : {', '.join(mobilises)}")
            else:
                print("\n🔧 Aucun agent mobilisé — réponse directe de l'orchestrateur.")

            if TRACE_DERNIER_TOUR:
                print("\n📎 Sources détaillées :")
                for entree in TRACE_DERNIER_TOUR:
                    print(f"\n  • {NOMS_AGENTS.get(entree['agent'], entree['agent'])}")
                    print(f"    Question transmise : {entree['question_transmise']}")
                    for outil in entree["outils_internes"]:
                        print(f"    → Outil interne : {outil['outil']}({outil['args']})")
                    source_formatee = formater_source(entree["agent"], entree["resultats_bruts"])
                    for ligne in source_formatee.split("\n"):
                        print(f"    {ligne}")

            reponse = extraire_texte(messages[-1].content)
            print(f"\n💡 IA :\n{reponse}", flush=True)

        except KeyboardInterrupt:
            print("\nFin de la session.", flush=True)
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}", flush=True)