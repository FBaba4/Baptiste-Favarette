"""
Agent de recherche internet pour MAF, basé sur Tavily.

Construit sur le même modèle que SQLAgent.py : create_agent, boucle CLI
propre (extraire_texte), mémoire de session en RAM. À TESTER SEUL avant
toute intégration dans App.py.

Prérequis :
1. pip install langchain-tavily
2. Créer un compte sur https://app.tavily.com (gratuit, pas de carte
   bancaire, 1000 crédits/mois) et récupérer la clé API.
3. Ajouter TAVILY_API_KEY=ta_cle dans le fichier .env, à côté de
   GOOGLE_API_KEY.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

model = init_chat_model("google_genai:gemini-3.1-flash-lite")

# max_results=5 : suffisant pour la plupart des questions sans gaspiller de
# crédits Tavily inutilement. topic="general" convient pour de l'actualité
# économique/financière générale — "news" serait plus adapté si tu veux
# filtrer sur de l'actualité récente uniquement.
outil_recherche_web = TavilySearch(
    max_results=5,
    topic="general",
)

TOOLS = [outil_recherche_web]

SYSTEM_PROMPT = """
Tu es l'agent de recherche internet de MAF (Mon Analyseur Financier). Tu as
accès à un moteur de recherche web (Tavily) pour trouver de l'information
récente ou publique que tu ne connais pas par toi-même — actualité
économique, cours de bourse, données macroéconomiques à jour, nouvelles sur
une entreprise, etc.

RÈGLES :
1. N'utilise la recherche web QUE quand l'information demandée est
   probablement récente, changeante, ou hors de tes connaissances (une date,
   un événement, un chiffre d'actualité). Ne cherche pas pour des questions
   de culture générale stable.
2. CITE TOUJOURS tes sources : indique le nom du site ou l'URL d'où vient
   chaque information que tu rapportes.
3. Si les résultats de recherche sont contradictoires ou peu fiables,
   dis-le explicitement plutôt que de trancher arbitrairement.
4. Ne confonds jamais une information trouvée sur le web (à vérifier, datée)
   avec une donnée de la base de MAF ou une prédiction du modèle ML — reste
   dans ton rôle de recherche web uniquement.
"""

agent = create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT)


def extraire_texte(contenu) -> str:
    """Même utilitaire que dans SQLAgent.py — certaines versions du SDK
    Gemini renvoient .content comme une liste de blocs structurés plutôt
    qu'une chaîne simple."""
    if isinstance(contenu, str):
        return contenu
    if isinstance(contenu, list):
        morceaux = []
        for bloc in contenu:
            if isinstance(bloc, dict) and bloc.get("type") == "text":
                morceaux.append(bloc.get("text", ""))
            elif isinstance(bloc, str):
                morceaux.append(bloc)
        return "".join(morceaux)
    return str(contenu)


if __name__ == "__main__":
    print("🔍 Agent de recherche web prêt (Tavily). Tapez 'exit' pour quitter.\n")

    messages = []

    while True:
        try:
            user_input = input("\n👤 Vous : ").strip()
            if user_input.lower() == "exit":
                print("Fin de la session.", flush=True)
                break
            if not user_input:
                continue

            print("🤖 Recherche en cours...", flush=True)
            messages.append({"role": "user", "content": user_input})

            resultat = agent.invoke({"messages": messages})
            messages = resultat["messages"]

            reponse = extraire_texte(messages[-1].content)
            print(f"\n💡 IA :\n{reponse}", flush=True)

        except KeyboardInterrupt:
            print("\nFin de la session.", flush=True)
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}", flush=True)