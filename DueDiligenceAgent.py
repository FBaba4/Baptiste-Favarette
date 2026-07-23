"""
DueDiligenceAgent.py — Checklist de due diligence financière pour MAF.

Ne construit AUCUN agent lui-même : reçoit un agent déjà prêt en paramètre
(construit par Monitor.construire_agent(..., _inclure_due_diligence=False)).
Ce découplage est nécessaire — Monitor.py expose ce module comme un outil
de son superviseur, donc ce module ne peut pas dépendre de Monitor.py au
niveau du module (import circulaire sinon).

À TESTER SEUL (python DueDiligenceAgent.py) — le bloc de test importe
Monitor.py localement, PAS au niveau du module, précisément pour cette
raison (l'import circulaire n'existe qu'au niveau module, pas à l'intérieur
d'une fonction/d'un bloc __main__ exécuté après le chargement complet des
deux fichiers).
"""

import time

# Pause anti-quota entre chaque question de la checklist — le plan gratuit
# Gemini limite gemini-3.1-flash-lite à 15 requêtes/minute, et CHAQUE
# question ici peut déclencher plusieurs appels internes (le superviseur
# réfléchit, délègue à un agent spécialiste, qui réfléchit à son tour) — 6
# questions sans pause épuisent le quota en quelques secondes.
PAUSE_ENTRE_QUESTIONS = 15  # secondes
PAUSE_APRES_QUOTA = 45      # secondes, si un 429 survient malgré la pause normale

CHECKLIST_DUE_DILIGENCE = [
    {
        "categorie": "Rentabilité",
        "question": (
            "Quelle est la marge nette, le ROE et le ROA de {entreprise} selon notre "
            "base de données, et comment se positionnent-ils par rapport à la moyenne "
            "du secteur ?"
        ),
    },
    {
        "categorie": "Structure financière & endettement",
        "question": (
            "Quel est le ratio dette/capitaux propres de {entreprise}, et comment se "
            "compare-t-il aux autres entreprises du secteur ?"
        ),
    },
    {
        "categorie": "Liquidité & risque",
        "question": (
            "Quel est le score de risque composite de {entreprise}, et y a-t-il des "
            "valeurs atypiques (outliers) sur ses ratios de liquidité ou d'endettement "
            "par rapport au panel ?"
        ),
    },
    {
        "categorie": "Valorisation",
        "question": (
            "Quel est le P/E ratio de {entreprise} et comment se compare-t-il à celui "
            "d'entreprises similaires du même secteur ?"
        ),
    },
    {
        "categorie": "Tendance boursière",
        "question": "Quelle est la tendance récente (momentum) du cours de {entreprise} ?",
    },
    {
        "categorie": "Actualité & contexte",
        "question": (
            "Y a-t-il eu des actualités récentes significatives concernant {entreprise} "
            "(résultats, acquisitions, litiges, changements de direction) ?"
        ),
    },
]

CHECKLIST_ITEM_DOCUMENT = {
    "categorie": "Cohérence avec les documents",
    "question": (
        "Les chiffres des documents attachés (chiffre d'affaires, bénéfice net, marges) "
        "sont-ils cohérents avec les données de notre base pour {entreprise}, ET entre "
        "eux si plusieurs documents couvrent des périodes différentes ? Signale tout "
        "écart notable."
    ),
}


def executer_due_diligence(
    agent,
    entreprise: str,
    documents: list | None,
    agents_mobilises,
    formater_source,
    extraire_texte,
    trace_dernier_tour: list,
) -> list[dict]:
    """
    Exécute la checklist pour UNE entreprise, dans une seule mémoire d'agent
    continue (chaque catégorie peut s'appuyer sur les réponses précédentes).
    documents : liste de {"index_name":..., "titre":...}, ou None.

    Paramètres injectés plutôt qu'importés (agent déjà construit,
    fonctions utilitaires de Monitor.py, trace partagée) — c'est ce qui
    permet à ce module de rester indépendant de Monitor.py au niveau import.
    """
    checklist = list(CHECKLIST_DUE_DILIGENCE)
    if documents:
        checklist = checklist + [CHECKLIST_ITEM_DOCUMENT]

    messages: list = []
    rapport = []

    for i, item in enumerate(checklist):
        question = item["question"].format(entreprise=entreprise)
        messages.append({"role": "user", "content": question})

        nb_avant = len(messages)
        trace_dernier_tour.clear()

        try:
            resultat = agent.invoke({"messages": messages})
        except Exception as e:
            message_erreur = str(e)
            if "RESOURCE_EXHAUSTED" in message_erreur or "429" in message_erreur:
                print(f"   ⏳ Quota API atteint sur « {item['categorie']} » — pause de "
                      f"{PAUSE_APRES_QUOTA}s avant une nouvelle tentative...", flush=True)
                time.sleep(PAUSE_APRES_QUOTA)
                try:
                    resultat = agent.invoke({"messages": messages})
                except Exception as e2:
                    rapport.append({
                        "categorie": item["categorie"], "question": question,
                        "reponse": f"❌ Échec après nouvelle tentative (quota API toujours atteint) : {e2}",
                        "agents_mobilises": [], "sources": [],
                    })
                    messages.pop()  # retire la question ratée, ne pollue pas le contexte des suivantes
                    continue
            else:
                rapport.append({
                    "categorie": item["categorie"], "question": question,
                    "reponse": f"❌ Erreur : {message_erreur}",
                    "agents_mobilises": [], "sources": [],
                })
                messages.pop()
                continue

        messages = resultat["messages"]

        mobilises = agents_mobilises(messages, nb_avant)
        sources = [
            {"agent": e["agent"], "source": formater_source(e["agent"], e["resultats_bruts"])}
            for e in trace_dernier_tour
        ]

        rapport.append({
            "categorie": item["categorie"],
            "question": question,
            "reponse": extraire_texte(messages[-1].content),
            "agents_mobilises": mobilises,
            "sources": sources,
        })

        if i < len(checklist) - 1:
            time.sleep(PAUSE_ENTRE_QUESTIONS)

    return rapport


if __name__ == "__main__":
    # Import LOCAL (pas en tête de fichier) : Monitor.py importe ce module,
    # donc un import de Monitor au niveau module créerait un cycle. Ici,
    # au moment où ce bloc s'exécute, les deux modules sont déjà chargés.
    from Monitor import construire_agent, agents_mobilises, formater_source, extraire_texte, TRACE_DERNIER_TOUR

    entreprise = input("Nom exact de l'entreprise à analyser (tel qu'en base) : ").strip()
    if not entreprise:
        print("Nom vide, arrêt.")
    else:
        agent = construire_agent(None, _inclure_due_diligence=False)
        print(f"\n🔎 Due diligence en cours pour {entreprise}... (6 questions séquentielles)\n")

        rapport = executer_due_diligence(
            agent, entreprise, None,
            agents_mobilises, formater_source, extraire_texte, TRACE_DERNIER_TOUR,
        )

        for section in rapport:
            print(f"\n### {section['categorie']}")
            print(f"Agents mobilisés : {', '.join(section['agents_mobilises']) or 'aucun'}")
            print(section["reponse"])