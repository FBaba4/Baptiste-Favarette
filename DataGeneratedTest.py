"""
GenerateurDonneesTest.py — Génère une entreprise fictive ET son "rapport
annuel" en PDF, pour tester le pipeline complet (indexation, RAG,
extraction) sans jamais influencer les moyennes de donnees_structurees.csv.

⚠️ SÉPARATION STRICTE, EN DEUX COUCHES :
1. Cosmétique — chaque entreprise générée est préfixée "TEST - " dans son
   nom, visible partout dans l'app (titres de conversation, aperçu
   d'extraction, base de données si jamais enregistrée).
2. Structurelle — DocumentExtractor.py REFUSE d'écrire dans
   donnees_structurees.csv toute entreprise dont le nom commence par ce
   préfixe, même si tu cliques par erreur sur "Enregistrer dans la base"
   après une extraction. Voir enregistrer_extraction() dans ce fichier.

Les PDF générés vont dans documents/, exactement comme les vrais — ils
suivent le pipeline normal (DemandingIngest.py, RAG, agent_document).
Seule l'écriture vers le CSV d'entraînement est bloquée.
"""

import random
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

PREFIXE_TEST = "TEST - "
DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

SECTEURS = ["Technologie", "Santé", "Industrie", "Finance", "Énergie", "Consommation"]
PAYS = ["France", "Allemagne", "États-Unis", "Royaume-Uni", "Pays-Bas"]

NOMS_FICTIFS = [
    "Norvégia Systems", "Delta Concept", "Aurore Industries", "Vertex Capital",
    "Solane Group", "Kestrel Dynamics", "Meridian Holdings", "Argos Materials",
    "Cassiopée Ventures", "Lumen Technologies",
]


def est_entreprise_test(nom_entreprise: str) -> bool:
    """Point de contrôle central — utilisé par DocumentExtractor.py pour
    bloquer l'écriture. Une seule fonction, un seul endroit à maintenir si
    le préfixe change un jour."""
    return nom_entreprise.strip().startswith(PREFIXE_TEST)


def generer_profil_fictif(nom: str | None = None) -> dict:
    """Génère un profil financier plausible — valeurs réalistes dans leurs
    ordres de grandeur respectifs, mais 100% inventées, sans lien avec une
    vraie entreprise."""
    nom = nom or random.choice(NOMS_FICTIFS)
    ca = round(random.uniform(500, 30000), 1)
    marge_brute = round(random.uniform(25, 65), 1)
    marge_nette = round(random.uniform(2, 22), 1)
    benefice_net = round(ca * marge_nette / 100, 1)
    croissance_ca = round(random.uniform(-5, 20), 1)
    ratio_dette = round(random.uniform(20, 140), 1)
    roe = round(random.uniform(5, 35), 1)
    roa = round(random.uniform(2, 15), 1)
    employes = random.randint(500, 150000)

    return {
        "entreprise": f"{PREFIXE_TEST}{nom}",
        "secteur": random.choice(SECTEURS),
        "pays": random.choice(PAYS),
        "employes": employes,
        "chiffre_affaires": ca,
        "benefice_net": benefice_net,
        "marge_brute": marge_brute,
        "marge_nette": marge_nette,
        "croissance_ca": croissance_ca,
        "ratio_dette_capital": ratio_dette,
        "roe": roe,
        "roa": roa,
    }


def _construire_recit(profil: dict) -> list:
    """
    Compose le texte du faux rapport annuel, réparti sur plusieurs pages —
    volontairement narratif (chiffres dispersés dans du texte en prose),
    comme un vrai rapport, plutôt qu'un simple tableau — pour tester
    correctement la récupération RAG plutôt qu'un cas trivial.
    """
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Rapport Annuel — {profil['entreprise']}", styles["Title"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        f"{profil['entreprise']} est une entreprise du secteur "
        f"{profil['secteur'].lower()}, dont le siège social est situé en "
        f"{profil['pays']}. Le Groupe emploie {profil['employes']:,} "
        f"collaborateurs à travers le monde.".replace(",", "\u202f"),
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Performance financière", styles["Heading1"]))
    story.append(Paragraph(
        f"Au cours de l'exercice, le Groupe a réalisé un chiffre d'affaires "
        f"de {profil['chiffre_affaires']:.1f} millions d'euros, en évolution "
        f"de {profil['croissance_ca']:+.1f}% par rapport à l'exercice "
        f"précédent. Le bénéfice net s'établit à {profil['benefice_net']:.1f} "
        f"millions d'euros.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"La marge brute ressort à {profil['marge_brute']:.1f}%, tandis que "
        f"la marge nette atteint {profil['marge_nette']:.1f}%. La "
        f"rentabilité des capitaux propres (ROE) s'élève à {profil['roe']:.1f}%, "
        f"et la rentabilité des actifs (ROA) à {profil['roa']:.1f}%.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Structure financière", styles["Heading1"]))
    story.append(Paragraph(
        f"Le ratio dette sur capitaux propres du Groupe s'établit à "
        f"{profil['ratio_dette_capital']:.1f}% à la clôture de l'exercice, "
        f"reflétant une structure de financement que la direction juge "
        f"maîtrisée dans le contexte macroéconomique actuel.",
        styles["Normal"],
    ))

    return story


def generer_pdf_fictif(profil: dict | None = None) -> tuple[dict, str]:
    """Génère le profil (si non fourni) + le PDF correspondant dans
    documents/. Retourne (profil, nom_fichier)."""
    profil = profil or generer_profil_fictif()
    nom_fichier = f"{profil['entreprise'].replace(' ', '_').replace('/', '-')}.pdf"
    chemin = DOCS_DIR / nom_fichier

    doc = SimpleDocTemplate(str(chemin), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    doc.build(_construire_recit(profil))

    return profil, nom_fichier


if __name__ == "__main__":
    n = input("Combien d'entreprises fictives générer ? [1] : ").strip()
    n = int(n) if n.isdigit() else 1

    for _ in range(n):
        profil, nom_fichier = generer_pdf_fictif()
        print(f"✅ {profil['entreprise']} → documents/{nom_fichier}")
        print(f"   CA={profil['chiffre_affaires']}M€, marge nette={profil['marge_nette']}%, ROE={profil['roe']}%")

    print(
        "\n⚠️ Ces PDF sont dans documents/ comme les vrais — indexe-les et "
        "teste normalement dans l'app (RAG, extraction). Mais même si tu "
        "cliques par erreur sur 'Enregistrer dans la base' après extraction, "
        "DocumentExtractor.py refuse structurellement d'écrire une entreprise "
        "préfixée 'TEST - ' dans donnees_structurees.csv."
    )