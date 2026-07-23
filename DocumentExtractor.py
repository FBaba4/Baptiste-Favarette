"""
RAG -> Base de données structurée (MAF)

⚠️ Statut épistémique : les valeurs extraites viennent d'un LLM qui lit un
texte, pas d'un parsing déterministe — c'est pourquoi chaque champ garde sa
page source, et pourquoi App.py DOIT montrer un aperçu avant d'enregistrer
(jamais d'écriture silencieuse dans la base).
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

CSV_PATH = "donnees_structurees.csv"
INDEXES_DIR = Path("faiss_indexes")


class ExtractionFinanciere(BaseModel):
    entreprise: str = Field(description="Nom de l'entreprise")
    secteur: Optional[str] = Field(None, description="Secteur d'activité")
    pays: Optional[str] = Field(None, description="Pays du siège social")
    nombre_employes: Optional[int] = Field(None, description="Nombre d'employés")
    chiffre_affaires: Optional[float] = Field(None, description="Chiffre d'affaires (montant brut)")
    benefice_net: Optional[float] = Field(None, description="Bénéfice net (montant brut)")
    marge_brute_pct: Optional[float] = Field(None, description="Marge brute en pourcentage")
    marge_nette_pct: Optional[float] = Field(None, description="Marge nette en pourcentage")
    croissance_ca_pct: Optional[float] = Field(None, description="Croissance du CA en % vs exercice précédent")
    ratio_dette_capital_pct: Optional[float] = Field(None, description="Ratio dette/capitaux propres en %")
    pages_sources: dict[str, int] = Field(
        default_factory=dict,
        description="Pour chaque champ rempli, la page du document où l'info a été trouvée",
    )


def _charger_retriever(index_name: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = FAISS.load_local(str(INDEXES_DIR / index_name), embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 15})


def extraire_donnees_du_document(index_name: str) -> ExtractionFinanciere:
    """
    Interroge le document avec plusieurs requêtes ciblées (une par thème
    financier), agrège le contexte récupéré (dédoublonné), puis demande au
    LLM une extraction structurée avec page source par champ.
    """
    retriever = _charger_retriever(index_name)

    requetes = [
        "chiffre d'affaires de l'entreprise",
        "bénéfice net résultat net",
        "marge brute marge nette",
        "croissance du chiffre d'affaires par rapport à l'exercice précédent",
        "endettement ratio dette capitaux propres",
        "secteur d'activité nombre d'employés pays siège social",
    ]

    chunks_uniques = {}
    for requete in requetes:
        for doc in retriever.invoke(requete):
            cle = (doc.metadata.get("page", "?"), doc.page_content[:60])
            chunks_uniques[cle] = doc

    contexte = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in chunks_uniques.values()
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    llm_structure = llm.with_structured_output(ExtractionFinanciere)

    prompt = (
        "Extrait les informations financières suivantes à partir des extraits de document "
        "ci-dessous. Pour chaque champ que tu remplis, indique sa page source dans "
        "'pages_sources' (clé = nom du champ, valeur = numéro de page). Si une information "
        "n'est pas présente dans les extraits, laisse le champ vide (null) — n'invente "
        "JAMAIS une valeur absente du texte.\n\n"
        f"Extraits du document :\n{contexte}"
    )

    return llm_structure.invoke(prompt)


def enregistrer_extraction(extraction: ExtractionFinanciere) -> dict:
    """
    Ajoute (ou met à jour si l'entreprise existe déjà) une ligne dans
    donnees_structurees.csv, avec mention explicite de la provenance.

    Sur une mise à jour, SEULS les champs réellement fournis par cette
    extraction sont modifiés — une ré-analyse partielle d'un document (qui
    ne mentionne par exemple que la marge nette) ne doit jamais écraser des
    valeurs déjà connues d'un passage précédent.
    """
    champs_bruts = {
        "Entreprise": extraction.entreprise,
        "Secteur": extraction.secteur,
        "Pays": extraction.pays,
        "Nombre employés": extraction.nombre_employes,
        "Chiffre d’affaires": extraction.chiffre_affaires,
        "Bénéfice Net": extraction.benefice_net,
        "Marge brute (%)": f"{extraction.marge_brute_pct}%" if extraction.marge_brute_pct is not None else None,
        "Marge nette (%)": f"{extraction.marge_nette_pct}%" if extraction.marge_nette_pct is not None else None,
        "Croissance CA (%)": f"{extraction.croissance_ca_pct}%" if extraction.croissance_ca_pct is not None else None,
        "Ratio Dette/Capital (%)": f"{extraction.ratio_dette_capital_pct}%" if extraction.ratio_dette_capital_pct is not None else None,
    }
    # Ne garder que les champs réellement extraits (pas de None) — c'est ce
    # sous-ensemble qui sera écrit, jamais les champs absents.
    champs_fournis = {k: v for k, v in champs_bruts.items() if v is not None}
    champs_fournis["Source des données"] = "RAG (document analysé)"
    champs_fournis["Pages sources (JSON)"] = json.dumps(extraction.pages_sources, ensure_ascii=False)

    if Path(CSV_PATH).exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["Entreprise"])

    for col in champs_fournis:
        if col not in df.columns:
            df[col] = ""
        # Colonnes en object : évite un crash si une colonne 100% numérique
        # (ex. déjà remplie par DataCollecting.py) reçoit une valeur texte
        # ("N/A", une chaîne avec %, etc.) lors d'un ajout ultérieur.
        if df[col].dtype != object:
            df[col] = df[col].astype(object)

    masque = df["Entreprise"] == extraction.entreprise if "Entreprise" in df.columns and not df.empty else pd.Series([], dtype=bool)

    if masque.any():
        idx = df[masque].index[0]
        for cle, valeur in champs_fournis.items():
            df.loc[idx, cle] = valeur
        action = "mise à jour (champs fournis uniquement, le reste est inchangé)"
    else:
        toutes_colonnes = set(df.columns) | set(champs_fournis.keys())
        ligne_complete = {col: champs_fournis.get(col, "N/A") for col in toutes_colonnes}
        df = pd.concat([df, pd.DataFrame([ligne_complete])], ignore_index=True)
        action = "ajoutée"

    df.to_csv(CSV_PATH, index=False)
    return {"action": action, "entreprise": extraction.entreprise, "champs_modifies": list(champs_fournis.keys())}