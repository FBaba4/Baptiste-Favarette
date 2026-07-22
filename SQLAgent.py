"""
sql_agent.py — Agent d'analyse unifié pour MAF (Mon Analyseur Financier)

Combine trois familles d'outils dans un seul agent :
- SQL (interrogation de maf.db, généré par migrate_to_sqlite.py)
- Prédiction ML (predictor.py — modèles entraînés sur le même panel)
- Analyse technique (analyse_technique.py — momentum boursier)

Adapté du tutoriel officiel LangChain (docs.langchain.com/oss/python/langchain/sql-agent).

⚠️ À TESTER SEUL (python sql_agent.py) avant toute intégration dans App.py :
create_agent() appartient à une famille d'API LangChain différente de celle
utilisée dans DKMQuery.py (bind_tools direct) — voir la note de compatibilité
transmise précédemment.
"""

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1 — Imports & configuration
# ─────────────────────────────────────────────────────────────────────────
import sqlite3
import difflib
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
load_dotenv()  # charge GOOGLE_API_KEY depuis .env — jamais codée en dur

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

DB_PATH = "maf.db"

if not Path(DB_PATH).exists():
    raise FileNotFoundError(
        f"{DB_PATH} introuvable — lance d'abord `python migrate_to_sqlite.py` "
        f"pour générer la base à partir de donnees_structurees.csv."
    )

# gemini-3.1-flash-lite : cohérent avec le modèle utilisé ailleurs dans MAF.
model = init_chat_model("google_genai:gemini-3.1-flash-lite")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — Accès à la base : connexion en lecture seule
# ─────────────────────────────────────────────────────────────────────────
def _connexion_lecture_seule() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — Outils SQL exposés au LLM
# ─────────────────────────────────────────────────────────────────────────
@tool
def sql_db_list_tables() -> str:
    """Input est une chaîne vide, output est la liste des tables de la base séparées par des virgules."""
    con = _connexion_lecture_seule()
    try:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        return ", ".join(tables)
    finally:
        con.close()


@tool
def sql_db_schema(table_names: str) -> str:
    """Input est une liste de tables séparées par des virgules, output est le schéma et des exemples de lignes.
    Vérifie que les tables existent bien en appelant sql_db_list_tables d'abord !
    Exemple d'input : table1, table2"""
    con = _connexion_lecture_seule()
    try:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        valid_tables = {row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")}

        results = []
        for table in table_names.split(","):
            table = table.strip()
            if table not in valid_tables:
                results.append(f"Erreur : table {table!r} introuvable dans la base")
                continue

            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            schema_row = cursor.fetchone()
            if not schema_row:
                continue
            results.append(schema_row[0])

            try:
                quoted_table = '"' + table.replace('"', '""') + '"'
                cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3;")
                rows = cursor.fetchall()
                if rows:
                    col_names = [d[0] for d in cursor.description]
                    apercu = (
                        f"/*\n3 lignes de {table} :\n"
                        + "\t".join(col_names) + "\n"
                        + "\n".join("\t".join(str(x) for x in row) for row in rows)
                        + "\n*/"
                    )
                    results.append(apercu)
            except Exception as e:
                results.append(f"Erreur en lisant des exemples : {e}")

        return "\n\n".join(results)
    finally:
        con.close()


@tool
def sql_db_query(query: str) -> str:
    """Input est une requête SQL correcte et détaillée, output est le résultat de la base.
    Si la requête est incorrecte, un message d'erreur est renvoyé — corrige et réessaie.
    Si une colonne est introuvable, utilise sql_db_schema pour vérifier les noms exacts."""
    con = _connexion_lecture_seule()
    try:
        cursor = con.cursor()
        cursor.execute(query)
        return str(cursor.fetchall())
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


@tool
def sql_db_query_checker(query: str) -> str:
    """Vérifie la requête SQL avant exécution — à utiliser SYSTÉMATIQUEMENT avant sql_db_query."""
    trigger_prompt = f"""{query}
Vérifie la requête SQLite ci-dessus pour les erreurs classiques :
- NOT IN avec des valeurs NULL
- UNION au lieu de UNION ALL
- BETWEEN pour des bornes exclusives
- incohérence de type dans les prédicats
- guillemets d'identifiants mal placés
- mauvais nombre d'arguments dans une fonction
- mauvaises colonnes dans une jointure

Si tu trouves une erreur, corrige la requête. Sinon, reproduis la requête d'origine.
Ne renvoie que la requête SQL finale, rien d'autre.

Requête SQL : """
    response = model.invoke(trigger_prompt)
    return response.text.strip()


@tool
def sql_db_statistiques(colonne: str, secteur: str = "") -> str:
    """
    Calcule moyenne, médiane, écart-type, min et max d'une colonne numérique
    de la table entreprises — éventuellement filtrée par secteur (laisser
    vide pour tout le panel). Utilise cet outil pour CONTEXTUALISER un
    chiffre (comparer une entreprise au reste du panel/secteur) plutôt que
    de donner une valeur brute sans point de comparaison.
    Exemple d'input : colonne='rentabilite_capitaux_roe', secteur='Technology'"""
    con = _connexion_lecture_seule()
    try:
        requete = "SELECT * FROM entreprises"
        params = ()
        if secteur:
            requete += " WHERE secteur = ?"
            params = (secteur,)

        df = pd.read_sql_query(requete, con, params=params)
        if colonne not in df.columns:
            colonnes_dispo = ", ".join(df.columns)
            return f"Erreur : colonne '{colonne}' introuvable. Colonnes disponibles : {colonnes_dispo}"

        valeurs = pd.to_numeric(
            df[colonne].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        ).dropna()

        if valeurs.empty:
            portee = f" dans le secteur {secteur!r}" if secteur else ""
            return f"Aucune valeur numérique exploitable pour '{colonne}'{portee}."

        portee = f" (secteur {secteur!r})" if secteur else " (tout le panel)"
        return (
            f"Statistiques pour '{colonne}'{portee} sur {len(valeurs)} entreprises :\n"
            f"Moyenne : {valeurs.mean():.2f} | Médiane : {valeurs.median():.2f} | "
            f"Écart-type : {valeurs.std():.2f} | Min : {valeurs.min():.2f} | Max : {valeurs.max():.2f}"
        )
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


@tool
def sql_db_valeurs_atypiques(colonne: str, secteur: str = "", seuil_zscore: float = 2.0) -> str:
    """
    Détecte les entreprises dont la valeur d'une colonne numérique s'écarte
    fortement de la moyenne du panel (ou du secteur), via un z-score.
    seuil_zscore=2.0 par défaut (au-delà de 2 écarts-types). Ne conclut
    JAMAIS automatiquement qu'un outlier est une anomalie ou une opportunité
    — signale le cas pour investigation, l'interprétation reste à faire.
    Exemple d'input : colonne='rentabilite_capitaux_roe', secteur='Technology'"""
    con = _connexion_lecture_seule()
    try:
        requete = "SELECT * FROM entreprises"
        params = ()
        if secteur:
            requete += " WHERE secteur = ?"
            params = (secteur,)
        df = pd.read_sql_query(requete, con, params=params)

        if colonne not in df.columns:
            return f"Erreur : colonne '{colonne}' introuvable. Colonnes disponibles : {', '.join(df.columns)}"
        if "entreprise" not in df.columns:
            return "Erreur : colonne 'entreprise' introuvable pour identifier les lignes."

        valeurs_num = pd.to_numeric(
            df[colonne].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df_valide = df.assign(_valeur=valeurs_num).dropna(subset=["_valeur"])

        if len(df_valide) < 3:
            return f"Pas assez de données valides sur '{colonne}' pour un calcul de z-score fiable."

        moyenne = df_valide["_valeur"].mean()
        ecart_type = df_valide["_valeur"].std()
        if ecart_type == 0:
            return f"Écart-type nul pour '{colonne}' — toutes les valeurs sont identiques, aucun outlier possible."

        df_valide = df_valide.assign(_zscore=(df_valide["_valeur"] - moyenne) / ecart_type)
        outliers = df_valide[df_valide["_zscore"].abs() >= seuil_zscore].sort_values("_zscore", ascending=False)

        if outliers.empty:
            return f"Aucune valeur atypique détectée pour '{colonne}' (seuil z-score = {seuil_zscore})."

        lignes = [
            f"- {row['entreprise']} : {row['_valeur']:.2f} (z-score {row['_zscore']:+.2f})"
            for _, row in outliers.iterrows()
        ]
        return (
            f"{len(outliers)} valeur(s) atypique(s) pour '{colonne}' "
            f"(moyenne={moyenne:.2f}, écart-type={ecart_type:.2f}) :\n" + "\n".join(lignes)
        )
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


def _resoudre_valeur_categorique(df: pd.DataFrame, colonne: str, valeur_saisie: str) -> tuple:
    """
    Résout une valeur saisie approximativement vers la valeur EXACTE présente
    dans UNE colonne catégorielle donnée (ex. 'entreprise', 'secteur') —
    SANS jamais halluciner : matching déterministe (difflib), aucune
    génération LLM impliquée dans cette fonction.

    Retourne (valeur_exacte, suggestions) :
    - Correspondance forte et unique -> (valeur_exacte, [])
    - Ambigu ou aucune correspondance suffisamment proche -> (None, [candidats])
      Dans ce cas, l'outil appelant DOIT renvoyer les suggestions et refuser
      de deviner — jamais filtrer/comparer sur une valeur non confirmée.
    """
    if colonne not in df.columns:
        return None, []
    valeurs_reelles = df[colonne].dropna().unique().tolist()
    valeur_normalisee = valeur_saisie.strip().lower()

    # 1. Correspondance exacte insensible à la casse — cas le plus fréquent
    for valeur_reelle in valeurs_reelles:
        if str(valeur_reelle).strip().lower() == valeur_normalisee:
            return valeur_reelle, []

    # 2. Correspondance par inclusion (ex. "bpce" contenu dans "Groupe BPCE",
    #    ou "tech" contenu dans "Technology")
    candidats_inclusion = [
        v for v in valeurs_reelles
        if valeur_normalisee in str(v).lower() or str(v).lower() in valeur_normalisee
    ]
    if len(candidats_inclusion) == 1:
        return candidats_inclusion[0], []

    # 3. Correspondance floue par distance d'édition (difflib, déterministe)
    proches = difflib.get_close_matches(valeur_saisie, [str(v) for v in valeurs_reelles], n=3, cutoff=0.6)
    if len(proches) == 1:
        return proches[0], []

    # Ambigu (plusieurs candidats à égalité) ou aucune correspondance fiable
    # -> ne jamais deviner, remonter les candidats pour que l'humain/LLM
    # demande confirmation plutôt que d'halluciner une réponse.
    return None, (proches or candidats_inclusion)[:3]


def _resoudre_nom_entreprise(df: pd.DataFrame, nom_saisi: str) -> tuple:
    """Raccourci de _resoudre_valeur_categorique pour la colonne 'entreprise'
    — conservé pour ne pas modifier les outils qui l'appellent déjà."""
    return _resoudre_valeur_categorique(df, "entreprise", nom_saisi)


@tool
def sql_db_comparer_entreprises(entreprises: str, colonnes: str = "") -> str:
    """
    Compare plusieurs entreprises côte à côte sur les colonnes indiquées.
    entreprises : noms séparés par des virgules — pas besoin de l'orthographe
    exacte, une reconnaissance approximative déterministe est appliquée
    (ex. 'bpce' reconnaît 'Groupe BPCE'). Si un nom est ambigu, l'outil
    liste des candidats plutôt que de deviner.
    colonnes : colonnes à comparer, séparées par des virgules. Si vide, utilise
    une sélection par défaut (secteur, marge nette, ROE, ROA, P/E, croissance CA).
    Exemple d'input : entreprises='Apple Inc.,Microsoft Corporation', colonnes=''"""
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)
        noms_saisis = [n.strip() for n in entreprises.split(",") if n.strip()]

        noms_resolus = []
        avertissements = []
        for nom_saisi in noms_saisis:
            nom_resolu, suggestions = _resoudre_nom_entreprise(df, nom_saisi)
            if nom_resolu is None:
                if suggestions:
                    avertissements.append(
                        f"⚠️ '{nom_saisi}' : ambigu ou introuvable avec certitude — "
                        f"candidats proches : {', '.join(suggestions)}. Précise laquelle."
                    )
                else:
                    avertissements.append(f"⚠️ '{nom_saisi}' : aucune entreprise correspondante trouvée dans la base.")
                continue
            if nom_resolu.lower() != nom_saisi.lower():
                avertissements.append(f"ℹ️ '{nom_saisi}' reconnu comme '{nom_resolu}'.")
            noms_resolus.append(nom_resolu)

        if not noms_resolus:
            return "Aucune entreprise résolue avec certitude.\n" + "\n".join(avertissements)

        df_filtre = df[df["entreprise"].isin(noms_resolus)]

        if colonnes.strip():
            cols_demandees = [c.strip() for c in colonnes.split(",")]
            cols_valides = [c for c in cols_demandees if c in df.columns]
            manquantes = set(cols_demandees) - set(cols_valides)
            if manquantes:
                return f"Colonnes introuvables : {', '.join(manquantes)}. Colonnes disponibles : {', '.join(df.columns)}"
        else:
            candidats_defaut = [
                "secteur", "marge_nette_pct", "rentabilite_capitaux_roe",
                "rentabilite_actifs_roa", "p_e_ratio_valorisation", "croissance_ca_pct",
            ]
            cols_valides = [c for c in candidats_defaut if c in df.columns]

        tableau = df_filtre[["entreprise"] + cols_valides].to_string(index=False)
        note = ("\n" + "\n".join(avertissements)) if avertissements else ""
        return f"Comparaison de {len(df_filtre)} entreprise(s) :\n\n{tableau}{note}"
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


@tool
def sql_db_score_risque(entreprise: str) -> str:
    """
    Calcule un score de risque composite (0=faible, 100=élevé) pour une
    entreprise, à partir de trois critères disponibles dans la base :
    liquidité générale, endettement (dette/capitaux), et beta (volatilité
    de marché). Chaque critère est converti en percentile par rapport au
    panel avant pondération, pour rendre les échelles comparables.

    ⚠️ CE N'EST PAS UN MODÈLE STATISTIQUE ENTRAÎNÉ (contrairement aux outils
    predict_*) — c'est une formule pondérée choisie arbitrairement (40%
    liquidité, 35% endettement, 25% beta), à présenter comme une méthode
    transparente et assumée, jamais comme une prédiction validée.

    Le nom d'entreprise n'a pas besoin d'être exact — une reconnaissance
    approximative déterministe est appliquée. Si le nom est ambigu, l'outil
    refuse de deviner et liste des candidats."""
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)

        colonnes_requises = ["entreprise", "ratio_liquidite_generale", "ratio_dette_capital_pct", "beta"]
        manquantes = [c for c in colonnes_requises if c not in df.columns]
        if manquantes:
            return f"Colonnes manquantes pour ce calcul : {manquantes}. Vérifie le schéma avec sql_db_schema."

        nom_resolu, suggestions = _resoudre_nom_entreprise(df, entreprise)
        if nom_resolu is None:
            if suggestions:
                return (
                    f"Entreprise '{entreprise}' ambiguë ou non trouvée avec certitude — "
                    f"candidats proches : {', '.join(suggestions)}. Précise laquelle plutôt "
                    f"que de deviner."
                )
            return f"Entreprise '{entreprise}' introuvable dans la base."

        note_reconnaissance = (
            f"(reconnu comme '{nom_resolu}' à partir de '{entreprise}')\n"
            if nom_resolu.lower() != entreprise.strip().lower() else ""
        )

        ligne = df[df["entreprise"] == nom_resolu].iloc[0]
        entreprise = nom_resolu  # pour que le reste de la fonction utilise le nom exact

        def _nettoyer(valeur):
            return pd.to_numeric(str(valeur).replace("%", "").replace(",", "."), errors="coerce")

        liquidite = _nettoyer(ligne["ratio_liquidite_generale"])
        endettement = _nettoyer(ligne["ratio_dette_capital_pct"])
        beta = _nettoyer(ligne["beta"])

        if pd.isna(liquidite) or pd.isna(endettement) or pd.isna(beta):
            return (
                f"Données insuffisantes pour '{entreprise}' "
                f"(liquidité={liquidite}, endettement={endettement}, beta={beta})."
            )

        def _percentile_dans_panel(colonne, valeur):
            serie = pd.to_numeric(
                df[colonne].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ).dropna()
            if serie.empty:
                return 50.0
            return (serie < valeur).mean() * 100

        pct_liquidite = _percentile_dans_panel("ratio_liquidite_generale", liquidite)
        pct_endettement = _percentile_dans_panel("ratio_dette_capital_pct", endettement)
        pct_beta = _percentile_dans_panel("beta", beta)

        score_risque = 0.40 * (100 - pct_liquidite) + 0.35 * pct_endettement + 0.25 * pct_beta

        if score_risque < 33:
            niveau = "Faible"
        elif score_risque < 66:
            niveau = "Modéré"
        else:
            niveau = "Élevé"

        return (
            f"{note_reconnaissance}"
            f"Score de risque composite pour {entreprise} : {score_risque:.0f}/100 — Risque {niveau}\n"
            f"Détail (percentile dans le panel, 0=meilleur, 100=pire) :\n"
            f"- Liquidité : {liquidite} (percentile {pct_liquidite:.0f}, pondération 40%)\n"
            f"- Endettement : {endettement} (percentile {pct_endettement:.0f}, pondération 35%)\n"
            f"- Beta : {beta} (percentile {pct_beta:.0f}, pondération 25%)\n"
            f"⚠️ Formule pondérée transparente et arbitraire — PAS un modèle statistique validé."
        )
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


@tool
def sql_db_screener(
    secteur: str = "",
    roe_min: float = None,
    roe_max: float = None,
    roa_min: float = None,
    marge_nette_min: float = None,
    dette_capital_max: float = None,
    pe_max: float = None,
    croissance_ca_min: float = None,
    beta_max: float = None,
    limite: int = 20,
) -> str:
    """
    Filtre le panel d'entreprises selon PLUSIEURS critères financiers
    combinés (ET logique), pour DÉCOUVRIR des candidats correspondant à un
    profil — à l'inverse de sql_db_comparer_entreprises ou
    sql_db_score_risque qui exigent de connaître déjà les noms des
    entreprises à examiner.

    Tous les critères sont optionnels et cumulatifs. Le secteur n'a pas
    besoin d'être exact — une reconnaissance approximative déterministe est
    appliquée (ex. 'tech' reconnaît 'Technology'), avec refus explicite si
    ambigu plutôt que de deviner. Exemple d'usage : "entreprises du secteur
    Technology avec un ROE > 20% et un endettement < 100%" ->
    secteur='Technology', roe_min=20, dette_capital_max=100.
    limite (défaut 20) plafonne le nombre de lignes affichées, PAS le
    nombre total trouvé (indiqué séparément si supérieur à la limite)."""
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)

        def _numerique(colonne):
            return pd.to_numeric(
                df[colonne].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )

        masque = pd.Series(True, index=df.index)
        criteres_appliques = []

        if secteur:
            if "secteur" not in df.columns:
                return "Erreur : colonne 'secteur' introuvable. Vérifie le schéma avec sql_db_schema."
            secteur_resolu, suggestions = _resoudre_valeur_categorique(df, "secteur", secteur)
            if secteur_resolu is None:
                if suggestions:
                    return (
                        f"Secteur '{secteur}' ambigu ou non reconnu avec certitude — "
                        f"secteurs proches dans la base : {', '.join(suggestions)}. "
                        f"Précise lequel plutôt que de deviner."
                    )
                return f"Secteur '{secteur}' introuvable dans la base."
            masque &= (df["secteur"] == secteur_resolu)
            note_secteur = f" (reconnu comme '{secteur_resolu}')" if secteur_resolu.lower() != secteur.strip().lower() else ""
            criteres_appliques.append(f"secteur = '{secteur_resolu}'{note_secteur}")

        # (colonne réelle, valeur du paramètre, nom du paramètre, opérateur)
        # — colonnes réutilisées telles quelles depuis sql_db_comparer_entreprises
        # et sql_db_score_risque, déjà confirmées fonctionnelles dans ce fichier.
        criteres_numeriques = [
            ("rentabilite_capitaux_roe", roe_min, "roe_min", ">="),
            ("rentabilite_capitaux_roe", roe_max, "roe_max", "<="),
            ("rentabilite_actifs_roa", roa_min, "roa_min", ">="),
            ("marge_nette_pct", marge_nette_min, "marge_nette_min", ">="),
            ("ratio_dette_capital_pct", dette_capital_max, "dette_capital_max", "<="),
            ("p_e_ratio_valorisation", pe_max, "pe_max", "<="),
            ("croissance_ca_pct", croissance_ca_min, "croissance_ca_min", ">="),
            ("beta", beta_max, "beta_max", "<="),
        ]

        for colonne, valeur, nom_param, operateur in criteres_numeriques:
            if valeur is None:
                continue
            if colonne not in df.columns:
                return f"Erreur : colonne '{colonne}' introuvable pour {nom_param}. Vérifie le schéma avec sql_db_schema."
            serie = _numerique(colonne)
            masque &= (serie >= valeur) if operateur == ">=" else (serie <= valeur)
            criteres_appliques.append(f"{nom_param} ({colonne} {operateur} {valeur})")

        if not criteres_appliques:
            return "Aucun critère fourni — précise au moins un filtre (secteur, roe_min, dette_capital_max, ...)."

        resultat = df[masque]
        if resultat.empty:
            return f"Aucune entreprise ne correspond aux critères : {', '.join(criteres_appliques)}."

        colonnes_affichage = [
            "entreprise", "secteur", "rentabilite_capitaux_roe", "rentabilite_actifs_roa",
            "marge_nette_pct", "ratio_dette_capital_pct", "p_e_ratio_valorisation",
            "croissance_ca_pct", "beta",
        ]
        colonnes_dispo = [c for c in colonnes_affichage if c in resultat.columns]
        tableau = resultat[colonnes_dispo].head(limite).to_string(index=False)

        note_limite = ""
        if len(resultat) > limite:
            note_limite = f"\n⚠️ {len(resultat)} entreprise(s) trouvée(s) au total — affichage limité aux {limite} premières."

        return (
            f"{len(resultat)} entreprise(s) correspondant aux critères ({', '.join(criteres_appliques)}) :\n\n"
            f"{tableau}{note_limite}"
        )
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


def _trouver_colonne(df: pd.DataFrame, candidats: list) -> str | None:
    """
    Retourne le premier nom de colonne parmi plusieurs variantes plausibles
    qui existe réellement dans df — utilisé pour les colonnes dont le nom
    SQL exact n'a jamais été confirmé dans le code (contrairement à
    'rentabilite_capitaux_roe', 'beta', etc. déjà utilisés ailleurs dans ce
    fichier). Évite de deviner un nom de colonne et de planter ou, pire,
    d'ignorer silencieusement un contrôle censé s'exécuter.
    """
    for candidat in candidats:
        if candidat in df.columns:
            return candidat
    return None


def _numerique_serie(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _controler_une_ligne(ligne: pd.Series, colonnes: dict) -> list:
    """
    Applique les contrôles de cohérence à UNE ligne (une entreprise) et
    retourne la liste des violations trouvées, chacune taguée
    'incohérence mathématique' (théoriquement impossible en comptabilité
    standard) ou 'anomalie à vérifier' (statistiquement suspect, mais pas
    strictement impossible — peut arriver sur des cas réels particuliers).
    """
    violations = []

    def _val(cle):
        col = colonnes.get(cle)
        if col is None or pd.isna(ligne.get(col)):
            return None
        return _numerique_serie(pd.Series([ligne[col]])).iloc[0]

    mb, mo, meb, mn = _val("marge_brute"), _val("marge_operationnelle"), _val("marge_ebitda"), _val("marge_nette")

    # Hiérarchie standard des marges : brute >= EBITDA >= opérationnelle >= nette.
    # Peut être violée par des éléments exceptionnels non récurrents (cessions,
    # dépréciations) — donc "à vérifier", pas "impossible à 100%", sauf le
    # premier cas (nette > brute) qui est structurellement impossible.
    if mb is not None and mn is not None and mn > mb:
        violations.append(("incohérence mathématique", f"Marge nette ({mn}%) > Marge brute ({mb}%) — structurellement impossible : la marge nette ne peut pas dépasser la marge brute."))
    if mb is not None and meb is not None and meb > mb + 1:  # tolérance 1pt pour arrondis
        violations.append(("anomalie à vérifier", f"Marge EBITDA ({meb}%) > Marge brute ({mb}%) — inhabituel, à vérifier (données possiblement mal renseignées)."))
    if meb is not None and mo is not None and mo > meb + 1:
        violations.append(("anomalie à vérifier", f"Marge opérationnelle ({mo}%) > Marge EBITDA ({meb}%) — inhabituel : l'EBITDA est normalement calculé avant D&A, donc >= marge opérationnelle."))

    prix, haut52, bas52 = _val("prix_actuel"), _val("plus_haut_52_sem"), _val("plus_bas_52_sem")
    if haut52 is not None and bas52 is not None and haut52 < bas52:
        violations.append(("incohérence mathématique", f"Plus haut 52 sem. ({haut52}) < Plus bas 52 sem. ({bas52}) — impossible par définition."))
    if prix is not None and haut52 is not None and prix > haut52 * 1.02:  # tolérance 2% (données pas forcément synchrones)
        violations.append(("anomalie à vérifier", f"Prix actuel ({prix}) > Plus haut 52 sem. ({haut52}) — données probablement désynchronisées dans le temps."))
    if prix is not None and bas52 is not None and prix < bas52 * 0.98:
        violations.append(("anomalie à vérifier", f"Prix actuel ({prix}) < Plus bas 52 sem. ({bas52}) — données probablement désynchronisées dans le temps."))

    beta = _val("beta")
    if beta is not None and abs(beta) > 5:
        violations.append(("anomalie à vérifier", f"Beta extrême ({beta}) — valeur rare, à vérifier avant de s'y fier."))

    return violations


@tool
def sql_db_controle_coherence(entreprise: str = "") -> str:
    """
    Vérifie la cohérence mathématique/logique interne des données d'une
    entreprise (ou de tout le panel si entreprise est vide) — complète
    sql_db_valeurs_atypiques (qui détecte des valeurs statistiquement
    inhabituelles) en détectant des INCOHÉRENCES LOGIQUES (ex. marge nette
    supérieure à la marge brute, ce qui est structurellement impossible),
    signe probable d'une donnée mal renseignée plutôt que d'une performance
    exceptionnelle réelle.

    Si entreprise est fourni, reconnaissance approximative du nom appliquée
    (comme les autres outils). Si vide, scanne tout le panel et liste les
    entreprises présentant au moins une incohérence.

    ⚠️ Certains contrôles portent sur des colonnes dont le nom SQL exact
    n'est pas garanti (marges opérationnelle/EBITDA, prix 52 semaines) — si
    une colonne attendue est introuvable, le contrôle correspondant est
    simplement ignoré (signalé explicitement), pas simulé ni deviné."""
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)

        colonnes = {
            "marge_brute": _trouver_colonne(df, ["marge_brute_pct", "marge_brute"]),
            "marge_operationnelle": _trouver_colonne(df, ["marge_operationnelle_pct", "marge_operationnelle"]),
            "marge_ebitda": _trouver_colonne(df, ["marge_ebitda_pct", "marge_ebitda"]),
            "marge_nette": _trouver_colonne(df, ["marge_nette_pct", "marge_nette"]),
            "prix_actuel": _trouver_colonne(df, ["prix_actuel"]),
            "plus_haut_52_sem": _trouver_colonne(df, ["plus_haut_52_sem", "plus_haut_52_sem_", "plus_haut_52sem"]),
            "plus_bas_52_sem": _trouver_colonne(df, ["plus_bas_52_sem", "plus_bas_52_sem_", "plus_bas_52sem"]),
            "beta": _trouver_colonne(df, ["beta"]),
        }
        colonnes_manquantes = [cle for cle, col in colonnes.items() if col is None]
        note_colonnes = (
            f"⚠️ Colonnes non trouvées, contrôles correspondants ignorés : {', '.join(colonnes_manquantes)}.\n\n"
            if colonnes_manquantes else ""
        )

        if entreprise:
            nom_resolu, suggestions = _resoudre_nom_entreprise(df, entreprise)
            if nom_resolu is None:
                if suggestions:
                    return f"Entreprise '{entreprise}' ambiguë ou non trouvée — candidats : {', '.join(suggestions)}."
                return f"Entreprise '{entreprise}' introuvable dans la base."

            ligne = df[df["entreprise"] == nom_resolu].iloc[0]
            violations = _controler_une_ligne(ligne, colonnes)

            if not violations:
                return f"{note_colonnes}✅ Aucune incohérence détectée pour {nom_resolu}."

            lignes_sortie = [f"{note_colonnes}⚠️ {len(violations)} problème(s) détecté(s) pour {nom_resolu} :"]
            for categorie, message in violations:
                lignes_sortie.append(f"  [{categorie}] {message}")
            return "\n".join(lignes_sortie)

        # Mode scan complet du panel
        resultats_par_entreprise = {}
        for _, ligne in df.iterrows():
            violations = _controler_une_ligne(ligne, colonnes)
            if violations:
                resultats_par_entreprise[ligne.get("entreprise", "?")] = violations

        if not resultats_par_entreprise:
            return f"{note_colonnes}✅ Aucune incohérence détectée sur les {len(df)} entreprises du panel."

        lignes_sortie = [
            f"{note_colonnes}⚠️ {len(resultats_par_entreprise)} entreprise(s) sur {len(df)} "
            f"présentent au moins une incohérence :"
        ]
        for nom, violations in list(resultats_par_entreprise.items())[:15]:
            resume = "; ".join(f"[{c}] {m}" for c, m in violations)
            lignes_sortie.append(f"  - {nom} : {resume}")
        if len(resultats_par_entreprise) > 15:
            lignes_sortie.append(f"  ... et {len(resultats_par_entreprise) - 15} autre(s), non affichée(s).")

        return "\n".join(lignes_sortie)
    except Exception as e:
        return f"Erreur : {e}"
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────
# TOOLS — purement SQL désormais. La prédiction ML et l'analyse technique
# ont leur propre agent dédié dans Monitor.py (agent_estimation,
# agent_technique) — les regrouper ici créait un chevauchement : une
# prédiction ML appelée via cet agent aurait été étiquetée à tort comme
# "donnée réelle en base" par l'orchestrateur, qui juge uniquement par le
# NOM de l'agent sollicité (agent_base_donnees), pas par l'outil interne
# réellement utilisé.
# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# Fonction pour l'interface visuelle (radar chart) — PAS un outil LLM,
# appelée directement par App.py. Calcul 100% SQL/pandas, aucun texte
# généré par un modèle de langage : les valeurs affichées dans le radar
# sont garanties exactes (ou absentes), jamais reformulées/résumées par un LLM.
# ─────────────────────────────────────────────────────────────────────────
AXES_RADAR = [
    ("rentabilite_capitaux_roe", "ROE"),
    ("rentabilite_actifs_roa", "ROA"),
    ("marge_brute_pct", "Marge brute"),
    ("marge_nette_pct", "Marge nette"),
    ("croissance_ca_pct", "Croissance CA"),
]


def donnees_radar_entreprise(nom_entreprise: str) -> dict | None:
    """
    Calcule les valeurs (entreprise vs moyenne du secteur) pour un radar
    chart de 5 ratios de rentabilité, tous en pourcentage donc comparables
    sur une même échelle. Retourne None si l'entreprise n'est pas résolue
    avec certitude (jamais de radar sur une entreprise devinée).

    Retourne un dict :
    {
        "entreprise": nom exact résolu,
        "secteur": secteur de l'entreprise,
        "labels": [noms d'axes affichés],
        "valeurs_entreprise": [valeurs, NaN si donnée absente],
        "valeurs_secteur": [moyennes sectorielles, NaN si non calculables],
        "colonnes_manquantes": [colonnes attendues absentes de la base],
    }
    """
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)

        nom_resolu, _ = _resoudre_nom_entreprise(df, nom_entreprise)
        if nom_resolu is None:
            return None

        ligne = df[df["entreprise"] == nom_resolu].iloc[0]
        secteur = ligne.get("secteur", None)
        df_secteur = df[df["secteur"] == secteur] if secteur is not None and "secteur" in df.columns else df

        labels, valeurs_entreprise, valeurs_secteur, colonnes_manquantes = [], [], [], []

        for colonne, label in AXES_RADAR:
            if colonne not in df.columns:
                colonnes_manquantes.append(colonne)
                continue
            labels.append(label)

            v_entreprise = _numerique_serie(pd.Series([ligne[colonne]])).iloc[0]
            valeurs_entreprise.append(v_entreprise)

            v_secteur = _numerique_serie(df_secteur[colonne]).mean()
            valeurs_secteur.append(v_secteur)

        return {
            "entreprise": nom_resolu,
            "secteur": secteur,
            "labels": labels,
            "valeurs_entreprise": valeurs_entreprise,
            "valeurs_secteur": valeurs_secteur,
            "colonnes_manquantes": colonnes_manquantes,
        }
    except Exception:
        return None
    finally:
        con.close()


def donnees_score_risque_graphique(entreprise: str) -> dict | None:
    """
    Version 'données structurées' de sql_db_score_risque, pour affichage en
    camembert (composition pondérée 40/35/25%) plutôt qu'en texte. Réutilise
    EXACTEMENT le même calcul que l'outil texte — aucune divergence possible
    entre ce que l'agent dit et ce que le graphique montre.
    """
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)
        colonnes_requises = ["entreprise", "ratio_liquidite_generale", "ratio_dette_capital_pct", "beta"]
        if any(c not in df.columns for c in colonnes_requises):
            return None

        nom_resolu, _ = _resoudre_nom_entreprise(df, entreprise)
        if nom_resolu is None:
            return None

        ligne = df[df["entreprise"] == nom_resolu].iloc[0]

        def _nettoyer(v):
            return pd.to_numeric(str(v).replace("%", "").replace(",", "."), errors="coerce")

        liquidite, endettement, beta = (
            _nettoyer(ligne["ratio_liquidite_generale"]),
            _nettoyer(ligne["ratio_dette_capital_pct"]),
            _nettoyer(ligne["beta"]),
        )
        if pd.isna(liquidite) or pd.isna(endettement) or pd.isna(beta):
            return None

        def _percentile(colonne, valeur):
            serie = _numerique_serie(df[colonne]).dropna()
            return 50.0 if serie.empty else (serie < valeur).mean() * 100

        pct_liquidite = _percentile("ratio_liquidite_generale", liquidite)
        pct_endettement = _percentile("ratio_dette_capital_pct", endettement)
        pct_beta = _percentile("beta", beta)

        contributions = {
            "Liquidité (pondération 40%)": 0.40 * (100 - pct_liquidite),
            "Endettement (pondération 35%)": 0.35 * pct_endettement,
            "Beta (pondération 25%)": 0.25 * pct_beta,
        }

        return {
            "entreprise": nom_resolu,
            "score_total": sum(contributions.values()),
            "contributions": contributions,
        }
    except Exception:
        return None
    finally:
        con.close()


def donnees_comparaison_graphique(entreprises: list, colonnes: list = None) -> dict | None:
    """
    Version 'données structurées' de sql_db_comparer_entreprises, pour
    affichage en diagramme en bâtons groupés plutôt qu'en tableau texte.
    """
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)

        noms_resolus = []
        for nom in entreprises:
            resolu, _ = _resoudre_nom_entreprise(df, nom)
            if resolu and resolu not in noms_resolus:
                noms_resolus.append(resolu)
        if not noms_resolus:
            return None

        if not colonnes:
            colonnes = ["marge_nette_pct", "rentabilite_capitaux_roe", "rentabilite_actifs_roa", "croissance_ca_pct"]
        colonnes_dispo = [c for c in colonnes if c in df.columns]
        if not colonnes_dispo:
            return None

        df_filtre = df[df["entreprise"].isin(noms_resolus)]
        valeurs = {
            colonne: {
                row["entreprise"]: _numerique_serie(pd.Series([row[colonne]])).iloc[0]
                for _, row in df_filtre.iterrows()
            }
            for colonne in colonnes_dispo
        }

        return {"entreprises": noms_resolus, "colonnes": colonnes_dispo, "valeurs": valeurs}
    except Exception:
        return None
    finally:
        con.close()


TOOLS = [
    sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker,
    sql_db_statistiques, sql_db_valeurs_atypiques,
    sql_db_comparer_entreprises, sql_db_score_risque, sql_db_screener,
    sql_db_controle_coherence,
]


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — System prompt de l'agent (recentré sur le SQL uniquement)
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Tu es l'agent base de données de MAF (Mon Analyseur Financier). Tu
interroges exclusivement une base SQLite de données réelles collectées sur
~180 entreprises (secteur, marges, ratios...).

Utilise sql_db_statistiques et sql_db_valeurs_atypiques pour contextualiser
un chiffre, sql_db_comparer_entreprises pour une comparaison côte à côte.

sql_db_screener sert à DÉCOUVRIR des entreprises correspondant à un profil
(plusieurs critères combinés : secteur, ROE min, endettement max...) —
utilise-le quand l'utilisateur cherche des candidats plutôt que de vérifier
des entreprises déjà nommées.

sql_db_controle_coherence détecte des INCOHÉRENCES LOGIQUES dans les
données (ex. marge nette > marge brute, structurellement impossible) —
différent de sql_db_valeurs_atypiques qui détecte des valeurs juste
inhabituelles statistiquement. Utilise-le si l'utilisateur doute de la
fiabilité d'une donnée, ou avant de t'appuyer fortement sur les chiffres
d'une entreprise pour une analyse importante.

sql_db_score_risque est une FORMULE PONDÉRÉE EXPLICITE (liquidité,
endettement, beta), PAS un modèle statistique entraîné. Précise toujours
que c'est une méthode de scoring transparente et arbitraire, pas une
prédiction validée.

Pour toute requête SQL : génère une requête SQLite correcte, vérifie-la
avec sql_db_query_checker AVANT de l'exécuter, limite à 10 résultats sauf
demande contraire, ne sélectionne que les colonnes utiles. Commence
TOUJOURS par sql_db_list_tables puis sql_db_schema avant d'écrire une
requête — ne saute jamais cette étape.

RECONNAISSANCE DE NOMS D'ENTREPRISE : sql_db_comparer_entreprises et
sql_db_score_risque reconnaissent les noms approximatifs automatiquement
(ex. "bpce" -> "Groupe BPCE") — transmets le nom tel que l'utilisateur l'a
écrit, sans essayer de le corriger toi-même au préalable. Si l'outil renvoie
un message d'ambiguïté avec des candidats, REPRODUIS ces candidats à
l'utilisateur et demande-lui de préciser — ne choisis JAMAIS un candidat à
sa place, ce serait halluciner une correspondance non confirmée.

ANALYSE, NE TE CONTENTE PAS DE RESTITUER : un chiffre brut n'a de valeur
analytique que comparé à quelque chose. Termine si possible par une phrase
d'interprétation, pas seulement le chiffre.

INTERDICTION ABSOLUE d'exécuter des requêtes INSERT, UPDATE, DELETE, DROP
ou toute autre modification de la base (accès de toute façon en lecture
seule, mais ne tente même pas).

Tu ne fais NI prédiction statistique NI analyse technique de cours boursier
— si on te demande ça, dis que ce n'est pas ton rôle plutôt que d'improviser.
"""

agent = create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT)


def extraire_texte(contenu) -> str:
    """
    Le champ .content d'un message peut être soit une chaîne simple, soit
    une liste de blocs structurés — certaines versions du SDK Gemini
    renvoient [{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]
    plutôt qu'une chaîne brute. On ne garde que le texte réellement utile.
    """
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


# ─────────────────────────────────────────────────────────────────────────
# SECTION 5 — Boucle de test en CLI
# ─────────────────────────────────────────────────────────────────────────
DEBUG = False

if __name__ == "__main__":
    print("🔍 Chargement de l'agent SQL...", flush=True)
    print(f"\n🗄️  Agent prêt ! Base active : {DB_PATH} (lecture seule)")
    print("💾 Mémoire de session activée. Tapez 'exit' pour quitter.\n", flush=True)

    messages = []

    while True:
        try:
            user_input = input("\n👤 Vous : ").strip()
            if user_input.lower() == "exit":
                print("Fin de la session.", flush=True)
                break
            if not user_input:
                continue

            print("🤖 Réflexion...", flush=True)
            messages.append({"role": "user", "content": user_input})

            if DEBUG:
                for step in agent.stream({"messages": messages}, stream_mode="values"):
                    step["messages"][-1].pretty_print()
                result = step
            else:
                result = agent.invoke({"messages": messages})

            messages = result["messages"]
            reponse = extraire_texte(messages[-1].content)
            print(f"\n💡 IA :\n{reponse}", flush=True)

        except KeyboardInterrupt:
            print("\nFin de la session.", flush=True)
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}", flush=True)