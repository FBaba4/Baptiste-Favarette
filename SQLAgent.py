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
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
load_dotenv()  # charge GOOGLE_API_KEY depuis .env — jamais codée en dur

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

# Outils des deux autres briques de MAF, ajoutés à ce même agent : requêtage
# structuré (SQL), prédiction ML (predictor.py) et analyse technique de
# cours boursier (TechnicalAnalysis.py) deviennent accessibles ensemble,
# au lieu d'être cloisonnés dans des scripts séparés.
from predictor import OUTILS_PREDICTION
from TechnicalAnalysis import analyser_momentum_action

DB_PATH = "maf.db"

if not Path(DB_PATH).exists():
    raise FileNotFoundError(
        f"{DB_PATH} introuvable — lance d'abord `python migrate_to_sqlite.py` "
        f"pour générer la base à partir de donnees_structurees.csv."
    )

# gemini-2.5-flash : cohérent avec le modèle utilisé ailleurs dans MAF
# (DKMQuery.py) — plus coûteux que flash-lite, à surveiller si tu recroises
# le quota gratuit comme précédemment.
model = init_chat_model("google_genai:gemini-3.1-flash-lite")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — Accès à la base : connexion en lecture seule
# ─────────────────────────────────────────────────────────────────────────
# Le tutoriel officiel avertit explicitement : "scope database connection
# permissions as narrowly as possible". On applique cette recommandation à
# la lettre via le mode=ro de l'URI SQLite : même si le LLM générait une
# requête DROP/DELETE par erreur, la connexion refuserait l'écriture au
# niveau du système de fichiers — une protection indépendante du prompt.
def _connexion_lecture_seule() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — Outils SQL exposés au LLM
# ─────────────────────────────────────────────────────────────────────────
# Quatre outils minimalistes, dans l'esprit du tutoriel officiel : lister
# les tables, lire un schéma, exécuter une requête, et vérifier une requête
# avant exécution. Chacun ouvre/ferme sa propre connexion (léger, mais
# évite tout risque de connexion partagée entre appels concurrents).

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


@tool
def sql_db_comparer_entreprises(entreprises: str, colonnes: str = "") -> str:
    """
    Compare plusieurs entreprises côte à côte sur les colonnes indiquées.
    entreprises : noms séparés par des virgules (doivent correspondre exactement
    à la colonne 'entreprise' — vérifie l'orthographe avec une requête si besoin).
    colonnes : colonnes à comparer, séparées par des virgules. Si vide, utilise
    une sélection par défaut (secteur, marge nette, ROE, ROA, P/E, croissance CA).
    Exemple d'input : entreprises='Apple Inc.,Microsoft Corporation', colonnes=''"""
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)
        noms = [n.strip() for n in entreprises.split(",") if n.strip()]
        df_filtre = df[df["entreprise"].isin(noms)]

        if df_filtre.empty:
            return f"Aucune entreprise trouvée pour : {entreprises}. Vérifie l'orthographe exacte (colonne 'entreprise')."

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
        entreprises_trouvees = df_filtre["entreprise"].tolist()
        manquantes_noms = set(noms) - set(entreprises_trouvees)
        note = f"\n⚠️ Non trouvées : {', '.join(manquantes_noms)}" if manquantes_noms else ""
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
    transparente et assumée, jamais comme une prédiction validée."""
    con = _connexion_lecture_seule()
    try:
        df = pd.read_sql_query("SELECT * FROM entreprises", con)

        colonnes_requises = ["entreprise", "ratio_liquidite_generale", "ratio_dette_capital_pct", "beta"]
        manquantes = [c for c in colonnes_requises if c not in df.columns]
        if manquantes:
            return f"Colonnes manquantes pour ce calcul : {manquantes}. Vérifie le schéma avec sql_db_schema."

        ligne = df[df["entreprise"] == entreprise]
        if ligne.empty:
            return f"Entreprise '{entreprise}' introuvable. Vérifie l'orthographe exacte."
        ligne = ligne.iloc[0]

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
                return 50.0  # valeur neutre si le panel n'a aucune donnée exploitable
            return (serie < valeur).mean() * 100

        pct_liquidite = _percentile_dans_panel("ratio_liquidite_generale", liquidite)
        pct_endettement = _percentile_dans_panel("ratio_dette_capital_pct", endettement)
        pct_beta = _percentile_dans_panel("beta", beta)

        # Liquidité inversée : un percentile élevé de liquidité = MOINS risqué.
        score_risque = 0.40 * (100 - pct_liquidite) + 0.35 * pct_endettement + 0.25 * pct_beta

        if score_risque < 33:
            niveau = "Faible"
        elif score_risque < 66:
            niveau = "Modéré"
        else:
            niveau = "Élevé"

        return (
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


TOOLS = [
    sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker,
    sql_db_statistiques, sql_db_valeurs_atypiques,
    sql_db_comparer_entreprises, sql_db_score_risque,
] + OUTILS_PREDICTION + [analyser_momentum_action]


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — System prompt de l'agent
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
Tu es l'agent d'analyse de MAF (Mon Analyseur Financier). Tu as accès à
quatre familles d'outils, à ne jamais confondre :

1. OUTILS SQL (sql_db_*) : interrogent une base SQLite de données réelles
   collectées sur ~180 entreprises (secteur, marges, ratios...). Utilise
   sql_db_statistiques et sql_db_valeurs_atypiques pour contextualiser un
   chiffre, sql_db_comparer_entreprises pour une comparaison côte à côte.

2. SCORE DE RISQUE (sql_db_score_risque) : une FORMULE PONDÉRÉE EXPLICITE
   (liquidité, endettement, beta), PAS un modèle statistique entraîné.
   Toujours présenter ce score en précisant que c'est une méthode de
   scoring transparente et arbitraire, pas une prédiction validée.

3. OUTILS DE PRÉDICTION (predict_*) : des modèles Random Forest entraînés
   sur le panel, qui ESTIMENT un indicateur à partir de trois paramètres
   d'entrée (chiffre d'affaires, bénéfice net, marge brute). À utiliser
   seulement pour une estimation à partir de paramètres hypothétiques.

4. ANALYSE TECHNIQUE (analyser_momentum_action) : indicateur descriptif de
   tendance de cours boursier, non validé statistiquement.

RÈGLE DE FOND : ne mélange jamais silencieusement ces quatre types de
résultats. Si tu combines plusieurs sources, précise explicitement laquelle
est laquelle et son statut (fait vérifié / formule transparente / modèle
statistique / indicateur descriptif).

Pour toute requête SQL : génère une requête SQLite correcte, vérifie-la
avec sql_db_query_checker AVANT de l'exécuter, limite à 10 résultats sauf
demande contraire, ne sélectionne que les colonnes utiles. Commence
TOUJOURS par sql_db_list_tables puis sql_db_schema avant d'écrire une
requête — ne saute jamais cette étape.

ANALYSE, NE TE CONTENTE PAS DE RESTITUER : un chiffre brut n'a de valeur
analytique que comparé à quelque chose. Termine si possible par une phrase
d'interprétation, pas seulement le chiffre.

INTERDICTION ABSOLUE d'exécuter des requêtes INSERT, UPDATE, DELETE, DROP
ou toute autre modification de la base (accès de toute façon en lecture
seule, mais ne tente même pas).
"""

agent = create_agent(model, TOOLS, system_prompt=SYSTEM_PROMPT)


def extraire_texte(contenu) -> str:
    """
    Le champ .content d'un message peut être soit une chaîne simple, soit
    une liste de blocs structurés — certaines versions du SDK Gemini
    renvoient [{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]
    plutôt qu'une chaîne brute (la signature sert à la vérification interne
    des appels d'outils par Google, aucun intérêt pour l'affichage).
    On ne garde que le texte réellement utile.
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
# Interface volontairement alignée sur celle de DKMQuery.py (👤 Vous / 💡 IA)
# pour rester cohérente dans tout le projet. On utilise agent.invoke() plutôt
# que agent.stream() : on ne veut que la réponse finale, pas le détail de
# chaque étape (appels d'outils, requêtes intermédiaires) affiché à l'écran.
#
# DEBUG=True réaffiche le détail complet (utile pour comprendre pourquoi une
# requête a échoué) sans revenir au code précédent.
DEBUG = False

if __name__ == "__main__":
    print("🔍 Chargement de l'agent SQL...", flush=True)
    print(f"\n🗄️  Agent prêt ! Base active : {DB_PATH} (lecture seule)")
    print("💾 Mémoire de session activée. Tapez 'exit' pour quitter.\n", flush=True)

    messages = []  # historique complet (avec appels d'outils) pour le contexte de l'agent

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

            messages = result["messages"]  # on garde tout pour le tour suivant (contexte de l'agent)
            reponse = extraire_texte(messages[-1].content)
            print(f"\n💡 IA :\n{reponse}", flush=True)

        except KeyboardInterrupt:
            print("\nFin de la session.", flush=True)
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# SECTION 6 (référence, non activée) — Contrôle humain avant exécution SQL
# ─────────────────────────────────────────────────────────────────────────
# La doc officielle propose un middleware human-in-the-loop qui met en
# pause l'agent avant chaque sql_db_query, pour validation manuelle — utile
# pour un vrai outil d'audit, mais ajoute de la complexité (checkpointer,
# reprise via Command). Pas activé par défaut ici ; à réintroduire plus tard
# si tu veux cette garantie supplémentaire :
#
#   from langchain.agents.middleware import HumanInTheLoopMiddleware
#   from langgraph.checkpoint.memory import InMemorySaver
#
#   agent = create_agent(
#       model, TOOLS, system_prompt=SYSTEM_PROMPT,
#       middleware=[HumanInTheLoopMiddleware(
#           interrupt_on={"sql_db_query": True},
#           description_prefix="Requête en attente de validation",
#       )],
#       checkpointer=InMemorySaver(),
#   )