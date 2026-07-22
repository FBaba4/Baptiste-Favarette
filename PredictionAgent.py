"""
PredictionAgent.py — Modèles ML de MAF (Mon Analyseur Financier)

Entraîne et expose 4 modèles Random Forest (marge nette, ROA, ROE,
croissance du CA), chacun avec ses propres features. Fournit aussi les
outils LangChain correspondants (OUTILS_PREDICTION pour une estimation
ponctuelle, OUTILS_SCENARIOS pour plusieurs variations d'une feature à la
fois — calcul garanti en Python, jamais confié au LLM).

Anciennement predictor.py — renommé pour suivre la convention de nommage
du reste du projet (SQLAgent.py, WebResearch.py, Monitor.py, ...).
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

CSV_PATH = "donnees_structurees.csv"

# ─────────────────────────────────────────────────────────────────────────
# Registre des indicateurs — CHAQUE indicateur a désormais SES PROPRES
# features, choisies pour avoir un lien économique réel avec la cible.
# Avant cette version, les 4 modèles partageaient les mêmes 3 features
# (CA, bénéfice net, marge brute) — pertinent pour marge nette/ROA, mais
# sans aucune justification théorique pour ROE ou la croissance, d'où les
# R² négatifs observés.
# ─────────────────────────────────────────────────────────────────────────
INDICATEURS_DISPONIBLES = {
    "marge_nette": {
        "target": "Marge nette (%)",
        "label": "Marge nette",
        "features": ["Chiffre d’affaires", "Bénéfice Net", "Marge brute (%)"],
        "descriptions": {
            "Chiffre d’affaires": "Chiffre d'affaires de l'entreprise",
            "Bénéfice Net": "Bénéfice net de l'entreprise",
            "Marge brute (%)": "Marge brute de l'entreprise, en pourcentage",
        },
    },
    "roa": {
        "target": "Rentabilité Actifs (ROA)",
        "label": "ROA (rentabilité des actifs)",
        "features": ["Chiffre d’affaires", "Bénéfice Net", "Marge brute (%)"],
        "descriptions": {
            "Chiffre d’affaires": "Chiffre d'affaires de l'entreprise",
            "Bénéfice Net": "Bénéfice net de l'entreprise",
            "Marge brute (%)": "Marge brute de l'entreprise, en pourcentage",
        },
    },
    # ROE dépend structurellement des capitaux propres et du levier
    # financier, pas du compte de résultat seul — features dédiées.
    "roe": {
        "target": "Rentabilité Capitaux (ROE)",
        "label": "ROE (rentabilité des capitaux propres)",
        "features": ["Bénéfice Net", "Capitaux propres estimés", "Ratio Dette/Capital (%)"],
        "descriptions": {
            "Bénéfice Net": "Bénéfice net de l'entreprise",
            "Capitaux propres estimés": "Capitaux propres (fonds propres) de l'entreprise",
            "Ratio Dette/Capital (%)": "Ratio dette/capitaux propres, en pourcentage (effet de levier)",
        },
    },
    # Croissance CA : sans historique multi-années dans ce CSV (photo à
    # l'instant T), on utilise des proxys de marché (spread BPA/PE
    # trailing-forward) qui reflètent les anticipations de croissance des
    # bénéfices — PAS "Croissance CA yfinance (%)", trop proche de la
    # cible elle-même (fuite de données).
    "croissance_ca": {
        "target": "Croissance CA (%)",
        "label": "Croissance du chiffre d'affaires",
        "features": ["BPA (trailing)", "BPA (forward)", "P/E Ratio (Valorisation)", "Forward P/E"],
        "descriptions": {
            "BPA (trailing)": "Bénéfice par action des 12 derniers mois",
            "BPA (forward)": "Bénéfice par action anticipé (12 prochains mois)",
            "P/E Ratio (Valorisation)": "Ratio cours/bénéfice sur les résultats passés",
            "Forward P/E": "Ratio cours/bénéfice sur les résultats anticipés",
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Feature engineering : colonnes dérivées non présentes telles quelles
# dans le CSV brut, mais nécessaires à certains modèles (ROE).
# ─────────────────────────────────────────────────────────────────────────
def _ajouter_features_derivees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Le CSV ne contient pas directement les capitaux propres. On les estime
    via Valeur comptable/action × Actions en circulation — la définition
    même de la valeur comptable des capitaux propres.
    """
    df = df.copy()
    if "Valeur comptable/action" in df.columns and "Actions en circulation" in df.columns:
        bv = pd.to_numeric(df["Valeur comptable/action"], errors="coerce")
        shares = pd.to_numeric(df["Actions en circulation"], errors="coerce")
        df["Capitaux propres estimés"] = bv * shares
    return df


def _slugifier(nom_colonne: str) -> str:
    """Nom de colonne -> identifiant Python valide, pour les arguments des
    outils LangChain (mêmes règles que la normalisation de migrate_to_sqlite.py,
    pour rester cohérent dans tout le projet)."""
    remplacements = {
        "é": "e", "è": "e", "ê": "e", "à": "a", "î": "i", "ô": "o", "ù": "u",
        "ç": "c", "’": "", "'": "", "(": "", ")": "", "%": "", "/": "_",
        "-": "_", " ": "_",
    }
    resultat = nom_colonne
    for ancien, nouveau in remplacements.items():
        resultat = resultat.replace(ancien, nouveau)
    return resultat.lower().strip("_")


class FinancialPredictor:
    def __init__(self, target_column: str, feature_columns: list[str]):
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, csv_path: str) -> dict:
        df = pd.read_csv(csv_path)
        df = _ajouter_features_derivees(df)

        cols_to_use = self.feature_columns + [self.target_column]
        colonnes_manquantes = [c for c in cols_to_use if c not in df.columns]
        if colonnes_manquantes:
            raise ValueError(f"Colonnes manquantes dans le CSV : {colonnes_manquantes}")

        for col in cols_to_use:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=cols_to_use)

        if len(df) < 5:
            raise ValueError(f"Pas assez de données valides après nettoyage. Lignes trouvées : {len(df)}")

        X = df[self.feature_columns]
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        preds = self.model.predict(X_test)
        return {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(self, features: dict) -> float:
        if not self.is_trained:
            raise RuntimeError("Le modèle doit être entraîné.")
        clean_features = {k: float(str(v).replace('%', '').replace(',', '.')) for k, v in features.items()}
        X = pd.DataFrame([clean_features])[self.feature_columns]
        return float(self.model.predict(X)[0])


# ─────────────────────────────────────────────────────────────────────────
# Registre des modèles entraînés + mapping param_slug -> nom de colonne réel
# (nécessaire puisque les arguments d'outils LangChain doivent être des
# identifiants Python valides, alors que les colonnes ont des accents/espaces)
# ─────────────────────────────────────────────────────────────────────────
_predicteurs: dict = {}
_metriques: dict = {}
_mappings_params: dict = {}  # {cle_indicateur: {param_slug: nom_colonne_reel}}


def entrainer_tous_les_modeles() -> dict:
    resultats = {}
    for cle, config in INDICATEURS_DISPONIBLES.items():
        predicteur = FinancialPredictor(target_column=config["target"], feature_columns=config["features"])
        _mappings_params[cle] = {_slugifier(col): col for col in config["features"]}
        try:
            metrics = predicteur.train(CSV_PATH)
            _predicteurs[cle] = predicteur
            _metriques[cle] = metrics
            resultats[cle] = metrics
            print(f"✅ Modèle '{config['label']}' entraîné : {metrics}")
        except Exception as e:
            _metriques[cle] = {"erreur": str(e)}
            resultats[cle] = {"erreur": str(e)}
            print(f"❌ Échec entraînement '{config['label']}' : {e}")
    return resultats


csv_file = Path(CSV_PATH)
if csv_file.exists():
    print(f"Chargement du fichier : {csv_file.absolute()}")
    entrainer_tous_les_modeles()
else:
    print(f"❌ ERREUR : Le fichier {CSV_PATH} est introuvable dans {Path('.').absolute()}")


def calculer(cle_indicateur: str, **kwargs) -> float:
    """
    Point d'entrée générique : kwargs utilise les noms d'arguments slugifiés
    (ex. 'benefice_net'), reconvertis ici vers les vrais noms de colonnes
    (ex. 'Bénéfice Net') avant de passer au modèle. Fonctionne pour
    n'importe quel indicateur, quelles que soient ses features propres.
    """
    if cle_indicateur not in _predicteurs:
        raise RuntimeError(f"Le modèle '{cle_indicateur}' n'est pas disponible (entraînement échoué ou absent).")
    mapping = _mappings_params[cle_indicateur]
    features = {mapping[slug]: valeur for slug, valeur in kwargs.items() if slug in mapping}
    return _predicteurs[cle_indicateur].predict(features)


def simuler_scenarios(cle_indicateur: str, feature_a_varier: str, variations_pct: list, **base_kwargs) -> list:
    """
    Calcule le même indicateur pour plusieurs variations en pourcentage
    d'UNE SEULE feature (les autres restent constantes) — calcul fait en
    Python garanti, jamais confié à l'arithmétique du LLM.

    feature_a_varier : nom slugifié de la feature à faire varier (ex.
    'chiffre_daffaires'). variations_pct : liste de %, ex. [-30, -20, -10, 0, 10].
    base_kwargs : valeurs de référence pour toutes les features du modèle.

    Détecte et signale explicitement les résultats hors du domaine plausible
    (marge > 100% en valeur absolue) — signe que le scénario combine des
    valeurs jamais observées ensemble dans les données d'entraînement
    (ex. bénéfice net réel + CA hypothétique 40x trop petit).
    """
    if cle_indicateur not in _mappings_params:
        raise RuntimeError(f"Le modèle '{cle_indicateur}' n'est pas disponible.")
    mapping = _mappings_params[cle_indicateur]

    if feature_a_varier not in mapping:
        raise ValueError(
            f"'{feature_a_varier}' n'est pas une feature de ce modèle. "
            f"Features valides : {list(mapping.keys())}"
        )

    valeur_base = base_kwargs.get(feature_a_varier)
    if valeur_base is None:
        raise ValueError(f"Valeur de référence manquante pour '{feature_a_varier}'.")

    resultats = []
    for pct in variations_pct:
        kwargs_scenario = dict(base_kwargs)
        kwargs_scenario[feature_a_varier] = valeur_base * (1 + pct / 100)

        try:
            valeur = calculer(cle_indicateur, **kwargs_scenario)
        except Exception as e:
            resultats.append({"variation_pct": pct, "erreur": str(e)})
            continue

        alerte = None
        if abs(valeur) > 100:
            alerte = (
                "⚠️ Résultat hors du domaine plausible (>100% en valeur absolue) — "
                "ce scénario combine probablement des valeurs jamais observées ensemble "
                "dans les données d'entraînement (n=180). Ne pas interpréter comme fiable."
            )

        resultats.append({
            "variation_pct": pct,
            feature_a_varier: round(kwargs_scenario[feature_a_varier], 2),
            "valeur_predite": round(valeur, 2),
            "alerte": alerte,
        })

    return resultats


class SimulateurMargeNette:
    """
    Simulateur de scénarios dédié à la marge nette — encapsule
    simuler_scenarios() derrière une interface orientée objet plus
    explicite : on fixe les valeurs de référence à la construction, puis on
    simule une ou plusieurs variations sans les répéter à chaque appel.

    C'est le seul indicateur du projet à avoir sa propre classe dédiée
    (les 3 autres — ROA, ROE, croissance — restent sur la fonction
    générique simuler_scenarios) : la marge nette est l'indicateur le plus
    fréquemment simulé en pratique, ce qui justifie une interface à part.
    """

    CLE_INDICATEUR = "marge_nette"

    def __init__(self, chiffre_affaires: float, benefice_net: float, marge_brute: float):
        self.chiffre_affaires = chiffre_affaires
        self.benefice_net = benefice_net
        self.marge_brute = marge_brute

    def _base_kwargs(self) -> dict:
        return {
            "chiffre_daffaires": self.chiffre_affaires,
            "benefice_net": self.benefice_net,
            "marge_brute": self.marge_brute,
        }

    def simuler(self, feature_a_varier: str, variations_pct: list) -> list:
        """Fait varier UNE des trois valeurs de référence, les deux autres
        restant constantes — délègue le calcul à simuler_scenarios()."""
        return simuler_scenarios(self.CLE_INDICATEUR, feature_a_varier, variations_pct, **self._base_kwargs())

    def simuler_chiffre_affaires(self, variations_pct: list) -> list:
        """Raccourci : fait varier le chiffre d'affaires (cas d'usage le plus fréquent)."""
        return self.simuler("chiffre_daffaires", variations_pct)

    def simuler_benefice_net(self, variations_pct: list) -> list:
        return self.simuler("benefice_net", variations_pct)

    def simuler_marge_brute(self, variations_pct: list) -> list:
        return self.simuler("marge_brute", variations_pct)

    def valeur_actuelle(self) -> float:
        """Marge nette prédite pour les valeurs de référence, sans variation (scénario 0%)."""
        return calculer(self.CLE_INDICATEUR, **self._base_kwargs())


def importances(cle_indicateur: str) -> dict:
    predicteur = _predicteurs.get(cle_indicateur)
    if predicteur is None or not predicteur.is_trained:
        return {}
    return dict(zip(predicteur.feature_columns, predicteur.model.feature_importances_.tolist()))


def metriques(cle_indicateur: str) -> dict:
    return _metriques.get(cle_indicateur, {})


def retrain_tous() -> dict:
    return entrainer_tous_les_modeles()


def get_csv_columns() -> list:
    if Path(CSV_PATH).exists():
        return list(pd.read_csv(CSV_PATH, nrows=0).columns)
    return []


def add_row_and_retrain(new_row: dict) -> dict:
    df = pd.read_csv(CSV_PATH)
    ligne = {col: new_row.get(col, "") for col in df.columns}
    df = pd.concat([df, pd.DataFrame([ligne])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return retrain_tous()


def merge_csv_and_retrain(uploaded_file) -> dict:
    df_existing = pd.read_csv(CSV_PATH)
    df_new = pd.read_csv(uploaded_file)

    colonnes_manquantes = set(df_existing.columns) - set(df_new.columns)
    if colonnes_manquantes:
        raise ValueError(
            f"Le fichier importé n'a pas les mêmes colonnes. Colonnes manquantes : "
            f"{', '.join(sorted(colonnes_manquantes))}"
        )

    df_new = df_new[df_existing.columns]
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged.to_csv(CSV_PATH, index=False)
    return retrain_tous()


# ─────────────────────────────────────────────────────────────────────────
# Génération dynamique des outils LangChain — un schéma d'arguments propre
# à CHAQUE indicateur, construit à partir de ses features réelles.
# ─────────────────────────────────────────────────────────────────────────
def _construire_args_schema(cle_indicateur: str, config: dict):
    """Construit dynamiquement un modèle pydantic dont les champs
    correspondent exactement aux features de CET indicateur — impossible
    à coder en dur puisque chaque indicateur a désormais ses propres
    features (contrairement à l'ancienne version à 3 features fixes)."""
    champs = {}
    for col in config["features"]:
        slug = _slugifier(col)
        description = config.get("descriptions", {}).get(col, col)
        champs[slug] = (float, Field(description=description))
    return create_model(f"Args_{cle_indicateur}", **champs)


def _fabriquer_tool(cle_indicateur: str, config: dict) -> StructuredTool:
    label = config["label"]
    args_schema = _construire_args_schema(cle_indicateur, config)

    def _fonction(**kwargs) -> str:
        try:
            valeur = calculer(cle_indicateur, **kwargs)
            return f"{label} prédit(e) : {valeur:.2f}%."
        except Exception as e:
            return f"Erreur de prédiction ({label}) : {str(e)}"

    liste_params = ", ".join(_slugifier(c) for c in config["features"])
    return StructuredTool.from_function(
        func=_fonction,
        name=f"predict_{cle_indicateur}",
        description=(
            f"Prédit {label.lower()} (en %) à partir de : {liste_params}."
        ),
        args_schema=args_schema,
    )


OUTILS_PREDICTION = [
    _fabriquer_tool(cle, config) for cle, config in INDICATEURS_DISPONIBLES.items()
]

# Rétrocompatibilité : ancien nom utilisé ailleurs
predict_marge_nette = next(t for t in OUTILS_PREDICTION if t.name == "predict_marge_nette")


# ─────────────────────────────────────────────────────────────────────────
# Outils de simulation de scénarios — un par indicateur, calcul Python
# garanti (voir simuler_scenarios ci-dessus), pas d'arithmétique confiée au LLM.
# ─────────────────────────────────────────────────────────────────────────
def _construire_args_schema_scenario(cle_indicateur: str, config: dict):
    """Comme _construire_args_schema, plus deux champs de contrôle du scénario :
    quelle feature faire varier, et selon quelles variations en %."""
    champs = {}
    for col in config["features"]:
        slug = _slugifier(col)
        description = config.get("descriptions", {}).get(col, col)
        champs[slug] = (float, Field(description=f"Valeur de référence — {description}"))

    slugs_valides = [_slugifier(c) for c in config["features"]]
    champs["feature_a_varier"] = (
        str,
        Field(description=f"Quelle variable faire varier — parmi : {', '.join(slugs_valides)}"),
    )
    champs["variations_pct"] = (
        list[float],
        Field(description="Liste des variations en pourcentage à tester, ex. [-30, -20, -10, 0, 10]"),
    )
    return create_model(f"ArgsScenario_{cle_indicateur}", **champs)


def _fabriquer_tool_scenario(cle_indicateur: str, config: dict) -> StructuredTool:
    label = config["label"]
    args_schema = _construire_args_schema_scenario(cle_indicateur, config)
    slugs_features = [_slugifier(c) for c in config["features"]]

    def _fonction(**kwargs) -> str:
        feature_a_varier = kwargs.pop("feature_a_varier")
        variations_pct = kwargs.pop("variations_pct")
        try:
            if cle_indicateur == "marge_nette":
                # Seul indicateur à passer par sa classe dédiée (voir
                # SimulateurMargeNette ci-dessus) plutôt que directement par
                # la fonction générique simuler_scenarios().
                simulateur = SimulateurMargeNette(
                    chiffre_affaires=kwargs.get("chiffre_daffaires"),
                    benefice_net=kwargs.get("benefice_net"),
                    marge_brute=kwargs.get("marge_brute"),
                )
                resultats = simulateur.simuler(feature_a_varier, variations_pct)
            else:
                resultats = simuler_scenarios(cle_indicateur, feature_a_varier, variations_pct, **kwargs)
        except Exception as e:
            return f"Erreur de simulation ({label}) : {str(e)}"

        lignes = [f"Simulation de scénarios pour {label} (variable modifiée : {feature_a_varier}) :"]
        for r in resultats:
            if "erreur" in r:
                lignes.append(f"  {r['variation_pct']:+.0f}% -> Erreur : {r['erreur']}")
                continue
            ligne = f"  {r['variation_pct']:+.0f}% ({feature_a_varier}={r[feature_a_varier]}) -> {label} : {r['valeur_predite']:.2f}%"
            if r["alerte"]:
                ligne += f"\n    {r['alerte']}"
            lignes.append(ligne)
        return "\n".join(lignes)

    return StructuredTool.from_function(
        func=_fonction,
        name=f"simuler_{cle_indicateur}",
        description=(
            f"Simule {label.lower()} pour PLUSIEURS variations en pourcentage d'une des "
            f"features ({', '.join(slugs_features)}), les autres restant constantes. "
            f"Calcul garanti en Python, pas d'arithmétique confiée au modèle de langage. "
            f"Utilise cet outil au lieu de predict_{cle_indicateur} dès que l'utilisateur "
            f"demande plusieurs scénarios (ex. '-10%, -20%, -30%') plutôt qu'une seule valeur."
        ),
        args_schema=args_schema,
    )


OUTILS_SCENARIOS = [
    _fabriquer_tool_scenario(cle, config) for cle, config in INDICATEURS_DISPONIBLES.items()
]
