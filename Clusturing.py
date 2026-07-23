import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CSV_PATH = "donnees_structurees.csv"
CSV_SORTIE = "donnees_avec_clusters.csv"

# Colonnes utilisées pour le clustering — volontairement plus larges que les
# 3 features du prédicteur ML, puisqu'on cherche ici à capturer un "profil"
# financier complet, pas à prédire une cible précise.
FEATURES_CLUSTERING = [
    "Marge brute (%)", "Marge nette (%)", "Marge opérationnelle (%)",
    "Rentabilité Capitaux (ROE)", "Rentabilité Actifs (ROA)",
    "Croissance CA (%)", "P/E Ratio (Valorisation)", "P/B Ratio (Valo Bancaire)",
    "Ratio Dette/Capital (%)", "Ratio liquidité générale", "Beta",
]


def _nettoyer_colonne(serie: pd.Series) -> pd.Series:
    """% et virgules -> float, comme partout ailleurs dans MAF."""
    return pd.to_numeric(
        serie.astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def preparer_donnees(csv_path: str = CSV_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Charge et nettoie les features numériques, avec imputation par la médiane
    plutôt que suppression des lignes (préserve la taille de l'échantillon)."""
    df = pd.read_csv(csv_path)

    colonnes_dispo = [c for c in FEATURES_CLUSTERING if c in df.columns]
    manquantes = set(FEATURES_CLUSTERING) - set(colonnes_dispo)
    if manquantes:
        print(f"⚠️ Colonnes absentes du CSV, ignorées : {manquantes}")

    df_num = df[colonnes_dispo].apply(_nettoyer_colonne)
    df_num = df_num.fillna(df_num.median(numeric_only=True))

    df_propre = df[["Entreprise", "Secteur"]].join(df_num).dropna()
    return df_propre, colonnes_dispo


def choisir_k(X_standardise: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """
    Inertie + score de silhouette pour plusieurs valeurs de k. Silhouette
    proche de 1 = clusters bien séparés ; proche de 0 = clusters qui se
    chevauchent. Sert à choisir k de façon défendable, pas arbitraire.
    """
    resultats = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_standardise)
        silhouette = silhouette_score(X_standardise, labels)
        resultats.append({"k": k, "inertie": round(kmeans.inertia_, 1), "silhouette": round(silhouette, 3)})
    return pd.DataFrame(resultats)


def executer_clustering(csv_path: str = CSV_PATH, k: int | None = None) -> dict:
    """
    Exécute le clustering complet : préparation, standardisation, choix de k
    (si non fourni), entraînement final, et interprétation des clusters.
    Retourne un dict avec le DataFrame enrichi + les diagnostics.
    """
    df, colonnes = preparer_donnees(csv_path)
    if len(df) < 10:
        raise ValueError(f"Pas assez d'entreprises exploitables ({len(df)}) pour un clustering pertinent.")

    # Standardisation INDISPENSABLE : k-means est basé sur des distances, et
    # nos features ont des échelles très différentes (un P/E ratio ~20 et un
    # ROE ~15% n'ont pas la même unité) — sans standardisation, la variable à
    # la plus grande échelle dominerait artificiellement le clustering.
    scaler = StandardScaler()
    X_standardise = scaler.fit_transform(df[colonnes])

    diagnostic_k = choisir_k(X_standardise)

    if k is None:
        k = int(diagnostic_k.loc[diagnostic_k["silhouette"].idxmax(), "k"])
        print(f"k choisi automatiquement (meilleure silhouette) : {k}")

    kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans_final.fit_predict(X_standardise)

    # Interprétation : profil moyen de chaque cluster (dans l'échelle
    # d'origine, pas standardisée — plus lisible pour un humain).
    profils = df.groupby("Cluster")[colonnes].mean().round(2)

    # Le résultat le plus intéressant pour ton dossier : le cluster
    # statistique correspond-il au secteur déclaré, ou pas ?
    correspondance_secteur = pd.crosstab(df["Cluster"], df["Secteur"])

    return {
        "donnees": df,
        "diagnostic_k": diagnostic_k,
        "k_retenu": k,
        "profils_clusters": profils,
        "correspondance_secteur": correspondance_secteur,
        "colonnes_utilisees": colonnes,
    }


if __name__ == "__main__":
    resultats = executer_clustering()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC — choix du nombre de clusters (k)")
    print("=" * 60)
    print(resultats["diagnostic_k"].to_string(index=False))
    print(f"\n✅ k retenu : {resultats['k_retenu']}")

    print("\n" + "=" * 60)
    print("PROFIL MOYEN DE CHAQUE CLUSTER")
    print("=" * 60)
    print(resultats["profils_clusters"].to_string())

    print("\n" + "=" * 60)
    print("CLUSTER STATISTIQUE vs SECTEUR DÉCLARÉ")
    print("(Un cluster qui mélange plusieurs secteurs = ressemblance financière")
    print(" statistique malgré une classification sectorielle différente)")
    print("=" * 60)
    print(resultats["correspondance_secteur"].to_string())

    resultats["donnees"].to_csv(CSV_SORTIE, index=False)
    print(f"\n✅ Résultats sauvegardés dans {CSV_SORTIE} (colonne 'Cluster' ajoutée).")