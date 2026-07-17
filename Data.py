"""
donnees.py — Chargement de l'historique de la courbe des taux

Deux sources :
  1. FRED (Federal Reserve Economic Data) — les taux souverains US quotidiens,
     gratuits et fiables. Nécessite internet. Aucune clé API : on utilise
     l'export CSV public de fredgraph.
  2. Mode synthétique (--demo) — une courbe simulée par le modèle de
     Nelson-Siegel, pour valider le pipeline sans réseau.

POURQUOI CES MATURITÉS
-----------------------
On prend les taux constants de maturité (Constant Maturity Treasury) :
3 mois, 6 mois, 1, 2, 3, 5, 7, 10, 20, 30 ans. C'est la courbe entière, du
monétaire au très long — il faut ce spectre complet pour que l'ACP puisse
séparer niveau, pente et courbure. Avec seulement 2 maturités, la "courbure"
n'existerait pas.

LE MODÈLE SYNTHÉTIQUE (mode démo)
----------------------------------
Nelson-Siegel décrit une courbe de taux par trois paramètres :

    y(τ) = β₀ + β₁·f₁(τ) + β₂·f₂(τ)

où f₁ décroît avec la maturité τ (facteur de pente) et f₂ est en bosse
(facteur de courbure). On fait évoluer (β₀, β₁, β₂) comme des processus
AR(1) corrélés — et on OBTIENT une courbe dont les mouvements sont, par
construction, gouvernés par trois facteurs.

C'est le test aller-retour idéal : si l'ACP retrouve ~3 facteurs dominants
avec les bonnes formes (plat / pente / bosse), le pipeline est correct.
⚠️ Ces données valident le CODE, pas le fait empirique — celui-ci ne se
démontre que sur les vraies données FRED.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Séries FRED des taux CMT, et maturité en années correspondante
SERIES_FRED = {
    "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0,
    "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}

CACHE = Path(__file__).resolve().parent.parent / "data" / "courbe_us.csv"


def charger_fred(debut: str = "2000-01-01", forcer: bool = False) -> pd.DataFrame:
    """
    Télécharge la courbe US complète depuis FRED et la met en cache.

    Retour : DataFrame indexé par date, une colonne par maturité (en années),
    valeurs en pourcents. Les jours fériés (NaN sur toute la ligne) sont
    retirés ; les NaN isolés sont interpolés PAR MATURITÉ dans le temps —
    jamais entre maturités, ce qui déformerait la structure de la courbe.
    """
    if CACHE.exists() and not forcer:
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        df.columns = [float(c) for c in df.columns]
        return df

    colonnes = {}
    for serie, maturite in SERIES_FRED.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"
        with urllib.request.urlopen(url, timeout=30) as reponse:
            brut = pd.read_csv(io.StringIO(reponse.read().decode()),
                               index_col=0, parse_dates=True)
        colonnes[maturite] = pd.to_numeric(brut.iloc[:, 0], errors="coerce")

    df = pd.DataFrame(colonnes).sort_index()
    df = df.loc[debut:]
    df = df.dropna(how="all")            # jours fériés
    df = df.interpolate(limit=3)          # trous isolés, par maturité
    df = df.dropna()                      # bords encore incomplets

    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    return df


# ─────────────────────────────────────────────────────────────────────────
# Mode démo : Nelson-Siegel simulé
# ─────────────────────────────────────────────────────────────────────────
def _facteurs_nelson_siegel(maturites: np.ndarray, lam: float = 0.5):
    """Les trois fonctions de charge de Nelson-Siegel."""
    x = lam * maturites
    f0 = np.ones_like(maturites)                       # niveau : constant
    f1 = (1 - np.exp(-x)) / x                          # pente : décroît avec τ
    f2 = f1 - np.exp(-x)                               # courbure : en bosse
    return f0, f1, f2


def courbe_synthetique(n_jours: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Simule ~10 ans de courbe des taux pilotée par 3 facteurs Nelson-Siegel
    évoluant en AR(1), plus un bruit idiosyncratique par maturité.

    Le bruit est essentiel : sans lui, l'ACP trouverait EXACTEMENT 3 facteurs
    à 100 % — trop beau pour tester quoi que ce soit. Avec lui, on vérifie que
    l'ACP sépare bien le signal (3 facteurs) du bruit (le reste).
    """
    rng = np.random.default_rng(seed)
    maturites = np.array(sorted(SERIES_FRED.values()))
    f0, f1, f2 = _facteurs_nelson_siegel(maturites)

    # Paramètres AR(1) : persistance forte (les taux sont très autocorrélés),
    # volatilités décroissantes (le niveau bouge plus que la courbure).
    betas = np.array([3.0, -1.0, 0.5])          # état initial (courbe croissante)
    phi = np.array([0.999, 0.998, 0.995])       # persistance
    vol = np.array([0.045, 0.035, 0.030])       # vol quotidienne des facteurs

    lignes = []
    for _ in range(n_jours):
        chocs = rng.standard_normal(3) * vol
        betas = phi * betas + chocs
        courbe = betas[0] * f0 + betas[1] * f1 + betas[2] * f2
        courbe = courbe + rng.standard_normal(len(maturites)) * 0.008  # bruit
        lignes.append(courbe)

    dates = pd.bdate_range("2015-01-01", periods=n_jours)
    return pd.DataFrame(lignes, index=dates, columns=maturites)