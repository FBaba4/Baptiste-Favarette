# TechnicalAnalysis.py

"""
analyse_technique.py — Momentum et accélération du cours boursier (MAF)

Distinct de predictor.py : ici on fait de l'analyse TECHNIQUE (série
temporelle du cours d'UNE entreprise), pas de l'analyse FONDAMENTALE en
coupe transversale (comparaison entre entreprises à un instant donné).

⚠️ Statut épistémique différent de predictor.py : ceci n'est PAS un modèle
entraîné/validé (pas de R², pas de train/test split). C'est un indicateur
descriptif de tendance, dans la tradition de l'analyse technique — à ne
jamais présenter avec le même niveau de confiance qu'une prédiction ML.
"""

import yfinance as yf
import pandas as pd
from langchain_core.tools import tool


def recuperer_historique(ticker: str, periode: str = "6mo") -> pd.Series:
    """Récupère le cours de clôture historique d'un ticker sur une période donnée.
    Périodes valides yfinance : 1mo, 3mo, 6mo, 1y, 2y, 5y, max."""
    hist = yf.Ticker(ticker).history(period=periode)
    if hist.empty:
        raise ValueError(f"Aucune donnée historique disponible pour le ticker '{ticker}'.")
    return hist["Close"]


def calculer_derivees(ticker: str, periode: str = "6mo", fenetre_lissage: int = 5) -> dict:
    """
    Calcule le momentum (dérivée première) et l'accélération (dérivée
    seconde) du cours de clôture, après lissage par moyenne mobile.

    fenetre_lissage : nombre de jours de la moyenne mobile appliquée avant
    dérivation. Plus elle est grande, plus le signal est stable mais moins
    réactif — 5 jours est un compromis standard en analyse technique.
    """
    cours = recuperer_historique(ticker, periode)

    # Lissage avant dérivation : indispensable, sinon la dérivée seconde
    # d'un cours brut est presque uniquement du bruit statistique.
    cours_lisse = cours.rolling(window=fenetre_lissage, min_periods=1).mean()

    derivee_1 = cours_lisse.diff()   # momentum : variation d'un jour à l'autre
    derivee_2 = derivee_1.diff()     # accélération : variation du momentum

    momentum_recent = float(derivee_1.iloc[-1])
    acceleration_recente = float(derivee_2.iloc[-1])

    if momentum_recent > 0 and acceleration_recente > 0:
        tendance = "accélération haussière"
    elif momentum_recent > 0 and acceleration_recente < 0:
        tendance = "hausse qui ralentit (essoufflement possible)"
    elif momentum_recent < 0 and acceleration_recente < 0:
        tendance = "accélération baissière"
    else:
        tendance = "baisse qui ralentit (possible retournement)"

    return {
        "ticker": ticker,
        "dernier_cours": float(cours.iloc[-1]),
        "momentum_recent": momentum_recent,
        "acceleration_recente": acceleration_recente,
        "tendance": tendance,
        "periode_analysee": periode,
        "nb_points": len(cours),
    }


@tool
def analyser_momentum_action(ticker: str) -> str:
    """
    Analyse le momentum et l'accélération du cours boursier d'une entreprise
    (ex: 'AAPL', 'MC.PA') à partir de son historique récent (6 mois).
    Renvoie une lecture technique de la tendance — PAS une prédiction
    statistique validée, contrairement aux outils predict_*.
    """
    try:
        resultat = calculer_derivees(ticker)
        return (
            f"Analyse technique de {resultat['ticker']} (sur {resultat['nb_points']} séances) :\n"
            f"- Dernier cours : {resultat['dernier_cours']:.2f}\n"
            f"- Momentum récent : {resultat['momentum_recent']:+.2f}/jour\n"
            f"- Accélération récente : {resultat['acceleration_recente']:+.3f}\n"
            f"- Lecture : {resultat['tendance']}\n"
            f"⚠️ Indicateur descriptif d'analyse technique, non validé statistiquement "
            f"(contrairement aux modèles predict_* entraînés sur donnees_structurees.csv)."
        )
    except Exception as e:
        return f"Erreur lors de l'analyse de {ticker} : {e}"


if __name__ == "__main__":
    # Test rapide en CLI, indépendant du reste de l'app
    ticker_test = input("Ticker à analyser (ex: AAPL) : ").strip() or "AAPL"
    print(analyser_momentum_action.invoke({"ticker": ticker_test}))