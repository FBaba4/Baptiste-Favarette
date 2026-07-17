"""
tracer_smile.py — La courbe : volatilité implicite en fonction du strike

C'est LA figure du projet. Le tableau de chiffres de smile.py ne dit rien à
l'œil ; la courbe, si.

CE QU'ELLE MONTRE
------------------
Axe X : le strike (ou le log-moneyness, plus rigoureux pour comparer des
        maturités).
Axe Y : la volatilité implicite.

Si Black-Scholes était vrai, ce serait une DROITE HORIZONTALE : une seule
volatilité, la même pour tous les strikes. La ligne pointillée grise du
graphe représente cette prédiction du modèle.

Ce qu'on observe : une courbe. Sur un indice actions, une pente descendante
(skew) — les strikes bas se traitent à une vol implicite plus élevée. L'écart
entre la droite théorique et la courbe observée, c'est la mesure visuelle de
la fausseté du modèle.

Lancer :  python python/tracer_smile.py SPY        (données réelles)
          python python/tracer_smile.py --demo     (synthétique, sans réseau)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smile import (
    charger_chaine_options, calculer_smile, chaine_synthetique, _annees_jusqu_a
)

DOSSIER = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
DOSSIER.mkdir(parents=True, exist_ok=True)


def tracer(df_calls, df_puts, spot, T, ticker, expiration):
    """
    Trace le smile. Deux panneaux :
      - gauche  : vol implicite vs strike (lecture directe)
      - droite  : vol implicite vs log-moneyness (axe normalisé, comparable
                  entre maturités et entre sous-jacents)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, x_calls, x_puts, xlabel in (
        (ax1, "strike", "strike", "Strike"),
        (ax2, "log_moneyness", "log_moneyness", "Log-moneyness  log(K/F)"),
    ):
        if not df_calls.empty:
            ax.plot(df_calls[x_calls], df_calls["vol_implicite"] * 100,
                    "o-", ms=5, lw=1.3, color="#2E86AB", label="Calls")
        if df_puts is not None and not df_puts.empty:
            ax.plot(df_puts[x_puts], df_puts["vol_implicite"] * 100,
                    "s-", ms=5, lw=1.3, color="#C0392B", label="Puts")

        # LA prédiction de Black-Scholes : une vol constante.
        # On prend la vol ATM comme référence — c'est la vol qu'un utilisateur
        # naïf du modèle appliquerait à toute la chaîne.
        if not df_calls.empty:
            i_atm = (df_calls["moneyness"] - 1.0).abs().idxmin()
            vol_atm = df_calls.loc[i_atm, "vol_implicite"] * 100
            ax.axhline(vol_atm, ls=":", color="gray", lw=1.8,
                       label=f"Prédiction Black-Scholes ({vol_atm:.1f} %)")

        # Repère du spot / de la monnaie
        if x_calls == "strike":
            ax.axvline(spot, ls="--", color="black", lw=0.8, alpha=0.4)
            ax.text(spot, ax.get_ylim()[1], " spot", fontsize=8,
                    va="top", alpha=0.6)
        else:
            ax.axvline(0.0, ls="--", color="black", lw=0.8, alpha=0.4)
            ax.text(0.0, ax.get_ylim()[1], " ATM forward", fontsize=8,
                    va="top", alpha=0.6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Volatilité implicite (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_title("Vol implicite vs strike", fontsize=10)
    ax2.set_title("Vol implicite vs log-moneyness\n(axe normalisé, comparable entre maturités)",
                  fontsize=10)

    fig.suptitle(
        f"Smile de volatilité — {ticker} @ {expiration}  "
        f"(spot = {spot:.2f}, T = {T*365:.0f} jours)\n"
        f"Si Black-Scholes était vrai, la courbe serait la ligne pointillée",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    chemin = DOSSIER / f"smile_{ticker}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


def _lecture(df, spot):
    """Quantifie la pente du smile — ce qu'on doit savoir commenter."""
    if len(df) < 3:
        return None
    bas, haut = df.iloc[0], df.iloc[-1]
    i_atm = (df["moneyness"] - 1.0).abs().idxmin()
    atm = df.loc[i_atm]
    pente = (haut["vol_implicite"] - bas["vol_implicite"]) * 100
    return {
        "vol_atm": atm["vol_implicite"] * 100,
        "vol_bas": bas["vol_implicite"] * 100,
        "vol_haut": haut["vol_implicite"] * 100,
        "moneyness_bas": bas["moneyness"],
        "moneyness_haut": haut["moneyness"],
        "amplitude": pente,
    }


def main():
    r = 0.05

    if "--demo" in sys.argv:
        print("MODE DÉMO — skew synthétique imposé (aucun réseau)")
        spot, T, ticker, expiration = 100.0, 0.25, "DEMO", "synthétique"
        brute = chaine_synthetique(spot=spot, T=T, r=r)
        df_calls = calculer_smile(brute, spot, T, r, is_call=True)
        df_puts = None
    else:
        ticker = next((a for a in sys.argv[1:] if not a.startswith("--")), "SPY")
        print(f"Chargement de la chaîne {ticker}...")
        spot, expiration, calls, puts = charger_chaine_options(ticker)
        T = _annees_jusqu_a(expiration)
        print(f"Spot = {spot:.2f} | {expiration} | T = {T*365:.0f} jours")
        df_calls = calculer_smile(calls, spot, T, r, is_call=True)
        df_puts = calculer_smile(puts, spot, T, r, is_call=False)

    if df_calls.empty:
        print("\n⚠️  Aucune cotation exploitable — impossible de tracer.")
        print("    Lance d'abord `python python/smile.py` pour le diagnostic")
        print("    des filtres, ou essaie pendant les heures de marché US")
        print("    (15h30–22h heure de Paris).")
        return

    chemin = tracer(df_calls, df_puts, spot, T, ticker, expiration)
    print(f"\n✓ Figure écrite : {chemin}")

    info = _lecture(df_calls, spot)
    if info:
        print(f"\nLECTURE DE LA COURBE (calls)")
        print(f"  strike bas  (K/S={info['moneyness_bas']:.2f}) : {info['vol_bas']:.2f} %")
        print(f"  à la monnaie                  : {info['vol_atm']:.2f} %")
        print(f"  strike haut (K/S={info['moneyness_haut']:.2f}) : {info['vol_haut']:.2f} %")
        print(f"  amplitude                     : {info['amplitude']:+.2f} points")
        if info["amplitude"] < -1:
            forme = "SKEW décroissant — typique des indices et actions"
            cause = ("les strikes bas coûtent plus cher en vol : queues épaisses, "
                     "effet de levier, et demande structurelle de puts de protection")
        elif info["amplitude"] > 1:
            forme = "SMILE croissant — plutôt typique du FX ou des matières premières"
            cause = "le marché price un risque symétrique dans les deux queues"
        else:
            forme = "quasi plat"
            cause = "rare ; vérifie que les filtres n'ont pas gardé qu'une plage étroite"
        print(f"  → {forme}")
        print(f"    {cause}")
        print(f"\n  Black-Scholes prédirait ces trois chiffres IDENTIQUES.")
        print(f"  L'écart de {abs(info['amplitude']):.1f} points, c'est la mesure de sa fausseté.")


if __name__ == "__main__":
    main()