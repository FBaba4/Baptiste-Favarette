"""
tracer_facteurs.py — Les figures du projet

  1. loadings.png : la forme des trois facteurs (niveau / pente / courbure)
     — LA figure du README, celle qui montre le résultat.
  2. variance.png : variance expliquée par composante (scree plot) et cumul.
  3. facteurs_temps.png : les scores dans le temps — PC2 raconte l'histoire
     de la politique monétaire.

Lancer : python python/tracer_facteurs.py [--demo]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donnees import courbe_synthetique, charger_fred
from acp_courbe import acp

DOSSIER = Path(__file__).resolve().parent.parent / "figures"
DOSSIER.mkdir(exist_ok=True)

COULEURS = ["#2E86AB", "#C0392B", "#1E8449"]
NOMS = ["PC1 — niveau", "PC2 — pente", "PC3 — courbure"]


def figure_loadings(res, suffixe=""):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i in range(3):
        ax.plot(res.maturites, res.loadings[:, i], "o-", ms=5, lw=1.6,
                color=COULEURS[i],
                label=f"{NOMS[i]}  ({res.variance_expliquee[i]:.1%})")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(res.maturites)
    ax.set_xticklabels([f"{m:g}" for m in res.maturites])
    ax.set_xlabel("Maturité (années, échelle log)")
    ax.set_ylabel("Charge (loading)")
    ax.set_title("Les trois facteurs de la courbe des taux\n"
                 f"({res.variance_cumulee(3):.1%} de la variance des variations quotidiennes)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    chemin = DOSSIER / f"loadings{suffixe}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


def figure_variance(res, suffixe=""):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    k = min(8, len(res.variance_expliquee))
    x = np.arange(1, k + 1)
    ax.bar(x, res.variance_expliquee[:k] * 100, color="#2E86AB", alpha=0.85,
           label="par composante")
    ax.plot(x, np.cumsum(res.variance_expliquee[:k]) * 100, "o-", color="#C0392B",
            lw=1.6, label="cumul")
    ax.axhline(95, ls=":", color="gray", lw=1.2)
    ax.text(k, 95, " 95 %", va="bottom", ha="right", fontsize=9, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x])
    ax.set_ylabel("Variance expliquée (%)")
    ax.set_title("Combien de facteurs pour expliquer la courbe ?", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    chemin = DOSSIER / f"variance{suffixe}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


def figure_facteurs_temps(res, suffixe=""):
    """Scores cumulés : PC1 suit le niveau des taux, PC2 le cycle de politique
    monétaire. Sur données réelles, on lit les hausses de 2022 à l'œil nu."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    cumul = res.scores.cumsum()  # les scores sont des variations → on cumule
    for i, ax in enumerate(axes):
        ax.plot(cumul.index, cumul.iloc[:, i], lw=0.9, color=COULEURS[i])
        ax.set_ylabel(NOMS[i].split("—")[1].strip(), fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title("Trajectoire cumulée des trois facteurs", fontsize=11)
    fig.tight_layout()
    chemin = DOSSIER / f"facteurs_temps{suffixe}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("MODE DÉMO — figures sur courbe synthétique")
        courbe, suffixe = courbe_synthetique(), "_DEMO"
    else:
        courbe, suffixe = charger_fred(), ""
    res = acp(courbe, sur_variations=True)
    for f in (figure_loadings, figure_variance, figure_facteurs_temps):
        print(f"  ✓ {f(res, suffixe).name}")