"""
Graph.py — Produit les figures du README

Deux figures, chacune répondant à une question qu'un lecteur se pose :

  1. convergence.png    → "vos trois méthodes donnent-elles le même prix ?"
  2. reduction_variance.png → "à quoi servent les variables antithétiques ?"

Lancer : python Graph.py
Sortie  : notebooks/figures/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans affichage — indispensable en CI
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from BlackScholes import prix_call
from Binomial import convergence as convergence_binomiale
from MonteCarlo import prix_monte_carlo

DOSSIER = Path(__file__).resolve().parent / "figures"
DOSSIER.mkdir(exist_ok=True)

# Paramètres de référence, identiques dans tout le projet
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
REFERENCE = prix_call(S, K, T, r, sigma)
SEED = 42


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 — convergence des trois méthodes
# ─────────────────────────────────────────────────────────────────────────
def figure_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panneau gauche : l'arbre binomial ---
    Ns, prix_arbre = convergence_binomiale(S, K, T, r, sigma, N_max=150)
    ax1.plot(Ns, prix_arbre, lw=1.0, color="#2E86AB", label="Arbre binomial (CRR)")
    ax1.axhline(REFERENCE, color="#C0392B", ls="--", lw=1.5,
                label=f"Black-Scholes = {REFERENCE:.4f}")
    ax1.set_xlabel("Nombre de pas N")
    ax1.set_ylabel("Prix du call")
    ax1.set_title("Arbre binomial → Black-Scholes\n(l'oscillation est structurelle, pas un bug)",
                  fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(REFERENCE - 0.6, REFERENCE + 0.6)

    # --- Panneau droit : le Monte-Carlo, avec son intervalle de confiance ---
    Ms = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    prix_mc, bornes_basses, bornes_hautes = [], [], []
    for M in Ms:
        res = prix_monte_carlo(S, K, T, r, sigma, M=M, seed=SEED)
        bas, haut = res.intervalle_confiance(0.95)
        prix_mc.append(res.prix)
        bornes_basses.append(bas)
        bornes_hautes.append(haut)

    ax2.fill_between(Ms, bornes_basses, bornes_hautes, alpha=0.25, color="#2E86AB",
                     label="IC 95 %")
    ax2.plot(Ms, prix_mc, "o-", ms=4, lw=1.2, color="#2E86AB", label="Monte-Carlo")
    ax2.axhline(REFERENCE, color="#C0392B", ls="--", lw=1.5,
                label=f"Black-Scholes = {REFERENCE:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Nombre de simulations M (échelle log)")
    ax2.set_ylabel("Prix du call")
    ax2.set_title("Monte-Carlo → Black-Scholes\n(la référence reste dans l'IC 95 %)",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Convergence des trois méthodes — call européen "
                 f"(S={S}, K={K}, T={T} an, r={r:.0%}, σ={sigma:.0%})",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    chemin = DOSSIER / "convergence.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — réduction de variance
# ─────────────────────────────────────────────────────────────────────────
def figure_reduction_variance():
    Ms = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    err_std, err_anti = [], []
    for M in Ms:
        err_std.append(prix_monte_carlo(S, K, T, r, sigma, M=M,
                                        antithetique=False, seed=SEED).erreur_standard)
        err_anti.append(prix_monte_carlo(S, K, T, r, sigma, M=M,
                                         antithetique=True, seed=SEED).erreur_standard)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(Ms, err_std, "o-", ms=5, color="#C0392B", label="Monte-Carlo standard")
    ax.loglog(Ms, err_anti, "s-", ms=5, color="#2E86AB", label="Variables antithétiques")

    # Droite de référence en 1/√M pour montrer que la PENTE est la même
    reference_pente = err_std[0] * np.sqrt(Ms[0]) / np.sqrt(np.array(Ms))
    ax.loglog(Ms, reference_pente, ":", color="gray", lw=1.5, label="pente théorique 1/√M")

    gain_moyen = np.mean(np.array(err_std) / np.array(err_anti))
    ax.set_xlabel("Nombre de simulations M")
    ax.set_ylabel("Erreur standard")
    ax.set_title("Variables antithétiques : même pente, meilleure constante\n"
                 f"gain moyen ≈ {gain_moyen:.2f}× ⟹ corrélation ρ ≈ {1/gain_moyen**2 - 1:+.2f}",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    chemin = DOSSIER / "reduction_variance.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin, gain_moyen


if __name__ == "__main__":
    print("Génération des figures...\n")
    c1 = figure_convergence()
    print(f"  ✓ {c1.name}")
    c2, gain = figure_reduction_variance()
    print(f"  ✓ {c2.name}  (gain antithétique mesuré : {gain:.3f}×)")
    print(f"\nFigures écrites dans {DOSSIER}/")