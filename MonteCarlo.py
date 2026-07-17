"""
monte_carlo.py — Pricing par simulation de Monte-Carlo

ÉTAPE 3 du pricer. Sur une option européenne, c'est la méthode la PLUS LENTE
et la MOINS PRÉCISE des trois. On la code quand même, pour deux raisons :

  1. C'est la seule qui se généralise. Dès qu'il n'y a plus de formule fermée
     — options asiatiques, barrières, panier, vol stochastique — Black-Scholes
     et l'arbre s'effondrent, et le Monte-Carlo continue. C'est la méthode de
     référence en salle de marché pour les produits exotiques, donc pour le
     structuring.

  2. C'est le module qu'on réécrira en C++ (étape 6) : le seul dont le coût
     de calcul justifie vraiment un langage compilé.

Ici, on le teste sur un cas qu'on sait résoudre exactement — c'est le seul
moyen de vérifier qu'il est juste.

PRINCIPE (le "pourquoi")
-------------------------
Le prix est une espérance sous la probabilité risque-neutre :

    C = e^{-rT}·E[payoff(S_T)]

Une espérance, ça s'estime par une moyenne empirique (loi des grands nombres).
On simule M trajectoires, on moyenne les payoffs, on actualise. C'est tout.

SOLUTION EXACTE DU GBM (pourquoi on ne discrétise pas le temps)
----------------------------------------------------------------
    S_T = S_0 · exp((r − q − σ²/2)·T + σ·√T·Z),   Z ~ N(0,1)

Point important : pour une option européenne, le payoff ne dépend QUE de S_T,
pas du chemin. On peut donc tirer S_T en UN SEUL PAS, sans discrétiser [0,T].
Il n'y a alors AUCUN biais de discrétisation — l'erreur est purement
statistique. (Pour une option path-dependent, il faudrait simuler le chemin
entier et un biais de discrétisation apparaîtrait : c'est une limite à citer.)

Le terme −σ²/2 est la correction d'Itô. C'est ce qui garantit
E[S_T] = S_0·e^{(r−q)T} : sans lui, l'actif ne rapporterait pas le taux sans
risque en moyenne, et le pricing serait faux. Question d'entretien classique.

LA VITESSE DE CONVERGENCE — la vraie limite de la méthode
-----------------------------------------------------------
L'erreur décroît en O(1/√M), par le théorème central limite. Conséquence
brutale : pour gagner UNE décimale, il faut 100× plus de simulations.
C'est pour ça qu'on réduit la variance (variables antithétiques ci-dessous)
plutôt que d'empiler les tirages.
"""

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np


@dataclass
class ResultatMC:
    """
    Un prix Monte-Carlo N'EST PAS un nombre, c'est une ESTIMATION.
    On renvoie donc systématiquement son incertitude — sinon le chiffre
    est ininterprétable. C'est la différence entre "le prix est 10,45" et
    "le prix est 10,45 ± 0,02 à 95 %".
    """
    prix: float
    erreur_standard: float
    n_simulations: int
    antithetique: bool

    def intervalle_confiance(self, niveau: float = 0.95) -> tuple[float, float]:
        """Intervalle de confiance (approximation normale, valide pour M grand)."""
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(niveau, 1.96)
        marge = z * self.erreur_standard
        return (self.prix - marge, self.prix + marge)

    def __str__(self) -> str:
        bas, haut = self.intervalle_confiance()
        mode = "antithétique" if self.antithetique else "standard"
        return (f"{self.prix:.6f} ± {1.96 * self.erreur_standard:.6f} "
                f"(IC 95 % : [{bas:.4f}, {haut:.4f}] | "
                f"M={self.n_simulations:,} | {mode})".replace(",", " "))


def prix_monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    M: int = 100_000,
    q: float = 0.0,
    is_call: bool = True,
    antithetique: bool = True,
    seed: int | None = None,
) -> ResultatMC:
    """
    Price une option européenne par simulation de Monte-Carlo.

    Paramètres
    ----------
    M : nombre de simulations. En mode antithétique, on tire M/2 normales et
        on les réutilise en miroir — le coût reste M évaluations de payoff.
    antithetique : réduction de variance (voir explication ci-dessous).
    seed : fixe le générateur aléatoire pour des résultats reproductibles.
           INDISPENSABLE dans un repo : un lecteur doit retrouver tes chiffres.

    Retour : un ResultatMC (prix + erreur standard), jamais un float nu.
    """
    if T <= 0:
        payoff = (S - K) if is_call else (K - S)
        return ResultatMC(max(payoff, 0.0), 0.0, 0, antithetique)

    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T   # le −σ²/2 est la correction d'Itô
    diffusion = sigma * sqrt(T)

    if antithetique:
        # VARIABLES ANTITHÉTIQUES : pour chaque tirage Z, on utilise aussi −Z.
        #
        # Pourquoi ça marche : Z et −Z ont la même loi (la normale est
        # symétrique), donc chaque moitié est un estimateur valide. Mais leurs
        # payoffs sont NÉGATIVEMENT corrélés : quand Z pousse le prix en haut,
        # −Z le pousse en bas. Or Var(A+B)/4 = [Var(A) + Var(B) + 2·Cov(A,B)]/4 :
        # une covariance négative RÉDUIT la variance de la moyenne.
        #
        # Bonus : la moyenne empirique des Z est exactement 0 par construction,
        # ce qui élimine un biais d'échantillonnage sur le drift.
        M_effectif = M // 2
        Z = rng.standard_normal(M_effectif)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(M)

    # Tirage direct de S_T — un seul pas, aucun biais de discrétisation.
    S_T = S * np.exp(drift + diffusion * Z)

    payoffs = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)
    payoffs_actualises = exp(-r * T) * payoffs

    prix = float(payoffs_actualises.mean())

    # Erreur standard = écart-type de l'échantillon / √M.
    # ⚠️ En mode antithétique, les tirages ne sont PAS indépendants : cette
    # formule sous-estimerait l'incertitude. On calcule donc l'écart-type sur
    # les PAIRES (moyenne de chaque couple Z/−Z), qui sont, elles, indépendantes.
    if antithetique:
        paires = 0.5 * (payoffs_actualises[:M_effectif] + payoffs_actualises[M_effectif:])
        erreur_standard = float(paires.std(ddof=1) / sqrt(M_effectif))
    else:
        erreur_standard = float(payoffs_actualises.std(ddof=1) / sqrt(M))

    return ResultatMC(prix, erreur_standard, len(payoffs), antithetique)


def delta_pathwise(S, K, T, r, sigma, M=100_000, q=0.0, is_call=True, seed=None) -> float:
    """
    Delta par la méthode "pathwise" — on dérive le payoff, pas le prix.

    Pourquoi c'est mieux que les différences finies : au lieu de repricer deux
    fois en (S+h) et (S−h) et de diviser par 2h (bruité, et sensible au choix
    de h), on dérive analytiquement sous l'espérance :

        ∂/∂S max(S_T − K, 0) = 1_{S_T > K} · (S_T / S)

    Un seul jeu de trajectoires, pas de paramètre h arbitraire, et une variance
    bien plus faible. C'est la méthode utilisée en production.

    Limite : elle exige un payoff différentiable presque partout — ce qui va
    pour un call/put vanille, mais casse sur une option digitale (payoff en
    escalier). À citer si on te pose la question.
    """
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * sqrt(T)
    Z = rng.standard_normal(M // 2)
    Z = np.concatenate([Z, -Z])
    S_T = S * np.exp(drift + diffusion * Z)

    if is_call:
        derivees = (S_T > K) * (S_T / S)
    else:
        derivees = -(S_T < K) * (S_T / S)
    return float(exp(-r * T) * derivees.mean())


def etude_convergence(S, K, T, r, sigma, q=0.0, is_call=True, seed=42):
    """
    Renvoie (liste_M, prix_standard, prix_antithetique, err_standard,
    err_antithetique) pour le graphe du README.

    Ce que le graphe doit montrer : l'erreur décroît en 1/√M dans les deux
    cas (une droite de pente −1/2 en échelle log-log), mais la courbe
    antithétique est DÉCALÉE VERS LE BAS — même vitesse, meilleure constante.
    """
    Ms = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    prix_std, prix_anti, err_std, err_anti = [], [], [], []
    for M in Ms:
        a = prix_monte_carlo(S, K, T, r, sigma, M, q, is_call, antithetique=False, seed=seed)
        b = prix_monte_carlo(S, K, T, r, sigma, M, q, is_call, antithetique=True, seed=seed)
        prix_std.append(a.prix); err_std.append(a.erreur_standard)
        prix_anti.append(b.prix); err_anti.append(b.erreur_standard)
    return Ms, prix_std, prix_anti, err_std, err_anti


# ─────────────────────────────────────────────────────────────────────────
# Vérification rapide
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from BlackScholes import prix_call, delta as delta_bs

    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    reference = prix_call(S, K, T, r, sigma)

    print("=" * 68)
    print("VALIDATION — le Monte-Carlo retrouve-t-il Black-Scholes ?")
    print("=" * 68)
    print(f"Référence Black-Scholes : {reference:.6f}\n")

    for M in (1_000, 10_000, 100_000, 1_000_000):
        res = prix_monte_carlo(S, K, T, r, sigma, M=M, seed=42)
        bas, haut = res.intervalle_confiance()
        dedans = "oui" if bas <= reference <= haut else "NON (⚠️)"
        print(f"M = {M:>9,} | {res.prix:>10.6f} ± {1.96*res.erreur_standard:.6f} "
              f"| BS dans l'IC 95 % : {dedans}".replace(",", " "))

    print("\n" + "=" * 68)
    print("RÉDUCTION DE VARIANCE — l'apport des variables antithétiques")
    print("=" * 68)
    print(f"{'M':>10} | {'err. standard':>13} | {'err. antithét.':>14} | {'gain':>7}")
    print("-" * 54)
    for M in (10_000, 100_000, 1_000_000):
        std = prix_monte_carlo(S, K, T, r, sigma, M=M, antithetique=False, seed=42)
        anti = prix_monte_carlo(S, K, T, r, sigma, M=M, antithetique=True, seed=42)
        gain = std.erreur_standard / anti.erreur_standard
        print(f"{M:>10,} | {std.erreur_standard:>13.6f} | "
              f"{anti.erreur_standard:>14.6f} | {gain:>6.2f}×".replace(",", " "))
    print("\n  → à budget de calcul égal, l'erreur est divisée par ce facteur.")
    print("    Rappel : sans réduction de variance, diviser l'erreur par 2")
    print("    coûterait 4× plus de simulations (convergence en 1/√M).")

    print("\n" + "=" * 68)
    print("VITESSE DE CONVERGENCE — vérification du 1/√M")
    print("=" * 68)
    print("Si l'erreur ∝ 1/√M, multiplier M par 100 doit la diviser par 10 :\n")
    e1 = prix_monte_carlo(S, K, T, r, sigma, M=10_000, seed=42).erreur_standard
    e2 = prix_monte_carlo(S, K, T, r, sigma, M=1_000_000, seed=42).erreur_standard
    print(f"  erreur(M=10 000)    = {e1:.6f}")
    print(f"  erreur(M=1 000 000) = {e2:.6f}")
    print(f"  ratio observé = {e1/e2:.2f}   (attendu ≈ 10)")

    print("\n" + "=" * 68)
    print("DELTA — méthode pathwise vs formule fermée")
    print("=" * 68)
    d_mc = delta_pathwise(S, K, T, r, sigma, M=1_000_000, seed=42)
    d_bs = delta_bs(S, K, T, r, sigma, is_call=True)
    print(f"  Delta pathwise (MC) : {d_mc:.6f}")
    print(f"  Delta Black-Scholes : {d_bs:.6f}")
    print(f"  Écart               : {abs(d_mc - d_bs):.2e}")