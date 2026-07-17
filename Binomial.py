"""
binomial.py — Pricing par arbre binomial (modèle Cox-Ross-Rubinstein, 1979)

ÉTAPE 2 du pricer. Deux objectifs, dans cet ordre :

  1. VALIDER : sur une option européenne, l'arbre doit converger vers le prix
     de Black-Scholes quand le nombre de pas augmente. C'est notre test de
     correction croisée (voir tests/test_convergence.py).

  2. DÉPASSER : l'arbre price les options AMÉRICAINES, ce que Black-Scholes
     ne sait pas faire. C'est la vraie raison d'être de ce module — sinon
     autant utiliser la formule fermée, infiniment plus rapide.

POURQUOI ÇA MARCHE (le "pourquoi" derrière le code)
----------------------------------------------------
On discrétise le temps en N pas. À chaque pas, le sous-jacent ne peut faire
que deux choses : monter d'un facteur u, ou descendre d'un facteur d. C'est
brutal comme hypothèse — mais quand N → ∞, la loi binomiale des trajectoires
converge vers la loi lognormale de Black-Scholes (théorème central limite).
L'arbre N'EST PAS une approximation grossière : c'est le même modèle, vu en
temps discret.

Le pricing se fait à REBOURS (backward induction) :
  - à maturité, on connaît le payoff exactement ;
  - à chaque nœud antérieur, la valeur est l'espérance risque-neutre des deux
    valeurs futures, actualisée d'un pas.

CALIBRAGE CRR (le choix de u, d, p)
------------------------------------
    u = e^{σ√Δt}       d = 1/u        p = (e^{(r−q)Δt} − d) / (u − d)

Deux propriétés à connaître pour un entretien :

  - d = 1/u rend l'arbre RECOMBINANT : monter puis descendre ramène au point
    de départ. Conséquence décisive : l'arbre a N(N+1)/2 nœuds au lieu de 2^N.
    Sans ça, N=1000 serait incalculable (2^1000 trajectoires) ; avec ça, c'est
    instantané.

  - p est la probabilité RISQUE-NEUTRE, pas une probabilité réelle. Comme en
    Black-Scholes, le rendement réel du sous-jacent n'intervient jamais. On
    la construit pour que E[S_{t+Δt}] = S_t·e^{(r−q)Δt} : sous cette mesure,
    l'actif rapporte le taux sans risque, par construction.

CONVENTIONS : identiques à black_scholes.py (T en années, r et sigma annualisés).
"""

from math import exp, sqrt

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Cœur : arbre CRR
# ─────────────────────────────────────────────────────────────────────────
def prix_binomial(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 500,
    q: float = 0.0,
    is_call: bool = True,
    americaine: bool = False,
) -> float:
    """
    Price une option par arbre binomial CRR.

    Paramètres
    ----------
    N : nombre de pas de temps. Plus N est grand, plus on est proche du
        modèle continu — mais le coût est en O(N²). N=500 est un bon
        compromis ; voir le graphe de convergence dans le README.
    americaine : si True, on autorise l'exercice anticipé à chaque nœud.

    Retour : le prix de l'option aujourd'hui (nœud racine).
    """
    if T <= 0:
        payoff = (S - K) if is_call else (K - S)
        return max(payoff, 0.0)
    if N < 1:
        raise ValueError("N doit valoir au moins 1.")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u

    # Probabilité risque-neutre. Si elle sort de [0, 1], le modèle est mal
    # calibré (pas de temps trop grand par rapport à la vol) et le prix
    # obtenu autoriserait un arbitrage : on refuse plutôt que de renvoyer
    # un chiffre silencieusement faux.
    p = (exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Probabilité risque-neutre hors de [0,1] (p={p:.4f}) : augmente N "
            f"ou vérifie les paramètres (condition : d < e^{{(r−q)Δt}} < u)."
        )

    actualisation = exp(-r * dt)

    # --- Étape 1 : les N+1 prix possibles du sous-jacent à maturité ---
    # Au nœud j de la dernière colonne : j hausses et (N−j) baisses.
    # On vectorise avec numpy : indispensable pour que N=1000 reste rapide.
    j = np.arange(N + 1)
    prix_finaux = S * (u ** j) * (d ** (N - j))

    # --- Étape 2 : payoff à maturité (la seule chose qu'on connaît avec certitude) ---
    if is_call:
        valeurs = np.maximum(prix_finaux - K, 0.0)
    else:
        valeurs = np.maximum(K - prix_finaux, 0.0)

    # --- Étape 3 : induction à rebours, colonne par colonne ---
    for i in range(N - 1, -1, -1):
        # Espérance risque-neutre des deux nœuds enfants, actualisée d'un pas.
        # valeurs[1:] = nœuds "hausse", valeurs[:-1] = nœuds "baisse".
        valeurs = actualisation * (p * valeurs[1:] + (1.0 - p) * valeurs[:-1])

        if americaine:
            # C'EST TOUT L'INTÉRÊT DE L'ARBRE : à chaque nœud, le détenteur
            # compare la valeur de continuation (ci-dessus) à celle d'exercer
            # immédiatement, et prend le maximum. Black-Scholes ne peut pas
            # faire ça : il n'a aucune notion des états intermédiaires.
            j = np.arange(i + 1)
            spots = S * (u ** j) * (d ** (i - j))
            intrinseque = (spots - K) if is_call else (K - spots)
            valeurs = np.maximum(valeurs, intrinseque)

    return float(valeurs[0])


# ─────────────────────────────────────────────────────────────────────────
# Commodités
# ─────────────────────────────────────────────────────────────────────────
def prime_exercice_anticipe(S, K, T, r, sigma, N=500, q=0.0, is_call=True) -> float:
    """
    Écart de prix entre l'option américaine et son équivalente européenne.

    Résultat à connaître : sur un CALL sans dividende, cette prime est NULLE.
    Il n'est jamais optimal d'exercer un call américain par anticipation —
    on perd la valeur temps et on paie le strike plus tôt. C'est un bon test
    de correction de l'implémentation, et une question d'entretien classique.

    Sur un PUT, en revanche, la prime est strictement positive : exercer tôt
    permet d'encaisser le strike et de le placer au taux sans risque.
    """
    americaine = prix_binomial(S, K, T, r, sigma, N, q, is_call, americaine=True)
    europeenne = prix_binomial(S, K, T, r, sigma, N, q, is_call, americaine=False)
    return americaine - europeenne


def convergence(S, K, T, r, sigma, q=0.0, is_call=True, N_max=200):
    """
    Renvoie (liste_N, liste_prix) pour tracer la convergence vers
    Black-Scholes. Sert à générer le graphe du README.

    Ce que le graphe montre : le prix OSCILLE autour de la valeur BS en
    convergeant. Cette oscillation est structurelle — elle vient de la
    position du strike par rapport aux nœuds terminaux, qui change avec la
    parité de N. C'est bien connu (effet "sawtooth" du modèle CRR) : ce
    n'est pas un bug, et le montrer prouve qu'on a compris le modèle.
    """
    Ns = list(range(1, N_max + 1))
    prix = [prix_binomial(S, K, T, r, sigma, N, q, is_call, americaine=False) for N in Ns]
    return Ns, prix


# ─────────────────────────────────────────────────────────────────────────
# Vérification rapide
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from BlackScholes import prix_call, prix_put

    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print("=" * 62)
    print("VALIDATION — convergence de l'arbre vers Black-Scholes (call européen)")
    print("=" * 62)
    reference = prix_call(S, K, T, r, sigma)
    print(f"Référence Black-Scholes : {reference:.6f}\n")
    print(f"{'N':>6} | {'prix binomial':>14} | {'écart':>10}")
    print("-" * 36)
    for N in (1, 5, 10, 50, 100, 500, 1000, 5000):
        p = prix_binomial(S, K, T, r, sigma, N=N, is_call=True)
        print(f"{N:>6} | {p:>14.6f} | {abs(p - reference):>10.2e}")

    print("\n" + "=" * 62)
    print("DÉPASSEMENT — ce que Black-Scholes ne sait pas faire")
    print("=" * 62)

    # Call américain sans dividende : la prime doit être nulle.
    prime_call = prime_exercice_anticipe(S, K, T, r, sigma, N=1000, is_call=True)
    print(f"Prime d'exercice anticipé, CALL (sans dividende) : {prime_call:.6f}")
    print("  → attendu ≈ 0 : jamais optimal d'exercer un call américain tôt.")

    # Put américain : la prime est strictement positive.
    prime_put = prime_exercice_anticipe(S, K, T, r, sigma, N=1000, is_call=False)
    put_eu = prix_put(S, K, T, r, sigma)
    put_us = prix_binomial(S, K, T, r, sigma, N=1000, is_call=False, americaine=True)
    print(f"\nPUT européen (Black-Scholes) : {put_eu:.6f}")
    print(f"PUT américain (arbre)        : {put_us:.6f}")
    print(f"Prime d'exercice anticipé    : {prime_put:.6f}")
    print("  → strictement positive : exercer tôt permet de placer le strike.")