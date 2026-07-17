"""
test_convergence.py — Validation croisée des trois méthodes de pricing

C'EST LA PIÈCE CENTRALE DU PROJET. Trois implémentations totalement
indépendantes (formule fermée, arbre, simulation) doivent produire le même
prix. Si elles s'accordent, il est très improbable qu'elles soient toutes
fausses de la même façon — c'est ce qui donne confiance dans le pricer.

On teste deux familles de choses :

  1. CONVERGENCE CROISÉE : les méthodes numériques (arbre, MC) convergent-elles
     vers la référence analytique (Black-Scholes) ?

  2. PROPRIÉTÉS THÉORIQUES : le pricer respecte-t-il des résultats qu'il ne
     "connaît" pas — parité call-put, bornes d'arbitrage, non-exercice du call
     américain ? Ces tests sont les plus forts : ils valident le pricer contre
     la THÉORIE, pas contre lui-même.

Lancer : pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
from math import exp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from BlackScholes import prix_call, prix_put, delta as delta_bs
from Binomial import prix_binomial, prime_exercice_anticipe
from MonteCarlo import prix_monte_carlo, delta_pathwise


# Jeu de paramètres balayant les cas usuels : à la monnaie, dans la monnaie,
# en dehors, court terme, long terme, vol basse, vol haute.
CAS = [
    # (S,   K,   T,    r,     sigma, description)
    (100, 100, 1.0, 0.05, 0.20, "à la monnaie, 1 an"),
    (100, 80, 1.0, 0.05, 0.20, "dans la monnaie"),
    (100, 120, 1.0, 0.05, 0.20, "en dehors de la monnaie"),
    (100, 100, 0.08, 0.05, 0.20, "court terme (1 mois)"),
    (100, 100, 5.0, 0.05, 0.20, "long terme (5 ans)"),
    (100, 100, 1.0, 0.05, 0.05, "volatilité basse (5 %)"),
    (100, 100, 1.0, 0.05, 0.80, "volatilité haute (80 %)"),
    (100, 100, 1.0, 0.00, 0.20, "taux nul"),
]

SEED = 42


# ─────────────────────────────────────────────────────────────────────────
# 1. Convergence croisée
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("S,K,T,r,sigma,desc", CAS)
def test_binomial_converge_vers_bs(S, K, T, r, sigma, desc):
    """L'arbre à N=2000 doit coller à Black-Scholes à ~1 centime près."""
    for is_call in (True, False):
        reference = (prix_call if is_call else prix_put)(S, K, T, r, sigma)
        arbre = prix_binomial(S, K, T, r, sigma, N=2000, is_call=is_call)
        assert arbre == pytest.approx(reference, abs=0.01), (
            f"{desc} | {'call' if is_call else 'put'} : "
            f"arbre={arbre:.4f} vs BS={reference:.4f}"
        )


@pytest.mark.parametrize("S,K,T,r,sigma,desc", CAS)
def test_monte_carlo_encadre_bs(S, K, T, r, sigma, desc):
    """
    Black-Scholes doit tomber dans l'intervalle de confiance à 99 % du MC.

    C'est le BON test pour une méthode stochastique : on ne teste pas "le prix
    est proche" (arbitraire), mais "la référence est statistiquement
    compatible avec l'estimation". On prend 99 % plutôt que 95 % pour éviter
    un échec aléatoire une fois sur vingt dans la CI.
    """
    for is_call in (True, False):
        reference = (prix_call if is_call else prix_put)(S, K, T, r, sigma)
        res = prix_monte_carlo(S, K, T, r, sigma, M=500_000, is_call=is_call, seed=SEED)
        bas, haut = res.intervalle_confiance(0.99)
        assert bas <= reference <= haut, (
            f"{desc} | {'call' if is_call else 'put'} : "
            f"BS={reference:.4f} hors de l'IC 99 % [{bas:.4f}, {haut:.4f}]"
        )


@pytest.mark.parametrize("S,K,T,r,sigma,desc", CAS)
def test_les_trois_methodes_s_accordent(S, K, T, r, sigma, desc):
    """
    Le test de synthèse : les trois méthodes s'accordent-elles ?

    ⚠️ PIÈGE ÉVITÉ ICI — la première version de ce test utilisait un seuil
    ABSOLU (|écart| < 0.05) et échouait à vol 80 %. Ce n'était pas un bug du
    pricer, mais du test : la variance du Monte-Carlo croît avec la
    volatilité, donc son erreur standard aussi. Exiger la même précision
    absolue à vol 5 % et à vol 80 % n'a pas de sens.

    La bonne tolérance est STATISTIQUE : on tolère 3 erreurs standard sur le
    MC (méthode stochastique) et un seuil serré sur l'arbre (méthode
    déterministe). Chaque méthode est jugée selon la nature de son erreur.
    """
    bs = prix_call(S, K, T, r, sigma)
    arbre = prix_binomial(S, K, T, r, sigma, N=2000)
    res_mc = prix_monte_carlo(S, K, T, r, sigma, M=500_000, seed=SEED)

    # Arbre : erreur déterministe, en O(1/N) → seuil absolu légitime.
    assert arbre == pytest.approx(bs, abs=0.01), (
        f"{desc} | arbre={arbre:.4f} vs BS={bs:.4f}"
    )

    # MC : erreur statistique → tolérance en multiples de l'erreur standard.
    tolerance = 3.0 * res_mc.erreur_standard
    assert abs(res_mc.prix - bs) < tolerance, (
        f"{desc} | MC={res_mc.prix:.4f} vs BS={bs:.4f} "
        f"(écart={abs(res_mc.prix - bs):.4f} > 3σ={tolerance:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. Propriétés théoriques — les tests les plus forts
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("S,K,T,r,sigma,desc", CAS)
def test_parite_call_put(S, K, T, r, sigma, desc):
    """
    C − P = S − K·e^{-rT}

    C'est une relation d'ARBITRAGE, indépendante de tout modèle : elle tient
    quelle que soit la dynamique du sous-jacent. La violer signifierait qu'on
    peut gagner de l'argent sans risque — donc que le pricer est faux.
    """
    ecart = prix_call(S, K, T, r, sigma) - prix_put(S, K, T, r, sigma)
    theorique = S - K * exp(-r * T)
    assert ecart == pytest.approx(theorique, abs=1e-10), desc


@pytest.mark.parametrize("S,K,T,r,sigma,desc", CAS)
def test_bornes_arbitrage(S, K, T, r, sigma, desc):
    """
    Bornes sans arbitrage du call européen :
        max(S − K·e^{-rT}, 0) ≤ C ≤ S

    Borne basse : sinon on achète l'option et vend le forward → profit sûr.
    Borne haute : le call ne peut valoir plus que l'action elle-même.
    """
    c = prix_call(S, K, T, r, sigma)
    assert c >= max(S - K * exp(-r * T), 0.0) - 1e-10, f"{desc} : borne basse violée"
    assert c <= S + 1e-10, f"{desc} : borne haute violée"


def test_call_americain_jamais_exerce_par_anticipation():
    """
    THÉORÈME : sans dividende, il n'est jamais optimal d'exercer un call
    américain avant l'échéance. Sa prime sur l'européen doit donc être NULLE.

    Test très fort : l'arbre ne "sait" pas ce théorème, il teste bêtement
    l'exercice à chaque nœud. S'il retrouve zéro, c'est que l'induction à
    rebours est correcte.
    """
    for S, K, T, r, sigma, desc in CAS:
        prime = prime_exercice_anticipe(S, K, T, r, sigma, N=1000, is_call=True)
        assert prime == pytest.approx(0.0, abs=1e-6), f"{desc} : prime={prime:.2e}"


def test_put_americain_vaut_plus_que_europeen():
    """
    À l'inverse, le put américain vaut STRICTEMENT plus : exercer tôt permet
    d'encaisser le strike et de le placer au taux sans risque. La prime doit
    être positive dès que r > 0.
    """
    for S, K, T, r, sigma, desc in CAS:
        if r <= 0:
            continue  # sans taux, l'avantage disparaît
        prime = prime_exercice_anticipe(S, K, T, r, sigma, N=1000, is_call=False)
        assert prime > 0, f"{desc} : prime du put américain = {prime:.2e}, attendue > 0"


def test_monotonie_prix_call_en_strike():
    """Le prix d'un call DÉCROÎT avec le strike (payoff décroissant en K)."""
    prix = [prix_call(100, K, 1.0, 0.05, 0.20) for K in range(80, 121, 5)]
    assert all(prix[i] > prix[i + 1] for i in range(len(prix) - 1))


def test_monotonie_prix_en_volatilite():
    """
    Le prix CROÎT avec la volatilité (vega > 0), pour le call comme pour le
    put : plus d'incertitude = plus de valeur, car le payoff est convexe.
    """
    for is_call in (True, False):
        fonction = prix_call if is_call else prix_put
        prix = [fonction(100, 100, 1.0, 0.05, s) for s in (0.05, 0.10, 0.20, 0.40, 0.80)]
        assert all(prix[i] < prix[i + 1] for i in range(len(prix) - 1))


# ─────────────────────────────────────────────────────────────────────────
# 3. Grecs
# ─────────────────────────────────────────────────────────────────────────
def test_delta_pathwise_vs_formule_fermee():
    """Le delta MC (méthode pathwise) doit retrouver celui de Black-Scholes."""
    for S, K, T, r, sigma, desc in CAS:
        d_mc = delta_pathwise(S, K, T, r, sigma, M=1_000_000, seed=SEED)
        d_bs = delta_bs(S, K, T, r, sigma, is_call=True)
        assert d_mc == pytest.approx(d_bs, abs=0.01), f"{desc}"


def test_delta_par_differences_finies():
    """
    Le delta analytique doit coïncider avec la dérivée numérique du prix.
    Valide que la formule du delta est bien la dérivée de la formule du prix
    (une erreur de signe ou de terme se verrait immédiatement).
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    h = 0.01
    numerique = (prix_call(S + h, K, T, r, sigma) - prix_call(S - h, K, T, r, sigma)) / (2 * h)
    analytique = delta_bs(S, K, T, r, sigma, is_call=True)
    assert numerique == pytest.approx(analytique, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# 4. Cas limites
# ─────────────────────────────────────────────────────────────────────────
def test_maturite_nulle_donne_valeur_intrinseque():
    """À T=0, le prix se réduit au payoff."""
    assert prix_call(120, 100, 0.0, 0.05, 0.20) == pytest.approx(20.0)
    assert prix_call(80, 100, 0.0, 0.05, 0.20) == pytest.approx(0.0)
    assert prix_put(80, 100, 0.0, 0.05, 0.20) == pytest.approx(20.0)


def test_volatilite_nulle_donne_forward_actualise():
    """
    À σ=0, il n'y a plus d'aléa : S_T = S·e^{rT} avec certitude, et le call
    vaut max(S − K·e^{-rT}, 0).
    """
    S, K, T, r = 100, 90, 1.0, 0.05
    attendu = max(S - K * exp(-r * T), 0.0)
    assert prix_call(S, K, T, r, 0.0) == pytest.approx(attendu)


def test_probabilite_risque_neutre_invalide_leve_erreur():
    """
    Avec trop peu de pas et une vol trop basse face au taux, p sort de [0,1]
    et le modèle autoriserait un arbitrage. On doit lever une erreur plutôt
    que de renvoyer un prix silencieusement faux.
    """
    with pytest.raises(ValueError, match="risque-neutre"):
        prix_binomial(100, 100, 1.0, r=0.50, sigma=0.01, N=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))