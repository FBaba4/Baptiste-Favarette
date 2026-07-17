"""
black_scholes.py — Pricing analytique d'options européennes (modèle de Black-Scholes)

ÉTAPE 1 du pricer. Ce module est la RÉFÉRENCE du projet : l'arbre binomial
(étape 2) et le Monte-Carlo (étape 3) devront converger vers ces valeurs.
C'est le seul des trois qui donne une solution exacte en forme fermée — d'où
son rôle de juge de paix.

RAPPEL DU MODÈLE (le "pourquoi" derrière les formules)
------------------------------------------------------
On suppose que le cours S_t suit un mouvement brownien géométrique :

    dS_t = r·S_t·dt + σ·S_t·dW_t        (sous la probabilité risque-neutre)

Deux idées à retenir pour un entretien :

1. POURQUOI r ET PAS LE RENDEMENT RÉEL μ ? Parce qu'on price sous la
   probabilité risque-neutre. Par l'argument de réplication (on couvre
   l'option par un portefeuille dynamique actions + cash), le rendement réel
   attendu de l'action DISPARAÎT du prix. C'est le résultat le plus
   contre-intuitif du modèle : le prix d'une option ne dépend pas de la
   hausse qu'on anticipe sur le sous-jacent.

2. D'OÙ VIENT LA FORMULE ? Le payoff actualisé est E[e^{-rT}·max(S_T-K, 0)].
   Comme S_T est lognormale sous la mesure risque-neutre, cette espérance
   s'intègre analytiquement — et donne les termes N(d1), N(d2) ci-dessous.

INTERPRÉTATION DE d1 ET d2 (question classique en entretien)
------------------------------------------------------------
    N(d2) = probabilité (risque-neutre) que l'option finisse dans la monnaie.
    N(d1) = delta du call ; c'est aussi la valeur actualisée de E[S_T | S_T>K]
            normalisée par S. Ce n'est PAS une probabilité, contrairement à
            l'erreur fréquente.

CONVENTIONS
-----------
    S     : prix spot du sous-jacent
    K     : strike (prix d'exercice)
    T     : maturité en ANNÉES (0.5 = 6 mois)
    r     : taux sans risque annualisé, en décimal (0.03 = 3 %)
    sigma : volatilité annualisée, en décimal (0.20 = 20 %)
    q     : taux de dividende continu (0.0 par défaut)

Les dividendes sont traités en continu (modèle de Merton) : c'est une
approximation — voir la section "Limites" du README.
"""

from dataclasses import dataclass
from math import log, sqrt, exp

from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────
# Termes intermédiaires
# ─────────────────────────────────────────────────────────────────────────
def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """
    Calcule d1 et d2, les deux termes centraux de la formule.

    Note numérique : on ne gère pas ici T=0 ou sigma=0 (division par zéro).
    Ces cas dégénérés sont traités en amont dans les fonctions de pricing,
    où le prix vaut simplement la valeur intrinsèque actualisée.
    """
    vol_sqrt_T = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def _valeur_intrinseque(S: float, K: float, r: float, T: float, q: float, is_call: bool) -> float:
    """
    Cas dégénéré (T→0 ou sigma→0) : plus d'incertitude, le prix se réduit à
    la valeur intrinsèque actualisée (forward vs strike).
    """
    forward = S * exp(-q * T)
    strike_actualise = K * exp(-r * T)
    if is_call:
        return max(forward - strike_actualise, 0.0)
    return max(strike_actualise - forward, 0.0)


# ─────────────────────────────────────────────────────────────────────────
# Prix
# ─────────────────────────────────────────────────────────────────────────
def prix_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Prix d'un CALL européen.

        C = S·e^{-qT}·N(d1) − K·e^{-rT}·N(d2)

    Lecture : on reçoit l'action (premier terme, pondéré par le delta) et on
    paie le strike (second terme, pondéré par la proba d'exercice).
    """
    if T <= 0 or sigma <= 0:
        return _valeur_intrinseque(S, K, r, T, q, is_call=True)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def prix_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Prix d'un PUT européen.

        P = K·e^{-rT}·N(−d2) − S·e^{-qT}·N(−d1)

    On pourrait aussi le déduire du call par parité — c'est justement ce que
    vérifie le test de parité plus bas (une implémentation indépendante est
    un meilleur test qu'une simple réécriture de la parité).
    """
    if T <= 0 or sigma <= 0:
        return _valeur_intrinseque(S, K, r, T, q, is_call=False)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


# ─────────────────────────────────────────────────────────────────────────
# Les Grecs — les sensibilités du prix
# ─────────────────────────────────────────────────────────────────────────
# Chaque Grec est une dérivée partielle du prix. Ce ne sont pas des
# curiosités académiques : c'est ce qu'un desk regarde toute la journée,
# parce que ce sont les risques qu'il doit couvrir.

def delta(S, K, T, r, sigma, q=0.0, is_call=True) -> float:
    """
    ∂C/∂S — sensibilité au spot. C'est LE Grec de la couverture : pour
    couvrir un call vendu, on détient delta actions (delta-hedging).
    Call : entre 0 et 1. Put : entre −1 et 0.
    """
    if T <= 0 or sigma <= 0:
        intrinseque = (S > K) if is_call else (S < K)
        return (1.0 if is_call else -1.0) * float(intrinseque)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if is_call:
        return exp(-q * T) * norm.cdf(d1)
    return exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0) -> float:
    """
    ∂²C/∂S² — la convexité, identique pour le call et le put.
    Mesure la vitesse à laquelle le delta bouge : un gamma élevé = une
    couverture à réajuster souvent. Maximal à la monnaie, proche de la
    maturité — d'où la nervosité des desks autour de l'expiration.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma, q=0.0) -> float:
    """
    ∂C/∂σ — sensibilité à la volatilité, identique call/put.

    ⚠️ CONVENTION : on renvoie la sensibilité pour +1 POINT de vol (+1 %),
    soit la dérivée divisée par 100. C'est la convention de marché ("le vega
    de cette option est de 0,12" = +0,12 € si la vol passe de 20 % à 21 %).
    La dérivée mathématique brute serait 100× plus grande.

    Vega > 0 toujours : plus d'incertitude = plus de valeur d'option, car le
    payoff est convexe (on capte la hausse, on est protégé à la baisse).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T) / 100.0


def theta(S, K, T, r, sigma, q=0.0, is_call=True) -> float:
    """
    ∂C/∂t — l'érosion temporelle.

    ⚠️ CONVENTION : on renvoie le theta PAR JOUR (dérivée annuelle / 365),
    comme sur un écran de trading. Généralement négatif pour un acheteur :
    chaque jour qui passe détruit de la valeur temps.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    terme_commun = -S * exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * sqrt(T))
    if is_call:
        theta_annuel = (terme_commun
                        - r * K * exp(-r * T) * norm.cdf(d2)
                        + q * S * exp(-q * T) * norm.cdf(d1))
    else:
        theta_annuel = (terme_commun
                        + r * K * exp(-r * T) * norm.cdf(-d2)
                        - q * S * exp(-q * T) * norm.cdf(-d1))
    return theta_annuel / 365.0


def rho(S, K, T, r, sigma, q=0.0, is_call=True) -> float:
    """
    ∂C/∂r — sensibilité au taux sans risque.

    ⚠️ CONVENTION : pour +1 POINT de taux (+1 %), donc dérivée / 100.
    Le moins suivi des Grecs en temps normal — mais il redevient intéressant
    quand les taux bougent vite, ce qui est précisément le cas depuis 2022.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if is_call:
        return K * T * exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * exp(-r * T) * norm.cdf(-d2) / 100.0


# ─────────────────────────────────────────────────────────────────────────
# Objet de commodité — regroupe prix + Grecs
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class OptionEuropeenne:
    """
    Regroupe les paramètres d'une option et expose prix + Grecs.
    Sert d'interface propre pour les étapes suivantes (binomial, MC, vol
    implicite) : elles pourront comparer leurs résultats à cette référence.
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    is_call: bool = True

    def prix(self) -> float:
        fonction = prix_call if self.is_call else prix_put
        return fonction(self.S, self.K, self.T, self.r, self.sigma, self.q)

    def grecs(self) -> dict:
        return {
            "delta": delta(self.S, self.K, self.T, self.r, self.sigma, self.q, self.is_call),
            "gamma": gamma(self.S, self.K, self.T, self.r, self.sigma, self.q),
            "vega": vega(self.S, self.K, self.T, self.r, self.sigma, self.q),
            "theta": theta(self.S, self.K, self.T, self.r, self.sigma, self.q, self.is_call),
            "rho": rho(self.S, self.K, self.T, self.r, self.sigma, self.q, self.is_call),
        }

    def resume(self) -> str:
        type_option = "CALL" if self.is_call else "PUT"
        g = self.grecs()
        return (
            f"{type_option} européen | S={self.S} K={self.K} T={self.T}a "
            f"r={self.r:.1%} σ={self.sigma:.1%} q={self.q:.1%}\n"
            f"  Prix  : {self.prix():.4f}\n"
            f"  Delta : {g['delta']:+.4f}   (par unité de spot)\n"
            f"  Gamma : {g['gamma']:+.4f}   (par unité de spot²)\n"
            f"  Vega  : {g['vega']:+.4f}   (pour +1 pt de vol)\n"
            f"  Theta : {g['theta']:+.4f}   (par jour)\n"
            f"  Rho   : {g['rho']:+.4f}   (pour +1 pt de taux)"
        )


# ─────────────────────────────────────────────────────────────────────────
# Vérification rapide
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    call = OptionEuropeenne(S=100, K=100, T=1.0, r=0.05, sigma=0.20, is_call=True)
    put = OptionEuropeenne(S=100, K=100, T=1.0, r=0.05, sigma=0.20, is_call=False)

    print(call.resume())
    print()
    print(put.resume())

    # Parité call-put : C − P = S·e^{-qT} − K·e^{-rT}
    # C'est une relation d'ARBITRAGE, pas un résultat du modèle : elle tient
    # quel que soit le modèle de dynamique du sous-jacent. Si elle est violée,
    # le pricer est faux (ou il y a une opportunité d'arbitrage sur le marché).
    ecart = call.prix() - put.prix()
    theorique = 100 * exp(-0.0 * 1.0) - 100 * exp(-0.05 * 1.0)
    print(f"\nParité call-put : C−P = {ecart:.6f} | théorique = {theorique:.6f} "
          f"| écart = {abs(ecart - theorique):.2e}")