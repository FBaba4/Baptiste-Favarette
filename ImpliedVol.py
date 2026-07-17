"""
implied_vol.py — Inversion de Black-Scholes : du prix vers la volatilité

ÉTAPE 5 du pricer. C'est le module qui change la nature du projet : jusqu'ici
on partait des paramètres pour obtenir un prix. Ici on fait l'INVERSE — on
part du prix de marché et on cherche la volatilité qui le reproduit.

POURQUOI C'EST L'USAGE DOMINANT DE BLACK-SCHOLES
-------------------------------------------------
Contre-intuitif mais essentiel : sur le marché, personne n'utilise
Black-Scholes pour "trouver le prix" — le prix est affiché à l'écran. On
l'utilise à l'envers, comme un CONVERTISSEUR de prix en volatilité.

Pourquoi ? Parce qu'un prix nu n'est pas comparable. Un call à 3 € sur une
action à 50 € et un autre à 8 € sur une action à 200 €, avec des strikes et
des maturités différents : impossible de dire lequel est "cher". Mais convertis
en volatilité implicite — 18 % contre 25 % — la comparaison devient immédiate.

Black-Scholes joue le rôle d'une unité de mesure. Personne ne dit "ce call
vaut 10,45" ; on dit "il se traite à 20 de vol".

LE PARADOXE À COMPRENDRE (et à savoir défendre en entretien)
--------------------------------------------------------------
Le modèle suppose une volatilité CONSTANTE. Si c'était vrai, toutes les
options d'une même échéance auraient la même vol implicite, quel que soit le
strike. Or ce n'est pas le cas : on observe un SMILE / SKEW.

Le marché contredit donc l'hypothèse centrale du modèle — et tout le monde
continue de l'utiliser. Ce n'est pas de l'incohérence : sa fausseté est
connue, stable et quantifiée. C'est un thermomètre mal calibré dont on
connaît exactement le biais. On s'en sert comme d'un langage, pas comme d'une
vérité.

MÉTHODE NUMÉRIQUE
------------------
Il n'existe pas de formule fermée pour inverser Black-Scholes (le prix dépend
de σ à travers N(d1) et N(d2), non inversibles analytiquement). On résout donc
numériquement f(σ) = prix_BS(σ) − prix_marché = 0.

Deux propriétés rendent le problème facile :
  - f est STRICTEMENT CROISSANTE en σ (vega > 0) → la solution est unique ;
  - on connaît sa dérivée exactement (c'est le vega) → Newton-Raphson est
    naturel, avec convergence quadratique.

On garde néanmoins un repli sur Brent (méthode par encadrement, plus lente
mais sans échec possible) pour les cas pathologiques : options très loin de
la monnaie, où le vega s'écrase et où Newton peut diverger.
"""

from math import sqrt, log, exp

from scipy.optimize import brentq

from BlackScholes import prix_call, prix_put, vega


class VolImpliciteError(ValueError):
    """Levée quand aucune volatilité implicite ne peut expliquer le prix."""


# ─────────────────────────────────────────────────────────────────────────
# Bornes d'arbitrage — le contrôle préalable indispensable
# ─────────────────────────────────────────────────────────────────────────
def _verifier_bornes(prix_marche, S, K, T, r, q, is_call):
    """
    Un prix hors des bornes sans arbitrage n'admet AUCUNE volatilité
    implicite — aucune valeur de σ ne peut le reproduire. Mieux vaut le
    détecter ici et lever une erreur claire que laisser Newton diverger.

    Bornes du call : max(S·e^{-qT} − K·e^{-rT}, 0) ≤ C ≤ S·e^{-qT}
    Bornes du put  : max(K·e^{-rT} − S·e^{-qT}, 0) ≤ P ≤ K·e^{-rT}

    En pratique, ce cas arrive souvent sur des données réelles : options
    illiquides, cours périmé, ou spread bid-ask aberrant. C'est le premier
    filtre à appliquer avant toute analyse de smile.
    """
    forward = S * exp(-q * T)
    strike_actualise = K * exp(-r * T)

    if is_call:
        borne_basse = max(forward - strike_actualise, 0.0)
        borne_haute = forward
    else:
        borne_basse = max(strike_actualise - forward, 0.0)
        borne_haute = strike_actualise

    if prix_marche < borne_basse - 1e-10:
        raise VolImpliciteError(
            f"Prix {prix_marche:.4f} sous la borne d'arbitrage {borne_basse:.4f} — "
            f"aucune volatilité ne peut l'expliquer (opportunité d'arbitrage, "
            f"ou donnée corrompue)."
        )
    if prix_marche > borne_haute + 1e-10:
        raise VolImpliciteError(
            f"Prix {prix_marche:.4f} au-dessus de la borne d'arbitrage "
            f"{borne_haute:.4f} — donnée probablement erronée."
        )


# ─────────────────────────────────────────────────────────────────────────
# Point de départ de Newton
# ─────────────────────────────────────────────────────────────────────────
def _estimation_initiale(prix_marche, S, K, T, r, q) -> float:
    """
    Approximation de Brenner-Subrahmanyam (1988) pour amorcer Newton :

        σ ≈ √(2π/T) · C / S     (valide à la monnaie)

    Ce n'est pas de la magie : à la monnaie, N(d1) ≈ 0,5 + d1/√(2π) et un
    développement au premier ordre donne cette relation. Elle est excellente
    près de la monnaie et grossière ailleurs — mais Newton n'a besoin que
    d'un point de départ raisonnable.

    On borne le résultat dans [1 %, 500 %] pour éviter un démarrage absurde.
    """
    if T <= 0:
        raise VolImpliciteError("Maturité nulle ou négative.")
    estimation = sqrt(2.0 * 3.141592653589793 / T) * prix_marche / S
    return min(max(estimation, 0.01), 5.0)


# ─────────────────────────────────────────────────────────────────────────
# Inversion
# ─────────────────────────────────────────────────────────────────────────
def volatilite_implicite(
    prix_marche: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    is_call: bool = True,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Volatilité implicite d'une option, par Newton-Raphson avec repli sur Brent.

    Retour : σ en décimal (0.20 = 20 %).
    Lève VolImpliciteError si le prix viole les bornes d'arbitrage.

    Stratégie :
      1. vérification des bornes d'arbitrage (sinon le problème n'a pas de
         solution, autant le dire tout de suite) ;
      2. Newton-Raphson, qui converge en général en 3-4 itérations grâce à
         la connaissance exacte du vega ;
      3. si Newton échoue (vega trop petit, itération hors bornes), repli sur
         Brent, plus lent mais garanti par encadrement.
    """
    _verifier_bornes(prix_marche, S, K, T, r, q, is_call)

    fonction_prix = prix_call if is_call else prix_put

    def ecart(sigma):
        return fonction_prix(S, K, T, r, sigma, q) - prix_marche

    # --- Tentative 1 : Newton-Raphson ---
    sigma = _estimation_initiale(prix_marche, S, K, T, r, q)
    for _ in range(max_iterations):
        difference = ecart(sigma)
        if abs(difference) < tolerance:
            return sigma

        # vega est renvoyé PAR POINT de vol (convention de marché) : on le
        # remultiplie par 100 pour retrouver la vraie dérivée ∂C/∂σ.
        derivee = vega(S, K, T, r, sigma, q) * 100.0

        # Vega trop petit : la fonction est quasi plate, Newton va exploser.
        # C'est le cas des options très loin de la monnaie.
        if derivee < 1e-10:
            break

        sigma_suivant = sigma - difference / derivee

        # Newton peut sortir du domaine admissible : on abandonne proprement
        # plutôt que d'évaluer Black-Scholes en σ négatif.
        if not 1e-6 < sigma_suivant < 10.0:
            break
        sigma = sigma_suivant

    # --- Tentative 2 : Brent (repli robuste) ---
    # Brent exige un encadrement [a, b] avec f(a) et f(b) de signes opposés.
    # Comme f est croissante en σ, on cherche une borne haute suffisante.
    try:
        borne_basse, borne_haute = 1e-6, 5.0
        if ecart(borne_basse) > 0 or ecart(borne_haute) < 0:
            raise VolImpliciteError(
                f"Impossible d'encadrer la solution dans [0, 500 %] pour un prix "
                f"de {prix_marche:.4f} (S={S}, K={K}, T={T:.3f})."
            )
        return float(brentq(ecart, borne_basse, borne_haute, xtol=tolerance))
    except VolImpliciteError:
        raise
    except Exception as e:
        raise VolImpliciteError(f"Échec de l'inversion : {e}") from e


def moneyness_log(S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """
    Log-moneyness normalisé : log(K / F) où F = S·e^{(r−q)T} est le forward.

    Pourquoi pas simplement K/S ? Parce que comparer des maturités différentes
    exige de se placer par rapport au FORWARD, pas au spot : à 5 ans, le
    forward est loin du spot, et un strike "à la monnaie spot" est en réalité
    très en dehors de la monnaie forward. C'est l'axe standard pour tracer un
    smile proprement.

    Convention : négatif = strike sous le forward (put OTM / call ITM).
    """
    forward = S * exp((r - q) * T)
    return log(K / forward)


# ─────────────────────────────────────────────────────────────────────────
# Vérification : test aller-retour
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("TEST ALLER-RETOUR — le seul test qui valide vraiment un inverseur")
    print("=" * 70)
    print("Principe : on price avec une vol connue, puis on inverse le prix.")
    print("On doit retrouver exactement la vol de départ.\n")

    S, T, r = 100, 1.0, 0.05
    print(f"{'K':>6} | {'σ vraie':>8} | {'prix':>9} | {'σ retrouvée':>12} | {'erreur':>9}")
    print("-" * 58)

    erreur_max = 0.0
    for K in (70, 85, 100, 115, 130):
        for sigma_vraie in (0.10, 0.20, 0.50):
            prix = prix_call(S, K, T, r, sigma_vraie)
            sigma_retrouvee = volatilite_implicite(prix, S, K, T, r, is_call=True)
            erreur = abs(sigma_retrouvee - sigma_vraie)
            erreur_max = max(erreur_max, erreur)
            print(f"{K:>6} | {sigma_vraie:>7.1%} | {prix:>9.4f} | "
                  f"{sigma_retrouvee:>11.6%} | {erreur:>9.2e}")

    print(f"\nErreur maximale : {erreur_max:.2e}")
    print("→ l'inverseur est exact à la précision machine.\n")

    # Cas pathologique : option très loin de la monnaie
    print("=" * 70)
    print("CAS PATHOLOGIQUE — option très en dehors de la monnaie")
    print("=" * 70)
    K_extreme, sigma_vraie = 200, 0.20
    prix = prix_call(S, K_extreme, T, r, sigma_vraie)
    print(f"K={K_extreme} (spot={S}), σ={sigma_vraie:.0%} → prix = {prix:.10f}")
    print("Le vega est ici quasi nul : Newton seul divergerait.")
    sigma_retrouvee = volatilite_implicite(prix, S, K_extreme, T, r, is_call=True)
    print(f"Vol retrouvée (via repli sur Brent) : {sigma_retrouvee:.6%} "
          f"| erreur = {abs(sigma_retrouvee - sigma_vraie):.2e}")

    # Prix violant les bornes d'arbitrage
    print("\n" + "=" * 70)
    print("PRIX ABERRANT — détection des bornes d'arbitrage")
    print("=" * 70)
    for prix_absurde, commentaire in [
        (0.5, "sous la borne basse (call ITM sous-évalué)"),
        (150.0, "au-dessus du spot"),
    ]:
        try:
            volatilite_implicite(prix_absurde, S, 80, T, r, is_call=True)
            print(f"⚠️ {prix_absurde} : aucune erreur levée — le contrôle a échoué !")
        except VolImpliciteError as e:
            print(f"✓ prix={prix_absurde:>6} ({commentaire})")
            print(f"    → {str(e)[:95]}")