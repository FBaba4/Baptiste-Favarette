"""
smile.py — Le smile de volatilité sur données de marché réelles

ÉTAPE 5b. C'est ici que le projet quitte l'exercice académique : on prend de
vraies options cotées, on inverse Black-Scholes sur chacune, et on regarde ce
que le marché dit vraiment de la volatilité.

CE QU'ON CHERCHE À MONTRER
---------------------------
Si Black-Scholes était vrai, la volatilité implicite serait CONSTANTE : même
valeur pour tous les strikes d'une même maturité, une droite horizontale.

On observe l'inverse : une courbe en sourire (smile) ou, sur indices actions,
une pente descendante (skew). Le marché price les strikes bas plus cher, en
vol, que les strikes hauts.

POURQUOI LE SKEW EXISTE (les trois explications à connaître)
--------------------------------------------------------------
1. LES QUEUES ÉPAISSES. Black-Scholes suppose des rendements lognormaux. Les
   vrais rendements ont des queues plus épaisses : les krachs sont bien plus
   fréquents que la loi normale ne le prédit. Le marché majore donc le prix
   des puts loin de la monnaie — d'où une vol implicite plus élevée sur les
   strikes bas.

2. L'EFFET DE LEVIER. Quand une action chute, le ratio dette/capitaux propres
   monte mécaniquement : l'entreprise devient plus risquée, donc plus volatile.
   Volatilité et rendement sont négativement corrélés — ce que le GBM à vol
   constante ne capture pas.

3. LA DEMANDE DE COUVERTURE. Structurellement, les gérants achètent des puts
   de protection et vendent des calls (covered call). Ce déséquilibre d'offre
   et de demande pousse la vol implicite des puts vers le haut. Le skew est
   donc AUSSI un phénomène de flux, pas seulement de modèle.

⚠️ Ces trois explications coexistent ; aucune n'est suffisante seule. C'est
exactement le type de question ouverte posée en entretien.

DONNÉES
-------
yfinance donne des chaînes d'options gratuitement, mais la qualité est
médiocre : cotations périmées, spreads absurdes, options non traitées. Le
FILTRAGE est plus important que le calcul — un smile tracé sans filtrage est
un nuage de points illisible. C'est le vrai travail de ce module.

Lancer :  python python/smile.py              (données réelles, ex. SPY)
          python python/smile.py --demo       (démo synthétique, sans réseau)
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp

import numpy as np
import pandas as pd

from BlackScholes import prix_call
from ImpliedVol import volatilite_implicite, VolImpliciteError, moneyness_log


# ─────────────────────────────────────────────────────────────────────────
# Paramètres de filtrage — le cœur du module
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class FiltresQualite:
    """
    Critères de rejet des cotations douteuses.

    Chaque seuil répond à un défaut précis des données gratuites. Les valeurs
    par défaut sont conservatrices : mieux vaut jeter des points douteux que
    tracer un smile pollué. Un smile propre sur 40 points vaut mieux qu'un
    nuage sur 300.
    """
    volume_min: int = 10
    # Une option sans volume n'a pas été traitée aujourd'hui : son "prix" est
    # un cours indicatif périmé, souvent aberrant.

    open_interest_min: int = 10
    # Pas de positions ouvertes = strike sans intérêt réel du marché.

    spread_relatif_max: float = 0.25
    # (ask − bid) / mid. Au-delà de 25 %, le "prix" ne veut plus rien dire :
    # l'incertitude sur le prix dépasse largement l'effet qu'on mesure.

    moneyness_min: float = 0.80
    moneyness_max: float = 1.20
    # K/S. Au-delà, les options valent quelques centimes : le vega s'écrase et
    # la vol implicite devient extrêmement sensible au bruit. C'est le filtre
    # le plus important pour un smile lisible.

    prix_min: float = 0.05
    # En dessous, le tick de cotation (1 centime) représente 20 % du prix :
    # le bruit d'arrondi domine le signal.

    vol_implicite_min: float = 0.01
    vol_implicite_max: float = 3.00
    # Garde-fou final : une vol implicite de 300 % signale une donnée corrompue,
    # pas un marché nerveux.


def _annees_jusqu_a(date_expiration: str) -> float:
    """
    Maturité en années, en base calendaire (365 jours).

    ⚠️ SIMPLIFICATION ASSUMÉE : les desks utilisent souvent une base en jours
    OUVRÉS (252) pour la vol, car il ne se passe rien le week-end. La base
    calendaire surestime légèrement T, donc sous-estime légèrement la vol
    implicite. L'écart est faible et systématique, donc il ne déforme pas la
    FORME du smile — ce qui nous intéresse ici. À citer dans les limites.
    """
    expiration = datetime.strptime(date_expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    maintenant = datetime.now(timezone.utc)
    return max((expiration - maintenant).total_seconds() / (365.0 * 24 * 3600), 1e-6)


# ─────────────────────────────────────────────────────────────────────────
# Récupération des données réelles
# ─────────────────────────────────────────────────────────────────────────
def charger_chaine_options(ticker: str, jours_cible: int = 45):
    """
    Récupère une chaîne d'options via yfinance.

    jours_cible : on choisit l'échéance la plus proche de ce nombre de jours.

    ⚠️ POURQUOI PAS UN INDEX ? Première version de cette fonction : on prenait
    la 3e expiration de la liste (index=2), en supposant des échéances
    mensuelles. Erreur — les sous-jacents liquides (SPY, QQQ, SPX) ont des
    expirations TROIS FOIS PAR SEMAINE. L'index 2 tombait donc sur une
    échéance à 4 jours, où toutes les options hors monnaie valent quelques
    centimes : spreads relatifs énormes, tous les filtres les rejetaient, et
    le smile sortait vide.

    Sélectionner par DURÉE et non par position est robuste quel que soit le
    calendrier du sous-jacent. 45 jours est le compromis usuel : assez loin
    pour un smile propre, assez proche pour rester liquide.

    Retour : (spot, date_expiration, DataFrame des calls, DataFrame des puts)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance requis : pip install yfinance\n"
            "(ou lancer avec --demo pour la version synthétique sans réseau)"
        )

    action = yf.Ticker(ticker)

    historique = action.history(period="1d")
    if historique.empty:
        raise ValueError(f"Aucune donnée pour {ticker} — ticker invalide ou marché fermé ?")
    spot = float(historique["Close"].iloc[-1])

    expirations = action.options
    if not expirations:
        raise ValueError(f"Aucune option cotée pour {ticker}.")

    # On choisit l'échéance dont la maturité est la plus proche de la cible.
    maintenant = datetime.now(timezone.utc)
    def _jours(date_str):
        exp = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (exp - maintenant).total_seconds() / (24 * 3600)

    candidates = [(e, _jours(e)) for e in expirations]
    candidates = [(e, j) for e, j in candidates if j > 1.0]  # on écarte le jour même
    if not candidates:
        raise ValueError(f"Aucune échéance future exploitable pour {ticker}.")
    expiration = min(candidates, key=lambda c: abs(c[1] - jours_cible))[0]

    chaine = action.option_chain(expiration)
    return spot, expiration, chaine.calls, chaine.puts


def calculer_smile(
    df: pd.DataFrame,
    spot: float,
    T: float,
    r: float,
    q: float = 0.0,
    is_call: bool = True,
    filtres: FiltresQualite | None = None,
) -> pd.DataFrame:
    """
    Calcule la vol implicite de chaque option d'une chaîne, après filtrage.

    Retour : DataFrame avec strike, moneyness, log_moneyness, prix_mid,
    vol_implicite — trié par strike. Les colonnes de diagnostic du rejet sont
    conservées pour pouvoir justifier ce qu'on a jeté.
    """
    filtres = filtres or FiltresQualite()
    lignes = []
    rejets = {"liquidité": 0, "spread": 0, "moneyness": 0, "prix": 0,
              "arbitrage": 0, "vol aberrante": 0}

    for _, option in df.iterrows():
        K = float(option["strike"])

        # Filtre 1 — moneyness (le plus discriminant)
        moneyness = K / spot
        if not filtres.moneyness_min <= moneyness <= filtres.moneyness_max:
            rejets["moneyness"] += 1
            continue

        # Filtre 2 — liquidité
        # ⚠️ PIÈGE : yfinance renvoie souvent NaN, pas 0. Or `NaN or 0` vaut
        # NaN (NaN est truthy !), et toute comparaison avec NaN est False —
        # donc `NaN < 10` = False et l'option passerait le filtre. On force
        # explicitement le NaN à 0 avec pd.isna.
        volume = option.get("volume", 0)
        volume = 0.0 if pd.isna(volume) else float(volume)
        open_interest = option.get("openInterest", 0)
        open_interest = 0.0 if pd.isna(open_interest) else float(open_interest)
        if volume < filtres.volume_min or open_interest < filtres.open_interest_min:
            rejets["liquidité"] += 1
            continue

        # Filtre 3 — spread bid-ask
        bid = option.get("bid", 0)
        bid = 0.0 if pd.isna(bid) else float(bid)
        ask = option.get("ask", 0)
        ask = 0.0 if pd.isna(ask) else float(ask)
        if bid <= 0 or ask <= 0 or ask < bid:
            rejets["spread"] += 1
            continue
        mid = 0.5 * (bid + ask)
        if (ask - bid) / mid > filtres.spread_relatif_max:
            rejets["spread"] += 1
            continue

        # Filtre 4 — prix minimal
        if mid < filtres.prix_min:
            rejets["prix"] += 1
            continue

        # Filtre 5 — inversion (les bornes d'arbitrage sont vérifiées dedans)
        try:
            vi = volatilite_implicite(mid, spot, K, T, r, q, is_call)
        except VolImpliciteError:
            rejets["arbitrage"] += 1
            continue

        if not filtres.vol_implicite_min <= vi <= filtres.vol_implicite_max:
            rejets["vol aberrante"] += 1
            continue

        lignes.append({
            "strike": K,
            "moneyness": moneyness,
            "log_moneyness": moneyness_log(spot, K, T, r, q),
            "prix_mid": mid,
            "spread_relatif": (ask - bid) / mid,
            "volume": volume,
            "vol_implicite": vi,
        })

    COLONNES = ["strike", "moneyness", "log_moneyness", "prix_mid",
                "spread_relatif", "volume", "vol_implicite"]

    if not lignes:
        # Cas fréquent sur données réelles : tout a été filtré (marché fermé,
        # échéance trop courte, sous-jacent illiquide). On renvoie un DataFrame
        # VIDE MAIS TYPÉ plutôt que de planter sur un sort_values : l'appelant
        # peut alors afficher le diagnostic des rejets, qui est justement
        # l'information utile dans ce cas.
        resultat = pd.DataFrame(columns=COLONNES)
    else:
        resultat = pd.DataFrame(lignes).sort_values("strike").reset_index(drop=True)

    resultat.attrs["rejets"] = rejets
    resultat.attrs["total_initial"] = len(df)
    return resultat


# ─────────────────────────────────────────────────────────────────────────
# Mode démo — données synthétiques, sans réseau
# ─────────────────────────────────────────────────────────────────────────
def chaine_synthetique(spot=100.0, T=0.25, r=0.05, seed=42) -> pd.DataFrame:
    """
    Fabrique une chaîne d'options avec un SKEW imposé, pour tester le pipeline
    sans réseau.

    On génère les prix avec une vol qui DÉPEND du strike :
        σ(K) = σ_ATM − pente · log(K/S) + courbure · log(K/S)²

    C'est exactement ce que Black-Scholes interdit (il postule σ constante).
    En inversant ces prix, on doit retrouver la courbe σ(K) qu'on a imposée :
    c'est un test aller-retour du pipeline complet.

    ⚠️ Ces données sont FABRIQUÉES, pas observées. Elles servent à valider le
    code, jamais à conclure quoi que ce soit sur un marché réel.
    """
    rng = np.random.default_rng(seed)
    strikes = np.arange(80, 121, 2.5)
    lignes = []
    for K in strikes:
        lm = np.log(K / (spot * exp(r * T)))
        # Skew typique d'un indice actions : vol décroissante en strike
        sigma_vraie = 0.20 - 0.30 * lm + 0.50 * lm**2
        prix = prix_call(spot, float(K), T, r, float(sigma_vraie))
        # Bruit de marché réaliste : spread bid-ask de ~2 %
        demi_spread = max(0.01 * prix, 0.01)
        lignes.append({
            "strike": float(K),
            "bid": prix - demi_spread,
            "ask": prix + demi_spread,
            "volume": int(rng.integers(50, 500)),
            "openInterest": int(rng.integers(100, 2000)),
            "sigma_vraie": float(sigma_vraie),  # pour vérifier l'aller-retour
        })
    return pd.DataFrame(lignes)


# ─────────────────────────────────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────────────────────────────────
def _afficher_smile(df, spot, T, titre, colonne_verite=None):
    print(f"\n{titre}")
    print("=" * len(titre))

    rejets = df.attrs.get("rejets", {})
    total = df.attrs.get("total_initial", 0)

    if df.empty:
        # Le diagnostic EST l'information : sans lui, on ne sait pas quel
        # filtre a tout mangé ni comment le corriger.
        print("\n⚠️  AUCUNE cotation n'a passé les filtres.\n")
        print(f"Cotations initiales : {total}")
        if any(rejets.values()):
            print("Motifs de rejet :")
            for motif, n in rejets.items():
                if n:
                    print(f"    {motif:>14} : {n:>4}")
        print("\nPistes :")
        print("  • marché fermé → bid/ask à zéro, tout part en « spread »")
        print("  • échéance trop courte → prix trop faibles, spreads relatifs énormes")
        print("  • sous-jacent illiquide → essaie SPY, QQQ, AAPL")
        print("  • filtres trop stricts → assouplis FiltresQualite (volume_min,")
        print("    spread_relatif_max) et relance")
        return

    print(f"{'strike':>8} | {'K/S':>6} | {'prix':>8} | {'vol impl.':>9}", end="")
    if colonne_verite is not None:
        print(f" | {'vol vraie':>9} | {'erreur':>8}")
    else:
        print()
    print("-" * (60 if colonne_verite is not None else 40))

    for _, r_ in df.iterrows():
        print(f"{r_['strike']:>8.1f} | {r_['moneyness']:>6.3f} | "
              f"{r_['prix_mid']:>8.3f} | {r_['vol_implicite']:>8.2%}", end="")
        if colonne_verite is not None:
            vraie = colonne_verite.loc[colonne_verite["strike"] == r_["strike"], "sigma_vraie"]
            if not vraie.empty:
                v = float(vraie.iloc[0])
                print(f" | {v:>8.2%} | {abs(v - r_['vol_implicite']):>8.2e}")
            else:
                print()
        else:
            print()

    print(f"\nRetenues : {len(df)} / {total} cotations")
    if any(rejets.values()):
        detail = ", ".join(f"{k}={v}" for k, v in rejets.items() if v)
        print(f"Rejetées : {detail}")


def main():
    mode_demo = "--demo" in sys.argv
    r = 0.05

    if mode_demo:
        print("MODE DÉMO — données synthétiques avec skew imposé (aucun réseau)")
        print("Ces prix sont FABRIQUÉS : ils valident le pipeline, rien d'autre.\n")
        spot, T = 100.0, 0.25
        brute = chaine_synthetique(spot=spot, T=T, r=r)
        df = calculer_smile(brute, spot, T, r, is_call=True)
        _afficher_smile(df, spot, T,
                        f"Smile reconstruit — spot={spot}, T={T:.2f} an",
                        colonne_verite=brute)
        erreur_max = max(
            abs(float(brute.loc[brute["strike"] == row["strike"], "sigma_vraie"].iloc[0])
                - row["vol_implicite"])
            for _, row in df.iterrows()
        )
        print(f"\nErreur maximale de reconstruction : {erreur_max:.2e}")
        print("→ le pipeline retrouve le skew qu'on a imposé. Il est correct.")
        print("\nLance sans --demo (et avec yfinance installé) pour un vrai marché.")
        return

    ticker = next((a for a in sys.argv[1:] if not a.startswith("--")), "SPY")
    print(f"Chargement de la chaîne d'options {ticker}...")
    spot, expiration, calls, puts = charger_chaine_options(ticker)
    T = _annees_jusqu_a(expiration)
    print(f"Spot = {spot:.2f} | expiration {expiration} | T = {T:.4f} an "
          f"({T*365:.0f} jours)")

    if T * 365 < 10:
        print("\n⚠️  Maturité très courte (< 10 jours) : le smile sera déformé")
        print("    (effet de pin, vol implicite explosive) et beaucoup d'options")
        print("    seront filtrées. Résultat peu représentatif.")

    df_calls = calculer_smile(calls, spot, T, r, is_call=True)
    _afficher_smile(df_calls, spot, T, f"SMILE — calls {ticker} @ {expiration}")

    if len(df_calls) >= 3:
        atm = df_calls.iloc[(df_calls["moneyness"] - 1.0).abs().argmin()]
        bas = df_calls.iloc[0]
        haut = df_calls.iloc[-1]
        print(f"\nLecture : vol implicite ATM = {atm['vol_implicite']:.2%}")
        print(f"          strike bas (K/S={bas['moneyness']:.2f}) : {bas['vol_implicite']:.2%}")
        print(f"          strike haut (K/S={haut['moneyness']:.2f}) : {haut['vol_implicite']:.2%}")
        pente = haut["vol_implicite"] - bas["vol_implicite"]
        forme = "SKEW décroissant (typique actions/indices)" if pente < -0.01 else (
                "SMILE en U" if pente > 0.01 else "quasi plat")
        print(f"          → {forme}")
        print("\nSi Black-Scholes était vrai, ces trois chiffres seraient IDENTIQUES.")


if __name__ == "__main__":
    main()