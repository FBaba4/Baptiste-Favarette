"""
acp_courbe.py — Analyse en composantes principales de la courbe des taux

LE RÉSULTAT QU'ON CHERCHE À RETROUVER
--------------------------------------
Litterman & Scheinkman (1991) : trois facteurs expliquent l'essentiel (~95 %)
des mouvements de la courbe des taux, et ils ont des formes interprétables :

  PC1 — NIVEAU    : charge ~constante sur toutes les maturités. Toute la
                    courbe monte ou descend en bloc. C'est le facteur dominant
                    (~80-90 % de la variance) : le marché révise le niveau
                    général des taux (inflation anticipée, prime de terme).

  PC2 — PENTE     : charges de signes opposés aux deux bouts. Le court monte
                    quand le long descend (aplatissement) ou l'inverse
                    (pentification). C'est le facteur de POLITIQUE MONÉTAIRE :
                    la banque centrale pilote le court, les anticipations de
                    croissance et d'inflation pilotent le long.

  PC3 — COURBURE  : charges en bosse (le ventre contre les ailes). Plus fin,
                    lié à la demande sur les maturités intermédiaires et à la
                    volatilité anticipée.

POURQUOI L'ACP MAISON (SVD) ET PAS sklearn
--------------------------------------------
Trois lignes de numpy suffisent, et on voit les maths : centrer, SVD, lire
les vecteurs singuliers. Utiliser sklearn ici cacherait précisément ce qu'on
veut montrer qu'on comprend. (Le test vérifie néanmoins l'équivalence avec
sklearn : même résultat, au signe près.)

LE PIÈGE CENTRAL : NIVEAUX vs VARIATIONS
------------------------------------------
Faire l'ACP sur les NIVEAUX des taux donne un premier facteur à ~99 % — un
résultat spectaculaire et VIDE. Les niveaux sont des séries très persistantes
(quasi non stationnaires) : tout est corrélé à tout parce que tout dérive
ensemble, et l'ACP capte cette dérive commune, pas la structure des
mouvements. C'est du spurious au sens de Granger-Newbold.

La question économique intéressante est : « quand la courbe BOUGE, comment
bouge-t-elle ? » — donc ACP sur les VARIATIONS quotidiennes. La fonction
comparer_niveaux_variations() démontre l'écart entre les deux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ResultatACP:
    """Résultat d'une ACP : tout ce qu'il faut pour l'analyse et les figures."""
    maturites: np.ndarray          # (p,) maturités en années
    loadings: np.ndarray           # (p, k) vecteurs propres (colonnes)
    variance_expliquee: np.ndarray # (k,) part de variance par composante
    scores: pd.DataFrame           # (T, k) séries temporelles des facteurs
    sur_variations: bool

    def variance_cumulee(self, k: int = 3) -> float:
        return float(self.variance_expliquee[:k].sum())

    def resume(self) -> str:
        noms = ["PC1 (niveau)", "PC2 (pente)", "PC3 (courbure)"]
        lignes = [
            f"  {noms[i] if i < 3 else f'PC{i+1}':<15}: "
            f"{self.variance_expliquee[i]:6.2%}  "
            f"(cumul {self.variance_expliquee[:i+1].sum():6.2%})"
            for i in range(min(5, len(self.variance_expliquee)))
        ]
        base = "variations quotidiennes" if self.sur_variations else "niveaux (⚠️ piège)"
        return f"ACP sur {base} :\n" + "\n".join(lignes)


def acp(courbe: pd.DataFrame, sur_variations: bool = True,
        standardiser: bool = False) -> ResultatACP:
    """
    ACP de la courbe des taux, par décomposition en valeurs singulières.

    sur_variations : True (défaut) → ACP sur les différences premières
                     quotidiennes, en points de base. C'est LE bon choix
                     (voir docstring du module).
    standardiser   : False (défaut) → ACP sur la matrice de covariance, pas
                     de corrélation. Choix délibéré : toutes les colonnes sont
                     dans la même unité (points de base), et l'écart de
                     volatilité entre maturités est une INFORMATION (le 2 ans
                     bouge plus que le 30 ans en régime de resserrement) —
                     la standardisation l'effacerait.
    """
    if sur_variations:
        X = courbe.diff().dropna() * 100.0     # en points de base
    else:
        X = courbe.copy()

    maturites = np.array([float(c) for c in courbe.columns])

    # Centrage (obligatoire), standardisation (optionnelle, non recommandée ici)
    X_centre = X - X.mean(axis=0)
    if standardiser:
        X_centre = X_centre / X_centre.std(axis=0, ddof=1)

    # SVD : X = U·S·Vᵀ. Les colonnes de V sont les vecteurs propres de la
    # covariance XᵀX/(n−1) ; les valeurs singulières au carré donnent les
    # variances. C'est plus stable numériquement que de diagonaliser XᵀX
    # (on évite de former la matrice de covariance, dont le conditionnement
    # est le carré de celui de X).
    U, S, Vt = np.linalg.svd(X_centre.to_numpy(), full_matrices=False)

    variances = S**2 / (len(X_centre) - 1)
    variance_expliquee = variances / variances.sum()
    loadings = Vt.T                                  # (p, k)

    # Convention de signe : la SVD est définie au signe près par composante.
    # On impose des signes interprétables et REPRODUCTIBLES :
    #   PC1 : charge moyenne positive (une hausse du facteur = hausse des taux)
    #   PC2 : pente positive (charge du long > charge du court)
    #   PC3 : bosse positive (ventre > ailes)
    if loadings[:, 0].mean() < 0:
        loadings[:, 0] *= -1
    if loadings.shape[1] > 1 and (loadings[-1, 1] - loadings[0, 1]) < 0:
        loadings[:, 1] *= -1
    if loadings.shape[1] > 2:
        ventre = loadings[len(maturites) // 2, 2]
        ailes = 0.5 * (loadings[0, 2] + loadings[-1, 2])
        if ventre - ailes < 0:
            loadings[:, 2] *= -1

    scores = pd.DataFrame(
        X_centre.to_numpy() @ loadings,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
    )

    return ResultatACP(maturites, loadings, variance_expliquee, scores, sur_variations)


def comparer_niveaux_variations(courbe: pd.DataFrame) -> str:
    """
    Démonstration du piège : la même ACP sur niveaux et sur variations.

    Sur les niveaux, PC1 capte ~99 % — résultat spectaculaire mais creux, qui
    reflète la persistance des séries, pas la structure des mouvements. Sur
    les variations, la décomposition 3-facteurs apparaît, et c'est elle qui
    a un sens économique.
    """
    sur_niveaux = acp(courbe, sur_variations=False)
    sur_variations = acp(courbe, sur_variations=True)

    return (
        "═" * 64 + "\n"
        "LE PIÈGE : niveaux vs variations\n" + "═" * 64 + "\n\n"
        + sur_niveaux.resume() + "\n\n"
        + sur_variations.resume() + "\n\n"
        f"Sur les NIVEAUX, PC1 affiche {sur_niveaux.variance_expliquee[0]:.1%} : "
        "impressionnant et vide —\nles taux dérivent ensemble (séries quasi "
        "non stationnaires), l'ACP capte\ncette dérive commune, pas la structure "
        "des mouvements.\n\n"
        f"Sur les VARIATIONS, PC1 tombe à {sur_variations.variance_expliquee[0]:.1%} "
        f"et il faut {_n_pour(sur_variations, 0.95)} facteurs pour 95 % :\n"
        "c'est la vraie granularité des mouvements de courbe — et c'est elle\n"
        "qui répond à la question économique « quand la courbe bouge, comment\n"
        "bouge-t-elle ? »"
    )


def _n_pour(res: ResultatACP, seuil: float) -> int:
    """Nombre de composantes nécessaires pour atteindre `seuil` de variance."""
    return int(np.searchsorted(np.cumsum(res.variance_expliquee), seuil) + 1)


def interpretation(res: ResultatACP) -> str:
    """
    Lecture économique des trois premiers facteurs — la partie qui distingue
    un analyste d'un utilisateur de sklearn.
    """
    ve = res.variance_expliquee
    L = res.loadings
    m = res.maturites

    # Diagnostics quantitatifs des formes
    dispersion_pc1 = L[:, 0].std() / abs(L[:, 0].mean())      # ~0 si plat
    pente_pc2 = L[-1, 1] - L[0, 1]                            # >0 si pente
    bosse_pc3 = L[len(m)//2, 2] - 0.5 * (L[0, 2] + L[-1, 2])  # >0 si bosse

    return (
        "═" * 64 + "\n"
        "INTERPRÉTATION ÉCONOMIQUE\n" + "═" * 64 + "\n\n"
        f"PC1 — NIVEAU ({ve[0]:.1%} de la variance)\n"
        f"  Charges quasi uniformes (dispersion relative {dispersion_pc1:.2f}).\n"
        "  Toute la courbe se déplace en bloc : révision du niveau général\n"
        "  des taux — inflation anticipée, prime de terme. Pour un desk :\n"
        "  c'est le risque de DURATION, couvert en premier.\n\n"
        f"PC2 — PENTE ({ve[1]:.1%})\n"
        f"  Charges opposées aux extrémités (écart long−court : {pente_pc2:+.2f}).\n"
        "  Aplatissement / pentification : le facteur de POLITIQUE MONÉTAIRE.\n"
        "  La banque centrale pilote le court ; croissance et inflation\n"
        "  anticipées pilotent le long. Une inversion de pente est le\n"
        "  signal récession le plus suivi du marché.\n\n"
        f"PC3 — COURBURE ({ve[2]:.1%})\n"
        f"  Le ventre contre les ailes (bosse : {bosse_pc3:+.2f}).\n"
        "  Mouvements du 2-5-10 : demande sur les maturités intermédiaires,\n"
        "  anticipations de volatilité. C'est le facteur des trades\n"
        "  BUTTERFLY (long les ailes, short le ventre, ou l'inverse).\n\n"
        f"Ensemble : {res.variance_cumulee(3):.1%} des mouvements de courbe\n"
        "expliqués par 3 facteurs — sur 10 maturités. C'est le résultat de\n"
        "Litterman-Scheinkman (1991), toujours vérifié 30 ans après."
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from donnees import courbe_synthetique, charger_fred

    if "--demo" in sys.argv:
        print("MODE DÉMO — courbe synthétique Nelson-Siegel (3 facteurs imposés)")
        print("Ces données valident le PIPELINE, pas le fait empirique.\n")
        courbe = courbe_synthetique()
    else:
        print("Chargement FRED (courbe US complète)...")
        courbe = charger_fred()
        print(f"{len(courbe)} jours, de {courbe.index[0].date()} "
              f"à {courbe.index[-1].date()}\n")

    print(comparer_niveaux_variations(courbe))
    print()
    res = acp(courbe, sur_variations=True)
    print(interpretation(res))