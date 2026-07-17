"""
test_acp.py — Validation de l'ACP

Trois familles :
  1. Propriétés mathématiques (vraies pour toute ACP correcte) ;
  2. Aller-retour : sur une courbe Nelson-Siegel à 3 facteurs imposés, l'ACP
     doit retrouver ~3 facteurs avec les bonnes formes ;
  3. Équivalence avec sklearn (notre SVD maison = leur implémentation).

Lancer : pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from donnees import courbe_synthetique
from acp_courbe import acp, _n_pour


@pytest.fixture(scope="module")
def courbe():
    return courbe_synthetique(n_jours=2500, seed=42)


@pytest.fixture(scope="module")
def res(courbe):
    return acp(courbe, sur_variations=True)


# ── 1. Propriétés mathématiques ─────────────────────────────────────────
def test_loadings_orthonormes(res):
    """Les vecteurs propres forment une base orthonormée : LᵀL = I."""
    produit = res.loadings.T @ res.loadings
    assert np.allclose(produit, np.eye(produit.shape[0]), atol=1e-10)

def test_variance_somme_a_un(res):
    assert res.variance_expliquee.sum() == pytest.approx(1.0, abs=1e-12)

def test_variance_decroissante(res):
    assert np.all(np.diff(res.variance_expliquee) <= 1e-12)

def test_scores_non_correles(res):
    """Les composantes principales sont décorrélées par construction."""
    corr = np.corrcoef(res.scores.to_numpy().T)
    hors_diag = corr - np.diag(np.diag(corr))
    assert np.abs(hors_diag).max() < 1e-8

def test_reconstruction_exacte(courbe, res):
    """Avec TOUTES les composantes, scores·Lᵀ reconstruit exactement les
    données centrées — l'ACP est une rotation, elle ne perd rien."""
    X = (courbe.diff().dropna() * 100.0)
    X_centre = (X - X.mean(axis=0)).to_numpy()
    reconstruction = res.scores.to_numpy() @ res.loadings.T
    assert np.allclose(reconstruction, X_centre, atol=1e-8)


# ── 2. Aller-retour sur facteurs imposés ────────────────────────────────
def test_trois_facteurs_dominent(res):
    """La courbe synthétique est pilotée par 3 facteurs (+ bruit) : l'ACP
    doit expliquer ≥ 95 % avec 3 composantes."""
    assert res.variance_cumulee(3) > 0.95

def test_forme_pc1_niveau(res):
    """PC1 : charges quasi uniformes — toutes de même signe, dispersion faible."""
    pc1 = res.loadings[:, 0]
    assert np.all(pc1 > 0)
    assert pc1.std() / pc1.mean() < 0.30

def test_forme_pc2_pente(res):
    """PC2 : signes opposés aux deux extrémités de la courbe."""
    pc2 = res.loadings[:, 1]
    assert pc2[0] * pc2[-1] < 0

def test_forme_pc3_courbure(res):
    """PC3 : le ventre s'oppose aux ailes (forme en bosse)."""
    pc3 = res.loadings[:, 2]
    ventre = pc3[len(pc3) // 2]
    ailes = 0.5 * (pc3[0] + pc3[-1])
    assert (ventre - ailes) > 0
    assert ventre * pc3[0] < 0 or ventre * pc3[-1] < 0

def test_piege_niveaux(courbe):
    """Sur les NIVEAUX, PC1 est artificiellement gonflé par la persistance :
    il doit dépasser sa valeur sur variations — c'est le piège documenté."""
    sur_niveaux = acp(courbe, sur_variations=False)
    sur_variations = acp(courbe, sur_variations=True)
    assert sur_niveaux.variance_expliquee[0] >= sur_variations.variance_expliquee[0] - 0.02
    # Et sur niveaux, 2 composantes suffisent presque toujours à 99 % :
    assert _n_pour(sur_niveaux, 0.99) <= 3


# ── 3. Équivalence avec sklearn ─────────────────────────────────────────
def test_equivalence_sklearn(courbe, res):
    """Notre SVD maison doit coïncider avec sklearn.decomposition.PCA —
    au SIGNE près par composante (l'ACP est définie au signe près ; nous
    imposons une convention, sklearn une autre)."""
    from sklearn.decomposition import PCA

    X = (courbe.diff().dropna() * 100.0)
    p = PCA()
    p.fit(X.to_numpy())

    assert np.allclose(p.explained_variance_ratio_, res.variance_expliquee, atol=1e-10)
    for i in range(3):
        notre = res.loadings[:, i]
        leur = p.components_[i]
        # colinéarité parfaite, signe libre
        assert min(np.abs(notre - leur).max(), np.abs(notre + leur).max()) < 1e-8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))