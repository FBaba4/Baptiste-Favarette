# Trois facteurs derrière toute la courbe des taux

Analyse en composantes principales des mouvements de la courbe des taux
souverains US. Résultat central, connu depuis Litterman & Scheinkman (1991)
et toujours vérifié aujourd'hui : **trois facteurs — niveau, pente,
courbure — expliquent l'essentiel des mouvements de dix maturités**, du
3 mois au 30 ans.

```bash
pip install -r requirements.txt
pytest tests/ -v                          # 11 tests
python python/tracer_facteurs.py          # les figures, sur données FRED
```

---

## Le résultat

![Les trois facteurs](figures/loadings_DEMO.png)

*(Figure de démonstration sur courbe synthétique — voir « Reproduire » pour
la version sur données FRED réelles.)*

Trois formes, trois lectures économiques :

**PC1 — niveau (91 %).** Charges quasi identiques sur toutes les maturités :
toute la courbe monte ou descend en bloc. C'est une révision du niveau
général des taux — inflation anticipée, prime de terme. Pour un desk, c'est
le risque de **duration**, celui qu'on couvre en premier.

**PC2 — pente (4 %).** Charges de signes opposés aux deux extrémités : le
court monte quand le long baisse, ou l'inverse. C'est le facteur de
**politique monétaire** — la banque centrale pilote le court terme, les
anticipations de croissance et d'inflation pilotent le long. Une inversion de
pente (2 ans au-dessus du 10 ans) est le signal de récession le plus suivi
du marché.

**PC3 — courbure (1 %).** Le ventre de la courbe (5-7 ans) contre les ailes
(court et long). Lié à la demande sur les maturités intermédiaires et aux
anticipations de volatilité — c'est le facteur des trades **butterfly**
(long les ailes, short le ventre, ou l'inverse).

**Trois facteurs, 96 % de variance expliquée, sur dix maturités.**

---

## Combien de facteurs, vraiment ?

![Variance expliquée](figures/variance_DEMO.png)

Le scree plot : après PC3, chaque composante additionnelle n'ajoute presque
rien. C'est la signature empirique du résultat — pas un artefact du choix de
seuil à 95 %.

---

## Le piège central : niveaux contre variations

Faire l'ACP sur les **niveaux** des taux (au lieu de leurs variations
quotidiennes) donne un PC1 à ~99 % — spectaculaire, et **vide de sens**.

Les niveaux de taux sont des séries très persistantes, quasi non
stationnaires : tout est corrélé à tout parce que tout dérive ensemble sur
le temps. L'ACP capte cette dérive commune, pas la structure des
*mouvements*. C'est le même mécanisme que les régressions fallacieuses de
Granger-Newbold (1974) : deux séries non stationnaires indépendantes
affichent souvent une corrélation élevée sans aucun lien réel.

La question économique intéressante n'est pas « les taux sont-ils tous
corrélés ? » (banalement oui) mais « **quand la courbe bouge, comment
bouge-t-elle ?** » — d'où l'ACP sur les variations quotidiennes, en points de
base.

```
ACP sur niveaux (⚠️ piège)      ACP sur variations
  PC1 : 92 %                     PC1 : 91 %
  PC2 :  8 %                     PC2 :  4 %
  PC3 :  0.1 %                   PC3 :  0.8 %
```

Le chiffre de PC1 est presque identique dans les deux cas ici (données
synthétiques propres) — l'écart est en général bien plus marqué sur données
réelles, où les niveaux sont fortement autocorrélés. Ce qui change
radicalement, c'est la **suite de la décomposition** : sur niveaux, 2
composantes suffisent à 99 %, ce qui n'apprend rien (les taux dérivent tous
ensemble, un point c'est tout) ; sur variations, il faut regarder PC2 et PC3
pour voir apparaître la vraie structure — pente et courbure. Beaucoup
d'étudiants tombent dans ce piège en confondant les deux.

---

## Ce qui se passe dans le temps

![Trajectoires des facteurs](figures/facteurs_temps_DEMO.png)

Les scores cumulés de chaque facteur. Sur données FRED réelles, PC1 (niveau)
retrace la trajectoire des taux directeurs — on y lit à l'œil le cycle de
resserrement monétaire 2022-2023. PC2 (pente) raconte le cycle : elle
s'aplatit puis s'inverse en fin de cycle de hausses, se repentifie en anticipation
de baisses.

---

## Pourquoi l'ACP « maison » (SVD) et pas `sklearn` directement

Trois lignes de `numpy.linalg.svd` suffisent, et elles montrent ce qu'on fait
réellement : centrer les données, décomposer, lire les vecteurs singuliers.
Utiliser `sklearn.decomposition.PCA` sans comprendre la mécanique en dessous
cacherait précisément ce que ce projet veut démontrer.

**Le test le plus fort du projet vérifie l'équivalence des deux** :
`test_equivalence_sklearn` compare loadings et variance expliquée entre
l'implémentation maison et sklearn — identiques à 10⁻¹⁰ près (au signe près
par composante, l'ACP étant définie ainsi). Ce n'est pas une coïncidence :
c'est la preuve qu'implémenter "à la main" et utiliser une bibliothèque de
référence donnent le même résultat, donc que l'implémentation est correcte.

---

## Tests

```bash
pytest tests/ -v    # 11 tests
```

**Propriétés mathématiques** — vraies pour toute ACP correcte, testées sans
dépendre des données : orthonormalité des loadings, somme des variances
expliquées = 1, décorrélation des scores, reconstruction exacte avec toutes
les composantes.

**Aller-retour sur facteurs imposés** — la courbe synthétique est pilotée par
3 facteurs Nelson-Siegel + bruit. L'ACP doit retrouver ≥ 95 % de variance
expliquée avec 3 composantes, et surtout les **bonnes formes** : PC1 de même
signe partout, PC2 de signes opposés aux extrémités, PC3 en bosse au centre.

**Piège niveaux/variations** — vérifie que l'ACP sur niveaux atteint 99 % en
2 composantes à peine (signature du problème), contrairement aux variations.

**Équivalence sklearn** — voir ci-dessus.

---

## Choix de conception

**ACP sur la covariance, pas la corrélation** (`standardiser=False` par
défaut). Toutes les colonnes sont dans la même unité (points de base) : la
standardisation effacerait une information réelle — le 2 ans bouge
structurellement plus que le 30 ans en période de resserrement monétaire, et
c'est un fait qu'on veut que l'ACP capture, pas qu'elle gomme.

**SVD plutôt que diagonalisation de la matrice de covariance.** Numériquement
plus stable : former XᵀX élève le conditionnement au carré, ce que la SVD
évite en travaillant directement sur X.

**Convention de signe explicite et reproductible.** L'ACP est indéterminée au
signe près par composante. On impose : PC1 de charge moyenne positive, PC2 de
pente positive (long > court), PC3 de bosse positive (ventre > ailes) — sinon
deux exécutions du même code pourraient afficher des graphes en miroir.

**Interpolation des données manquantes par maturité dans le temps, jamais
entre maturités** — interpoler le 5 ans à partir du 2 ans et du 10 ans un
jour donné inventerait une forme de courbe qui n'a pas existé.

---

## Limites assumées

- **Régime unique implicite.** L'ACP suppose une structure de covariance
  stable. Or elle change entre régimes (taux zéro post-2008 vs resserrement
  2022) — une ACP glissante (rolling window) serait l'extension naturelle
  pour le montrer.
- **Linéarité.** L'ACP ne capture que les relations linéaires ; des non-
  linéarités (effets de seuil près de zéro, asymétrie hausse/baisse) lui
  échappent.
- **Un seul pays.** Pas de facteurs communs entre courbes souveraines
  (US/zone euro), qui existent et sont suivis par les desks macro cross-asset.
- **Base FRED gratuite** : données quotidiennes de clôture, pas intraday ;
  suffisant pour une analyse de moyen terme, pas pour du trading haute
  fréquence.

---

## Reproduire

```bash
pip install -r requirements.txt

pytest tests/ -v                              # 11 tests
python python/acp_courbe.py                   # données FRED réelles
python python/acp_courbe.py --demo            # synthétique, sans réseau
python python/tracer_facteurs.py              # figures, données FRED
python python/tracer_facteurs.py --demo       # figures, synthétique
```

Le premier lancement sur FRED télécharge et met en cache l'historique complet
(`data/courbe_us.csv`) ; les suivants sont instantanés.

---

## Structure

```
yield-curve-pca/
├── python/
│   ├── donnees.py          # chargement FRED + génération synthétique
│   ├── acp_courbe.py        # ACP par SVD, piège niveaux/variations, interprétation
│   └── tracer_facteurs.py   # les trois figures
├── tests/
│   └── test_acp.py          # 11 tests : maths, aller-retour, équivalence sklearn
└── figures/
```

---

## Prochaines étapes

- **ACP glissante (rolling)** — suivre l'évolution de la variance expliquée
  par régime, pour montrer que la structure 3-facteurs elle-même varie dans
  le temps.
- **Comparaison multi-pays** — US, zone euro, UK : un facteur commun global
  existe-t-il derrière les facteurs domestiques ?
- **Reconstruction et détection d'anomalies** — un point qui s'écarte
  fortement de sa reconstruction à 3 facteurs signale un mouvement
  idiosyncratique (une maturité spécifique sous tension), utile pour repérer
  des dislocations de marché.
