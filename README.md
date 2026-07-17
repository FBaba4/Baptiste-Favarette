# Pricer d'options — trois méthodes, une validation croisée

Implémentation et **validation croisée** de trois méthodes de pricing d'options :
formule fermée de Black-Scholes, arbre binomial (Cox-Ross-Rubinstein), et
simulation de Monte-Carlo. Puis inversion du modèle sur données de marché
réelles pour observer le **smile de volatilité**.

Le projet n'a pas pour but de « trouver le prix d'une option » — le marché
l'affiche déjà. Il a pour but de montrer *comment on sait qu'un pricer est
juste*, et *où le modèle se trompe*.

```bash
git clone <repo> && cd options-pricer
pip install -r requirements.txt
pytest tests/ -v                    # 49 tests
python python/tracer_smile.py SPY   # le smile, sur vraies données
```

---

## Le résultat central : trois méthodes indépendantes s'accordent

Call européen, `S=100, K=100, T=1 an, r=5 %, σ=20 %` :

| Méthode | Prix | Erreur vs référence | Temps |
|---|---|---|---|
| Black-Scholes (formule fermée) | **10.450584** | — (référence) | < 0.1 ms |
| Arbre binomial (N=5000) | 10.450184 | 4.0 × 10⁻⁴ | ~180 ms |
| Monte-Carlo (M=10⁶, antithétique) | 10.449871 ± 0.020 | dans l'IC 95 % | ~20 ms |

Trois implémentations qui ne partagent aucun code produisent le même prix.
C'est ce qui donne confiance : elles ne peuvent pas être fausses de la même
façon par hasard.

![Convergence des trois méthodes](notebooks/figures/convergence.png)

**À gauche**, l'arbre binomial converge vers Black-Scholes en oscillant. Cette
oscillation (effet *sawtooth*) n'est pas un bug : elle vient de la position du
strike par rapport aux nœuds terminaux, qui bascule avec la parité de N. La
convergence est en **O(1/N)** — l'erreur est divisée par 2 quand N double,
ce que le tableau ci-dessous vérifie :

| N | 100 | 500 | 1000 | 5000 |
|---|---|---|---|---|
| erreur | 2.0 × 10⁻² | 4.0 × 10⁻³ | 2.0 × 10⁻³ | 4.0 × 10⁻⁴ |

**À droite**, le Monte-Carlo et son intervalle de confiance à 95 %, qui se
resserre en 1/√M. Black-Scholes reste dans l'IC à chaque taille d'échantillon.

---

## Ce que l'arbre sait faire et que Black-Scholes ne peut pas

Black-Scholes ne connaît que la maturité. L'arbre teste l'exercice à **chaque
nœud**, ce qui lui permet de pricer les options **américaines** :

| | Européenne | Américaine | Prime |
|---|---|---|---|
| Call (sans dividende) | 10.4506 | 10.4506 | **0.000000** |
| Put | 5.5735 | 6.0896 | **+0.5181** |

La prime nulle sur le call n'est pas un hasard numérique : c'est un
**théorème** — il n'est jamais optimal d'exercer un call américain sans
dividende avant l'échéance (on perdrait la valeur temps et on paierait le
strike plus tôt). L'arbre ne « connaît » pas ce résultat, il teste bêtement
l'exercice à chaque nœud — et retrouve zéro à l'erreur machine près. C'est
une validation contre la théorie, pas contre soi-même.

Sur le put, la prime est strictement positive : exercer tôt permet d'encaisser
le strike et de le placer au taux sans risque.

---

## Réduction de variance : ce que mesure vraiment le gain

![Réduction de variance](notebooks/figures/reduction_variance.png)

Les **variables antithétiques** (utiliser Z *et* −Z) donnent un gain mesuré de
**1.41×** sur l'erreur standard, stable quel que soit M.

Ce 1.41 ≈ √2 n'est pas arbitraire. On peut montrer que le gain vaut
`1/√(1+ρ)` où ρ est la corrélation entre les payoffs antithétiques. Un gain de
√2 signifie donc **ρ ≈ −0.5** : la corrélation négative n'est pas parfaite
(−1) parce que le payoff est **tronqué** à zéro — le `max(S_T − K, 0)` casse
la symétrie.

Les deux droites du graphe sont **parallèles** en échelle log-log : les
antithétiques ne changent pas la *vitesse* de convergence (toujours 1/√M),
seulement la *constante*.

---

## Le smile : là où le modèle se trompe

C'est la partie qui quitte l'exercice académique.

On **inverse** Black-Scholes : au lieu de partir de σ pour trouver un prix, on
part du prix de marché et on cherche le σ qui le reproduit. C'est d'ailleurs
l'usage dominant du modèle en salle de marché — personne ne dit « ce call vaut
10,45 », on dit « il se traite à 20 de vol ». Black-Scholes sert de
**convertisseur**, pas d'oracle.

![Smile de volatilité](notebooks/figures/smile_DEMO.png)

*(Figure de démonstration sur données synthétiques — voir « Reproduire » pour
la version sur données de marché réelles.)*

Si Black-Scholes était vrai, la volatilité implicite serait **constante** : la
ligne pointillée grise. On observe une courbe.

**Trois explications coexistent** — aucune n'est suffisante seule :

1. **Queues épaisses.** Les rendements réels ne sont pas lognormaux ; les
   krachs sont bien plus fréquents que la loi normale ne le prédit. Le marché
   majore donc les puts loin de la monnaie.
2. **Effet de levier.** Quand une action chute, son ratio dette/capitaux monte
   mécaniquement : elle devient plus risquée, donc plus volatile. Volatilité
   et rendement sont négativement corrélés — ce que le GBM à volatilité
   constante ne capture pas.
3. **Demande de couverture.** Les gérants achètent structurellement des puts
   de protection et vendent des calls. Ce déséquilibre pousse la vol implicite
   des puts vers le haut. Le skew est donc *aussi* un phénomène de flux.

**Le paradoxe à retenir :** tout le monde sait que le modèle est faux, et tout
le monde l'utilise. Parce que sa fausseté est connue, stable et quantifiée.
C'est un thermomètre mal calibré dont on connaît exactement le biais — un
langage commun, pas une vérité.

---

## Le vrai travail : filtrer les données

Sur données de marché réelles, **le filtrage est plus long à écrire que le
calcul**. L'inversion tient en 30 lignes ; les filtres de qualité en font le
double.

| Filtre | Seuil | Pourquoi |
|---|---|---|
| Moneyness | 0.80 ≤ K/S ≤ 1.20 | au-delà, le vega s'écrase et la vol implicite devient extrêmement sensible au bruit |
| Volume | ≥ 10 | une option non traitée n'a qu'un cours indicatif périmé |
| Open interest | ≥ 10 | pas de positions ouvertes = strike sans intérêt réel |
| Spread bid-ask | ≤ 25 % du mid | au-delà, l'incertitude sur le prix dépasse l'effet mesuré |
| Prix mid | ≥ 0.05 | en dessous, le tick de cotation représente 20 % du prix |
| Bornes d'arbitrage | vérifiées | un prix hors bornes n'admet **aucune** vol implicite |

Un smile propre sur 40 points vaut mieux qu'un nuage illisible sur 300.

**Deux pièges rencontrés, documentés dans le code :**

- **Sélection d'échéance par index.** Prendre « la 3ᵉ expiration » suppose un
  calendrier mensuel. SPY expire *trois fois par semaine* : l'index 2 tombait
  sur une échéance à 4 jours, où tout est filtré. On sélectionne désormais par
  **durée cible** (~45 jours), robuste quel que soit le sous-jacent.
- **`NaN or 0` vaut `NaN`.** yfinance renvoie souvent `NaN`, pas `0`. Or NaN
  est *truthy* en Python, et `NaN < 10` vaut `False` — l'option passait donc
  le filtre de liquidité au lieu d'être rejetée. Bug silencieux qui aurait
  pollué le smile sans jamais lever d'erreur. Il faut `pd.isna()` explicite.

---

## Tests

```bash
pytest tests/ -v    # 49 tests
```

Deux familles, dont la seconde est la plus intéressante :

**Convergence croisée** — l'arbre et le MC retrouvent-ils Black-Scholes, sur
8 jeux de paramètres (ATM, ITM, OTM, 1 mois à 5 ans, vol 5 % à 80 %, taux nul) ?

**Propriétés théoriques** — le pricer respecte-t-il des résultats qu'il ne
connaît pas ?
- parité call-put (relation d'**arbitrage**, indépendante de tout modèle)
- bornes sans arbitrage
- prime nulle du call américain sans dividende
- monotonie en strike et en volatilité (vega > 0)
- cas limites : T → 0, σ → 0

> **Un test mal conçu, corrigé — et gardé documenté.** La première version du
> test de synthèse utilisait un seuil **absolu** (`|écart| < 0.05`) et échouait
> à vol 80 %. Ce n'était pas un bug du pricer mais du test : la variance du
> Monte-Carlo croît avec la volatilité, donc son erreur standard aussi. Exiger
> la même précision absolue à vol 5 % et à vol 80 % n'a pas de sens. La
> tolérance est désormais **statistique** (3 erreurs standard) pour le MC, et
> absolue pour l'arbre — chaque méthode jugée selon la nature de son erreur.

---

## Choix de conception

**Un prix Monte-Carlo n'est jamais renvoyé nu.** `prix_monte_carlo` renvoie un
`ResultatMC` avec son erreur standard. Une estimation sans son incertitude est
ininterprétable.

**L'erreur standard antithétique est calculée sur les paires.** Les tirages Z
et −Z ne sont pas indépendants : la formule σ/√M sous-estimerait
l'incertitude. On calcule l'écart-type sur la moyenne de chaque couple.

**Les Grecs suivent les conventions de marché**, pas les dérivées brutes :
vega et rho **pour +1 point**, theta **par jour** — ce que lit un écran de
trading.

**Le delta MC est calculé par méthode *pathwise***, pas par différences
finies : on dérive le payoff sous l'espérance plutôt que de repricer en
(S±h). Moins bruité, pas de paramètre `h` arbitraire. Écart avec la formule
fermée : 8.6 × 10⁻⁵.

**Newton-Raphson puis Brent** pour la vol implicite. Newton converge en 3-4
itérations grâce au vega connu analytiquement, mais diverge quand le vega
s'écrase (options loin de la monnaie) ; Brent est plus lent mais ne peut pas
échouer. On prend la vitesse *et* la robustesse.

**Le smile est tracé en log-moneyness `log(K/F)`**, contre le forward et non
le spot : à 5 ans, un strike « ATM spot » est en réalité très OTM forward.
C'est le seul axe qui permet de comparer plusieurs maturités.

**Seeds fixés partout** — un lecteur doit pouvoir reproduire les chiffres.

---

## Limites assumées

**Le modèle**
- **Volatilité constante** — précisément ce que le smile réfute. Un modèle à
  volatilité stochastique (Heston, SABR) serait l'extension naturelle.
- **Pas de sauts** — le GBM a des trajectoires continues ; les vrais prix
  sautent (résultats, annonces). Merton (1976) ajoute un processus de Poisson.
- **Dividendes continus** (modèle de Merton) — les vrais dividendes sont
  discrets et datés. L'approximation est acceptable sur indice, discutable sur
  action individuelle.
- **Taux constant et connu** — on utilise r = 5 % en dur. Un pricer sérieux
  interpolerait une courbe zéro-coupon.
- **Pas de coûts de transaction ni de contraintes de couverture** —
  l'argument de réplication suppose une couverture continue et sans frais.

**L'implémentation**
- **Base calendaire (365 j)** pour la maturité. Les desks utilisent souvent
  une base en jours ouvrés (252) : il ne se passe rien le week-end. L'écart
  est faible et systématique, donc il ne déforme pas la *forme* du smile,
  mais il sous-estime légèrement la vol implicite.
- **Monte-Carlo mono-thread**, non parallélisé.
- **Le MC ne price que des payoffs européens** — on tire S_T en un seul pas.
  Une option path-dependent exigerait de simuler le chemin complet, et un
  biais de discrétisation apparaîtrait.
- **Données yfinance** : gratuites, donc bruitées. Cotations parfois périmées,
  spreads aberrants. D'où l'importance des filtres.

---

## Reproduire

```bash
pip install -r requirements.txt

pytest tests/ -v                          # 49 tests
python python/black_scholes.py            # prix + Grecs + parité
python python/binomial.py                 # convergence + options américaines
python python/monte_carlo.py              # MC + réduction de variance
python python/implied_vol.py              # test aller-retour de l'inverseur
python notebooks/generer_graphes.py       # figures de convergence

# Sur données de marché réelles :
python python/smile.py SPY                # tableau + diagnostic des filtres
python python/tracer_smile.py SPY         # la figure du smile
```

> ⚠️ **Lancer le smile pendant les heures de marché US** (15h30–22h heure de
> Paris). Hors séance, les bid/ask tombent à zéro et tout est filtré — le
> script affiche alors un diagnostic expliquant quel filtre a tout rejeté.

Un mode `--demo` permet de tester le pipeline sans réseau, sur une chaîne
synthétique à skew imposé (test aller-retour : on retrouve le skew imposé à
2 × 10⁻⁹ près).

---

## Structure

```
options-pricer/
├── python/
│   ├── black_scholes.py    # formule fermée + Grecs (la référence)
│   ├── binomial.py         # arbre CRR + options américaines
│   ├── monte_carlo.py      # simulation + variables antithétiques
│   ├── implied_vol.py      # inversion : Newton-Raphson + repli Brent
│   ├── smile.py            # chaîne d'options réelles + filtrage
│   └── tracer_smile.py     # la figure du smile
├── tests/
│   └── test_convergence.py # 49 tests : convergence + propriétés théoriques
└── notebooks/
    ├── generer_graphes.py
    └── figures/
```

---

## Prochaines étapes

- **Portage du Monte-Carlo en C++** (en cours — C++ étudié cette année).
  Le gain sur un call vanille serait modeste : NumPy vectorise et délègue à du
  C compilé. L'écart se creuse sur les produits **path-dependent**, où la
  vectorisation ne s'applique plus, et sur la mémoire (NumPy alloue M doubles ;
  une boucle scalaire accumule à mémoire constante). C'est la raison pour
  laquelle les bibliothèques de pricing en banque sont en C++ avec des
  bindings Python : Python pour orchestrer, C++ pour calculer.
- **Options asiatiques et à barrière** — le cas qui justifie vraiment le
  Monte-Carlo, et où l'arbre comme la formule fermée s'effondrent.
- **Surface de volatilité** — étendre le smile à plusieurs maturités.
- **Modèle de Heston** — volatilité stochastique, pour capturer le smile au
  lieu de le subir.
