"""
Glossaire.py — Contenu de la page glossaire de MAF.

Séparé de App.py pour rester facile à enrichir sans toucher au reste de
l'interface. Chaque entrée : nom du terme, définition, formule
mathématique (LaTeX, rendue nativement par st.latex()), et pour les
concepts les plus théoriques, une "explication" plus longue.

⚠️ Couverture : ce glossaire couvre tous les champs dont le nom exact a été
vu dans le code du projet (CSV d'origine, PredictionAgent.py, SQLAgent.py,
DocumentExtractor.py). DataCollecting.py a été étendu à un moment du projet
avec des colonnes supplémentaires (PEG ratio, payout ratio, 52-week
high/low, moyennes mobiles, volume...) dont les noms français exacts
n'ont jamais été communiqués — elles ne sont donc PAS incluses ici, pour
éviter d'inventer un nom de colonne qui ne correspondrait pas à la réalité
du CSV. À compléter si besoin avec la liste exacte de get_csv_columns().

⚠️ Les formules "descriptives" (WACC, BFR, EV/EBITDA, CET1) sont des
définitions générales d'analyse financière — MAF ne les calcule pas
lui-même à partir de la base, il les explique seulement quand on les
mentionne dans une question.
"""

GLOSSAIRE = [
    # ─────────────────────────────────────────────────────────────────
    # Champs d'identification — pas de théorie mathématique, mais inclus
    # pour la couverture complète des colonnes de la base.
    # ─────────────────────────────────────────────────────────────────
    {
        "categorie": "Identification & profil de l'entreprise",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Entreprise",
                "definition": "Raison sociale de l'entreprise, telle que collectée (souvent avec sa forme juridique : 'Apple Inc.', 'Microsoft Corporation'). Clé d'identification principale utilisée par tous les outils SQL — voir la reconnaissance approximative de noms (_resoudre_nom_entreprise) qui tolère les variantes.",
            },
            {
                "nom": "Ticker",
                "definition": "Code boursier court identifiant l'entreprise sur sa place de cotation (ex. AAPL pour Apple sur le NASDAQ). Utilisé par yfinance pour la collecte initiale des données.",
            },
            {
                "nom": "Date création",
                "definition": "Date de fondation de l'entreprise — donnée contextuelle, sans usage dans les modèles prédictifs.",
            },
            {
                "nom": "Siège social",
                "definition": "Adresse ou localisation du siège social légal de l'entreprise.",
            },
            {
                "nom": "Pays",
                "definition": "Pays du siège social — utilisé pour le contexte géographique et macroéconomique (à mettre en perspective avec contexte_macro.json).",
            },
            {
                "nom": "Ville",
                "definition": "Ville du siège social.",
            },
            {
                "nom": "Secteur",
                "definition": "Secteur d'activité au sens large (ex. Technology, Financial Services) — utilisé comme filtre dans sql_db_statistiques et sql_db_valeurs_atypiques pour comparer une entreprise à ses pairs plutôt qu'à tout le panel.",
            },
            {
                "nom": "Industrie",
                "definition": "Classification plus fine que le secteur (ex. au sein de 'Technology' : 'Software—Infrastructure', 'Semiconductors'...).",
            },
            {
                "nom": "Nombre employés",
                "definition": "Effectif total de l'entreprise — donnée de taille, corrélée en général (mais pas systématiquement) au chiffre d'affaires.",
            },
            {
                "nom": "Devise",
                "definition": (
                    "Devise dans laquelle les montants financiers de l'entreprise sont "
                    "exprimés (EUR, USD, GBP, CHF...). ⚠️ Important pour toute "
                    "comparaison internationale : comparer un chiffre d'affaires en USD "
                    "à un chiffre d'affaires en EUR sans conversion fausse toute analyse "
                    "— sql_db_statistiques et sql_db_comparer_entreprises ne font "
                    "actuellement AUCUNE conversion de devise automatique."
                ),
            },
            {
                "nom": "Bourse",
                "definition": (
                    "Place boursière sur laquelle l'action est cotée (NASDAQ, NYSE, "
                    "Euronext Paris, Xetra...) — donnée contextuelle, liée au ticker "
                    "(le suffixe '.PA', '.DE', '.SW' etc. dans le ticker yfinance "
                    "l'indique déjà implicitement)."
                ),
            },
        ],
    },
]

GLOSSAIRE = GLOSSAIRE + [
    {
        "categorie": "Taille & structure du capital",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Capitalisation boursière (Market Cap)",
                "definition": (
                    "Valeur totale de l'entreprise telle qu'évaluée par le marché boursier "
                    "— le prix qu'il faudrait payer pour racheter 100% des actions au cours "
                    "actuel. Différent de la valeur comptable (capitaux propres) : la "
                    "capitalisation reflète les anticipations futures du marché, pas "
                    "seulement les actifs nets actuels."
                ),
                "formule": r"\text{Capitalisation} = \text{Cours de l'action} \times \text{Nombre d'actions en circulation}",
            },
            {
                "nom": "Actions en circulation",
                "definition": (
                    "Nombre total d'actions émises par l'entreprise et détenues par les "
                    "actionnaires (hors autodétention/rachat). Dénominateur de nombreux "
                    "ratios par action (BPA, valeur comptable/action)."
                ),
            },
            {
                "nom": "Valeur comptable par action (Book Value per Share)",
                "definition": (
                    "Part des capitaux propres comptables revenant à chaque action — ce "
                    "que recevrait théoriquement un actionnaire si l'entreprise liquidait "
                    "tous ses actifs et remboursait toutes ses dettes. MAF l'utilise pour "
                    "ESTIMER les capitaux propres totaux (colonne dérivée 'Capitaux propres "
                    "estimés'), en l'absence de cette donnée directement dans le CSV."
                ),
                "formule": r"\text{Valeur comptable/action} = \frac{\text{Capitaux propres}}{\text{Actions en circulation}}",
                "note": (
                    "MAF utilise cette formule À L'ENVERS pour reconstituer les capitaux "
                    "propres : Capitaux propres estimés = Valeur comptable/action × "
                    "Actions en circulation (voir _ajouter_features_derivees dans "
                    "PredictionAgent.py)."
                ),
            },
            {
                "nom": "Capitaux propres (Equity)",
                "definition": (
                    "Ressources appartenant réellement aux actionnaires : la différence "
                    "entre tout ce que possède l'entreprise (actifs) et tout ce qu'elle "
                    "doit (dettes). C'est le dénominateur du ROE et un des trois piliers "
                    "du WACC. Dans MAF, cette valeur n'est pas collectée directement mais "
                    "ESTIMÉE (voir 'Valeur comptable par action' ci-dessus) — à traiter "
                    "comme une approximation, pas une donnée comptable exacte."
                ),
                "formule": r"\text{Capitaux propres} = \text{Total actifs} - \text{Total dettes}",
            },
        ],
    },

    {
        "categorie": "Rentabilité",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Chiffre d'affaires (CA)",
                "definition": (
                    "Total des ventes de biens et/ou services réalisées par l'entreprise "
                    "sur une période, avant déduction de charges. Indicateur de taille "
                    "d'activité — 'top line' du compte de résultat, à ne pas confondre "
                    "avec le bénéfice ('bottom line')."
                ),
            },
            {
                "nom": "Bénéfice Net (Résultat Net)",
                "definition": (
                    "Ce qu'il reste du chiffre d'affaires une fois TOUTES les charges "
                    "déduites (coûts de production, frais généraux, charges financières, "
                    "impôts, éléments exceptionnels). C'est la 'bottom line' — l'indicateur "
                    "de rentabilité ultime, mais qui peut être affecté par des éléments "
                    "non récurrents (cessions d'actifs, dépréciations) qui brouillent la "
                    "lecture de la performance opérationnelle réelle."
                ),
            },
            {
                "nom": "Marge brute",
                "definition": (
                    "Performance de production ou de service, avant prise en compte des "
                    "frais fixes et administratifs. Mesure l'efficacité du cœur de "
                    "métier — combien reste-t-il après avoir payé ce qui a été vendu, "
                    "avant les coûts de structure."
                ),
                "formule": r"\text{Marge brute (\%)} = \frac{\text{Chiffre d'affaires} - \text{Coût des ventes}}{\text{Chiffre d'affaires}} \times 100",
            },
            {
                "nom": "Marge nette",
                "definition": (
                    "Part du chiffre d'affaires qui se transforme en bénéfice net, une "
                    "fois toutes les charges déduites. L'indicateur de rentabilité le "
                    "plus synthétique, mais aussi le plus sensible aux éléments "
                    "exceptionnels du bénéfice net."
                ),
                "formule": r"\text{Marge nette (\%)} = \frac{\text{Bénéfice net}}{\text{Chiffre d'affaires}} \times 100",
                "explication": (
                    "Comparer des marges nettes entre secteurs n'a souvent pas de sens : "
                    "un distributeur (grande vitesse, faible marge) et une entreprise "
                    "pharmaceutique (marges élevées, cycles longs) ne sont pas "
                    "comparables sur ce seul critère. C'est pourquoi sql_db_statistiques "
                    "permet de filtrer par secteur avant de juger une marge nette."
                ),
            },
            {
                "nom": "ROE — Rentabilité des capitaux propres",
                "definition": (
                    "Mesure la rentabilité générée pour les actionnaires, rapportée aux "
                    "capitaux qu'ils ont investis (pas au total des actifs)."
                ),
                "formule": r"ROE = \frac{\text{Bénéfice net}}{\text{Capitaux propres}} \times 100",
                "explication": (
                    "Le ROE peut être décomposé en trois leviers (décomposition de "
                    "DuPont), ce qui explique pourquoi deux entreprises au même ROE "
                    "peuvent avoir des profils de risque très différents : "
                    r"$ROE = \underbrace{\frac{\text{Bénéfice net}}{CA}}_{\text{marge}} \times "
                    r"\underbrace{\frac{CA}{\text{Actifs}}}_{\text{rotation des actifs}} \times "
                    r"\underbrace{\frac{\text{Actifs}}{\text{Capitaux propres}}}_{\text{levier financier}}$. "
                    "Un ROE élevé porté surtout par le troisième terme (beaucoup de "
                    "dette relativement aux capitaux propres) est plus risqué qu'un ROE "
                    "élevé porté par la marge ou la rotation des actifs — c'est "
                    "précisément pour cette raison que MAF croise toujours le ROE avec "
                    "le ratio Dette/Capital plutôt que de le lire isolément."
                ),
            },
            {
                "nom": "ROA — Rentabilité des actifs",
                "definition": (
                    "Mesure la capacité de l'entreprise à générer du profit à partir de "
                    "l'ensemble de ses actifs, indépendamment de la façon dont ils sont "
                    "financés (dette ou capitaux propres) — moins sensible au levier que "
                    "le ROE, donc souvent plus comparable entre entreprises à structures "
                    "financières différentes."
                ),
                "formule": r"ROA = \frac{\text{Bénéfice net}}{\text{Total des actifs}} \times 100",
            },
            {
                "nom": "Croissance du chiffre d'affaires",
                "definition": (
                    "Évolution du chiffre d'affaires par rapport à l'exercice précédent. "
                    "Le modèle de prédiction de cet indicateur dans MAF n'utilise PAS "
                    "d'historique multi-années (le CSV est une photographie à un instant "
                    "T) mais des proxys de marché — le spread entre BPA trailing/forward "
                    "et P/E trailing/forward — qui reflètent les anticipations de "
                    "croissance déjà intégrées dans le cours par les investisseurs."
                ),
                "formule": r"\text{Croissance CA (\%)} = \frac{CA_t - CA_{t-1}}{CA_{t-1}} \times 100",
            },
            {
                "nom": "Croissance CA yfinance (%)",
                "definition": (
                    "⚠️ Colonne DISTINCTE de 'Croissance CA (%)' dans la base — même "
                    "concept économique, mais calculée directement par l'API yfinance "
                    "(méthodologie propre à Yahoo Finance) plutôt que reconstituée par "
                    "MAF. Volontairement EXCLUE des features des modèles prédictifs de "
                    "croissance : trop proche de la cible elle-même, son utilisation "
                    "créerait une fuite de données (data leakage) — le modèle "
                    "'prédirait' en réalité une valeur qu'il connaît déjà sous une autre forme."
                ),
            },
            {
                "nom": "Marge opérationnelle",
                "definition": (
                    "Rentabilité du cœur d'activité après les coûts d'exploitation "
                    "courants (production, ventes, administration), mais AVANT charges "
                    "financières et impôts — se situe entre la marge brute et la marge "
                    "nette dans le compte de résultat, et reflète mieux la performance "
                    "opérationnelle pure qu'une marge nette parfois brouillée par des "
                    "éléments financiers ou exceptionnels."
                ),
                "formule": r"\text{Marge opérationnelle (\%)} = \frac{\text{Résultat opérationnel (EBIT)}}{\text{Chiffre d'affaires}} \times 100",
            },
            {
                "nom": "Marge EBITDA",
                "definition": (
                    "Rentabilité opérationnelle brute, AVANT dépréciation et "
                    "amortissement — utile pour comparer des entreprises à intensité "
                    "capitalistique différente (une usine lourdement amortie vs une "
                    "entreprise de services), puisque ces charges comptables n'affectent "
                    "pas directement la trésorerie générée."
                ),
                "formule": r"\text{Marge EBITDA (\%)} = \frac{EBITDA}{\text{Chiffre d'affaires}} \times 100",
            },
            {
                "nom": "Croissance des bénéfices",
                "definition": (
                    "Évolution du bénéfice net (ou du BPA) par rapport à l'exercice "
                    "précédent — à ne pas confondre avec la croissance du CHIFFRE "
                    "D'AFFAIRES : une entreprise peut voir ses bénéfices croître "
                    "beaucoup plus vite que ses ventes (effet de levier opérationnel, "
                    "amélioration de marge), ou l'inverse (marges qui se compriment "
                    "malgré des ventes en hausse)."
                ),
                "formule": r"\text{Croissance bénéfices (\%)} = \frac{\text{Bénéfice}_t - \text{Bénéfice}_{t-1}}{|\text{Bénéfice}_{t-1}|} \times 100",
            },
            {
                "nom": "Croissance des bénéfices trimestrielle",
                "definition": (
                    "Même logique que la croissance annuelle des bénéfices, mais "
                    "mesurée trimestre par trimestre — plus volatile et plus sensible à "
                    "la saisonnalité, mais capte plus tôt une inflexion de tendance "
                    "qu'une lecture uniquement annuelle."
                ),
            },
        ],
    },

    {
        "categorie": "Structure financière & risque",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Ratio Dette/Capital",
                "definition": (
                    "Mesure le poids de la dette dans la structure financière de "
                    "l'entreprise, par rapport à ses capitaux propres — un indicateur "
                    "classique d'effet de levier et de risque financier."
                ),
                "formule": r"\text{Ratio Dette/Capital (\%)} = \frac{\text{Dette totale}}{\text{Capitaux propres}} \times 100",
                "explication": (
                    "L'effet de levier financier amplifie le ROE dans les deux sens : si "
                    "le rendement des actifs (ROA) dépasse le coût de la dette, "
                    "emprunter augmente le ROE (levier positif) ; si le ROA est inférieur "
                    "au coût de la dette, emprunter le RÉDUIT (levier négatif). Un ratio "
                    "élevé n'est donc ni bon ni mauvais en soi — il amplifie ce qui existe "
                    "déjà, en bien comme en mal."
                ),
            },
            {
                "nom": "Ratio de liquidité générale",
                "definition": (
                    "Capacité de l'entreprise à honorer ses dettes à court terme avec ses "
                    "actifs à court terme — un ratio inférieur à 1 signale un risque de "
                    "tension de trésorerie à court terme. Utilisé dans sql_db_score_risque "
                    "(critère 'liquidité', 40% de la pondération, inversé : plus la "
                    "liquidité est élevée, moins le risque associé est élevé)."
                ),
                "formule": r"\text{Ratio de liquidité générale} = \frac{\text{Actifs court terme}}{\text{Passifs court terme}}",
            },
            {
                "nom": "Ratio de liquidité immédiate (Quick Ratio)",
                "definition": (
                    "Version plus stricte du ratio de liquidité générale : exclut les "
                    "stocks (souvent les moins facilement convertibles en cash) du "
                    "numérateur. Un écart important entre liquidité générale et "
                    "immédiate signale une dépendance forte aux stocks pour honorer les "
                    "dettes court terme — pertinent notamment pour juger la solidité "
                    "réelle d'un distributeur ou d'un industriel."
                ),
                "formule": r"\text{Ratio de liquidité immédiate} = \frac{\text{Actifs court terme} - \text{Stocks}}{\text{Passifs court terme}}",
            },
            {
                "nom": "Ratio de distribution (Payout Ratio)",
                "definition": (
                    "Part du bénéfice net reversée aux actionnaires sous forme de "
                    "dividendes plutôt que réinvestie dans l'entreprise. Un payout "
                    "proche de 100% (ou au-delà) laisse peu de marge de manœuvre : le "
                    "dividende devient vulnérable à la moindre baisse de bénéfice, et "
                    "peu de capital reste disponible pour financer la croissance."
                ),
                "formule": r"\text{Payout Ratio (\%)} = \frac{\text{Dividendes versés}}{\text{Bénéfice net}} \times 100",
            },
            {
                "nom": "Trésorerie totale",
                "definition": (
                    "Liquidités et équivalents de trésorerie détenus par l'entreprise "
                    "(cash, placements très court terme) — le 'coussin de sécurité' "
                    "immédiatement disponible, indépendamment de sa capacité future à "
                    "générer du cash-flow."
                ),
            },
            {
                "nom": "Dette totale",
                "definition": (
                    "Ensemble des emprunts et dettes financières portant intérêt "
                    "(court et long terme) — base de calcul du ratio Dette/Capital et "
                    "de la dette nette (Dette totale − Trésorerie totale), un indicateur "
                    "souvent plus parlant que la dette brute seule."
                ),
            },
            {
                "nom": "Beta (β) — volatilité de marché",
                "definition": (
                    "Mesure la sensibilité du cours d'une action aux mouvements du marché "
                    "dans son ensemble."
                ),
                "formule": r"\beta = \frac{\text{Cov}(R_{\text{action}}, R_{\text{marché}})}{\text{Var}(R_{\text{marché}})}",
                "explication": (
                    r"$\beta = 1$ : l'action évolue comme le marché. $\beta > 1$ : plus "
                    r"volatile — amplifie les hausses ET les baisses (ex. $\beta = 1{,}5$ "
                    "signifie qu'une hausse de 10% du marché s'accompagne en moyenne "
                    r"d'une hausse de 15% de l'action, et inversement à la baisse). "
                    r"$\beta < 1$ : moins volatile que le marché (souvent les secteurs "
                    "défensifs : santé, biens de consommation courante). "
                    r"$\beta < 0$ (rare) : évolue à l'inverse du marché. C'est le "
                    "troisième critère de sql_db_score_risque (25% de la pondération)."
                ),
            },
            {
                "nom": "Score de risque composite (MAF)",
                "definition": (
                    "⚠️ Spécifique à MAF — PAS un indicateur financier standard, ni un "
                    "modèle statistique entraîné : une formule pondérée arbitraire et "
                    "transparente (sql_db_score_risque), combinant trois critères "
                    "convertis en percentile par rapport au panel avant pondération."
                ),
                "formule": r"\text{Score} = 0{,}40 \times (100 - P_{\text{liquidité}}) + 0{,}35 \times P_{\text{endettement}} + 0{,}25 \times P_{\text{beta}}",
                "explication": (
                    "Les pondérations (40/35/25) sont un choix de conception assumé, pas "
                    "une calibration statistique — elles reflètent une priorité donnée à "
                    "la liquidité (risque de défaut à court terme), puis à l'endettement "
                    "(risque structurel), puis à la volatilité de marché (risque perçu par "
                    "les investisseurs). Le score va de 0 (risque le plus faible du panel) "
                    "à 100 (risque le plus élevé) — c'est un classement RELATIF au panel "
                    "actuel, pas une probabilité de défaut absolue."
                ),
            },
            {
                "nom": "Percentile (dans le panel)",
                "definition": (
                    "⚠️ Notion statistique, pas un indicateur financier — utilisée par "
                    "sql_db_score_risque pour rendre comparables trois critères d'échelles "
                    "différentes (liquidité, endettement, beta) avant de les pondérer."
                ),
                "formule": r"P(x) = \frac{\text{nombre de valeurs du panel} < x}{\text{taille du panel}} \times 100",
                "note": "Exemple : percentile de liquidité = 80 signifie que 80% du panel a une liquidité plus faible que cette entreprise.",
            },
            {
                "nom": "Z-score (détection de valeurs atypiques)",
                "definition": (
                    "Mesure de combien d'écarts-types une valeur s'éloigne de la moyenne "
                    "du panel (ou du secteur) — utilisé par sql_db_valeurs_atypiques pour "
                    "repérer les entreprises dont un ratio sort nettement de la norme."
                ),
                "formule": r"z = \frac{x - \mu}{\sigma}",
                "explication": (
                    r"Sous une hypothèse de distribution normale, environ 95% des valeurs "
                    r"se situent entre $z=-2$ et $z=+2$ — le seuil par défaut de MAF "
                    "(seuil_zscore=2.0). Un z-score au-delà de ce seuil signale une valeur "
                    "statistiquement atypique, SANS présumer qu'il s'agit d'une anomalie "
                    "comptable, d'une opportunité, ou d'un simple effet de petit "
                    "échantillon — l'interprétation reste à faire au cas par cas."
                ),
            },
        ],
    },

    {
        "categorie": "Valorisation boursière",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Bénéfice par action (BPA / EPS)",
                "definition": (
                    "Part du bénéfice net attribuable à chaque action en circulation. "
                    "MAF utilise à la fois le BPA 'trailing' (12 derniers mois, réalisé) "
                    "et 'forward' (12 prochains mois, anticipé par le marché) comme "
                    "proxys de croissance dans le modèle de prédiction de croissance du CA."
                ),
                "formule": r"BPA = \frac{\text{Bénéfice net} - \text{Dividendes préférentiels}}{\text{Actions en circulation}}",
            },
            {
                "nom": "P/E Ratio (Price/Earnings, trailing)",
                "definition": (
                    "Combien de fois le bénéfice annuel PASSÉ (12 derniers mois) les "
                    "investisseurs sont prêts à payer pour détenir une action."
                ),
                "formule": r"P/E = \frac{\text{Cours de l'action}}{\text{BPA (trailing)}}",
                "explication": (
                    "Un P/E élevé peut signaler soit des attentes de croissance fortes "
                    "(le marché anticipe des bénéfices futurs bien supérieurs), soit une "
                    "survalorisation. Un P/E bas peut signaler soit une opportunité "
                    "(sous-valorisation), soit une inquiétude du marché sur la pérennité "
                    "des bénéfices actuels — le chiffre seul ne tranche jamais, d'où "
                    "l'intérêt de le comparer au secteur (sql_db_statistiques)."
                ),
            },
            {
                "nom": "Forward P/E",
                "definition": (
                    "Même logique que le P/E trailing, mais rapporté au bénéfice ANTICIPÉ "
                    "des 12 prochains mois plutôt qu'au bénéfice passé — reflète "
                    "directement les attentes de croissance du marché."
                ),
                "formule": r"\text{Forward P/E} = \frac{\text{Cours de l'action}}{\text{BPA (forward)}}",
                "note": "Un Forward P/E nettement inférieur au P/E trailing signale que le marché anticipe une forte croissance des bénéfices.",
            },
            {
                "nom": "PEG Ratio",
                "definition": (
                    "Corrige le P/E Ratio par le taux de croissance anticipé des "
                    "bénéfices — répond à la limite du P/E seul (un P/E de 30 est-il "
                    "cher ? Ça dépend si les bénéfices croissent de 5% ou de 40% par "
                    "an). Un PEG proche de 1 est souvent interprété comme un équilibre "
                    "raisonnable entre le prix payé et la croissance attendue ; un PEG "
                    "très supérieur à 1 peut signaler une survalorisation même avec un "
                    "P/E d'apparence modérée."
                ),
                "formule": r"PEG = \frac{P/E}{\text{Taux de croissance annuel des bénéfices (\%)}}",
            },
            {
                "nom": "P/B Ratio (Price-to-Book, Valorisation Bancaire)",
                "definition": (
                    "Combien de fois la valeur comptable des capitaux propres les "
                    "investisseurs sont prêts à payer. Particulièrement utilisé pour "
                    "valoriser les banques (d'où sa mention explicite 'Valo Bancaire' "
                    "dans le CSV), car leurs actifs (créances, titres) sont plus proches "
                    "de leur valeur de marché que ceux d'une entreprise industrielle."
                ),
                "formule": r"P/B = \frac{\text{Capitalisation boursière}}{\text{Capitaux propres}}",
                "explication": (
                    "P/B < 1 signifie que le marché valorise l'entreprise EN DESSOUS de "
                    "sa valeur comptable — soit une sous-valorisation potentielle, soit "
                    "un doute du marché sur la qualité réelle des actifs comptabilisés "
                    "(fréquent pour les banques en période de stress financier)."
                ),
            },
            {
                "nom": "Rendement du dividende",
                "definition": (
                    "Revenu annuel versé aux actionnaires sous forme de dividendes, "
                    "rapporté au cours de l'action — mesure le rendement 'cash' immédiat "
                    "d'un investissement, indépendamment de la plus-value potentielle."
                ),
                "formule": r"\text{Rendement dividende (\%)} = \frac{\text{Dividende annuel par action}}{\text{Cours de l'action}} \times 100",
                "note": "Un rendement très élevé peut aussi signaler un cours en chute (le dénominateur baisse) plutôt qu'une générosité de l'entreprise — à croiser avec la tendance du cours (agent_technique).",
            },
            {
                "nom": "Valeur d'entreprise (Enterprise Value, EV)",
                "definition": (
                    "Ce qu'il faudrait débourser pour racheter TOUTE l'entreprise, dette "
                    "comprise, en tenant compte de sa trésorerie disponible — contrairement "
                    "à la capitalisation boursière qui ne reflète que la part actionnaires. "
                    "Base de calcul de tous les multiples 'EV/...' (EV/CA, EV/EBITDA)."
                ),
                "formule": r"EV = \text{Capitalisation boursière} + \text{Dette totale} - \text{Trésorerie totale}",
            },
            {
                "nom": "EV/Chiffre d'affaires",
                "definition": (
                    "Multiple de valorisation rapportant la valeur d'entreprise totale "
                    "au chiffre d'affaires — utile pour valoriser des entreprises pas "
                    "encore rentables (EBITDA ou bénéfice négatifs, rendant EV/EBITDA ou "
                    "P/E inutilisables), au prix d'une information plus pauvre (ne dit "
                    "rien de la rentabilité)."
                ),
                "formule": r"EV/CA = \frac{EV}{\text{Chiffre d'affaires}}",
            },
            {
                "nom": "Prix/Ventes (Price/Sales, 12 mois)",
                "definition": (
                    "Version 'capitalisation boursière' du EV/Chiffre d'affaires — ne "
                    "tient pas compte de la dette, contrairement à EV/CA. Les deux "
                    "mesurent une idée proche (combien le marché paie pour 1€ de ventes) "
                    "mais divergent fortement pour les entreprises très endettées."
                ),
                "formule": r"\text{Prix/Ventes} = \frac{\text{Capitalisation boursière}}{\text{Chiffre d'affaires (12 derniers mois)}}",
            },
            {
                "nom": "Prix actuel, 52 sem. haut/bas, moyennes mobiles 50j/200j",
                "definition": (
                    "⚠️ Données de marché brutes, pas des ratios financiers — le cours "
                    "actuel, les extrêmes sur 52 semaines, et les moyennes mobiles à 50 "
                    "et 200 jours (lissage du cours sur ces périodes, utilisé en analyse "
                    "technique pour identifier une tendance de fond au-delà du bruit "
                    "quotidien). Le croisement moyenne 50j / moyenne 200j est un signal "
                    "technique classique ('golden cross' si la 50j dépasse la 200j en "
                    "montant, 'death cross' dans le cas inverse) — MAF ne calcule PAS ce "
                    "signal automatiquement, agent_technique se limite au momentum."
                ),
            },
            {
                "nom": "Volume moyen (10 jours)",
                "definition": (
                    "Nombre moyen d'actions échangées par jour sur les 10 derniers jours "
                    "— indicateur de liquidité du titre. Un volume anormalement élevé "
                    "accompagnant un mouvement de cours en renforce la significativité "
                    "(beaucoup d'investisseurs sont d'accord avec le mouvement) ; un "
                    "mouvement de cours à faible volume est statistiquement moins fiable."
                ),
            },
            {
                "nom": "Objectif de cours des analystes (moyen)",
                "definition": (
                    "⚠️ Opinion humaine agrégée, pas un calcul MAF — moyenne des "
                    "objectifs de cours à 12 mois publiés par les analystes financiers "
                    "suivant le titre. À traiter avec la même prudence qu'une prévision : "
                    "les analystes se trompent régulièrement, et leurs objectifs peuvent "
                    "être influencés par des biais (relations avec l'entreprise, "
                    "conformisme de marché)."
                ),
            },
            {
                "nom": "Nombre d'analystes / Recommandation analystes",
                "definition": (
                    "Nombre d'analystes suivant activement le titre, et consensus de "
                    "leurs recommandations (strong_buy / buy / hold / sell / "
                    "strong_sell). Un consensus construit sur peu d'analystes est "
                    "statistiquement moins robuste qu'un consensus sur un grand nombre."
                ),
            },
            {
                "nom": "Détention insiders / institutionnels (%)",
                "definition": (
                    "Part du capital détenue respectivement par les dirigeants/fondateurs "
                    "(insiders) et par les investisseurs institutionnels (fonds, "
                    "assureurs...). Une détention insiders élevée aligne en théorie les "
                    "intérêts des dirigeants sur ceux des actionnaires ; une détention "
                    "institutionnelle élevée signale une confiance des investisseurs "
                    "professionnels, mais peut aussi créer une pression court-termiste "
                    "sur les résultats trimestriels."
                ),
            },
        ],
    },

    {
        "categorie": "Trésorerie & financement",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Flux de trésorerie opérationnel (Operating Cash Flow)",
                "definition": (
                    "Cash réellement généré par l'activité courante de l'entreprise, "
                    "AVANT investissements — souvent jugé plus fiable que le bénéfice "
                    "net comptable, car moins sensible aux choix comptables "
                    "(amortissements, provisions) qui n'impliquent aucun mouvement de "
                    "cash réel."
                ),
            },
            {
                "nom": "Flux de trésorerie disponible (Free Cash Flow)",
                "definition": (
                    "Cash qu'il reste à l'entreprise après avoir financé ses "
                    "investissements courants (maintenance et développement de son "
                    "outil de production) — ce qui reste réellement disponible pour "
                    "rembourser de la dette, verser des dividendes, ou racheter des "
                    "actions. Un des indicateurs les plus regardés en valorisation "
                    "(base des modèles DCF), car moins manipulable comptablement que "
                    "le bénéfice net."
                ),
                "formule": r"FCF = \text{Flux de trésorerie opérationnel} - \text{Dépenses d'investissement (CAPEX)}",
            },
        ],
    },

    {
        "categorie": "Trésorerie & financement (théorique — non calculé par MAF)",
        "partie": "II — Théorie complémentaire (non calculée par MAF)",
        "termes": [
            {
                "nom": "BFR — Besoin en Fonds de Roulement",
                "definition": (
                    "Montant que l'entreprise doit financer en permanence pour couvrir "
                    "le décalage entre ses décaissements (achats, salaires) et ses "
                    "encaissements (ventes) liés à son cycle d'exploitation."
                ),
                "formule": r"BFR = \text{Stocks} + \text{Créances clients} - \text{Dettes fournisseurs}",
                "explication": (
                    "Un BFR croissant plus vite que le chiffre d'affaires est un signal "
                    "d'alerte classique : l'entreprise vend plus, mais immobilise "
                    "proportionnellement plus de trésorerie dans son cycle d'exploitation "
                    "(stocks qui s'accumulent, clients qui paient plus lentement) — un "
                    "phénomène qui peut précéder des tensions de liquidité même quand la "
                    "rentabilité comptable reste bonne."
                ),
            },
            {
                "nom": "WACC — Coût moyen pondéré du capital",
                "definition": (
                    "Taux de rendement minimum qu'une entreprise doit générer sur ses "
                    "investissements pour satisfaire à la fois ses actionnaires et ses "
                    "créanciers — sert de taux d'actualisation dans les modèles de "
                    "valorisation (DCF)."
                ),
                "formule": r"WACC = \frac{E}{E+D} \times r_e \;+\; \frac{D}{E+D} \times r_d \times (1 - T)",
                "note": r"$E$ = capitaux propres, $D$ = dette, $r_e$ = coût des fonds propres, $r_d$ = coût de la dette, $T$ = taux d'imposition.",
                "explication": (
                    "Le terme $(1-T)$ existe parce que les intérêts de la dette sont "
                    "fiscalement déductibles (le 'bouclier fiscal') alors que les "
                    "dividendes versés aux actionnaires ne le sont pas — ce qui rend la "
                    "dette structurellement moins chère que les capitaux propres, "
                    "jusqu'à un certain niveau où le risque de surendettement fait "
                    "remonter $r_e$ et $r_d$ eux-mêmes."
                ),
            },
            {
                "nom": "EV/EBITDA",
                "definition": (
                    "Multiple de valorisation comparant la valeur d'entreprise totale "
                    "(dette incluse) à sa rentabilité opérationnelle brute — souvent "
                    "préféré au P/E pour comparer des entreprises à structures "
                    "d'endettement différentes, car l'EBITDA est calculé avant charges "
                    "financières."
                ),
                "formule": r"EV/EBITDA = \frac{EV}{EBITDA}, \quad EV = \text{Capitalisation} + \text{Dette nette}",
                "note": "EBITDA = Bénéfice avant intérêts, impôts, dépréciation et amortissement — approxime le cash-flow opérationnel brut.",
            },
        ],
    },

    {
        "categorie": "Analyse technique",
        "partie": "III — Comment fonctionnent les agents et modèles de MAF",
        "termes": [
            {
                "nom": "Momentum boursier",
                "definition": (
                    "⚠️ Indicateur DESCRIPTIF de tendance récente d'un cours de bourse "
                    "(agent_technique), jamais un modèle statistique validé."
                ),
                "formule": r"\text{Momentum} \approx \frac{P_t - P_{t-1}}{P_{t-1}}",
                "note": "Où $P_t$ est le cours à l'instant $t$ — variation relative sur une période récente.",
                "explication": (
                    "Le momentum correspond à la dérivée première du cours (sa vitesse "
                    "de variation) ; son accélération correspond à la dérivée seconde "
                    "(la tendance se renforce-t-elle ou s'essouffle-t-elle ?). Un momentum "
                    "positif mais en décélération peut précéder un retournement, ce que "
                    "le momentum seul (sans sa dérivée) ne révèle pas."
                ),
            },
        ],
    },

    {
        "categorie": "Modèles & statistiques (comprendre l'onglet « État des modèles »)",
        "partie": "III — Comment fonctionnent les agents et modèles de MAF",
        "termes": [
            {
                "nom": "R² — Coefficient de détermination",
                "definition": (
                    "⚠️ Statistique de qualité d'un modèle ML, PAS un indicateur "
                    "financier — c'est le chiffre affiché dans « État des modèles » "
                    "(sidebar). Mesure la part de variance de l'indicateur cible que le "
                    "modèle parvient à expliquer."
                ),
                "formule": r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}",
                "note": r"$y_i$ = valeur réelle, $\hat{y}_i$ = valeur prédite, $\bar{y}$ = moyenne des valeurs réelles.",
                "explication": (
                    r"$R^2 = 1$ : prédiction parfaite. $R^2 = 0$ : le modèle ne fait pas "
                    r"mieux que prédire la moyenne pour toutes les entreprises. $R^2$ "
                    "négatif (possible sur un jeu de test, jamais sur l'entraînement) : "
                    "le modèle fait PIRE que la moyenne — signe que les features "
                    "choisies n'ont pas de lien réel avec la cible sur cet échantillon. "
                    r"Avec $n \approx 180$ entreprises et seulement 3-4 variables "
                    "explicatives, des R² entre 0,3 et 0,5 (observés dans MAF) sont "
                    "attendus et acceptables : la rentabilité d'une entreprise dépend en "
                    "réalité de bien plus de facteurs (qualité du management, position "
                    "concurrentielle, cycle économique...) que ce qu'un modèle aussi "
                    "simple peut capturer."
                ),
            },
            {
                "nom": "MAE — Erreur absolue moyenne",
                "definition": (
                    "Autre statistique de qualité d'un modèle ML, dans la même unité que "
                    "l'indicateur prédit (points de pourcentage ici) — plus intuitive que "
                    "le R² pour juger 'de combien le modèle se trompe en moyenne'."
                ),
                "formule": r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|",
                "note": "Une MAE de 9,8 sur la marge nette signifie une erreur moyenne d'environ 9,8 points de pourcentage entre prédiction et réalité sur l'échantillon de test.",
            },
        ],
    },

    {
        "categorie": "Réglementation bancaire",
        "partie": "II — Théorie complémentaire (non calculée par MAF)",
        "termes": [
            {
                "nom": "CET1 — Common Equity Tier 1",
                "definition": (
                    "⚠️ Spécifique au secteur bancaire — MAF ne le calcule pas (absent "
                    "des données collectées), mais c'est un indicateur clé si tu analyses "
                    "des rapports de banques (BPCE, BNP Paribas, Société Générale...)."
                ),
                "formule": r"\text{CET1 ratio} = \frac{\text{Fonds propres Common Equity Tier 1}}{\text{Actifs pondérés du risque (RWA)}} \times 100",
                "explication": (
                    "Exigé par la réglementation prudentielle Bâle III, avec un minimum "
                    "réglementaire de 4,5% (souvent porté à 7-10,5% avec les coussins de "
                    "conservation et contracycliques). Les grandes banques visent "
                    "généralement 12-15% pour rassurer marché et régulateur. Ce ratio "
                    "explique pourquoi le ROE et le ROA classiques sont moins pertinents "
                    "pour juger une banque que pour une entreprise industrielle : le "
                    "'capital' réglementaire d'une banque n'est pas comparable à des "
                    "capitaux propres ordinaires."
                ),
            },
        ],
    },

    {
        "categorie": "Colonne technique",
        "partie": "I — Données financières de la base",
        "termes": [
            {
                "nom": "Données brutes (JSON)",
                "definition": (
                    "⚠️ Pas un indicateur financier — une colonne de repli technique. "
                    "yfinance renvoie 80-100+ champs par entreprise ; DataCollecting.py "
                    "n'en extrait individuellement qu'une cinquantaine (les plus "
                    "pertinents pour l'analyse financière, tous couverts par ce "
                    "glossaire). Cette colonne conserve L'INTÉGRALITÉ des données "
                    "brutes renvoyées par yfinance, sous forme de texte JSON — y "
                    "compris les champs non repris individuellement ailleurs. À parser "
                    "avec json.loads(...) si un besoin ponctuel porte sur un champ "
                    "absent des colonnes dédiées."
                ),
            },
        ],
    },

    {
        "categorie": "Random Forest — principe général",
        "partie": "III — Comment fonctionnent les agents et modèles de MAF",
        "termes": [
            {
                "nom": "Arbre de décision (Decision Tree)",
                "definition": (
                    "Brique de base du Random Forest. Un arbre de décision découpe "
                    "successivement les données en sous-groupes de plus en plus "
                    "homogènes, en posant des questions du type 'Marge brute > 45% ?' "
                    "à chaque nœud. La prédiction finale d'une feuille est la moyenne "
                    "des valeurs cibles des exemples d'entraînement tombés dans cette feuille."
                ),
                "explication": (
                    "Un seul arbre de décision, utilisé seul, surapprend facilement "
                    "(il colle trop précisément aux données d'entraînement et généralise "
                    "mal) — c'est exactement le problème que le Random Forest résout en "
                    "combinant beaucoup d'arbres légèrement différents plutôt qu'un seul "
                    "arbre très profond."
                ),
            },
            {
                "nom": "Bootstrap Aggregating (Bagging)",
                "definition": (
                    "Principe central du Random Forest : au lieu d'entraîner UN arbre "
                    "sur TOUTES les données, on entraîne PLUSIEURS arbres, chacun sur un "
                    "échantillon tiré aléatoirement AVEC REMISE (bootstrap) des données "
                    "d'entraînement — chaque arbre voit donc une version légèrement "
                    "différente du jeu de données."
                ),
                "formule": r"D_b = \{(x_i, y_i) \text{ tirés avec remise depuis } D_{\text{train}}\}, \quad b = 1, \ldots, B",
                "note": r"$D_b$ = échantillon bootstrap pour l'arbre $b$, $B$ = nombre total d'arbres (100 dans MAF, voir n_estimators).",
                "explication": (
                    "Avec le tirage avec remise, chaque échantillon bootstrap contient "
                    "en moyenne environ 63% des observations originales (certaines "
                    "répétées, d'autres absentes) — cette diversité entre arbres est ce "
                    "qui permet, une fois les prédictions moyennées, de réduire la "
                    "variance globale du modèle sans trop augmenter son biais."
                ),
            },
            {
                "nom": "Agrégation des prédictions",
                "definition": (
                    "Une fois les $B$ arbres entraînés (chacun sur son propre "
                    "échantillon bootstrap), la prédiction finale du Random Forest pour "
                    "une nouvelle observation est simplement la MOYENNE des prédictions "
                    "de tous les arbres — c'est cette moyenne qui lisse les erreurs "
                    "individuelles de chaque arbre et stabilise la prédiction globale."
                ),
                "formule": r"\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)",
                "note": r"$T_b(x)$ = prédiction de l'arbre $b$ pour l'observation $x$, $B = 100$ dans MAF (RandomForestRegressor(n_estimators=100)).",
            },
            {
                "nom": "Critère de division des nœuds (impureté)",
                "definition": (
                    "À chaque nœud d'un arbre, l'algorithme choisit la feature ET le "
                    "seuil qui, une fois utilisés pour séparer les données en deux "
                    "groupes, réduisent le plus possible la variance de la cible dans "
                    "chaque groupe résultant — un split est 'bon' s'il rend les deux "
                    "sous-groupes plus homogènes qu'avant la séparation."
                ),
                "formule": r"\text{MSE}(\text{nœud}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2",
                "note": "Le split retenu est celui qui minimise la somme pondérée des MSE des deux nœuds enfants — c'est un choix glouton (optimal localement à chaque étape), pas une optimisation globale de l'arbre entier.",
            },
            {
                "nom": "Importance des variables (Feature Importance)",
                "definition": (
                    "Mesure, pour chaque feature utilisée par le modèle, sa contribution "
                    "totale à la réduction d'impureté sur l'ensemble des arbres de la "
                    "forêt — c'est ce que renvoie importances() dans PredictionAgent.py, "
                    "affiché sous forme de barres de progression après chaque prédiction "
                    "dans l'interface."
                ),
                "formule": r"\text{Importance}(x_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in \text{nœuds de } T_b \text{ divisant sur } x_j} \frac{n_t}{n} \Delta\text{MSE}(t)",
                "explication": (
                    "⚠️ Piège classique d'interprétation : l'importance d'une feature "
                    "mesure SA contribution AU MODÈLE ENTRAÎNÉ, pas nécessairement sa "
                    "causalité économique réelle — deux features corrélées entre elles "
                    "(ex. chiffre d'affaires et bénéfice net) peuvent se 'partager' "
                    "artificiellement l'importance, chacune paraissant moins "
                    "déterminante qu'elle ne l'est isolément."
                ),
            },
            {
                "nom": "Découpage entraînement / test (Train/Test Split)",
                "definition": (
                    "Avant l'entraînement, les données sont divisées en deux groupes "
                    "disjoints : 80% pour ENTRAÎNER le modèle (il voit ces exemples et "
                    "leur cible réelle), 20% pour le TESTER ensuite (le modèle prédit "
                    "sans jamais avoir vu ces exemples). R² et MAE affichés dans « État "
                    "des modèles » sont calculés UNIQUEMENT sur ce jeu de test — sinon, "
                    "un modèle pourrait sembler excellent simplement en 'récitant' des "
                    "données déjà vues, sans réelle capacité de généralisation."
                ),
                "note": "test_size=0.2, random_state=42 dans PredictionAgent.py — le random_state fixe la graine aléatoire du découpage pour que l'entraînement soit reproductible d'une exécution à l'autre.",
            },
            {
                "nom": "Pourquoi un Random Forest plutôt qu'une régression linéaire ?",
                "definition": (
                    "Choix de conception assumé dans MAF, avec trois justifications "
                    "principales : (1) capture des relations NON-LINÉAIRES entre "
                    "features et cible, sans avoir à les spécifier à l'avance (une "
                    "régression linéaire suppose une relation additive et proportionnelle) ; "
                    "(2) robustesse relative aux valeurs extrêmes/aberrantes dans les "
                    "features, chaque arbre n'étant qu'un vote parmi 100 ; (3) aucune "
                    "exigence de mise à l'échelle des features (contrairement à beaucoup "
                    "d'autres modèles), pratique quand les features ont des unités très "
                    "différentes (un chiffre d'affaires en milliards, une marge en "
                    "pourcentage)."
                ),
                "note": "Contrepartie : un Random Forest reste une 'boîte moins transparente' qu'une régression linéaire (pas de coefficient unique et interprétable par feature) — feature_importances_ compense partiellement, mais reste une mesure globale, pas un coefficient causal par variable.",
            },
            {
                "nom": "Limite fondamentale : extrapolation hors du domaine d'entraînement",
                "definition": (
                    "Un Random Forest ne peut JAMAIS prédire une valeur en dehors de la "
                    "plage des valeurs cibles vues à l'entraînement — chaque feuille "
                    "d'arbre renvoie une moyenne d'exemples réels, donc la prédiction "
                    "finale est toujours une moyenne pondérée de valeurs déjà observées. "
                    "C'est pourquoi simuler_scenarios() (voir PredictionAgent.py) "
                    "détecte et signale les résultats hors du domaine plausible (>100% "
                    "en valeur absolue) : un tel résultat révèle un scénario dont la "
                    "COMBINAISON de features n'a jamais été observée ensemble dans le "
                    "panel d'entraînement, même si chaque feature individuellement "
                    "reste dans une plage normale."
                ),
                "explication": (
                    "Avec n≈180 entreprises et seulement 3-4 features par modèle, le "
                    "'volume' de l'espace des combinaisons plausibles réellement couvert "
                    "par l'entraînement est limité — un scénario hypothétique un peu "
                    "inhabituel (mais individuellement raisonnable sur chaque variable) "
                    "peut très bien se retrouver hors de ce domaine couvert, sans que "
                    "rien dans les paramètres d'entrée ne le signale a priori."
                ),
            },
        ],
    },
]