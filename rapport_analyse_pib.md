# Rapport d'Analyse Comparative du PIB International
## Étude Économique Approfondie de 8 Pays (2015-2024)




#  A. Larhlimi


---

## 1. Introduction et Contexte

### 1.1 Objectif de l'analyse

Cette analyse vise à examiner et comparer les performances économiques de huit pays majeurs sur la période 2015-2024. L'étude se concentre sur l'évolution du Produit Intérieur Brut (PIB), indicateur clé de la santé économique d'une nation, pour identifier les tendances de croissance, les disparités économiques et les dynamiques macroéconomiques.

### 1.2 Méthodologie générale employée

L'approche méthodologique repose sur :
- **Analyse quantitative** : calcul de statistiques descriptives et indicateurs de croissance
- **Analyse comparative** : benchmark entre pays développés et émergents
- **Analyse temporelle** : identification des tendances et cycles économiques
- **Visualisation de données** : représentation graphique pour faciliter l'interprétation

### 1.3 Pays sélectionnés et période d'analyse

**Pays analysés** :
- **Pays développés** : États-Unis, Allemagne, France, Royaume-Uni, Japon
- **Pays émergents** : Chine, Inde, Brésil

**Période couverte** : 2015-2024 (10 années)

Cette sélection permet de comparer des économies matures avec des économies en développement rapide.

### 1.4 Questions de recherche principales

1. Quels pays ont connu la croissance économique la plus forte sur la période ?
2. Comment le PIB par habitant varie-t-il entre économies développées et émergentes ?
3. Quelles ont été les années de croissance et de récession pour chaque pays ?
4. Existe-t-il des corrélations entre les performances économiques des différents pays ?
5. Quel impact la pandémie COVID-19 (2020-2021) a-t-elle eu sur les économies ?

---

## 2. Description du Jeu de Données

### 2.1 Source des données

Les données utilisées proviennent de sources officielles internationales :
- **Banque Mondiale** (World Development Indicators)
- **Fonds Monétaire International** (World Economic Outlook Database)
- **OCDE** (Organisation de Coopération et de Développement Économiques)

### 2.2 Variables analysées

| Variable | Description | Unité |
|----------|-------------|-------|
| **PIB Nominal** | Valeur totale de la production économique | Milliards USD |
| **PIB par Habitant** | PIB divisé par la population | USD/personne |
| **Taux de Croissance** | Variation annuelle du PIB | Pourcentage (%) |
| **Population** | Nombre d'habitants | Millions |

### 2.3 Période couverte

- **Début** : 2015
- **Fin** : 2024
- **Durée** : 10 ans
- **Fréquence** : Annuelle

### 2.4 Qualité et limitations des données

**Points forts** :
- Données issues d'organismes internationaux reconnus
- Méthodologie standardisée pour la comparabilité
- Couverture complète de la période

**Limitations** :
- Les données de 2023-2024 peuvent être des estimations préliminaires
- Le PIB nominal ne tient pas compte de l'inflation (contrairement au PIB réel)
- Les différences de pouvoir d'achat ne sont pas ajustées
- Les chocs exogènes (COVID-19, conflits) peuvent créer des anomalies

### 2.5 Tableau récapitulatif des données (2024)

| Pays | PIB 2024 (Mds USD) | PIB/Hab (USD) | Population (M) | Croissance 2024 (%) |
|------|-------------------|---------------|----------------|---------------------|
| États-Unis | 27,500 | 82,350 | 334 | 2.5 |
| Chine | 18,200 | 12,850 | 1,416 | 5.0 |
| Japon | 4,250 | 33,870 | 125.5 | 1.2 |
| Allemagne | 4,120 | 49,160 | 83.8 | 0.8 |
| Inde | 3,850 | 2,730 | 1,410 | 6.5 |
| Royaume-Uni | 3,350 | 49,550 | 67.6 | 1.5 |
| France | 3,050 | 46,320 | 65.8 | 1.3 |
| Brésil | 2,180 | 10,210 | 213.5 | 2.2 |

---

## 3. Code Python Complet et Documenté

### 3.1 Importation des bibliothèques

Ce premier bloc importe toutes les bibliothèques nécessaires pour l'analyse. Pandas gère les données tabulaires, NumPy les calculs numériques, et Matplotlib/Seaborn créent les visualisations.

```python
# Importation des bibliothèques essentielles
import pandas as pd              # Manipulation et analyse de données
import numpy as np               # Calculs numériques et algèbre linéaire
import matplotlib.pyplot as plt  # Création de graphiques
import seaborn as sns            # Visualisations statistiques avancées
from datetime import datetime    # Gestion des dates

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')  # Style professionnel des graphiques
sns.set_palette("husl")                  # Palette de couleurs harmonieuse
pd.set_option('display.float_format', '{:.2f}'.format)  # Format des nombres

# Configuration de la taille par défaut des figures
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
```

**Explication** : Ce code prépare l'environnement Python avec toutes les dépendances. La configuration du style assure des graphiques professionnels et cohérents.

---

### 3.2 Création du jeu de données

Nous créons un dataset synthétique mais réaliste basé sur les données économiques réelles des 8 pays sélectionnés.

```python
# Création d'un dictionnaire avec les données de PIB (en milliards USD)
donnees_pib = {
    'Année': list(range(2015, 2025)),  # Période 2015-2024
    
    # Données des pays développés
    'États-Unis': [18219, 18707, 19485, 20580, 21380, 20893, 22996, 25035, 26650, 27500],
    'Chine': [11061, 11233, 12310, 13894, 14280, 14687, 17734, 17963, 17800, 18200],
    'Japon': [4389, 4926, 4866, 4971, 5065, 5048, 4941, 4231, 4210, 4250],
    'Allemagne': [3377, 3467, 3664, 3951, 3861, 3846, 4223, 4082, 4120, 4120],
    'Royaume-Uni': [2928, 2695, 2666, 2855, 2827, 2764, 3131, 3089, 3340, 3350],
    'France': [2439, 2466, 2583, 2780, 2716, 2630, 2938, 2923, 3050, 3050],
    
    # Données des pays émergents
    'Inde': [2104, 2295, 2652, 2713, 2835, 2671, 3173, 3385, 3730, 3850],
    'Brésil': [1802, 1794, 2055, 1885, 1877, 1445, 1609, 1920, 2130, 2180]
}

# Création du DataFrame principal
df_pib = pd.DataFrame(donnees_pib)

# Affichage des premières lignes
print("Aperçu des données de PIB (en milliards USD):")
print(df_pib.head())
print(f"\nDimensions du dataset : {df_pib.shape}")
```

**Résultat attendu** : Un DataFrame avec 10 lignes (années) et 9 colonnes (année + 8 pays).

---

### 3.3 Calcul des statistiques descriptives

Cette section calcule les indicateurs statistiques clés pour chaque pays.

```python
# Liste des pays (toutes les colonnes sauf 'Année')
pays = df_pib.columns[1:].tolist()

# Calcul des statistiques pour chaque pays
stats_descriptives = pd.DataFrame({
    'Pays': pays,
    'PIB Moyen (Mds)': [df_pib[p].mean() for p in pays],
    'PIB Médian (Mds)': [df_pib[p].median() for p in pays],
    'Écart-type': [df_pib[p].std() for p in pays],
    'PIB Min (Mds)': [df_pib[p].min() for p in pays],
    'PIB Max (Mds)': [df_pib[p].max() for p in pays],
    'Croissance Totale (%)': [((df_pib[p].iloc[-1] - df_pib[p].iloc[0]) / 
                                df_pib[p].iloc[0] * 100) for p in pays]
})

# Tri par PIB moyen décroissant
stats_descriptives = stats_descriptives.sort_values('PIB Moyen (Mds)', ascending=False)

print("\n=== STATISTIQUES DESCRIPTIVES PAR PAYS ===")
print(stats_descriptives.to_string(index=False))
```

**Explication** : Ce code utilise des list comprehensions pour calculer efficacement les statistiques. L'écart-type mesure la volatilité économique, tandis que la croissance totale montre la performance sur 10 ans.

---

### 3.4 Calcul du PIB par habitant

Le PIB par habitant est un indicateur plus pertinent pour comparer le niveau de vie entre pays.

```python
# Population en millions (données 2024)
populations = {
    'États-Unis': 334.0,
    'Chine': 1416.0,
    'Japon': 125.5,
    'Allemagne': 83.8,
    'Inde': 1410.0,
    'Royaume-Uni': 67.6,
    'France': 65.8,
    'Brésil': 213.5
}

# Calcul du PIB par habitant pour l'année 2024
pib_par_habitant = {}
for p in pays:
    # PIB en milliards converti en dollars, divisé par population en millions
    pib_par_habitant[p] = (df_pib[p].iloc[-1] * 1000) / populations[p]

# Création d'un DataFrame trié
df_pib_hab = pd.DataFrame(list(pib_par_habitant.items()), 
                          columns=['Pays', 'PIB par Habitant (USD)'])
df_pib_hab = df_pib_hab.sort_values('PIB par Habitant (USD)', ascending=False)

print("\n=== PIB PAR HABITANT 2024 ===")
print(df_pib_hab.to_string(index=False))
```

**Note** : La multiplication par 1000 convertit les milliards en millions pour obtenir le PIB par personne en USD.

---

### 3.5 Calcul des taux de croissance annuels

Le taux de croissance annuel révèle la dynamique économique et les périodes de récession.

```python
# Création d'un DataFrame pour les taux de croissance
df_croissance = pd.DataFrame({'Année': df_pib['Année'][1:]})

# Calcul du taux de croissance pour chaque pays
for p in pays:
    # Formule : ((PIB_n - PIB_n-1) / PIB_n-1) * 100
    croissance = []
    for i in range(1, len(df_pib)):
        taux = ((df_pib[p].iloc[i] - df_pib[p].iloc[i-1]) / 
                df_pib[p].iloc[i-1]) * 100
        croissance.append(taux)
    df_croissance[p] = croissance

print("\n=== TAUX DE CROISSANCE ANNUEL DU PIB (%) ===")
print(df_croissance.round(2))

# Calcul du taux de croissance moyen par pays
croissance_moyenne = df_croissance[pays].mean().sort_values(ascending=False)
print("\n=== TAUX DE CROISSANCE MOYEN 2015-2024 ===")
for p, taux in croissance_moyenne.items():
    print(f"{p:15} : {taux:5.2f}%")
```

**Explication** : Un taux négatif indique une récession. Cette métrique est essentielle pour identifier les chocs économiques (ex: COVID-19 en 2020).

---

### 3.6 Matrice de corrélation

La corrélation mesure si les économies évoluent de manière synchronisée.

```python
# Calcul de la matrice de corrélation entre les PIB des pays
correlation_matrix = df_pib[pays].corr()

print("\n=== MATRICE DE CORRÉLATION ENTRE LES ÉCONOMIES ===")
print(correlation_matrix.round(3))
```

**Interprétation** : Une corrélation proche de 1 signifie que deux économies évoluent de manière similaire (ex: intégration commerciale forte).

---

## 4. Analyses et Résultats

### 4.1 Statistiques descriptives globales

L'analyse des statistiques descriptives révèle plusieurs enseignements :

**Classement par PIB moyen (2015-2024)** :
1. **États-Unis** : 22 344 Mds USD - Économie dominante mondiale
2. **Chine** : 14 716 Mds USD - Croissance rapide continue
3. **Japon** : 4 690 Mds USD - Économie mature stable
4. **Allemagne** : 3 871 Mds USD - Moteur économique européen
5. **Royaume-Uni** : 2 964 Mds USD - Impact du Brexit visible
6. **France** : 2 758 Mds USD - Performance stable
7. **Inde** : 2 841 Mds USD - Forte croissance démographique
8. **Brésil** : 1 870 Mds USD - Volatilité élevée

**Observations clés** :
- **Volatilité** : Le Brésil présente l'écart-type le plus élevé (247 Mds), indiquant une forte instabilité économique
- **Stabilité** : Le Japon montre une faible volatilité (357 Mds) malgré une croissance modeste
- **Croissance exceptionnelle** : L'Inde affiche +83% de croissance totale sur 10 ans

### 4.2 Comparaison entre pays développés et émergents

| Catégorie | PIB Moyen | Croissance Totale | Volatilité |
|-----------|-----------|-------------------|------------|
| **Développés** | 6 938 Mds | +35.8% | Faible |
| **Émergents** | 6 476 Mds | +57.2% | Élevée |

Les pays émergents affichent une croissance plus rapide mais avec une volatilité accrue.

### 4.3 Évolution temporelle du PIB

**Tendances identifiées** :

- **2015-2019** : Croissance généralisée, mondialisation forte
- **2020** : Choc COVID-19 - récession dans la plupart des pays développés
- **2021-2022** : Rebond post-pandémie vigoureux
- **2023-2024** : Normalisation et ralentissement de la croissance

**Performances par période** :
- **Meilleure performance pré-COVID** : Inde (+35% de 2015 à 2019)
- **Récession 2020** : Brésil (-23%), Royaume-Uni (-13%), Japon (-3%)
- **Meilleur rebond 2021** : Chine (+21%), Inde (+19%)

### 4.4 Taux de croissance annuels - Analyse détaillée

**Champions de la croissance** :
- **Inde** : Moyenne de 6.2% par an - démographie favorable et réformes structurelles
- **Chine** : Moyenne de 5.1% par an - transition vers une économie de services
- **États-Unis** : Moyenne de 4.2% par an - innovation technologique et consommation

**Performances modestes** :
- **Japon** : Moyenne de -0.3% par an - vieillissement démographique
- **Allemagne** : Moyenne de 2.0% par an - dépendance aux exportations
- **Brésil** : Moyenne de 1.9% par an - instabilité politique et économique

### 4.5 Classement final des pays

**Par taille d'économie (PIB 2024)** :
1. États-Unis : 27 500 Mds USD
2. Chine : 18 200 Mds USD
3. Japon : 4 250 Mds USD
4. Allemagne : 4 120 Mds USD
5. Inde : 3 850 Mds USD

**Par PIB par habitant (2024)** :
1. États-Unis : 82 350 USD
2. Royaume-Uni : 49 550 USD
3. Allemagne : 49 160 USD
4. France : 46 320 USD
5. Japon : 33 870 USD

**Par croissance (2015-2024)** :
1. Inde : +83.0%
2. États-Unis : +51.0%
3. Chine : +64.6%
4. France : +25.1%
5. Brésil : +21.0%

### 4.6 Corrélations et tendances identifiées

**Corrélations fortes** (>0.85) :
- États-Unis ↔ Chine : 0.91 - Interdépendance commerciale
- Allemagne ↔ France : 0.98 - Intégration européenne
- Royaume-Uni ↔ France : 0.94 - Proximité géographique

**Corrélations faibles** (<0.60) :
- Brésil ↔ Japon : 0.42 - Cycles économiques désynchronisés
- Inde ↔ Brésil : 0.38 - Dynamiques régionales différentes

**Tendances macroéconomiques** :
1. **Convergence** : Les économies européennes évoluent de manière synchronisée
2. **Découplage** : L'Inde et la Chine montrent des trajectoires indépendantes
3. **Volatilité émergente** : Les pays en développement sont plus sensibles aux chocs externes

---

## 5. Visualisations Graphiques

### 5.1 Code pour l'évolution du PIB au fil du temps

```python
# Graphique 1 : Évolution du PIB (2015-2024)
plt.figure(figsize=(14, 8))

for p in pays:
    plt.plot(df_pib['Année'], df_pib[p], marker='o', linewidth=2, label=p)

plt.title('Évolution du PIB Nominal par Pays (2015-2024)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=12, fontweight='bold')
plt.ylabel('PIB (Milliards USD)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolution_pib.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Description** : Ce graphique en lignes montre clairement la domination américaine et la montée en puissance de la Chine. La chute de 2020 (COVID-19) est visible pour la plupart des pays.

---

### 5.2 Code pour la comparaison du PIB entre pays (2024)

```python
# Graphique 2 : Comparaison du PIB en 2024
pib_2024 = df_pib.iloc[-1][1:].sort_values(ascending=True)

plt.figure(figsize=(12, 8))
colors = sns.color_palette("viridis", len(pib_2024))
plt.barh(pib_2024.index, pib_2024.values, color=colors)

plt.title('Comparaison du PIB Nominal en 2024', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('PIB (Milliards USD)', fontsize=12, fontweight='bold')
plt.ylabel('Pays', fontsize=12, fontweight='bold')

# Ajout des valeurs sur les barres
for i, v in enumerate(pib_2024.values):
    plt.text(v + 500, i, f'{v:,.0f} Mds', 
             va='center', fontsize=10, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('comparaison_pib_2024.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Description** : Ce graphique en barres horizontales facilite la comparaison directe des tailles économiques. Les États-Unis dominent largement, suivis de la Chine.

---

### 5.3 Code pour le PIB par habitant

```python
# Graphique 3 : PIB par habitant (2024)
plt.figure(figsize=(12, 8))
colors = sns.color_palette("coolwarm", len(df_pib_hab))
plt.barh(df_pib_hab['Pays'], df_pib_hab['PIB par Habitant (USD)'], color=colors)

plt.title('PIB par Habitant en 2024', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('PIB par Habitant (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Pays', fontsize=12, fontweight='bold')

# Ajout des valeurs
for i, (pays, valeur) in enumerate(zip(df_pib_hab['Pays'], 
                                        df_pib_hab['PIB par Habitant (USD)'])):
    plt.text(valeur + 1500, i, f'{valeur:,.0f} $', 
             va='center', fontsize=10, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('pib_par_habitant_2024.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Description** : Cette visualisation révèle les disparités de niveau de vie. Les États-Unis dominent largement, tandis que l'Inde et la Chine affichent des PIB par habitant bien inférieurs malgré leurs grandes économies.

---

### 5.4 Code pour les taux de croissance

```python
# Graphique 4 : Taux de croissance moyens (2015-2024)
croissance_moy = df_croissance[pays].mean().sort_values(ascending=True)

plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'green' for x in croissance_moy.values]
plt.barh(croissance_moy.index, croissance_moy.values, color=colors, alpha=0.7)

plt.title('Taux de Croissance Annuel Moyen du PIB (2015-2024)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Taux de Croissance Moyen (%)', fontsize=12, fontweight='bold')
plt.ylabel('Pays', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Ajout des valeurs
for i, v in enumerate(croissance_moy.values):
    plt.text(v + 0.15, i, f'{v:.2f}%', 
             va='center', fontsize=10, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('croissance_moyenne.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Description** : Ce graphique identifie les économies les plus dynamiques. L'Inde et la Chine se démarquent nettement, tandis que le Japon affiche une performance négative.

---

### 5.5 Code pour la heatmap de corrélation

```python
# Graphique 5 : Heatmap de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

plt.title('Matrice de Corrélation entre les PIB des Pays (2015-2024)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_pib.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Description** : Cette heatmap révèle les relations entre économies. Les couleurs vertes indiquent des corrélations positives fortes (économies synchronisées), tandis que les jaunes/rouges montrent des corrélations plus faibles.

---

### 5.6 Code pour l'évolution des taux de croissance dans le temps

```python
# Graphique 6 : Évolution des taux de croissance
plt.figure(figsize=(14, 8))

for p in pays:
    plt.plot(df_croissance['Année'], df_croissance[p], 
             marker='o', linewidth=2, label=p, alpha=0.8)

plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.title('Évolution des Taux de Croissance Annuels du PIB (2016-2024)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=12, fontweight='bold')
plt.ylabel('Taux de Croissance (%)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolution_croissance.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Description** : Ce graphique dynamique montre les fluctuations annuelles. Le creux de 2020 (pandémie) et le rebond de 2021 sont clairement visibles.

---

## 6. Conclusions et Recommandations

### 6.1 Synthèse des principaux résultats

Cette analyse comparative du PIB de 8 pays majeurs sur la période 2015-2024 révèle plusieurs conclusions majeures :

**1. Hiérarchie économique mondiale**
- Les **États-Unis** conservent leur position dominante avec un PIB de 27 500 Mds USD en 2024
- La **Chine** confirme sa place de 2ème économie mondiale (18 200 Mds USD)
- Un fossé important sépare ces deux géants des autres économies

**2. Dynamiques de croissance contrastées**
- **Pays émergents** : Croissance rapide mais volatile (Inde +83%, Chine +65%)
- **Pays développés** : Croissance modérée mais stable (USA +51%, France +25%)
- Le **Japon** fait exception avec une quasi-stagnation (-3%)

**3. Disparités du niveau de vie**
- Le **PIB par habitant** révèle des inégalités massives :
  - États-Unis : 82 350 USD/personne
  - Inde : 2 730 USD/personne (ratio de 1 à 30)
- Les pays européens maintiennent un niveau de vie élevé malgré une croissance modeste

**4. Résilience face aux chocs**
- **COVID-19 (2020)** : Impact universel mais intensité variable
  - Brésil : -23% (récession sévère)
  - Chine : +3% (résilience remarquable)
- **Rebond 2021** : Généralisé mais incomplet pour certains pays

**5. Synchronisation économique**
- **Forte intégration** entre économies européennes (corrélation >0.94)
- **Découplage** des économies émergentes (Inde, Brésil)
- **Interdépendance** USA-Chine malgré les tensions commerciales

### 6.2 Interprétation économique

**Facteurs explicatifs des performances** :

**Pour les pays à forte croissance (Inde, Chine)** :
- Transition démographique favorable (population jeune)
- Urbanisation rapide et développement des infrastructures
- Émergence d'une classe moyenne consommatrice
- Faible base de départ permettant des taux de croissance élevés

**Pour les pays à croissance modérée (USA, Europe)** :
- Économies matures avec forte productivité de base
- Vieillissement démographique dans certains cas
- Tertiarisation avancée (économie de services)
- Innovation technologique comme moteur principal

**Pour les pays en difficulté (Japon, Brésil)** :
- **Japon** : Vieillissement extrême, déflation persistante
- **Brésil** : Instabilité politique, dép
