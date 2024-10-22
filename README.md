# Projet : Système d'Apprentissage Fédéré Décentralisé avec TensorBoard et Suivi des Performances

Bienvenue dans ce projet de système d'apprentissage fédéré décentralisé. Ce projet implémente un environnement complet permettant d'entraîner des modèles de machine learning de manière décentralisée, en simulant un apprentissage fédéré sur plusieurs clients tout en suivant les performances via TensorBoard. 

Le projet inclut plusieurs composants essentiels pour l'initialisation des modèles, la gestion des métriques, l'entraînement, et l'évaluation dans un cadre décentralisé, le tout avec des outils de visualisation pour suivre les performances.

## Table des matières

- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Exécution du projet](#exécution-du-projet)
- [Structure du projet](#structure-du-projet)
- [Utilisation des composants principaux](#utilisation-des-composants-principaux)
  - [Entraînement et Evaluation](#entraînement-et-évaluation)
  - [Gestion des Clients et Serveurs](#gestion-des-clients-et-serveurs)
  - [Initialisation des Poids](#initialisation-des-poids)
  - [Suivi des Performances avec TensorBoard](#suivi-des-performances-avec-tensorboard)
- [Contributeurs](#contributeurs)

---

## Introduction

Ce projet est un framework d'apprentissage fédéré qui permet de simuler un environnement où plusieurs clients participent à l'entraînement d'un modèle global. Chaque client utilise ses propres données locales pour entraîner un modèle local, et un serveur central (ou un client élu) agrège les poids des modèles locaux pour mettre à jour le modèle global.

L'apprentissage fédéré permet de préserver la confidentialité des données puisque les données restent sur les clients et seules les mises à jour des modèles (et non les données elles-mêmes) sont partagées.

---

## Fonctionnalités

1. **Apprentissage Fédéré Décentralisé** : Simulation d'une architecture d'apprentissage fédéré où les clients sont indépendants et utilisent leurs propres données locales.
2. **Suivi des métriques** : Suivi des performances du modèle à l'aide de métriques personnalisées.
3. **Initialisation des Poids** : Plusieurs techniques d'initialisation des poids (normal, xavier, kaiming, etc.).
4. **TensorBoard** : Suivi en temps réel des performances du modèle avec des graphiques et des statistiques via TensorBoard.
5. **Gestion multi-processus** : Gestion des clients et des processus d'entraînement grâce à des pools de threads pour améliorer les performances.
6. **Split des données** : Divise stratégiquement le dataset pour maintenir un équilibre entre les classes lors de la création des sous-ensembles d'entraînement et de test.

---

## Prérequis

Avant d'exécuter ce projet, assurez-vous d'avoir installé les logiciels et bibliothèques suivants :

- Python 3.7+
- PyTorch
- NumPy
- TQDM
- TensorBoard
- Pandas (facultatif pour certains types de datasets)

---

## Installation

Clonez le dépôt GitHub et installez les dépendances nécessaires.

```bash
git clone https://github.com/nom_du_projet/projet-federated-learning.git
cd projet-federated-learning

# Installation des dépendances
pip install -r requirements.txt
```

Assurez-vous que tous les packages requis, notamment PyTorch et TensorBoard, sont bien installés.

---

## Exécution du projet

L'exécution du projet se fait via un script principal, `main.py`, qui orchestre l'entraînement fédéré et les différentes étapes de communication entre clients et serveur.

### Exemple d'exécution :

```bash
python main.py --config config.yaml
```

Les paramètres d'entraînement, comme le nombre de clients, les hyperparamètres du modèle, et les configurations du serveur, sont spécifiés dans le fichier de configuration (`config.yaml`).

---

## Structure du projet

```bash
├── src
│   ├── main.py                    # Script principal pour démarrer l'apprentissage fédéré
│   ├── client.py                  # Définit la classe Client pour gérer les clients dans l'apprentissage fédéré
│   ├── server.py                  # Gère les opérations du serveur dans un cadre centralisé ou décentralisé
│   ├── metrics.py                 # Implémente les métriques de suivi de performance
│   ├── utils.py                   # Fonctions utilitaires (initialisation des poids, gestion des processus, etc.)
│   ├── basenode.py                # Classe de base pour gérer les nœuds (clients/serveurs)
│   └── split.py                   # Gère le découpage des datasets pour les clients
├── README.md                      # Document d'explication du projet
└── requirements.txt               # Liste des dépendances nécessaires
```

---

## Utilisation des composants principaux

### Entraînement et Évaluation

L'entraînement se fait sur plusieurs clients de manière indépendante. Chaque client utilise un sous-ensemble de données pour entraîner son propre modèle. Le serveur central (ou un client élu dans le cas décentralisé) agrège les mises à jour des modèles locaux.

**Fonction principale :**
```python
def train_round():
    """ Effectue une itération complète de formation en fédéré """
```

### Gestion des Clients et Serveurs

Chaque client possède un modèle local et des données privées. Après chaque cycle d'entraînement local, les clients envoient leurs modèles au serveur, qui les agrège pour mettre à jour le modèle global.

**Exemple de création de clients :**
```python
self.clients = self._create_clients(client_datasets)
```

### Initialisation des Poids

Vous pouvez initialiser les poids des modèles à l'aide de différentes techniques d'initialisation (normal, xavier, kaiming, etc.). Cela permet de garantir une bonne convergence du modèle.

**Fonction d'initialisation des poids :**
```python
def init_weights(model, init_type, init_gain):
    """ Initialise les poids du modèle en fonction du type spécifié """
```

### Suivi des Performances avec TensorBoard

L'exécution de TensorBoard est gérée par le script. Vous pouvez suivre en temps réel les performances du modèle (perte, précision, etc.) via des graphiques.

**Exemple de lancement de TensorBoard :**
```bash
tensorboard --logdir logs/
```

Vous pouvez accéder à l'interface TensorBoard via votre navigateur en utilisant le port spécifié.

---

## Contributeurs

- **Takam Rushclin** - [GitHub](https://github.com/TakamRushclin)

N'hésitez pas à contribuer au projet en ouvrant des pull requests ou en signalant des problèmes dans les issues.

---

Merci d'avoir utilisé ce framework d'apprentissage fédéré décentralisé. Bonne expérimentation !
