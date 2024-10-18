import numpy as np

# Initialisation des nœuds (clients) et de leurs propriétés
class Node:
    def __init__(self, node_id, compute_power, resources, reliability):
        self.node_id = node_id
        self.compute_power = compute_power  # Charge computationnelle C_k
        self.resources = resources  # Ressources disponibles R_k
        self.reliability = reliability  # Fiabilité F_k
        self.local_model = np.random.rand(10)  # Exemple de modèle local (vecteur de poids)
        self.data_size = np.random.randint(50, 200)  # Taille des données locales
    
    # Mise à jour locale du modèle par descente de gradient
    def local_update(self, learning_rate=0.01):
        gradient = np.random.randn(10)  # Simuler un gradient
        self.local_model -= learning_rate * gradient

# Création d'un ensemble de nœuds (clients)
def initialize_nodes(num_nodes):
    nodes = []
    for i in range(num_nodes):
        compute_power = np.random.rand()  # Valeur entre 0 et 1
        resources = np.random.rand()  # Valeur entre 0 et 1
        reliability = np.random.rand()  # Valeur entre 0 et 1
        nodes.append(Node(i, compute_power, resources, reliability))
    return nodes

# Calcul du score de vote pour chaque nœud
def calculate_scores(nodes, alpha=0.5, beta=0.3, gamma=0.2):
    scores = []
    for node in nodes:
        score = alpha * node.compute_power + beta * node.resources + gamma * node.reliability
        scores.append(score)
    return scores

# Sélection du nœud agrégateur en fonction des scores de vote
def select_aggregator(nodes, scores):
    max_score_idx = np.argmax(scores)
    return nodes[max_score_idx]

# Agrégation des modèles locaux pour produire un modèle global
def aggregate_models(nodes):
    total_data_size = sum([node.data_size for node in nodes])
    global_model = np.zeros_like(nodes[0].local_model)
    
    # Moyenne pondérée des modèles en fonction de la taille des données locales
    for node in nodes:
        weight = node.data_size / total_data_size
        global_model += weight * node.local_model
    
    return global_model

# Synchronisation des modèles : diffusion du modèle global à tous les nœuds
def synchronize_models(nodes, global_model):
    for node in nodes:
        node.local_model = global_model

# Simuler une itération d'apprentissage fédéré
def federated_learning_iteration(nodes):
    # Étape 1 : Mise à jour locale pour chaque nœud
    for node in nodes:
        node.local_update()
    
    # Étape 2 : Calcul des scores et vote pour sélectionner l'agrégateur
    scores = calculate_scores(nodes)
    aggregator = select_aggregator(nodes, scores)
    print(f"Nœud agrégateur sélectionné : {aggregator.node_id} avec un score de {max(scores)}")
    
    # Étape 3 : L'agrégateur reçoit les modèles et effectue l'agrégation
    global_model = aggregate_models(nodes)
    
    # Étape 4 : Synchronisation : diffuser le modèle global à tous les nœuds
    synchronize_models(nodes, global_model)
    
    return global_model

# Exécution de plusieurs itérations d'apprentissage fédéré
def run_federated_learning(num_nodes, num_iterations=5):
    nodes = initialize_nodes(num_nodes)
    for iteration in range(num_iterations):
        print(f"\n--- Itération {iteration + 1} ---")
        global_model = federated_learning_iteration(nodes)
        print(f"Modèle global après l'itération {iteration + 1}: {global_model}")

# Lancer l'apprentissage fédéré
run_federated_learning(num_nodes=5, num_iterations=5000)
