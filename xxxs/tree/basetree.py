from abc import ABC, abstractmethod

class BaseTree(ABC):
    """
    Abstract Base Class for a Binary Tree structure in V-Tree Learning.
    Defines the required methods for managing the tree and its operations.
    """

    def __init__(self, nodes):
        """
        Initialize the tree with a list of nodes.
        Each node is expected to have properties like node_id, data_size, model, etc.
        """
        self.nodes = nodes  # List of all nodes in the tree
        self.root = None    # Root node (to be set by the concrete implementation)

    @abstractmethod
    def build_tree(self):
        """
        Build the initial binary tree structure.
        Organizes the nodes into a balanced binary tree.
        """
        pass

    @abstractmethod
    def select_server(self):
        """
        Elect a server node based on specific criteria (e.g., data size, latency).
        Returns the selected node.
        """
        pass

    @abstractmethod
    def reorganize_tree(self, new_root):
        """
        Reorganize the tree structure to make the selected server the root node.
        Adjusts parent-child relationships accordingly.
        """
        pass

    @abstractmethod
    def left_rotate(self, node):
        """
        Perform a left rotation on the given node.
        Adjust parent-child pointers to reflect the rotation.
        """
        pass

    @abstractmethod
    def right_rotate(self, node):
        """
        Perform a right rotation on the given node.
        Adjust parent-child pointers to reflect the rotation.
        """
        pass

    @abstractmethod
    def broadcast_model(self, model):
        """
        Broadcast the global model from the root to all nodes in the tree.
        Ensures each node receives the updated model.
        """
        pass

    @abstractmethod
    def aggregate_models(self, node):
        """
        Perform model aggregation at the given node.
        Combine the models from the node's children with its local model,
        using a weighted aggregation based on data sizes.
        """
        pass

    @abstractmethod
    def print_tree(self):
        """
        Display the current structure of the tree for debugging purposes.
        Shows parent-child relationships and relevant node properties.
        """
        pass
