from basetree import BaseTree 

class Node:
    """Class representing a node in the binary tree."""
    def __init__(self, node_id, data_size, model=None):
        self.node_id = node_id        # Unique identifier for the node
        self.data_size = data_size    # Size of local data
        self.model = model            # Model associated with this node
        self.left = None              # Left child
        self.right = None             # Right child
        self.parent = None            # Parent node

# class BinaryTree(BaseTree):
#     """Class representing the binary tree structure."""
#     def __init__(self, nodes):
#         self.nodes = nodes  # List of all nodes
#         self.root = None    # Root of the tree (server node)

#     def build_tree(self):
#         """Organize nodes into a binary tree."""
#         if not self.nodes:
#             return None
#         self.root = self.nodes[0]  # Set the first node as the initial root
#         queue = [self.root]
#         idx = 1
#         while queue and idx < len(self.nodes):
#             current = queue.pop(0)
#             if idx < len(self.nodes):
#                 current.left = self.nodes[idx]
#                 current.left.parent = current
#                 queue.append(current.left)
#                 idx += 1
#             if idx < len(self.nodes):
#                 current.right = self.nodes[idx]
#                 current.right.parent = current
#                 queue.append(current.right)
#                 idx += 1

#     def select_server(self):
#         """
#         Elect the node with the highest score as the new root.
#         Example scoring: data size.
#         """
#         elected_node = max(self.nodes, key=lambda n: n.data_size)
#         print(f"Node {elected_node.node_id} selected as the new server.")
#         return elected_node

#     def reorganize_tree(self, new_root):
#         """
#         Reorganize the tree to make the selected node the root.
#         Uses tree rotations to bring the new root to the top.
#         """
#         path_to_root = []
#         current = new_root

#         # Traverse upwards to collect the path to the root
#         while current is not None:
#             path_to_root.append(current)
#             current = current.parent

#         # Perform rotations to bring the new root to the top
#         for node in reversed(path_to_root):
#             if node.parent is None:
#                 break
#             parent = node.parent
#             grandparent = parent.parent
#             if parent.left == node:  # Right rotation
#                 self.right_rotate(parent)
#             elif parent.right == node:  # Left rotation
#                 self.left_rotate(parent)

#             # Update the grandparent's connection
#             if grandparent:
#                 if grandparent.left == parent:
#                     grandparent.left = node
#                 else:
#                     grandparent.right = node
#                 node.parent = grandparent
#             else:
#                 node.parent = None  # New root
#                 self.root = node

#     def left_rotate(self, node):
#         """Perform a left rotation on a node."""
#         new_parent = node.right
#         if new_parent is None:
#             return
#         node.right = new_parent.left
#         if new_parent.left:
#             new_parent.left.parent = node
#         new_parent.left = node

#         # Update parent pointers
#         new_parent.parent = node.parent
#         if node.parent:
#             if node.parent.left == node:
#                 node.parent.left = new_parent
#             else:
#                 node.parent.right = new_parent
#         node.parent = new_parent

#     def right_rotate(self, node):
#         """Perform a right rotation on a node."""
#         new_parent = node.left
#         if new_parent is None:
#             return
#         node.left = new_parent.right
#         if new_parent.right:
#             new_parent.right.parent = node
#         new_parent.right = node

#         # Update parent pointers
#         new_parent.parent = node.parent
#         if node.parent:
#             if node.parent.left == node:
#                 node.parent.left = new_parent
#             else:
#                 node.parent.right = new_parent
#         node.parent = new_parent

#     def print_tree(self, node=None, level=0):
#         """Helper function to display the tree structure."""
#         if node is None:
#             node = self.root
#         if node.right:
#             self.print_tree(node.right, level + 1)
#         print(" " * 4 * level + f"-> Node {node.node_id} (Data size: {node.data_size})")
#         if node.left:
#             self.print_tree(node.left, level + 1)
    
#     def aggregate_models(self, node):
#         return super().aggregate_models(node)
    
#     def broadcast_model(self, model):
#         return super().broadcast_model(model)


class Tree:
    def __init__(self, val=None):
        self.value = val
        if self.value:
            self.left = Tree()
            self.right = Tree()
        else:
            self.left = None
            self.right = None
    
    def isempty(self):
        return self.value == None
    
    def isleaf(self):
        # Check if the tree node is a leaf node (both left and right children are None)
        if self.left.left == None and self.right.right == None:
            return True
        else:
            return False
    
    def insert(self, data):
        if self.isempty():
            # If the current node is empty, 
            #insert the data as its value
            self.value = data
            # Create empty left and right children
            self.left = Tree()
            self.right = Tree()
        elif self.value == data:
            # If the data already exists in the tree, return
            return
        elif data < self.value:
            # If the data is less than the current node's value, 
            #insert it into the left subtree
            self.left.insert(data)
            return
        elif data > self.value:
            # If the data is greater than the current node's value, 
            #insert it into the right subtree
            self.right.insert(data)
            return
    
    def find(self, v):
        if self.isempty():
            # If the tree is empty, the value is not found
            print("{} is not found".format(v))
            return False
        if self.value == v:
            # If the value is found at the current node, 
            #print a message and return True
            print("{} is found".format(v))
            return True
        if v < self.value:
            # If the value is less than the current node's value, 
            #search in the left subtree
            return self.left.find(v)
        else:
            # If the value is greater than the current node's value, 
            #search in the right subtree
            return self.right.find(v)
    
    def inorder(self):
        if self.isempty():
            # If the tree is empty, return an empty list
            return []
        else:
            # Return the inorder traversal of the tree (left subtree, root, right subtree)
            return self.left.inorder() + [self.value] + self.right.inorder()


if __name__ == "__main__":
    # Create nodes with example data sizes
    nodes = [Node(node_id=i, data_size=(i + 1) * 10) for i in range(7)]

    # Initialize and build the tree
    # tree = BinaryTree(nodes)
    # tree.build_tree()
    # tree.print_tree()

    # # Select a new server and reorganize the tree
    # new_server = tree.select_server()
    # tree.reorganize_tree(new_server)

    # # Print the updated tree structure
    # print("\nTree after reorganization:")
    # tree.print_tree()

    # Example usage
    t = Tree(20)
    t.insert(15)
    t.insert(25)
    t.insert(8)
    t.insert(16)
    t.find(20)
    print(t.inorder())


