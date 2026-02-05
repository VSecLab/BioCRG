from .node import TreeNode

class Tree: 
    def __init__(self, L=int):
        self.root = TreeNode(value="root", prob=0.0, suffix=None)
        self.L = L # max length of the tree

    def add_structural_node(self, node=None, depth=0, alphabet=list): 
        if node is None:
            node = self.root 

        for child in node.children: 
            if not child.structural:
                for a in alphabet: 
                    new_value = a + "_" + child.value  # Add separator
                    if not self.search_node(new_value) and not child.is_leaf():
                        new_node = TreeNode(value=new_value, prob=0.0, suffix=child.value, structural=True)
                        child.addChild(new_node)
                        print(f"Added structural node '{new_value}' under node '{child.value}'")
            self.add_structural_node(child, depth + 1, alphabet)

    def search_node(self, node_value): 
        """ Find a node in the tree by its value. """
        current = self.root
        if current.value == node_value:
            return True
        
        queue = [current]
        while queue:
            current = queue.pop(0)
            if current.value == node_value:
                return True
            queue.extend(current.children)
        
        return False
    
    def len_tree(self): 
        """ Compute the number of nodes in the tree. """
        count = 0
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            count += 1
            queue.extend(current.children)
        return count
    
    def find_father_node(self, fatherNode_value): 
        """ Find the father node of a given node value. """
        current = self.root
        if current.value == fatherNode_value:
            return current
        
        queue = [current]
        while queue:
            current = queue.pop(0)
            if current.value == fatherNode_value:
                return current
            queue.extend(current.children)
        
        return None

    def is_father_root(self, node_value): 
        """ Check if the father node of a given node value is the root. """
        father_node = self.find_father_node(node_value)
        if father_node and father_node == self.root:
            return True
        return False
    
    def print_tree(self, node=None, prefix="", is_last=True):
        """
        Stampa l'albero in modo gerarchico con linee di connessione.
        
        Esempio:
        root
        ├── a
        │   ├── aa
        │   └── ab
        └── b
            └── bb
        """
        if node is None:
            node = self.root
            print(f"Suffix Tree (max length: {self.L})")
            print(f"└── {node.value} (t_prob: {node.transitions_probs}, suffix: None)")
            prefix = "    "
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self.print_tree(child, prefix, is_last_child)
        else:
            connector = "└── " if is_last else "├── "
            prob_str = f"{node.transitions_probs}" if node.prob is not None else "None"
            suffix_str = f"'{node.suffix}'" if node.suffix is not None else "None"
            
            print(f"{prefix}{connector}{node.value} (t_prob: {prob_str}, suffix: {suffix_str})")
            
            if node.children:
                extension = "    " if is_last else "│   "
                new_prefix = prefix + extension
                for i, child in enumerate(node.children):
                    is_last_child = (i == len(node.children) - 1)
                    self.print_tree(child, new_prefix, is_last_child)
    
    def print_tree_simple(self, node=None, level=0):
        """Stampa l'albero in modo semplice con indentazione."""
        if node is None:
            node = self.root
            print(f"\n{'='*50}")
            print(f"Suffix Tree (L={self.L}, Total nodes: {self.len_tree()})")
            print(f"{'='*50}")
        
        indent = "  " * level
        prob_str = f"{node.prob:.4f}" if node.prob is not None else "None"
        suffix_str = f"'{node.suffix}'" if node.suffix else "root"
        children_count = len(node.children)
        
        print(f"{indent}{'└─' if level > 0 else ''}[{node.value}] prob={prob_str}, suffix={suffix_str}, children={children_count}")
        
        for child in node.children:
            self.print_tree_simple(child, level + 1)




        