from node import TreeNode
from tree import Tree
import random

def random_seq():
    sequences = []
    for _ in range(5):
        sequence = ''.join(random.choice(['a', 'b']) for _ in range(10))
        sequences.append(sequence)
    return sequences

def add_node(node_value, prob: None, father_node: TreeNode):
    tmp_node = TreeNode(value=node_value, prob=prob, suffix=father_node.value)
    father_node.addChild(tmp_node)
    print(f"Added node '{node_value}' under father node '{father_node.value}'")

def find_all_suffixes(s):
    """ 
        Find all suffixes of s.
        E.g., for s = "abc", it returns ["bc", "c"].
    """
    suffixes = []
    for i in range(len(s) - 1):
        suffixes.append(s[i+1:])
    return suffixes

def find_father_suffix(s): 
    # TODO: da fare bene e capire in che ordine esaminare i contesti.
    """ 
        Find the longest suffix of s.
        Theorically, it should be the value of the father node in the tree.
    """
    return s[1:]

def compute_context_probabilities(border, sequences):
    """ The probabilities of contextes are used only to filter rare contexts."""
    s_prob = {s: 0.0 for s in border}
    for s in border:
        len_s = len(s)
        window_size = 0
        cont = 0
        tmp = 0
        for seq in sequences:
            tmp = 0
            for i in range(len(seq) - len(s) + 1):
                if seq[i:i+len(s)] == s:
                    tmp += 1
                    cont += 1
            print(f"Suffix '{s}' found {tmp} times in the sequence '{seq}'.")

            window_size += len(seq) - len_s + 1
        
        print(f"Total number of context windows for suffix '{s}': {window_size}\nTotal occurrences: {cont}\n")

        p_s = cont / window_size if window_size > 0 else 0
        s_prob[s] = p_s
        print(f"Probability of suffix '{s}': {p_s}\n")
    return s_prob
         
def main(): 
    alphabet = ['a', 'b']
    border = {"a", "b", "aa", "ab", "ba", "bb"} # just for the probabilities used fot the significativity test

    sequences = random_seq()
    print("Generated sequences:")
    for seq in sequences:
        print(seq)
    
    # compute suffix probabilities
    s_prob = compute_context_probabilities(border, sequences) 
    #print(f"Final suffix probabilities: {s_prob}\n")

    suffix_tree = Tree(L=4)
    s_sign = ["aa", "ab", "bb", "aaa", "abb", "abbb"]

    #for s in s_sign.copy(): 
    go = True
    while go:
        s = s_sign[0]
        print(f"\n----- Processing suffix '{s}' -----")
        s_sign.remove(s)
        s_suffix = s[1:]
        #print(f"s: {s}, s_suffix: {s_suffix}")

        # TODO: test di significatività 
        all_suffixes = find_all_suffixes(s)[::-1]
        
        print(f"All suffixes of '{s}': {all_suffixes}")

        for x in all_suffixes: # check the presence of all suffixes of s in the tree
            
            find = suffix_tree.search_node(x)
            #print(f"Finding node '{x}': {find}")
            if not find: # add it to the tree if not present
                if x[1:] == "":
                    father_node = suffix_tree.root
                else:
                    father_node = suffix_tree.find_father_node(x[1:])
                #print(f"Father node of '{x}': '{father_node.value}'")
                add_node(node_value=x, prob=None, father_node=father_node)
        
        # Aggiungi il nodo s solo se non è già presente nell'albero
        if not suffix_tree.search_node(s):
            father_node = suffix_tree.find_father_node(s[1:])
            add_node(node_value=s, prob=None, father_node=father_node)

        # border expansion 
        print(f"----- Expanding border for suffix '{s}' -----")
        if len(s) < suffix_tree.L:
            for a in alphabet:
                new_suffix = a + s
                if new_suffix not in s_sign:
                    print(f"len(s_sign): {len(s_sign)}")

                    s_sign.append(new_suffix)
                    print(f"len(s_sign) after append: {len(s_sign)}")
                    print(f"Border expanded with new suffix: '{new_suffix}'")
                    print(f"New border: {s_sign}\n")
        if len(s_sign) == 0:
            go = False
    suffix_tree.print_tree()



             
        
if __name__ == "__main__":
    main()  
    
    


