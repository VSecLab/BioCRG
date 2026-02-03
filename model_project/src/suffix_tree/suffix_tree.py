from node import TreeNode
from tree import Tree
import random

L = 4  # Max length of the suffix tree
Pmin = 0.000001
gamma_min = 0.0
eps = 0.0

import math

def PST_Probability(PST, sequence, L):
    """
    Compute the log-probability of a sequence given a Probabilistic Suffix Tree (PST).
    """
    log_prob = 0.0
    T = len(sequence)
    print(f"\n\nComputing log-probability for sequence: '{sequence}'")
    
    for t in range(T):
        if t == 0:
            # First symbol: always use root
            node = PST.root
            symbol = sequence[t]
            print(f"\n[{t}] First symbol, using root")
        else:
            # Get history: last min(L, t) symbols before position t
            history_length = min(L, t)
            history = sequence[t - history_length:t]
            
            candidate = history
            print(f"\nHistory for position {t}: '{history}'")
            node = None
            
            # Search for the longest suffix in the PST
            while candidate != "":
                inverted_candidate = candidate[::-1]
                if PST.search_node(inverted_candidate):
                    node = PST.find_father_node(inverted_candidate)
                    print(f"[{t}] Found Candidate: {candidate} - (inverted candidate '{inverted_candidate}')")
                    break
                # Get suffix by removing first character
                candidate = candidate[1:]
            
            # If no candidate found, use root
            if candidate == "" or node is None:
                node = PST.root
            
            symbol = sequence[t]
        
        # Get transition probability for this symbol
        if symbol in node.transitions_probs:
            prob = node.transitions_probs[symbol]
            print(f"[{t}] P({symbol}|{node.value}) = {prob}")

            if prob > 0:
                log_prob += math.log(prob)
            else:
                # Handle zero probability (should not happen with gamma_min)
                log_prob += math.log(1e-10)  # Use small value to avoid log(0)
        else:
            # Symbol not in transitions (should not happen if alphabet is complete)
            log_prob += math.log(1e-10)
    
    print(f"\n\nTotal log-probability: {log_prob}")
    self_info =  - log_prob / T
    print(f"Self-information (average log-probability per symbol): {self_info}\n")
    return log_prob

def random_seq(nos:5):
    sequences = []
    for _ in range(nos):
        sequence = ''.join(random.choice(['a', 'b']) for _ in range(10))
        sequences.append(sequence)
    return sequences

def define_border(alphabet, contexts_prob): 
    border = []
    for a in alphabet: 
        a_prob = contexts_prob.get(a, 0)
        if a_prob > Pmin:  # Threshold for including in the border
            border.append(a)
    return border

def add_node(node_value, prob, father_node: TreeNode):
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

def border_expansion(suffix_tree, border, s, alphabet, emp): 
    # border expansion 
    if len(s) < L:
        for a in alphabet:
            new_suffix = a + s

            # probability check 
            p_new_suffix = get_context_probabilities(emp).get(new_suffix, 0)
            if p_new_suffix > Pmin and new_suffix not in border:

                border.append(new_suffix)
                print(f"Border expanded: {border}")

def add_suffixes_and_node(suffix_tree, s, emp):
    all_suffixes = find_all_suffixes(s)[::-1]
    if len(all_suffixes) == 0 and not suffix_tree.search_node(s):
        # there are no suffixes to add
        father_node = suffix_tree.root
        add_node(node_value=s, prob=emp.get(s, (0, {})), father_node=father_node)
        return
    elif not suffix_tree.search_node(s): 
        for x in all_suffixes:
            find = suffix_tree.search_node(x)
            if not find:
                print(f"Adding suffix '{x}' to the tree - suffix: {x[1:]}")
                if x[1:] == "":
                    father_node = suffix_tree.root
                else:
                    father_node = suffix_tree.find_father_node(x[1:])
                add_node(node_value=x, prob=emp.get(x, (0, {})), father_node=father_node)
        father_node = suffix_tree.find_father_node(s[1:])
        add_node(node_value=s, prob=emp.get(s, (0, {})), father_node=father_node)  

def significativity_test(s, emp, parent_suffix, alphabet): 
    find = False 
    for a in alphabet:
        conditional_prob = get_single_conditional_prob(emp=emp, s=s, sigma=a)
        parent_cond_prob = get_single_conditional_prob(emp=emp, s=parent_suffix, sigma=a)
        #print(f"Significativity test for suffix '{s}' with symbol '{a}': P({a}|{s}) = {conditional_prob}, P({a}|{parent_suffix}) = {parent_cond_prob}")
        if (parent_cond_prob == 0) and (conditional_prob > gamma_min):
            find = True
            break
        elif conditional_prob > gamma_min and (conditional_prob / parent_cond_prob) > (1 + eps):    
            find = True
            break
    return find

def compute_context_probabilities(contexts, sequences):
    """ The probabilities of contextes are used only to filter rare contexts."""
    s_prob = {s: 0.0 for s in contexts}
    for s in contexts:
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

def get_single_conditional_prob(emp, s, sigma):
    """
    Extract the conditional probability P(sigma|s) from the empirical probabilities dictionary.
    
    Args:
        emp: Dictionary where each value is a tuple (prob, conditional_probs)
        s: The context string
        sigma: The symbol for which to get the conditional probability
    Returns:
        The conditional probability P(sigma|s)
    """
    if s in emp:
        _, conditional_probs = emp[s]
        return conditional_probs.get(sigma, 0.0)
    return 0.0
    
def get_conditional_probabilities(emp):
    """
    Extract conditional probabilities from the empirical probabilities dictionary.
    
    Args:
        emp: Dictionary where each value is a tuple (prob, conditional_probs)
    Returns:
        Dictionary mapping context to its conditional probabilities P(sigma|s)
    """
    cond_probs = {}
    for context, (_, conditional_probs) in emp.items():
        cond_probs[context] = conditional_probs
    return cond_probs

def get_context_probabilities(emp):
    """
    Extract context probabilities from the empirical probabilities dictionary.
    
    Args:
        emp: Dictionary where each value is a tuple (prob, conditional_probs)
    
    Returns:
        Dictionary mapping context to its probability P(s)
    """
    context_probs = {}
    for context, (prob, _) in emp.items():
        context_probs[context] = prob
    return context_probs

def empirical_probs(context_occ, sequences, alphabet):
    """
        Compute empirical probabilities for all contexts:
        P(s) = #s / total_windows
        P(sigma|s) = #(sigma * s) / #s for each sigma in the alphabet.
    """
    emp = {}
    extended_context_occ = compute_context_plus(sequences)

    for context, count in context_occ.items():
        extendend_cout = extended_context_occ.get(context, 0)
        scp = single_context_prob(context, extendend_cout, sequences) # i need to consider all the occurrences including last symbols
        pss = single_conditional_probs(context, context_occ, extended_context_occ, alphabet) 
        emp[context] = (scp, pss)
        print(f"Context: '{context}', P(s): {scp}, P(sigma|s): {pss}")
    return emp

def single_conditional_probs(context, context_occ, extended_context_occ, alphabet):
    """
        for the given context s, compute P(sigma|context) = #(sigma * s) / #s for each sigma in the alphabet.
        Where #(sigma * s) is the number of occurrences of the extended context (sigma + s) and #s is the number of occurrences of s.

    """
    p_sigma_s = {}
    context_count = context_occ.get(context, 0)

    for sigma in alphabet:
        extended_context = context + sigma
        extended_count = extended_context_occ.get(extended_context, 0)

        if context_count > 0:
            p_sigma_given_s = extended_count / context_count
        else:
            p_sigma_given_s = 0.0

        p_sigma_s[sigma] = p_sigma_given_s
    return p_sigma_s

def single_context_prob(s, s_count, sequences): 
    """ 
        For the given context s, compute P(context) = #s / total_windows.
        Where #s is the number of occurrences of s in the sequences,
    """
    if len(s) > L:
        return None
    window_lenght = sum(len(seq) - len(s) + 1 for seq in sequences)
    p_s = s_count / window_lenght if window_lenght > 0 else 0

    return p_s

def compute_context_plus(sequences): 
    """ 
        Compute all subsequences of length up to L from the given sequences and their occurrences, also with last symbols.  

        E.g.: "aabbaababba", the "a" occures 6 times.
    """
    subsequences_occurences = {}

    for seq in sequences:
        seq_len = len(seq)
        for i in range(seq_len):
            for l in range(1, L + 1):
                if i + l <= seq_len:
                    context = seq[i:i + l]
                    if context in subsequences_occurences:
                        subsequences_occurences[context] += 1
                    else:
                        subsequences_occurences[context] = 1
    return subsequences_occurences

def compute_context(sequences): 
    """ 
        Compute all subsequences of length up to L from the given sequences and their occurrences, without last symbols.

        E.g.: "aabbaababba", the "a" occures 5 times.
    """
    context_occ = {}

    for seq in sequences:
        seq_len = len(seq)
        for i in range(1, seq_len):  # i is the index of the predicted symbol
            for l in range(1, L + 1):
                if i - l >= 0:
                    context = seq[i - l:i]
                    context_occ[context] = context_occ.get(context, 0) + 1

    return context_occ

def compute_trans_probs(tree, node=None, alphabet=list, emp=dict): 
    cond_probs = get_conditional_probabilities(emp)

    if node is None:
        node = tree.root

    for a in alphabet: 
        if node is not tree.root:
            father = node.suffix
            if father == "root": 
                p = tree.root.transitions_probs.get(a, 0)
            else:
                p = cond_probs.get(father, {}).get(a, 0) # P( a | father)
            #print(f"P({a}|{father}) = {p} for node '{node.value}'")
            node.transitions_probs[a] = (1 - len(alphabet) * gamma_min) * p + gamma_min
            print(f"Node {node.value} - Transition probability P({a}|{node.value}) = {node.transitions_probs[a]}")

        else:
            p = emp.get(a, (0, {}))[0]  # P(a) for root node
            node.transitions_probs[a] = (1 - len(alphabet) * gamma_min) * p + gamma_min

    for child in node.children:
        compute_trans_probs(tree, node=child, alphabet=alphabet, emp=emp)
        
def phace_two(alphabet, emp):
    suffix_tree = Tree(L=L)
    border = define_border(alphabet, get_context_probabilities(emp))
    print(f"Initial border: {border}\n")
    go = True
    while go:
        s = border[0]
        print(f"\n----- Processing suffix '{s}' -----")
        border.remove(s)
        s_suffix = s[1:]

        bool_sig_test = False
        if s_suffix == "":
            bool_sig_test = True
        else:
            bool_sig_test = significativity_test(s, emp, s_suffix, alphabet)
        print(f"Significativity test for suffix '{s}': {bool_sig_test}")

        if bool_sig_test:
            add_suffixes_and_node(suffix_tree, s, emp)
            
            border_expansion(suffix_tree, border, s, alphabet, emp)

        if len(border) == 0:
            go = False
    return suffix_tree

def phase_one(sequences, alphabet): 

    context_occ = compute_context(sequences)
    emp = empirical_probs(context_occ, sequences, alphabet)
    
    return context_occ, emp

def phase_zero(): 
    # Phase 0: Initialize alphabet of actions, generate sequences
    alphabet = ['a', 'b']
    sequences = random_seq(nos=2)

    print("Generated sequences:")
    for seq in sequences:
        print(seq)
    
    return alphabet, sequences

def main(): 
    

    # Phase 0
    alphabet, sequences = phase_zero()

    # Phase 1
    context_occ, emp = phase_one(sequences, alphabet)
    print(f"\n\n-----Empirical probabilities:\n{emp}-----\n\n")
    
    # Phase 2 
    # PST building 
    suffix_tree = phace_two(alphabet, emp)

    print(f"\n\n-----Suffix Tree:-----\n")
    suffix_tree.print_tree()
    
    # Phase 3 
    # Adding structural node and compute probabilities
    suffix_tree.add_structural_node(alphabet=alphabet)
    print(f"\n\n-----Suffix Tree after adding structural nodes:-----\n")
    suffix_tree.print_tree()
    print("\n")
    compute_trans_probs(suffix_tree, alphabet=alphabet, emp=emp)
    print("\n\n\n")
    suffix_tree.print_tree()

    PST_Probability(suffix_tree, "abbababbabbaba", L)

      
if __name__ == "__main__":
    main()  
    
    


