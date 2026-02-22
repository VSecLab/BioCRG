from .node import TreeNode
from .tree import Tree
import numpy as np
import random
import math

#L = 4  # Max length of the suffix tree
Pmin = 0.000001
gamma_min = 0.0
eps = 0.0



def ghost_func(n: int, N: int, exp: int, states: int) -> float:
    m = math.ceil(states / N)
    return np.exp(- (m) * (n ** exp) / N)

def PST_Probability_ghost(PST, sequence:list, L:int, states, ghost_exp=4, avg_sequence_length:int=1): 

    N_states = len([s for s in sequence])
    ghost_state = "c0"
    ghost_count = sequence.count(ghost_state)

    if N_states == ghost_count:
        # print(f"All states are ghost states. Returning infinite self-information.")
        return np.inf, N_states, ghost_count
    elif N_states == 1: 
        # print(f"Only one state present. Returning infinite self-information.")
        return np.inf, N_states, ghost_count
    elif N_states == 0:
        # print(f"No states present. Returning infinite self-information.")
        return None, N_states, ghost_count

    
    # Calcolo della penalità da applicare nei salti ghost
    ghost_penalty = ghost_func(ghost_count, N_states, exp=ghost_exp, states=avg_sequence_length)

    # print(f"N_states: {N_states} - Ghost Count: {ghost_count} - Ghost Penalty: {ghost_penalty} - States: {states}")

    prob = PST_Probability(PST, sequence, L) * ghost_penalty
    if prob == 0: 
        # print(f"Probability is zero after applying ghost penalty. Returning infinite self-information.")
        return np.inf, N_states, ghost_count
    self_info = np.abs(np.log(prob)) / (N_states)
    # print(f"Log-probability with ghost penalty: {prob} - Self-information: {self_info}")
    
    return self_info, N_states, ghost_count


def PST_Probability(PST, sequence, L):
    """
    Compute the log-probability of a sequence given a Probabilistic Suffix Tree (PST).
    """
    tot_prob = 1
    T = len(sequence)
    # print(f"\n\nComputing log-probability for sequence: {sequence}")
    
    for t in range(T):
        if t == 0:
            # First symbol: always use root
            node = PST.root
            symbol = sequence[t]
            # print(f"\n[{t}] First symbol, using root")
        else:
            # Get history: last min(L, t) symbols before position t
            history_length = min(L, t)
            history = tuple(sequence[t - history_length:t])  # Convert to tuple
            
            candidate = history
            # print(f"\nHistory for position {t}: {history}")
            node = None
            
            # Search for the longest suffix in the PST
            while len(candidate) > 0:
                inverted_candidate = candidate[::-1]
                # Convert tuple to string for node search (assuming node values are stored as strings)
                inverted_str = '_'.join(inverted_candidate)
                if PST.search_node(inverted_str):
                    node = PST.find_father_node(inverted_str)
                    # print(f"[{t}] Found Candidate: {candidate} - (inverted candidate {inverted_candidate})")
                    break
                # Get suffix by removing first element
                candidate = candidate[1:]
            
            # If no candidate found, use root
            if len(candidate) == 0 or node is None:
                node = PST.root
            
            symbol = sequence[t]
        
        # Get transition probability for this symbol
        if symbol in node.transitions_probs:
            prob = node.transitions_probs[symbol]
            # print(f"[{t}] P({symbol}|{node.value}) = {prob}")

            if prob > 0:
                tot_prob *= prob 
                # print(f"[{t}] Cumulative probability: {tot_prob}")
            else:
                # Handle zero probability (should not happen with gamma_min)
                tot_prob *= 1e-10  # Use small value to avoid log(0)
        else:
            # Symbol not in transitions (should not happen if alphabet is complete)
            tot_prob *= 1e-10
    

    # print(f"\n\nTotal log-probability: {tot_prob}")
    #print(f"Self-information (average log-probability per symbol): {self_info}\n")
    return tot_prob

def random_seq(nos=5, alphabet=None):
    """Generate random sequences as lists of symbols."""
    if alphabet is None:
        alphabet = ['a', 'b']
    sequences = []
    for _ in range(nos):
        sequence = [random.choice(alphabet) for _ in range(10)]
        sequences.append(sequence)
    return sequences

def define_border(alphabet, contexts_prob, pmin=0.000001): 
    border = []
    for a in alphabet: 
        # Convert single symbol to tuple for dictionary lookup
        a_tuple = (a,)
        a_prob = contexts_prob.get(a_tuple, 0)
        if a_prob > pmin:  # Threshold for including in the border
            border.append(a_tuple)
    return border

def add_node(node_value, prob, father_node: TreeNode):
    tmp_node = TreeNode(value=node_value, prob=prob, suffix=father_node.value)
    father_node.addChild(tmp_node)
    # print(f"Added node '{node_value}' under father node '{father_node.value}'")

def find_all_suffixes(s):
    """ 
        Find all suffixes of s.
        E.g., for s = ('a', 'b', 'c'), it returns [('b', 'c'), ('c',)].
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

def border_expansion(suffix_tree, border, s, alphabet, emp, L, pmin=0.000001): 
    # border expansion 
    if len(s) < L:
        for a in alphabet:
            new_suffix = (a,) + s  # Prepend symbol to tuple

            # probability check 
            p_new_suffix = get_context_probabilities(emp).get(new_suffix, 0)
            if p_new_suffix > pmin and new_suffix not in border:

                border.append(new_suffix)
                # print(f"Border expanded: {border}")

def add_suffixes_and_node(suffix_tree, s, emp, L):
    # s is now a tuple of symbols
    # Convert tuple to string for node value (using '_' as separator)
    s_str = '_'.join(s) if len(s) > 0 else ''
    all_suffixes = find_all_suffixes(s)[::-1]
    if len(all_suffixes) == 0 and not suffix_tree.search_node(s_str):
        # there are no suffixes to add
        father_node = suffix_tree.root
        add_node(node_value=s_str, prob=emp.get(s, (0, {})), father_node=father_node)
        return
    elif not suffix_tree.search_node(s_str): 
        for x in all_suffixes:
            x_str = '_'.join(x) if len(x) > 0 else ''
            find = suffix_tree.search_node(x_str)
            if not find:
                # print(f"Adding suffix {x} to the tree - suffix: {x[1:]}")
                if len(x[1:]) == 0:
                    father_node = suffix_tree.root
                else:
                    suffix_str = '_'.join(x[1:])
                    father_node = suffix_tree.find_father_node(suffix_str)
                add_node(node_value=x_str, prob=emp.get(x, (0, {})), father_node=father_node)
        suffix_str = '_'.join(s[1:]) if len(s) > 1 else ''
        father_node = suffix_tree.find_father_node(suffix_str) if suffix_str else suffix_tree.root
        add_node(node_value=s_str, prob=emp.get(s, (0, {})), father_node=father_node)  

def significativity_test(s, emp, parent_suffix, alphabet, gamma_min=0.0, eps=0.0): 
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
                if tuple(seq[i:i+len(s)]) == s:
                    tmp += 1
                    cont += 1
            # print(f"Suffix {s} found {tmp} times in the sequence {seq}.")

            window_size += len(seq) - len_s + 1
        
        # print(f"Total number of context windows for suffix {s}: {window_size}\nTotal occurrences: {cont}\n")

        p_s = cont / window_size if window_size > 0 else 0
        s_prob[s] = p_s
        # print(f"Probability of suffix {s}: {p_s}\n")
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

def empirical_probs(context_occ, sequences, alphabet, L):
    """
        Compute empirical probabilities for all contexts:
        P(s) = #s / total_windows
        P(sigma|s) = #(sigma * s) / #s for each sigma in the alphabet.
    """
    emp = {}
    extended_context_occ = compute_context_plus(sequences, L)

    for context, count in context_occ.items():
        extendend_cout = extended_context_occ.get(context, 0)
        scp = single_context_prob(context, extendend_cout, sequences, L) # i need to consider all the occurrences including last symbols
        pss = single_conditional_probs(context, context_occ, extended_context_occ, alphabet) 
        emp[context] = (scp, pss)
        # print(f"Context: {context}, P(s): {scp}, P(sigma|s): {pss}")
    return emp

def single_conditional_probs(context, context_occ, extended_context_occ, alphabet):
    """
        for the given context s, compute P(sigma|context) = #(sigma * s) / #s for each sigma in the alphabet.
        Where #(sigma * s) is the number of occurrences of the extended context (sigma + s) and #s is the number of occurrences of s.

    """
    p_sigma_s = {}
    context_count = context_occ.get(context, 0)

    for sigma in alphabet:
        # Append sigma to context tuple
        extended_context = context + (sigma,)
        extended_count = extended_context_occ.get(extended_context, 0)

        if context_count > 0:
            p_sigma_given_s = extended_count / context_count
        else:
            p_sigma_given_s = 0.0

        p_sigma_s[sigma] = p_sigma_given_s
    return p_sigma_s

def single_context_prob(s, s_count, sequences, L): 
    """ 
        For the given context s, compute P(context) = #s / total_windows.
        Where #s is the number of occurrences of s in the sequences,
    """
    if len(s) > L:
        return None
    window_lenght = sum(len(seq) - len(s) + 1 for seq in sequences)
    p_s = s_count / window_lenght if window_lenght > 0 else 0

    return p_s

def compute_context_plus(sequences, L): 
    """ 
        Compute all subsequences of length up to L from the given sequences and their occurrences, also with last symbols.  

        E.g.: ["a","a","b","b","a","a","b","a","b","b","a"], the ("a",) occurs 6 times.
    """
    subsequences_occurences = {}

    for seq in sequences:
        seq_len = len(seq)
        for i in range(seq_len):
            for l in range(1, L + 1):
                if i + l <= seq_len:
                    context = tuple(seq[i:i + l])  # Convert to tuple
                    if context in subsequences_occurences:
                        subsequences_occurences[context] += 1
                    else:
                        subsequences_occurences[context] = 1
    return subsequences_occurences

def compute_context(sequences, L=None): 
    """ 
        Compute all subsequences of length up to L from the given sequences and their occurrences, without last symbols.

        E.g.: ["a","a","b","b","a","a","b","a","b","b","a"], the ("a",) occurs 5 times.
    """
    context_occ = {}

    for seq in sequences:
        seq_len = len(seq)
        for i in range(1, seq_len):  # i is the index of the predicted symbol
            for l in range(1, L + 1):
                if i - l >= 0:
                    context = tuple(seq[i - l:i])  # Convert to tuple
                    context_occ[context] = context_occ.get(context, 0) + 1

    return context_occ

def compute_trans_probs(tree, node=None, alphabet=list, emp=dict, gamma_min=0.0): 
    cond_probs = get_conditional_probabilities(emp)

    if node is None:
        node = tree.root

    for a in alphabet: 
        if node is not tree.root:
            father = node.suffix
            if father == "root": 
                p = tree.root.transitions_probs.get(a, 0)
            else:
                # Convert father string to tuple for lookup in emp
                father_tuple = tuple(father.split('_'))
                p = cond_probs.get(father_tuple, {}).get(a, 0) # P( a | father)
            #print(f"P({a}|{father}) = {p} for node '{node.value}'")
            node.transitions_probs[a] = (1 - len(alphabet) * gamma_min) * p + gamma_min
            # print(f"Node {node.value} - Transition probability P({a}|{node.value}) = {node.transitions_probs[a]}")

        else:
            # For root node, get P(a) - convert symbol to tuple
            p = emp.get((a,), (0, {}))[0]  # P(a) for root node
            node.transitions_probs[a] = (1 - len(alphabet) * gamma_min) * p + gamma_min

    for child in node.children:
        compute_trans_probs(tree, node=child, alphabet=alphabet, emp=emp, gamma_min=gamma_min)

def phace_three(suffix_tree, alphabet, emp, gamma_min=0.0):
    suffix_tree.add_structural_node(alphabet=alphabet)
    #print(f"\n\n-----Suffix Tree after adding structural nodes:-----\n")
    #suffix_tree.print_tree()
    #print("\n")
    compute_trans_probs(suffix_tree, alphabet=alphabet, emp=emp, gamma_min=gamma_min)
    # print("\n\n\n")
    suffix_tree.print_tree()
 
def phace_two(alphabet, emp, L, pmin=0.000001, gamma_min=0.0, eps=0.0):
    suffix_tree = Tree(L=L)
    border = define_border(alphabet, get_context_probabilities(emp), pmin=pmin)
    # print(f"Initial border: {border}\n")
    go = True
    while go:
        s = border[0]
        # print(f"\n----- Processing suffix {s} -----")
        border.remove(s)
        s_suffix = s[1:]

        bool_sig_test = False
        if len(s_suffix) == 0:  # Check length instead of comparing to ""
            bool_sig_test = True
        else:
            bool_sig_test = significativity_test(s, emp, s_suffix, alphabet, gamma_min=gamma_min, eps=eps)
        # print(f"Significativity test for suffix {s}: {bool_sig_test}")

        if bool_sig_test:
            add_suffixes_and_node(suffix_tree, s, emp, L)
            
            border_expansion(suffix_tree, border, s, alphabet, emp, L, pmin=pmin)

        if len(border) == 0:
            go = False
    return suffix_tree

def phase_one(sequences, alphabet, L=None): 

    context_occ = compute_context(sequences, L)
    emp = empirical_probs(context_occ, sequences, alphabet, L)
    
    return context_occ, emp

def phase_zero(alphabet=None, sequences=None): 
    # Phase 0: Initialize alphabet of actions, generate sequences
    if alphabet is None:    
        alphabet = ['a', 'b']
    if sequences is None:
        sequences = random_seq(nos=2)

    #for seq in sequences:
        # print(seq)
    
    return alphabet, sequences

def create_tree(sequences = None, alphabet = None, L=int, pmin=0.000001, gamma_min=0.0, eps=0.0): 
    # Phase 0
    alphabet, sequences = phase_zero(alphabet, sequences)

    # Phase 1
    context_occ, emp = phase_one(sequences, alphabet, L)
    # print(f"\n\n-----Empirical probabilities:\n{emp}-----\n\n")
    
    # Phase 2 
    # PST building 
    suffix_tree = phace_two(alphabet, emp, L, pmin=pmin, gamma_min=gamma_min, eps=eps)

    #print(f"\n\n-----Suffix Tree:-----\n")
    #suffix_tree.print_tree()
    
    # Phase 3 
    # Adding structural node and compute probabilities
    phace_three(suffix_tree, alphabet, emp, gamma_min=gamma_min)

    return suffix_tree

def main(): 
    L = 4
    # Example usage with list sequences
    alphabet = ['a', 'b']
    # sequences = [["a", "b", "b", "a", "b", "a", "b", "b", "a", "b", "b", "a", "b", "a"]]
    # Or use random sequences
    sequences = random_seq(nos=2, alphabet=alphabet)
    
    suffix_tree = create_tree(sequences=sequences, alphabet=alphabet, L=L)
    
    test_sequence = ["a", "b", "b", "a", "b", "a", "b", "b", "a", "b", "b", "a", "b", "a"]
    PST_Probability(suffix_tree, test_sequence, L)
     
if __name__ == "__main__":
    main()  
    
    


