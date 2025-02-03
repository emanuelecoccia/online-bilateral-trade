import numpy as np
from typing import Union, Tuple, Callable

def easy_function(s_dot:Union[float, np.ndarray], b_dot:Union[float, np.ndarray], *args, **qwargs)\
    ->Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    This function generates a simple function that is Lipschitz with respect to the context.
    It returns a pair of private valuations.
    """
    s = s_dot + 0.6 * (b_dot - s_dot)
    b = s + 0.2 * (b_dot - s_dot)
    return s, b

def sinusoidal_function(s_dot:Union[float, np.ndarray], b_dot:Union[float, np.ndarray], Lipschitz_constant:float)\
    ->Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    This function generates a wobbly function that is Lipschitz with respect to the context.
    It returns a pair of private valuations.
    """
    sinusoidal_residual = (0.5*np.sin(s_dot*2*Lipschitz_constant)+0.5) * (0.5*np.sin(b_dot*2*Lipschitz_constant)+0.5)
    s = s_dot + sinusoidal_residual * (b_dot - s_dot) * 0.75
    b = s + (b_dot - s_dot) * 0.25
    return s, b

def triangle_wave_function(s_dot:Union[float, np.ndarray], b_dot:Union[float, np.ndarray], Lipschitz_constant:float)\
    ->Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    This function generates a wobbly function with pointy edges 
    that is Lipschitz with respect to the context.
    It returns a pair of private valuations.
    """
    triangle_residual = 2*np.abs(s_dot*0.5*Lipschitz_constant-np.floor(s_dot*0.5*Lipschitz_constant+0.5))\
        * 2*np.abs(b_dot*0.5*Lipschitz_constant-np.floor(b_dot*0.5*Lipschitz_constant+0.5))
    s = s_dot + triangle_residual * (b_dot - s_dot) * 0.75
    b = s + (b_dot - s_dot) * 0.25
    return s, b

def construct_sequence_with_lipschitz_valuations(T:int, Lipschitz_constant:float, 
                                                 Lipschitz_function:Callable=sinusoidal_function, *args, **kwargs)\
                                                 ->Tuple[np.ndarray, np.ndarray]:
    """
    This function constructs a sequence of valuations and a sequence of contexts,
    such that the valuations are Lipschitz continuous with respect to the contexts.
    """
    s_dot = np.random.random(size=T)
    b_dot = np.random.random(size=T)
    s = np.zeros(T)
    b = np.zeros(T)
    for t in range(T):
        if s_dot[t] > b_dot[t]:
            s_dot[t], b_dot[t] = b_dot[t], s_dot[t]

    s, b = Lipschitz_function(s_dot, b_dot, Lipschitz_constant)
    contexts = np.stack([s_dot, b_dot], axis=1)
    valuations = np.stack([s, b], axis=1)
    return contexts, valuations

def construct_sqrt_lower_bound(T:int, Lipschitz_constant:float, *args, **kwargs)\
    ->Tuple[np.ndarray, np.ndarray]:
    """
    This function constructs a sequence of valuations and a sequence of contexts,
    such that the valuations are Lipschitz continuous with respect to the contexts.
    And the regret lower bound is O(L sqrt(T)).
    """
    # Contexts are inside the square (0, 1)-(0.25, 0.75)
    # inside a grid of size ceil(sqrt(T)) x ceil(sqrt(T))
    # chosen boustrophedon.
    m = int(np.ceil(np.sqrt(T)))
    counter = 0
    row = 0
    contexts = []
    while row < m and counter < T:
        col = 0
        while col < m and counter < T:
            # s and b bounded between 0 and m-1
            if row % 2 == 0:
                s_dot = col
                b_dot = m-row-1
            else:
                s_dot = m-col-1
                b_dot = m-row-1
            # Rescale the contexts to be inside the square
            s_dot = (s_dot + 0.5)/m * 0.25
            b_dot = 0.75 + (b_dot + 0.5)/m * 0.25
            # Append the context
            contexts.append(np.array([s_dot, b_dot]))
            col += 1
            counter += 1
        row += 1

    # Convert the list to numpy array
    contexts = np.array(contexts)

    # The valuations are chosen at random during odd turns
    # and at 0.5 and 0.5 during even turns
    EPSILON = 1e-5
    valuations = np.zeros((T, 2))
    for t in range(T):
        s = 0.5
        b = 0.5
        if t % 2 == 0: # odd turn
            if np.random.randint(2) == 1:
                s = s + EPSILON - min(0.25*Lipschitz_constant/m, 0.25)
                b = b - EPSILON
            else:
                s = s + EPSILON
                b = b - EPSILON + min(0.25*Lipschitz_constant/m, 0.25)
        valuations[t] = np.array([s, b])

    return contexts, valuations


def construct_logarithmic_lower_bound(T:int, Lipschitz_constant:float, *args, **kwargs)\
    ->Tuple[np.ndarray, np.ndarray]:
    """
    This function constructs a sequence of valuations and a sequence of contexts,
    such that the valuations are Lipschitz continuous with respect to the contexts.
    Fake and actual contexts are used in this construction to understand better the 
    logarithmic order of the lower bound.
    This construction would give a regret lower bound of O(L log(T/L)).
    """
    # The contexts are:
    # (0, 1), 
    # (0, 0.5), (0.5, 1), 
    # (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1),
    # ...
    # However if the Lipschitz constant allows it,
    # we can create M contexts equally spaced between parent and child.
    # Assuming the distance between two points is the l1 norm:
    if Lipschitz_constant < 4:
        raise ValueError("The Lipschitz constant is too small to construct the sequence.")
    M = int(Lipschitz_constant/4)

    # Initialize two lists of contexts:
    # One is for rescaling the valuations, the other is for the actual contexts
    contexts_for_rescaling_vals = [np.array([0, 1])]
    actual_contexts = [np.array([0, 1])]

    # Construct the contexts
    counter = 1
    i = 1 # level of the tree
    while counter < T:
        j = 0
        while j < 2**i and counter < T:
            # Append the fake contexts
            m = 0
            secondary_counter = counter
            while m < M and secondary_counter < T:
                contexts_for_rescaling_vals.append(np.array([j/(2**i), (j+1)/(2**i)]))
                m += 1
                secondary_counter += 1
            # Append the actual contexts
            if j%2==0:
                m = 0
                while m < M and counter < T:
                    x = j/(2**i)
                    y = (j+1)/(2**i) + (m/M)/(2**i)
                    actual_contexts.append(np.array([x, y]))
                    m += 1
                    counter += 1
            else:
                m = 0
                while m < M and counter < T:
                    x = j/(2**i) - (m/M)/(2**i)
                    y = (j+1)/(2**i)
                    actual_contexts.append(np.array([x, y]))
                    m += 1
                    counter += 1
            j += 1
        i += 1

    # Convert the lists to numpy arrays
    contexts_for_rescaling_vals = np.array(contexts_for_rescaling_vals)
    actual_contexts = np.array(actual_contexts)

    # The valuations are [0, 1/4] and [3/4, 1] at random, and rescaled to the "fake" contexts
    left_point = np.array([0, 1/4])
    right_point = np.array([3/4, 1])
    points = np.vstack([left_point, right_point])
    valuation_sequence_indices = np.random.choice(2, size=T)
    valuation_sequence = points[valuation_sequence_indices]

    # Rescale the valuation sequence to be inside the order book
    scaling_factors = (contexts_for_rescaling_vals[:, 1] - contexts_for_rescaling_vals[:, 0])
    rescaled_s = contexts_for_rescaling_vals[:, 0] + valuation_sequence[:, 0] * scaling_factors
    rescaled_b = contexts_for_rescaling_vals[:, 1] - (1 - valuation_sequence[:, 1]) * scaling_factors
    valuation_sequence = np.vstack([rescaled_s, rescaled_b]).T

    return actual_contexts, valuation_sequence


def construct_Lsq_logT_lower_bound(T:int, Lipschitz_constant:float, *args, **kwargs)\
    ->Tuple[np.ndarray, np.ndarray]:
    """
    This function constructs a sequence of valuations and a sequence of contexts,
    such that the valuations are Lipschitz continuous with respect to the contexts.
    This time we don't have fake or actual contexts. We only have the actual contexts.
    This construction would give a regret lower bound of O(L^2 log(T/L^2)).
    """
    # The contexts are:
    # (0, 1), 
    # (0, 0.5), (0.5, 1), 
    # (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1),
    # ...
    # However if the Lipschitz constant allows it,
    # we can create M^2 contexts in the triangle between parent and two children.
    # Assuming the distance between two points is the l1 norm:
    if Lipschitz_constant < 4:
        raise ValueError("The Lipschitz constant is too small to construct the sequence.")
    M = int(Lipschitz_constant/4)

    # Initialize the list of contexts:
    actual_contexts = [np.array([0, 1])] # root context

    # Construct the contexts
    counter = 1
    queue = [(0, 1, 1)] # (x, y, i) where i is the level of the tree

    # BFS
    while counter < T:
        # Take the first element from the queue
        # Create the children and append them to the queue
        parent = queue.pop(0)
        x, y, i = parent
        left_child = (x, y - 1/2**i, i+1)
        right_child = (x + 1/2**i, y, i+1)
        queue.append(left_child)
        queue.append(right_child)

        # Create the M^2 contexts in the upper left triangle
        # between the parent and the two children
        # and append them to the list of contexts
        m = 0
        while m < M+1 and counter < T:
            n = m
            while n < M+1 - m and counter < T:
                if m == 0 and n == 0: # Skip the parent context since it is already in the list
                    n += 1
                    continue

                else:
                    x = parent[0] + n/(M*2**i)
                    y = parent[1] - m/(M*2**i)
                    actual_contexts.append(np.array([x, y]))
                    n += 1
                    counter += 1
            m += 1

    # Convert the list to numpy array
    actual_contexts = np.array(actual_contexts)

    # The valuations are [0, 1/4] and [3/4, 1] at random
    left_point = np.array([0, 1/4])
    right_point = np.array([3/4, 1])
    points = np.vstack([left_point, right_point])
    valuation_sequence_indices = np.random.choice(2, size=T)
    valuation_sequence = points[valuation_sequence_indices]

    # Rescale the valuation sequence to be inside the order book
    scaling_factors = (actual_contexts[:, 1] - actual_contexts[:, 0])
    rescaled_s = actual_contexts[:, 0] + valuation_sequence[:, 0] * scaling_factors
    rescaled_b = actual_contexts[:, 1] - (1 - valuation_sequence[:, 1]) * scaling_factors
    valuation_sequence = np.vstack([rescaled_s, rescaled_b]).T

    return actual_contexts, valuation_sequence