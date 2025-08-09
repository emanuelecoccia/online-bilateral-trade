import numpy as np
from learners.experts import BaseAlgorithm
from environments.contextual import ContextualEnvironment
from tqdm import tqdm
from utils.data_structures import NTreeConquerDivide, NNodeConquerDivide
from math import floor, ceil
import random

class ConquerAndDivide(BaseAlgorithm):
    def __init__(self, T: int, d: int, L: float, environment: ContextualEnvironment) -> None:
        self.T = T
        self.d = d
        self.L = L
        self.environment = environment
        
        self.gft: float = 0
        self.K = ceil(T**(1/d))
        self.context_tree = NTreeConquerDivide(d=self.d, K=self.K)
        

    def run(self) -> None:
        for i in tqdm(range(self.T)):
            context: np.ndarray = self.environment.get_context(i)
            node = self.context_tree.get_node(context=context)
            if node.is_routine_one_on:
                self.reduce_feasible_area(i, node)
            else:
                self.blind_guess(i, node)

    def reduce_feasible_area(self, i: int, node: NNodeConquerDivide) -> None:
        # First visit initialization
        if node.current_index_routine_one is None:
            self.initialize_node(node)

        # Play
        p: float = node.current_index_routine_one / self.K
        action = np.array([p, p])
        feedback: bool = self.environment.get_one_bit_feedback(i, action)
        # If we receive positive feedback we ignore this turn, else we update the turn
        if feedback:
            self.gft += self.environment.get_turn_gft(i)
        else:
            if node.current_index_routine_one == node.price_boundary_index_b:
                node.is_routine_one_on = False # switch to routine 2
            else:
                node.current_index_routine_one += 1

    def blind_guess(self, i: int, node: NNodeConquerDivide) -> None:
        p_index = random.randint(node.price_boundary_index_a, node.price_boundary_index_b)
        p: float = p_index / self.K
        action = np.array([p, p])
        feedback: bool = self.environment.get_one_bit_feedback(i, action)
        if feedback:
            self.gft += self.environment.get_turn_gft(i)
            node.p_solution = p

    def initialize_node(self, node: NNodeConquerDivide) -> None:
        p_solution = node.parent.p_solution
        parent_boundaries = node.parent.boundaries
        max_context_distance = parent_boundaries[0, 1] - parent_boundaries[0, 0]
        h = self.L * max_context_distance
        price_boundary_a = max(0, p_solution - h)
        price_boundary_b = min(1, p_solution + h)
        node.price_boundary_index_a = max(floor(price_boundary_a * self.K), node.parent.price_boundary_index_a)
        node.price_boundary_index_b = min(ceil(price_boundary_b * self.K), node.parent.price_boundary_index_b)
        node.current_index_routine_one = node.price_boundary_index_a

    def get_final_gft(self) -> float:
        return self.gft
