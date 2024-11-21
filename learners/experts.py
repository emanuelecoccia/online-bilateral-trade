import numpy as np
from environments.base import DummyEnvironment
from typing import List
from tqdm import tqdm

class DummyAlgorithm:
    def __init__(self, T:int, environment:DummyEnvironment)->None:
        self.T = T
        self.environment = environment

    def run(self)->None:
        pass

    def get_final_gft(self)->float:
        return 0

class Hedge:
    def __init__(self, experts:list, T:int):
        self.experts = np.array(experts)
        self.T = T
        self.no_experts = len(experts)
        self.weights = np.ones(self.no_experts) / self.no_experts
        self.epsilon = np.sqrt(np.log(self.no_experts) / T)
        self.gft = np.zeros(self.no_experts)

    def choose_action(self):
        probabilities = self.weights/np.sum(self.weights)
        expert_index = np.random.choice(self.no_experts, p=probabilities)
        return self.experts[expert_index]
        
    def update_weights(self, hidden_s, hidden_b):

        indicator_s = hidden_s <= self.experts[:, 0]
        indicator_b = self.experts[:, 1] <= hidden_b
        gft_diff = hidden_b - hidden_s

        gft = (indicator_s*indicator_b) * gft_diff

        if hidden_b >= hidden_s:
            gft_losses = (1 - (indicator_s*indicator_b)) * gft_diff
        else:
            gft_losses = gft

        self.weights *= np.exp(-self.epsilon * gft_losses)
        self.gft += gft

    def update_weights_with_rescaling(self, hidden_s, hidden_b, s_dot, b_dot):
        # Rescale the experts (actions)
        rescaling_factor = (b_dot - s_dot)**2
        p = s_dot + self.experts[:, 0] * rescaling_factor
        q = b_dot - (1 - self.experts[:, 1]) * rescaling_factor

        indicator_s = hidden_s <= p
        indicator_b = q <= hidden_b
        gft_diff = hidden_b - hidden_s

        gft = (indicator_s*indicator_b) * gft_diff

        if hidden_b >= hidden_s:
            gft_losses = (1 - (indicator_s*indicator_b)) * gft_diff
        else:
            gft_losses = gft

        self.weights *= np.exp(-self.epsilon * gft_losses)
        self.gft += gft

    def get_best_expert(self):
        best_expert_index = np.argmax(self.gft)
        return self.experts[best_expert_index], self.gft[best_expert_index]

class GFTMax(DummyAlgorithm):
    """
    Use this class only once for each run.
    """
    def __init__(self, T:int, environment:DummyEnvironment)->None:
        self.budget = 0
        self.gft = 0
        self.budget_threshold = np.sqrt(T)
        self.K = int(np.sqrt(T))
        self.T = T
        self.environment = environment
        self.run_profit_max = True
        self.price_grid_F:List = self.create_multiplicative_grid(self.K)
        self.price_grid_H:List = self.create_additive_grid(self.K)
        self.hedge_profit = Hedge(self.price_grid_F, self.T)
        self.hedge_gft = Hedge(self.price_grid_H, self.T)

    def create_multiplicative_grid(self, K):
        grid = []
        for k in range(K+1):
            g_k = k/K
            grid.append((g_k, g_k))
            for i in range(int(np.log(K)+1)):
                if g_k - 2**-i >= 0:
                    grid.append((g_k - 2**-i, g_k))
                if g_k + 2**-i <= 1:
                    grid.append((g_k, g_k + 2**-i))
        return grid

    def create_additive_grid(self, K):
        grid = []
        for i in range(K):
            grid.append(((i + 1)/K, i/K))
        return grid

    def run(self):
        for i in range(self.T):
            # Budget accumulation
            if self.run_profit_max:
                self.profit_max(i)
            # GFT maximization
            else:
                self.gft_max(i)

    def profit_max(self, i):
        # Decide action
        action = self.hedge_profit.choose_action()
        # Get valuations from the environment
        feedback = self.environment.get_valuations(i)
        # Update weights
        self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)
        if self.budget >= self.budget_threshold:
            self.run_profit_max = False

    def gft_max(self, i):
        # Decide action
        action = self.hedge_gft.choose_action()
        # Get valuations from the environment
        feedback = self.environment.get_valuations(i)
        # Update weights (also of the profit hedge for determining the best expert)
        self.hedge_gft.update_weights(feedback[0], feedback[1])
        self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)

    def update_budget(self, action, feedback):
        if feedback[0] <= action[0] and action[1] <= feedback[1]:
            self.budget += action[1] - action[0]

    def update_gft(self, action, feedback):
        if feedback[0] <= action[0] and action[1] <= feedback[1]:
            self.gft += feedback[1] - feedback[0]

    def get_final_gft(self):
        return self.gft
    

class ConstrainedGFTMax(GFTMax):
    def __init__(self, T, environment):
        super().__init__(T, environment)

    def rescale_action(self, action, s_dot, b_dot):
        # Add rescaling
        rescaling_factor = (b_dot - s_dot)**2
        p = s_dot + action[0] * rescaling_factor
        q = b_dot - (1 - action[1]) * rescaling_factor
        return np.array([p, q])
        
    def profit_max(self, i):
        # Decide action
        action = self.hedge_profit.choose_action()
        # Get valuations from the environment
        feedback = self.environment.get_valuations(i)
        # Get constraints from the environment
        s_dot, b_dot = self.environment.get_constraints(i)
        # If action is outside the constraints, rescale actions
        if action[0] < s_dot or b_dot < action[1]:
            action = self.rescale_action(action, s_dot, b_dot)
            self.hedge_profit.update_weights_with_rescaling(feedback[0], feedback[1], s_dot, b_dot)
        else:
            # Update weights
            self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)
        if self.budget >= self.budget_threshold:
            self.run_profit_max = False

    def gft_max(self, i):
        # Decide action
        action = self.hedge_gft.choose_action()
        # Get valuations from the environment
        feedback = self.environment.get_valuations(i)
        # Get constraints from the environment
        s_dot, b_dot = self.environment.get_constraints(i)
        # If action is outside the constraints, rescale actions
        if action[0] < s_dot or b_dot < action[1]:
            action = self.rescale_action(action, s_dot, b_dot)
            self.hedge_gft.update_weights_with_rescaling(feedback[0], feedback[1], s_dot, b_dot)
            self.hedge_profit.update_weights_with_rescaling(feedback[0], feedback[1], s_dot, b_dot)
        else:
            # Update weights
            self.hedge_gft.update_weights(feedback[0], feedback[1])
            self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)


class EstimateDeterministicLipschitzValuations(DummyAlgorithm):
    def __init__(self, T:int, environment:DummyEnvironment, L:float)->None:
        self.T = T
        self.environment = environment
        self.L = L
        self.gft = 0

    def get_final_gft(self):
        return self.gft

    def run(self):
        # For the first round, just guess (0.5, 0.5)
        feedback = self.environment.get_valuations(0)
        if feedback[0] <= 0.5 and 0.5 <= feedback[1]:
            self.gft += feedback[1] - feedback[0]

        for i in tqdm(range(1, self.T)):
            constraints = self.environment.get_constraints(i)
            # Find the distance of previous constraints to the current constraints
            past_constraints = self.environment.order_book[:i]
            distances = np.sqrt(np.sum((past_constraints - constraints)**2, axis = 1))
            # Retrieve the past valuations and create upper and lower bounds
            past_valuations = self.environment.valuation_sequence[:i]
            lower_bound_s = np.max(past_valuations[:, 0] - self.L * distances)
            upper_bound_s = np.min(past_valuations[:, 0] + self.L * distances)
            lower_bound_b = np.max(past_valuations[:, 1] - self.L * distances)
            upper_bound_b = np.min(past_valuations[:, 1] + self.L * distances)
            # Take the point in the middle of the bounds, whether they cross or not
            p = (upper_bound_s + lower_bound_b) / 2
            q = p
            # Get valuations from the environment
            feedback = self.environment.get_valuations(i)
            # Update gft
            if feedback[0] <= p and q <= feedback[1]:
                self.gft += feedback[1] - feedback[0]
            
            # Check if the valuations are within the bounds
            if feedback[0] < lower_bound_s:
                print("Valuation s is below the lower bound")
            if upper_bound_s < feedback[0]:
                print("Valuation s is above the upper bound")
            if feedback[1] < lower_bound_b:
                print("Valuation b is below the lower bound")
            if upper_bound_b < feedback[1]:
                print("Valuation b is above the upper bound")