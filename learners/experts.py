import numpy as np

class Hedge:
    def __init__(self, experts:list, T:int):
        self.experts = experts
        self.T = T
        self.no_experts = len(experts)
        self.weights = np.ones(self.no_experts) / self.no_experts
        self.epsilon = np.sqrt(np.log(self.no_experts) / T)

    def choose_action(self):
        expert_index = np.random.choice(self.no_experts, p=self.weights)
        return self.experts[expert_index]
        
    def update_weights(self, hidden_s, hidden_b):
        for i, (p, q) in enumerate(self.no_experts):
            if hidden_s <= p and q <= hidden_b:
                loss = max(0, hidden_s - hidden_b)
            else:
                loss = max(0, hidden_b - hidden_s)

            self.weights[i] *= np.exp(-self.epsilon * loss)


class GFTMax:
    """
    Use this class only once for each run.
    """
    def __init__(self, T, environment):
        self.budget = 0
        self.budget_threshold = ... # Some function of T
        self.K = ... # Some function of T
        self.turn = 0
        self.T = T
        self.environment = environment
        self.run_profit_max = True
        self.price_grid_F:list = self.create_multiplicative_grid(self.K)
        self.price_grid_H:list = self.create_additive_grid(self.K)
        self.hedge_profit = Hedge(self.price_grid_F, self.T)
        self.hedge_gft = Hedge(self.price_grid_H, self.T)

    def create_multiplicative_grid(self, K):
        pass

    def create_additive_grid(self, K):
        pass

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
        # Pass it to the environment
        # Get the feedback and the max GFT
        # Calculate loss
        # Update weights
        if self.budget >= self.budget_threshold:
            self.run_profit_max = False
        pass

    def gft_max(self, i):
        # Decide action
        # Pass it to the environment
        # Get the feedback and the max GFT
        # Calculate loss
        # Update weights
        pass
    