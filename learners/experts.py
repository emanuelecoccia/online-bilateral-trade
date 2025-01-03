import numpy as np
from environments.base import BaseEnvironment
from environments.contextual import ContextualEnvironment
from utils.data_structures import TwoDimensionalNode, TwoDimensionalTree
from tqdm import tqdm

class BaseAlgorithm:
    def __init__(self)->None:
        raise NotImplementedError

    def run(self)->None:
        raise NotImplementedError

    def get_final_gft(self)->float:
        raise NotImplementedError

class Hedge:
    def __init__(self, experts:list, T:int)->None:
        """
        This class implements the Hedge algorithm, suitable for bilateral trade.
        """
        self.experts:np.ndarray = np.array(experts)
        self.T:int = T
        self.no_experts:int = len(experts)
        self.weights:np.ndarray = np.ones(self.no_experts) / self.no_experts
        self.epsilon:float = np.sqrt(np.log(self.no_experts) / T)
        self.gft:np.ndarray = np.zeros(self.no_experts)

    def choose_action(self)->np.ndarray[float, float]:
        """
        We choose the action to take by sampling one expert according to their current weight.
        """
        probabilities:np.ndarray = self.weights/np.sum(self.weights)
        expert_index:int = np.random.choice(self.no_experts, p=probabilities)
        chosen_expert:np.ndarray[float, float] = self.experts[expert_index]
        return chosen_expert
        
    def update_weights(self, hidden_s:float, hidden_b:float)->None:
        """
        We update the weights based on the reward each expert would have missed.
        """
        # Calculate the (unrealized) Gain From Trade of each expert
        indicator_s:np.ndarray[bool] = hidden_s <= self.experts[:, 0]
        indicator_b:np.ndarray[bool] = self.experts[:, 1] <= hidden_b
        gft_diff:float = hidden_b - hidden_s
        gft:np.ndarray[float] = (indicator_s*indicator_b) * gft_diff

        # If the trade is accepted and the GFT is positive
        # Experts that would have missed out receive GFT as loss
        if hidden_b >= hidden_s:
            gft_losses:np.ndarray[float] = (1 - (indicator_s*indicator_b)) * gft_diff
        # If the trade is accepted and the GFT is negative
        # Experts that would have traded receive GFT as loss
        else:
            gft_losses:np.ndarray[float] = gft

        # Changing values in-place is bad practice, but none is here to stop me
        self.weights *= np.exp(-self.epsilon * gft_losses)
        self.gft += gft

    def get_best_expert(self)->tuple[np.ndarray[float, float], float]:
        """
        This function returns the expert which accumulated the most Gain From Trade
        """
        best_expert_index:int = np.argmax(self.gft)
        best_expert:np.ndarray = self.experts[best_expert_index], 
        best_expert_gft:float = self.gft[best_expert_index]
        return best_expert, best_expert_gft

class RescaleAndHedge(Hedge):
    """
    We use a different weight updating algorithm for this class:
    if the expert selected is outside the context (s_dot, b_dot),
    we "rescale" it inside the context.
    """

    def update_weights_with_rescaling(self, hidden_s:float, hidden_b:float, s_dot:float, b_dot:float)->None:
        """
        This method updates the weights, but first it rescales everything inside the context. 
        Also the losses become smaller because they are rescaled. 
        """
        # Rescale the experts (actions)
        rescaling_factor:float = (b_dot - s_dot)**2
        p:np.ndarray[float] = s_dot + self.experts[:, 0] * rescaling_factor
        q:np.ndarray[float] = b_dot - (1 - self.experts[:, 1]) * rescaling_factor

        # The following code replicates what is inside the "update_weights" method in the parent class
        # Calculate the (unrealized) Gain From Trade of each expert
        indicator_s:np.ndarray[bool] = hidden_s <= p
        indicator_b:np.ndarray[bool] = q <= hidden_b
        gft_diff:float = hidden_b - hidden_s
        gft:np.ndarray[float] = (indicator_s*indicator_b) * gft_diff

        # If the trade is accepted and the GFT is positive
        # Experts that would have missed out receive GFT as loss
        if hidden_b >= hidden_s:
            gft_losses:np.ndarray[float] = (1 - (indicator_s*indicator_b)) * gft_diff
        # If the trade is accepted and the GFT is negative
        # Experts that would have traded receive GFT as loss
        else:
            gft_losses:np.ndarray[float] = gft

        # Changing values in-place is really bad practice, but none is here to stop me
        self.weights *= np.exp(-self.epsilon * gft_losses)
        self.gft += gft
    
class GFTMax(BaseAlgorithm):
    """
    This class implements the GFTMax algorithm from Bernasconi et al. 2023. 
    Use a separate object for each run. 
    """
    def __init__(self, T:int, environment:BaseEnvironment)->None:
        self.budget:float = 0
        self.gft:float = 0
        self.budget_threshold:float = np.sqrt(T)
        self.K:int = int(np.sqrt(T))
        self.T:int = T
        self.environment:BaseEnvironment = environment
        self.run_profit_max:bool = True
        self.price_grid_F:list[tuple[float, float]] = self.create_multiplicative_grid(self.K)
        self.price_grid_H:list[tuple[float, float]] = self.create_additive_grid(self.K)
        self.hedge_profit:Hedge = Hedge(self.price_grid_F, self.T)
        self.hedge_gft:Hedge = Hedge(self.price_grid_H, self.T)

    def create_multiplicative_grid(self, K:int)->list[tuple[float, float]]:
        """
        Creates a grid with points exponentially distant from the main diagonal,
        both with respect to the s and the b directions.
        """
        grid:list[tuple[float, float]] = []
        for k in range(K+1):
            g_k:float = k/K
            grid.append((g_k, g_k))
            for i in range(int(np.log(K)+1)):
                if g_k - 2**-i >= 0:
                    grid.append((g_k - 2**-i, g_k))
                if g_k + 2**-i <= 1:
                    grid.append((g_k, g_k + 2**-i))
        return grid

    def create_additive_grid(self, K:int)->list[tuple[float, float]]:
        """
        Creates a grid with points placed slightly below the main diagonal.
        """
        grid:list[tuple[float, float]] = []
        for i in range(K):
            grid.append(((i + 1)/K, i/K))
        return grid
    
    def update_budget(self, action:np.ndarray[float, float], feedback:np.ndarray[float, float])->None:
        """
        This method is called to keep track of the budget.
        """
        if feedback[0] <= action[0] and action[1] <= feedback[1]:
            self.budget += action[1] - action[0]
        if self.budget < 0:
            raise ValueError("The budget is negative. This should not happen by design.")

    def update_gft(self, action:np.ndarray[float, float], feedback:np.ndarray[float, float])->None:
        """
        This method is called to keep track of the Gain From Trade.
        """
        if feedback[0] <= action[0] and action[1] <= feedback[1]:
            self.gft += feedback[1] - feedback[0]

    def run(self)->None:
        """
        Run method. Runs for T iterations.
        It calls profit_max - it uses the multiplicative grid -
        until the budget is accumulated. At that point it switches to gft_max,
        which uses the additive grid - loses budget but has less regret.
        """
        for i in range(self.T):
            # Budget accumulation
            if self.run_profit_max:
                self.profit_max(i)
            # GFT maximization
            else:
                self.gft_max(i)

    def profit_max(self, i:int)->None:
        """
        This algorithm maximizes the profit while guaranteeing sublinear regret.
        """
        # Decide action
        action:np.ndarray[float, float] = self.hedge_profit.choose_action()
        # Get valuations from the environment
        feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
        # Update weights of the hedge
        self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)
        # If the budget is over the threshold, we can start the GFT maximization phase
        if self.budget >= self.budget_threshold:
            self.run_profit_max = False

    def gft_max(self, i:int)->None:
        """
        This algorithm spends the budget to minimize regret.
        """
        # Decide action
        action:np.ndarray[float, float] = self.hedge_gft.choose_action()
        # Get valuations from the environment
        feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
        # Update weights (also of the profit hedge for determining the best expert)
        self.hedge_gft.update_weights(feedback[0], feedback[1])
        self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)

    def get_final_gft(self)->float:
        """
        This method returns the Gain From Trade accumulated during the run.
        """
        return self.gft
    
class ContextualGFTMax(GFTMax):
    def __init__(self, T:int, environment:ContextualEnvironment, *args, **kwargs)->None:
        """
        This version of GFTMax rescales the experts if they are outside the context. 
        """
        super().__init__(T, environment)
        self.hedge_profit:RescaleAndHedge = RescaleAndHedge(self.price_grid_F, self.T)
        self.hedge_gft:RescaleAndHedge = RescaleAndHedge(self.price_grid_H, self.T)
        self.environment:ContextualEnvironment # just type hint

    def rescale_action(self, action:tuple[float, float], s_dot:float, b_dot:float)->np.ndarray[float, float]:
        """
        This method rescales the selected action inside the context.
        """
        # Rescaling
        rescaling_factor:float = (b_dot - s_dot)**2
        p:float = s_dot + action[0] * rescaling_factor
        q:float = b_dot - (1 - action[1]) * rescaling_factor
        return np.array([p, q])
        
    def profit_max(self, i:int)->None:
        """
        Same method of the parent class, but this time if the action is outside the context,
        it gets rescaled. This method is called inside the run method (see parent class). 
        """
        # Decide action
        action:np.ndarray[float, float] = self.hedge_profit.choose_action()
        # Get valuations from the environment
        feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
        # Get context from the environment
        s_dot: float
        b_dot: float
        s_dot, b_dot = self.environment.get_context(i)
        # If action is outside the context, rescale actions
        if action[0] < s_dot or b_dot < action[1]:
            action:np.ndarray[float, float] = self.rescale_action(action, s_dot, b_dot)
            self.hedge_profit.update_weights_with_rescaling(feedback[0], feedback[1], s_dot, b_dot)
        else:
            # Update weights
            self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)
        # If the budget is over the threshold, we can start the GFT maximization phase
        if self.budget >= self.budget_threshold:
            self.run_profit_max = False

    def gft_max(self, i:int)->None:
        # Decide action
        action:np.ndarray[float, float] = self.hedge_gft.choose_action()
        # Get valuations from the environment
        feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
        # Get context from the environment
        s_dot:float
        b_dot:float
        s_dot, b_dot = self.environment.get_context(i)
        # If action is outside the context, rescale actions
        if action[0] < s_dot or b_dot < action[1]:
            action:np.ndarray[float, float] = self.rescale_action(action, s_dot, b_dot)
            # Update weights - hedge profit update is superfluous
            self.hedge_gft.update_weights_with_rescaling(feedback[0], feedback[1], s_dot, b_dot)
            self.hedge_profit.update_weights_with_rescaling(feedback[0], feedback[1], s_dot, b_dot)
        else:
            # Update weights - hedge profit update is superfluous
            self.hedge_gft.update_weights(feedback[0], feedback[1])
            self.hedge_profit.update_weights(feedback[0], feedback[1])
        # Update budget
        self.update_budget(action, feedback)
        # Update gft
        self.update_gft(action, feedback)


class EDLV(BaseAlgorithm):
    """
    Estimate Deterministic Lipschitz Valuations.
    This algorithm estimates a deterministic Lipschitz function 
    that maps contexts to hidden valuations.
    """
    def __init__(self, T:int, environment:ContextualEnvironment, L:float, *args, **kwargs)->None:
        self.T:int = T
        self.environment:ContextualEnvironment = environment
        self.L:float = L
        self.gft:float = 0

    def get_final_gft(self)->float:
        return self.gft

    def run(self)->None:
        """
        The algorithm works by estimating the bounds on the current hidden valuations 
        from the past contexts.
        """
        # For the first round, just guess the central point on the diagonal, given the context 
        # and update the current GFT based on the feedback
        context:np.ndarray[float, float] = self.environment.get_context(0)
        feedback:np.ndarray[float, float] = self.environment.get_valuations(0)
        action = (context[0] + context[1]) / 2
        if feedback[0] <= action and action <= feedback[1]:
            self.gft += feedback[1] - feedback[0]

        # We iterate through the remaining T-1 turns
        for i in tqdm(range(1, self.T)):
            context:np.ndarray[float, float] = self.environment.get_context(i)
            # Find the distance of previous context to the current context
            past_context:np.ndarray = self.environment.order_book[:i]
            distances:np.ndarray = np.sqrt(np.sum((past_context - context)**2, axis = 1))
            # Retrieve the past valuations and create upper and lower bounds
            past_valuations:np.ndarray = self.environment.valuation_sequence[:i]
            lower_bound_s:float = np.max(past_valuations[:, 0] - self.L * distances)
            upper_bound_s:float = np.min(past_valuations[:, 0] + self.L * distances)
            lower_bound_b:float = np.max(past_valuations[:, 1] - self.L * distances)
            upper_bound_b:float = np.min(past_valuations[:, 1] + self.L * distances)
            # Use the context to refine the bounds
            lower_bound_s = max(lower_bound_s, context[0])
            upper_bound_s = min(upper_bound_s, context[1])
            lower_bound_b = max(lower_bound_b, context[0])
            upper_bound_b = min(upper_bound_b, context[1])
            # Take the point in the middle of the bounds, whether they cross or not
            p:float = (upper_bound_s + lower_bound_b) / 2
            q:float = p
            # Get valuations from the environment
            feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
            # Update gft
            if feedback[0] <= p and q <= feedback[1]:
                self.gft += feedback[1] - feedback[0]
            
            # Check if the valuations are within the bounds.
            # It would be really weird if they did, it could only be explained 
            # by some numerical rounding errors.
            if feedback[0] < lower_bound_s:
                print("Valuation s is below the lower bound:")
                print(f"Hidden s:{feedback[0]}, lower bound: {lower_bound_s}\n")
            if upper_bound_s < feedback[0]:
                print("Valuation s is above the upper bound")
                print(f"Hidden s:{feedback[0]}, upper bound: {upper_bound_s}\n")
            if feedback[1] < lower_bound_b:
                print("Valuation b is below the lower bound")
                print(f"Hidden b:{feedback[1]}, lower bound: {lower_bound_b}\n")
            if upper_bound_b < feedback[1]:
                print("Valuation b is above the upper bound")
                print(f"Hidden b:{feedback[1]}, upper bound: {upper_bound_b}\n")

class FastEDLV(EDLV):
    """
    This class is a "hopefully" faster version of EDLV.
    """
    def run(self)->None:
        # Initialize a tree
        self.tree = TwoDimensionalTree()
        # Set the radius
        radius:float = 1/self.L
        # For the first round, just guess the central point on the diagonal, given the context 
        # and update the current GFT based on the feedback
        context:np.ndarray[float, float] = self.environment.get_context(0)
        feedback:np.ndarray[float, float] = self.environment.get_valuations(0)
        action = (context[0] + context[1]) / 2
        if feedback[0] <= action and action <= feedback[1]:
            self.gft += feedback[1] - feedback[0]

        # Store the data in the tree
        self.tree.insert(TwoDimensionalNode(context[0], context[1], feedback))

        # Iterate through the remaining turns
        for i in tqdm(range(1, self.T)):
            # Get the context
            context:np.ndarray[float, float] = self.environment.get_context(i)
            # Find the close contexts
            close_nodes = self.tree.query(context[0], context[1], radius, i)

            # If there are close nodes, we can estimate the bounds
            if close_nodes:
                # Find the distance of previous context to the current context
                distances:np.ndarray = np.array([node.temporary_distance["distance"] for node in close_nodes\
                                                        if node.temporary_distance["current_iteration"] == i])
                # Retrieve the past valuations
                valuations:np.ndarray = np.array([node.valuations for node in close_nodes])

                # Create upper and lower bounds
                lower_bound_s:float = np.max(valuations[:, 0] - self.L * distances)
                upper_bound_s:float = np.min(valuations[:, 0] + self.L * distances)
                lower_bound_b:float = np.max(valuations[:, 1] - self.L * distances)
                upper_bound_b:float = np.min(valuations[:, 1] + self.L * distances)

                # Use the context to refine the bounds
                lower_bound_s = max(lower_bound_s, context[0])
                upper_bound_s = min(upper_bound_s, context[1])
                lower_bound_b = max(lower_bound_b, context[0])
                upper_bound_b = min(upper_bound_b, context[1])

                # Take the point in the middle of the bounds, whether they cross or not
                p:float = (upper_bound_s + lower_bound_b) / 2
                q:float = p
                # Get valuations from the environment
                feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
                # Insert the new node
                self.tree.insert(TwoDimensionalNode(context[0], context[1], feedback))
                # Update gft
                if feedback[0] <= p and q <= feedback[1]:
                    self.gft += feedback[1] - feedback[0]
                
                # Check if the valuations are within the bounds.
                # It would be really weird if they did, it could only be explained 
                # by some numerical rounding errors.
                if feedback[0] < lower_bound_s:
                    print("Valuation s is below the lower bound:")
                    print(f"Hidden s:{feedback[0]}, lower bound: {lower_bound_s}\n")
                if upper_bound_s < feedback[0]:
                    print("Valuation s is above the upper bound")
                    print(f"Hidden s:{feedback[0]}, upper bound: {upper_bound_s}\n")
                if feedback[1] < lower_bound_b:
                    print("Valuation b is below the lower bound")
                    print(f"Hidden b:{feedback[1]}, lower bound: {lower_bound_b}\n")
                if upper_bound_b < feedback[1]:
                    print("Valuation b is above the upper bound")
                    print(f"Hidden b:{feedback[1]}, upper bound: {upper_bound_b}\n")

            else:
                # If there are no close nodes, we just guess the central point on the diagonal
                # and update the current GFT based on the feedback
                action = (context[0] + context[1]) / 2
                feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
                if feedback[0] <= action and action <= feedback[1]:
                    self.gft += feedback[1] - feedback[0]
                # Insert the new node
                self.tree.insert(TwoDimensionalNode(context[0], context[1], feedback))


class FastEDLV2(EDLV):
    """
    This is a faster version of FastEDLV.
    We only retrieve the closest neighbor instead of 
    all the neighbors within a certain radius.
    """
    def run(self)->None:
        # Initialize a tree
        self.tree = TwoDimensionalTree()
        # For the first round, just guess the central point on the diagonal, given the context 
        # and update the current GFT based on the feedback
        context:np.ndarray[float, float] = self.environment.get_context(0)
        feedback:np.ndarray[float, float] = self.environment.get_valuations(0)
        action = (context[0] + context[1]) / 2
        if feedback[0] <= action and action <= feedback[1]:
            self.gft += feedback[1] - feedback[0]
        
        # Store the data in the tree
        self.tree.insert(TwoDimensionalNode(context[0], context[1], feedback))

        # Iterate through the remaining turns
        # Iterate through the remaining turns
        for i in tqdm(range(1, self.T)):
            # Get the context
            context:np.ndarray[float, float] = self.environment.get_context(i)
            # Find the closest context
            closest_neigh:TwoDimensionalNode
            distance:float
            closest_neigh, distance = self.tree.find_nearest_neighbor(context[0], context[1])
            # Create upper and lower bounds
            lower_bound_s:float = closest_neigh.valuations[0] - self.L * distance
            upper_bound_s:float = closest_neigh.valuations[0] + self.L * distance
            lower_bound_b:float = closest_neigh.valuations[1] - self.L * distance
            upper_bound_b:float = closest_neigh.valuations[1] + self.L * distance

            # Use the context to refine the bounds
            lower_bound_s = max(lower_bound_s, context[0])
            upper_bound_s = min(upper_bound_s, context[1])
            lower_bound_b = max(lower_bound_b, context[0])
            upper_bound_b = min(upper_bound_b, context[1])

            # Take the point in the middle of the bounds, whether they cross or not
            p:float = (upper_bound_s + lower_bound_b) / 2
            q:float = p
            # Get valuations from the environment
            feedback:np.ndarray[float, float] = self.environment.get_valuations(i)
            # Insert the new node
            self.tree.insert(TwoDimensionalNode(context[0], context[1], feedback))
            # Update gft
            if feedback[0] <= p and q <= feedback[1]:
                self.gft += feedback[1] - feedback[0]

            # Check if the valuations are within the bounds.
            # It would be really weird if they did, it could only be explained 
            # by some numerical rounding errors.
            if feedback[0] < lower_bound_s:
                print("Valuation s is below the lower bound:")
                print(f"Hidden s:{feedback[0]}, lower bound: {lower_bound_s}\n")
            if upper_bound_s < feedback[0]:
                print("Valuation s is above the upper bound")
                print(f"Hidden s:{feedback[0]}, upper bound: {upper_bound_s}\n")
            if feedback[1] < lower_bound_b:
                print("Valuation b is below the lower bound")
                print(f"Hidden b:{feedback[1]}, lower bound: {lower_bound_b}\n")
            if upper_bound_b < feedback[1]:
                print("Valuation b is above the upper bound")
                print(f"Hidden b:{feedback[1]}, upper bound: {upper_bound_b}\n")