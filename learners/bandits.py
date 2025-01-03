import numpy as np
from learners.experts import BaseAlgorithm
from environments.contextual import ContextualEnvironment
from tqdm import tqdm

class TwoBitsEDLV(BaseAlgorithm):
    """
    This class implements the EDLV algorithm with a two-bits feedback.
    """
    def __init__(self, T:int, environment:ContextualEnvironment, L:float, *args, **kwargs)->None:
        self.T:int = T
        self.environment:ContextualEnvironment = environment
        self.L:float = L
        self.gft:float = 0
        # The histories keeps track of the lower and upper bounds on the valuations
        # Each time we receive a feedback, we slash each interval and keep one part
        self.history_s:np.ndarray = np.zeros((T, 2))
        self.history_b:np.ndarray = np.zeros((T, 2))
        # Set upper bound as 1
        self.history_s[:, 1] = 1
        self.history_b[:, 1] = 1

    def get_final_gft(self)->float:
        return self.gft
    
    def update_history(self, index:int, feedback:np.ndarray[float, float], action:np.ndarray[float, float])->None:
        """
        This function updates the history based on the feedback.
        We always post p = q (action).
        """
        # feedback[0] = s <= action[0] and feedback[1] = action[1] <= b
        if feedback[0]:
            self.history_s[index, 1] = action[0]
        else:
            self.history_s[index, 0] = action[0]
        if feedback[1]:
            self.history_b[index, 0] = action[1]
        else:
            self.history_b[index, 1] = action[1]
    
    def run(self)->None:
        """
        The algorithm works by estimating the bounds on the current hidden valuations 
        from the past contexts and feedbacks.
        For each new contexts, we estimate the bounds on the valuations.
        But then, we also update all the past bounds based on the feedback.
        """
        # For the first round, just guess the central point on the diagonal, given the context 
        # and update the current GFT based on the feedback
        context:np.ndarray[float, float] = self.environment.get_context(0)
        p = (context[0] + context[1]) / 2
        q = p
        action:np.ndarray[float, float] = np.array([p, q])
        feedback:np.ndarray[bool, bool] = self.environment.get_two_bits_feedback(0, action)
        if feedback[0] and feedback[1]:
            self.gft += self.environment.get_turn_gft(0)
        # Update the history
        self.history_s[0] = context
        self.history_b[0] = context
        self.update_history(0, feedback, action)

        # Iterate through the remaining rounds
        # We iterate through the remaining T-1 turns
        for i in tqdm(range(1, self.T)):
            context:np.ndarray[float, float] = self.environment.get_context(i)
            # Find the distance of previous context to the current context
            past_context:np.ndarray = self.environment.order_book[:i]
            distances:np.ndarray = np.sqrt(np.sum((past_context - context)**2, axis = 1))
            # Use history to create upper and lower bounds
            lower_bound_s:float = np.max(self.history_s[:i, 0] - self.L * distances)
            upper_bound_s:float = np.min(self.history_s[:i, 1] + self.L * distances)
            lower_bound_b:float = np.max(self.history_b[:i, 0] - self.L * distances)
            upper_bound_b:float = np.min(self.history_b[:i, 1] + self.L * distances)
            # Use the context to refine the bounds
            lower_bound_s = max(lower_bound_s, context[0])
            upper_bound_s = min(upper_bound_s, context[1])
            lower_bound_b = max(lower_bound_b, context[0])
            upper_bound_b = min(upper_bound_b, context[1])
            # Take the point in the middle of the bounds
            p:float = (upper_bound_s + lower_bound_b) / 2
            q:float = p
            action:np.ndarray[float, float] = np.array([p, q])
            # Get feedback from the environment
            feedback:np.ndarray[bool, bool] = self.environment.get_two_bits_feedback(i, action)
            if feedback[0] and feedback[1]:
                self.gft += self.environment.get_turn_gft(i) # Update gft
            # Update the history
            self.history_s[i] = [lower_bound_s, upper_bound_s]
            self.history_b[i] = [lower_bound_b, upper_bound_b]
            self.update_history(i, feedback, action)
                
            # Check if the valuations are within the bounds.
            # It would be really weird if they did, it could only be explained 
            # by some numerical rounding errors.
            valuations = self.environment.get_valuations(i)
            if valuations[0] < lower_bound_s:
                print("Valuation s is below the lower bound:")
                print(f"Hidden s:{valuations[0]}, lower bound: {lower_bound_s}\n")
            if upper_bound_s < valuations[0]:
                print("Valuation s is above the upper bound")
                print(f"Hidden s:{valuations[0]}, upper bound: {upper_bound_s}\n")
            if valuations[1] < lower_bound_b:
                print("Valuation b is below the lower bound")
                print(f"Hidden b:{valuations[1]}, lower bound: {lower_bound_b}\n")
            if upper_bound_b < valuations[1]:
                print("Valuation b is above the upper bound")
                print(f"Hidden b:{valuations[1]}, upper bound: {upper_bound_b}\n")

            # Update the past history based on the feedback
            # (with the feedback we updated history[i])
            lower_bounds_s = self.history_s[i, 0] - self.L * distances
            upper_bounds_s = self.history_s[i, 1] + self.L * distances
            lower_bounds_b = self.history_b[i, 0] - self.L * distances
            upper_bounds_b = self.history_b[i, 1] + self.L * distances
            self.history_s[:i, 0] = np.maximum(self.history_s[:i, 0], lower_bounds_s)
            self.history_s[:i, 1] = np.minimum(self.history_s[:i, 1], upper_bounds_s)
            self.history_b[:i, 0] = np.maximum(self.history_b[:i, 0], lower_bounds_b)
            self.history_b[:i, 1] = np.minimum(self.history_b[:i, 1], upper_bounds_b)
            

class OneBitEDLV(BaseAlgorithm):
    """
    This class implements the EDLV algorithm with a one-bit feedback.
    """
    def __init__(self, T:int, environment:ContextualEnvironment, L:float, *args, **kwargs)->None:
        self.T:int = T
        self.environment:ContextualEnvironment = environment
        self.L:float = L
        self.gft:float = 0
        # The histories keeps track of the lower and upper bounds on the valuations
        self.history_s:np.ndarray = np.zeros((T, 2))
        self.history_b:np.ndarray = np.zeros((T, 2))
        # Remember the contexts as well
        self.contexts:np.ndarray = np.zeros((T, 2))
        
        # Add intervals from negative feedback 
        self.negative_history_s:np.ndarray = np.zeros((T, 2))
        self.negative_history_b:np.ndarray = np.zeros((T, 2))
        self.negative_contexts:np.ndarray = np.zeros((T, 2))
        # Keep track of how many negative histories we filled
        self.negative_index:int = 0


    def get_final_gft(self)->float:
        return self.gft
    
    def double_array(self, array:np.ndarray)->np.ndarray:
        """
        This function increases the size of the array and fills it with zeros.
        We designed it to increase the size of the "negative" 
        history and context arrays when we run out of space.
        """
        new_array:np.ndarray = np.zeros(2*len(array))
        new_array[:len(array)] = array
        return new_array
    
    def update_histories(self, index:int, context:np.ndarray[float, float], 
                         feedback:bool, action:np.ndarray[float, float])->None:
        """
        This function updates the history based on the feedback.
        """
        # Update the contex
        self.contexts[index] = context
        # History is filled anyways
        self.history_s[index] = context
        self.history_b[index] = context

        # feedback = s <= action[0] and action[1] <= b
        if feedback:
            self.history_s[index, 1] = action[0]
            self.history_b[index, 0] = action[1]
        else:
            # Check that the arrays can contain the next two elements
            if len(self.negative_contexts) - 1 < self.negative_index + 2:
                # Double the size of the arrays
                self.negative_contexts = self.double_array(self.negative_contexts)
                self.negative_history_s = self.double_array(self.negative_history_s)
                self.negative_history_b = self.double_array(self.negative_history_b)

            # Split into two possibilities
            self.negative_contexts[index] = context
            self.negative_contexts[index+1] = context
            self.negative_history_s[index] = context[0], action[0]
            self.negative_history_s[index+1] = action[0], context[1]
            self.negative_history_b[index] = context[0], action[1]
            self.negative_history_b[index+1] = action[1], context[1]
            self.negative_index += 2
    
    def run(self)->None:
        """
        The algorithm works by estimating the bounds on the current hidden valuations 
        from the past contexts and feedbacks.
        For each new contexts, we estimate the bounds on the valuations.
        But then, we also update all the past bounds based on the feedback.
        """
        # For the first round, just guess the central point on the diagonal, given the context 
        # and update the current GFT based on the feedback
        context:np.ndarray[float, float] = self.environment.get_context(0)
        p = (context[0] + context[1]) / 2
        q = p
        action = np.array([p, q])
        feedback:bool = self.environment.get_one_bit_feedback(0, action)
        if feedback:
            self.gft += self.environment.get_turn_gft(0)

        # Update the histories
        self.update_histories(0, context, feedback, action)

        # Iterate through the remaining rounds
        # We iterate through the remaining T-1 turns
        for i in tqdm(range(1, self.T)):
            # Get the context
            context:np.ndarray[float, float] = self.environment.get_context(i)
            # Find the distance of previous context to the current context
            past_context:np.ndarray = self.contexts[:i]
            distances:np.ndarray = np.sqrt(np.sum((past_context - context)**2, axis = 1))
            # Use history to create upper and lower bounds
            lower_bound_s:float = np.max(self.history_s[:i, 0] - self.L * distances)
            upper_bound_s:float = np.min(self.history_s[:i, 1] + self.L * distances)
            lower_bound_b:float = np.max(self.history_b[:i, 0] - self.L * distances)
            upper_bound_b:float = np.min(self.history_b[:i, 1] + self.L * distances)
            # Use the context to refine the bounds
            lower_bound_s = max(lower_bound_s, context[0])
            upper_bound_s = min(upper_bound_s, context[1])
            lower_bound_b = max(lower_bound_b, context[0])
            upper_bound_b = min(upper_bound_b, context[1])
            # Take into account the intervals, if there are any
            if self.negative_index > 0:
                pass
                past_negative_context:np.ndarray = self.negative_contexts[:self.negative_index]
                negative_distances:np.ndarray = np.sqrt(np.sum((past_negative_context - context)**2, axis = 1))
                negative_lower_bounds_s:float = np.maximum(
                    self.negative_history_s[:self.negative_index, 0] - self.L * negative_distances,
                    np.zeros_like(negative_distances) + lower_bound_s
                    )
                negative_upper_bounds_s:float = np.minimum(
                    self.negative_history_s[:self.negative_index, 1] + self.L * negative_distances,
                    np.zeros_like(negative_distances) + upper_bound_s
                    )
                negative_lower_bounds_b:float = np.maximum(
                    self.negative_history_b[:self.negative_index, 0] - self.L * negative_distances,
                    np.zeros_like(negative_distances) + lower_bound_b
                    )
                negative_upper_bounds_b:float = np.minimum(
                    self.negative_history_b[:self.negative_index, 1] + self.L * negative_distances,
                    np.zeros_like(negative_distances) + upper_bound_b
                    )
                
                # Now these are all our intervals, indexed by the contexts. 
                # Put everything in a pandas dataframe?

            # Take the point in the middle of the bounds, whether they cross or not
            p:float = (upper_bound_s + lower_bound_b) / 2
            q:float = p
            action:np.ndarray[float, float] = np.array([p, q])
            # Get feedback from the environment
            feedback:bool = self.environment.get_one_bit_feedback(i, action)
            if feedback:
                self.gft += self.environment.get_turn_gft(i) # Update gft
            
            # Update the histories
            self.update_histories(i, context, feedback, action)
                
            # Check if the valuations are within the bounds.
            # It would be really weird if they did, it could only be explained 
            # by some numerical rounding errors.
            valuations = self.environment.get_valuations(i)
            if valuations[0] < lower_bound_s:
                print("Valuation s is below the lower bound:")
                print(f"Hidden s:{valuations[0]}, lower bound: {lower_bound_s}\n")
            if upper_bound_s < valuations[0]:
                print("Valuation s is above the upper bound")
                print(f"Hidden s:{valuations[0]}, upper bound: {upper_bound_s}\n")
            if valuations[1] < lower_bound_b:
                print("Valuation b is below the lower bound")
                print(f"Hidden b:{valuations[1]}, lower bound: {lower_bound_b}\n")
            if upper_bound_b < valuations[1]:
                print("Valuation b is above the upper bound")
                print(f"Hidden b:{valuations[1]}, upper bound: {upper_bound_b}\n")

            # Update the past histories based on the feedback
            lower_bounds_s = self.history_s[i, 0] - self.L * distances
            upper_bounds_s = self.history_s[i, 1] + self.L * distances
            lower_bounds_b = self.history_b[i, 0] - self.L * distances
            upper_bounds_b = self.history_b[i, 1] + self.L * distances
            self.history_s[:i, 0] = np.maximum(self.history_s[:i, 0], lower_bounds_s)
            self.history_s[:i, 1] = np.minimum(self.history_s[:i, 1], upper_bounds_s)
            self.history_b[:i, 0] = np.maximum(self.history_b[:i, 0], lower_bounds_b)
            self.history_b[:i, 1] = np.minimum(self.history_b[:i, 1], upper_bounds_b)
            self.negative_history_s[:self.negative_index, 0] = np.maximum(self.negative_history_s[:self.negative_index, 0], lower_bounds_s)
            self.negative_history_s[:self.negative_index, 1] = np.minimum(self.negative_history_s[:self.negative_index, 1], upper_bounds_s)
            self.negative_history_b[:self.negative_index, 0] = np.maximum(self.negative_history_b[:self.negative_index, 0], lower_bounds_b)
            self.negative_history_b[:self.negative_index, 1] = np.minimum(self.negative_history_b[:self.negative_index, 1], upper_bounds_b)