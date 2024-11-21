import numpy as np
import pandas as pd
from .base import DummyEnvironment
from typing import Union

class OrderBookEnvironment(DummyEnvironment):
    def __init__(self, T:int, order_book:Union[np.ndarray, None]=None, valuation_sequence:Union[np.ndarray, None]=None)->None:
        """
        This class adds the order book to the DummyEnvironment.
        The order book is a sequence of constraints on the private valuations.
        It is best to input the order book and the valuation sequence as numpy arrays.
        """
        self.T = T
        if order_book is None:
            self.order_book = self.construct_order_book()
        else:
            self.order_book = order_book
        if valuation_sequence is None:
            self.valuation_sequence = self.construct_valuation_sequence()
        else:
            self.valuation_sequence = valuation_sequence

    def construct_order_book(self)->np.ndarray:
        # Sample at random points (0, 1/4), (3/4, 1) from a bernoulli distribution
        left_point = np.array([0, 1/4])
        right_point = np.array([3/4, 1])
        points = np.vstack([left_point, right_point])
        order_book_indices = np.random.choice(2, size=self.T)
        order_book = points[order_book_indices]
        return order_book

    def construct_valuation_sequence(self)->np.ndarray:
        # Sample at random points (0, 1/4), (3/4, 1) from a bernoulli distribution
        # and rescale them to be inside the constraints sequence
        left_point = np.array([0, 1/4])
        right_point = np.array([3/4, 1])
        points = np.vstack([left_point, right_point])
        valuation_sequence_indices = np.random.choice(2, size=self.T)
        valuation_sequence = points[valuation_sequence_indices]

        # Rescale the valuation sequence to be inside the order book
        scaling_factors = (self.order_book[:, 1] - self.order_book[:, 0])**2
        rescaled_s = self.order_book[:, 0] + valuation_sequence[:, 0] * scaling_factors
        rescaled_b = self.order_book[:, 1] - (1 - valuation_sequence[:, 1]) * scaling_factors
        valuation_sequence = np.vstack([rescaled_s, rescaled_b]).T

        return valuation_sequence

    def get_constraints(self, index:int)->np.ndarray:
        """
        This function returns the constraints at the given time.
        """
        return self.order_book[index]
    
    def get_policy_gft_having_adhoc_valuations(self)->float:
        """
        This function calculates the policy regret when the valuations are deterministic
        and already account for the Lipschitnezz of the policy.
        So we just need to sum the GFT of each turn. 
        BEWARE: we assume that the valuations have the constraint that b >= s.
        """
        return np.sum(self.valuation_sequence[:, 1] - self.valuation_sequence[:, 0])
    
    def get_policy_gft(self)->Union[dict, float]:
        """
        This function computes the aggregated reward on the unit square and returns it.
        We assume that the policy is able to follow precisely the valuation sequence 
        (Lipschitz constraints are the same for the policy and for the valuation sequence,
        the best expert in each context is also the output of the optimal policy).
        """
        # For each context, the policy chooses the best price pair
        # Work with pandas for easier sorting
        df = pd.DataFrame(np.hstack([self.order_book, self.valuation_sequence]), columns=['s_dot', 'b_dot', 's', 'b'])

        cumulative_max_reward = 0
        max_reward_tuples = dict()

        # For each context:
        for context, context_df in df.groupby(['s_dot', 'b_dot']):
            
            # Drop the context columns
            context_df = context_df.drop(columns=['s_dot', 'b_dot']) 
            
            # Sort the price pairs by b in descending order and s in ascending order
            sorted_df = context_df.sort_values(by=['b', 's'], ascending=[False, True]) # might be redundant

            # Eliminate all the price pairs with b <= s
            filtered_df = sorted_df[sorted_df['b'] >= sorted_df['s']]
            # Group by b
            grouped = filtered_df.groupby('b')

            # Get sorted group keys in descending order
            sorted_keys = sorted(grouped.groups.keys(), reverse=True)

            prev_curr_rewards = []
            max_reward = 0
            max_reward_tuple = (0, 0)

            for b in sorted_keys:
                group = grouped.get_group(b)
                reward_index = 0
                baseline_reward = 0
                reward_max_index = len(prev_curr_rewards) - 1 
                curr_rewards = []

                # Iterate through the s values
                for s in group['s']:
                    if prev_curr_rewards: # if the list is not empty
                        # Swipe through prev_s < s
                        while reward_index <= reward_max_index:
                            s_prev, reward_prev = prev_curr_rewards[reward_index]
                            if s > s_prev:
                                new_reward = reward_prev + baseline_reward
                                # Append the curr_rewards before s
                                curr_rewards.append((s_prev, new_reward))
                                reward_index += 1
                                # Update max reward
                                if new_reward > max_reward:
                                    max_reward = new_reward
                                    max_reward_tuple = (s_prev, b)
                            elif s == s_prev:
                                if reward_index > 0:
                                    baseline_reward += reward_prev - prev_curr_rewards[reward_index-1][1]
                                else:
                                    baseline_reward += reward_prev
                            else:
                                break

                    # Increment the baseline reward with the current s
                    baseline_reward += b-s
                    # If the current s is the same as the previous s, update the reward
                    if curr_rewards and s == curr_rewards[-1][0]:
                        curr_rewards[-1] = (s, baseline_reward)
                    else:
                        if prev_curr_rewards:
                            reward_s  = prev_curr_rewards[reward_index][1] + baseline_reward
                        else:
                            reward_s = baseline_reward
                        curr_rewards.append((s, reward_s))
                    # Update max reward
                    if reward_s > max_reward:
                        max_reward = reward_s
                        max_reward_tuple = (s, b)

                # Swipe through prev_s > s if prev_s < b
                if prev_curr_rewards:
                    while reward_index <= reward_max_index:
                        s_prev, reward_prev = prev_curr_rewards[reward_index]
                        if reward_index <= reward_max_index and s_prev <= b:
                            new_reward = reward_prev + baseline_reward
                            curr_rewards.append((s_prev, new_reward))
                            reward_index += 1
                            # Update max reward
                            if new_reward > max_reward:
                                max_reward = new_reward
                                max_reward_tuple = (s_prev, b)
                        else:
                            break
                
                # print(curr_rewards) # Debugging
                prev_curr_rewards = curr_rewards

            # Add the max reward to the cumulative max reward
            cumulative_max_reward += max_reward
            max_reward_tuples[tuple(context)] = max_reward_tuple

        return max_reward_tuples, cumulative_max_reward