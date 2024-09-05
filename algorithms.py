import numpy as np
from typing import Tuple

class EllipsoidPricing:
    """
    This is the parent class for the EllipsoidPricing algorithms.
    We assume that the uncertainty sets are bounded by the ellipsoid of center 0 and matrix I.

    Attributes:
    epsilon: float, determines the exploration-exploitation trade-off.
    d: int, the dimension of the feature space.
    E_center_s: np.array, the center of the ellipsoid for the uncertainty set of the seller's valuation.
    E_matrix_s: np.array, the matrix of the ellipsoid for the uncertainty set of the seller's valuation.
    E_center_b: np.array, the center of the ellipsoid for the uncertainty set of the buyer's valuation.
    E_matrix_b: np.array, the matrix of the ellipsoid for the uncertainty set of the buyer's valuation.
    s_floor: float, the lower bound of the uncertainty set of the seller's valuation.
    s_ceil: float, the upper bound of the uncertainty set of the seller's valuation.
    b_floor: float, the lower bound of the uncertainty set of the buyer's valuation.
    b_ceil: float, the upper bound of the uncertainty set of the buyer's valuation.
    beta_s: float, the radius of the ellipsoid for the seller's valuation.
    beta_b: float, the radius of the ellipsoid for the buyer's valuation.
    x: np.array, the feature vector at time t.
    update_s: bool, whether we need to update the ellipsoid for the seller's valuation.
    update_b: bool, whether we need to update the ellipsoid for the buyer's valuation.
    budget: float, the total revenues accumulated so far.
    """
    def __init__(self, epsilon:float, d:int)->None:
        self.epsilon:float = epsilon
        self.d:float = d
        self.E_center_s:np.array = np.zeros(d)
        self.E_matrix_s:np.array= np.eye(d)
        self.E_center_b:np.array = np.zeros(d)
        self.E_matrix_b:np.array = np.eye(d)
        self.s_floor = None
        self.s_ceil = None
        self.b_floor = None
        self.b_ceil = None
        self.beta_s = None
        self.beta_b = None
        self.x = None
        self.update_s:bool = False
        self.update_b:bool = False
        self.budget:float = 0

        
    def update_ellipsoids(self, feedback_s:bool, feedback_b:bool)->None:
        """
        We update the ellipsoids separately based on the two-bits feedback.
        """
        if self.update_s:
            gamma_s = self.E_matrix_s @ self.x / self.beta_s
            if feedback_s:
                self.E_center_s = self.E_center_s - 1/(self.d + 1) * gamma_s
            else:
                self.E_center_s = self.E_center_s + 1/(self.d + 1) * gamma_s
            self.E_matrix_s = self.d**2/(self.d**2 - 1) * (self.E_matrix_s - 2/(self.d + 1) * np.outer(gamma_s, gamma_s))

        if self.update_b:
            gamma_b = self.E_matrix_b @ self.x / self.beta_b
            if feedback_b:
                self.E_center_b = self.E_center_b + 1/(self.d + 1) * gamma_b
            else:
                self.E_center_b = self.E_center_b - 1/(self.d + 1) * gamma_b
            self.E_matrix_b = self.d**2/(self.d**2 - 1) * (self.E_matrix_b - 2/(self.d + 1) * np.outer(gamma_b, gamma_b))

    def update_budget(self, revenues:float)->None:
        """
        Updates the cumulative revenues.
        """
        self.budget += revenues


class EllipsoidPricingSBB(EllipsoidPricing):
    def __init__(self, epsilon, d):
        super().__init__(epsilon, d)

    def get_price(self, x:np.array)->Tuple[float, float]:
        """
        This function calculates and returns the prices for the seller and the buyer.
        """
        self.x = x
        self.beta_s = np.sqrt(np.dot(np.dot(self.x, self.E_matrix_s), self.x))
        self.beta_b = np.sqrt(np.dot(np.dot(self.x, self.E_matrix_b), self.x))

        self.s_floor = x.T @ self.E_center_s - self.beta_s
        self.s_ceil = x.T @ self.E_center_s + self.beta_s
        self.b_floor = x.T @ self.E_center_b - self.beta_b
        self.b_ceil = x.T @ self.E_center_b + self.beta_b

        s_mid = (self.s_ceil + self.s_floor) / 2
        b_mid = (self.b_ceil + self.b_floor) / 2
        s_b_mid = (self.s_ceil + self.b_floor) / 2

        if self.s_ceil < self.b_floor:
            self.update_s = False
            self.update_b = False
            return s_b_mid, s_b_mid
        elif self.s_ceil - self.s_floor >= self.epsilon:
            self.update_s = True
            self.update_b = False
            return s_mid, s_mid
        elif self.b_ceil - self.b_floor >= self.epsilon:
            self.update_s = False
            self.update_b = True
            return b_mid, b_mid
        else:
            self.update_s = False
            self.update_b = False
            return s_b_mid, s_b_mid


class EllipsoidPricingWBB(EllipsoidPricing):
    def __init__(self, epsilon, d):
        super().__init__(epsilon, d)

    def get_price(self, x:np.array)->Tuple[float, float]:
        """
        This function calculates and returns the prices for the seller and the buyer.
        """
        self.x = x
        self.beta_s = np.sqrt(np.dot(np.dot(self.x, self.E_matrix_s), self.x))
        self.beta_b = np.sqrt(np.dot(np.dot(self.x, self.E_matrix_b), self.x))

        self.s_floor = x.T @ self.E_center_s - self.beta_s
        self.s_ceil = x.T @ self.E_center_s + self.beta_s
        self.b_floor = x.T @ self.E_center_b - self.beta_b
        self.b_ceil = x.T @ self.E_center_b + self.beta_b

        s_mid = (self.s_ceil + self.s_floor) / 2
        b_mid = (self.b_ceil + self.b_floor) / 2
        s_b_mid = (self.s_ceil + self.b_floor) / 2

        if self.s_ceil < self.b_floor:
            self.update_s = False
            self.update_b = False
            return self.s_ceil, self.b_floor
        elif self.s_ceil - self.s_floor >= self.epsilon and\
                self.b_ceil - self.b_floor >= self.epsilon and\
                b_mid - s_mid >= 0:
            self.update_s = True
            self.update_b = True
            return s_mid, b_mid
        elif self.s_ceil - self.s_floor >= self.epsilon:
            self.update_s = True
            self.update_b = False
            return s_mid, s_mid
        elif self.b_ceil - self.b_floor >= self.epsilon:
            self.update_s = False
            self.update_b = True
            return b_mid, b_mid
        else:
            self.update_s = False
            self.update_b = False
            return s_b_mid, s_b_mid
        
class EllipsoidPricingSBBCuts(EllipsoidPricingSBB):
    def __init__(self, epsilon, d):
        super().__init__(epsilon, d)

    def update_ellipsoids_with_cuts(self, feedback_s:bool, feedback_b:bool, p, q)->None:
        """
        Now we always update the ellipsoids with shallow/deep cuts if possible.
        It is worth-noting that very shallow/deep cuts lead to numerical instability, 
        so we limit the depth of the cut to 0.5.
        Some key references:
        - Cohen et al. 2020 (shallow cuts)
        - Bland et al. 1981 (deep cuts)
        """
        if self.s_ceil - self.s_floor >= self.epsilon and p > self.s_floor and p < self.s_ceil:
            # Update the ellipsoid for the seller's valuation
            gamma_s = self.E_matrix_s @ self.x / self.beta_s
            # Calculate depth of the cut and update the center
            delta = (self.s_ceil + self.s_floor)/2 - p
            depth = - delta / self.beta_s

            if abs(depth) < 0.5: # We limit the depth of the cuts for numerical stability
                
                if depth <= 0 and feedback_s:
                    depth = -depth # deep cut, depth becomes positive
                    self.E_center_s = self.E_center_s - (1+self.d*depth)/(self.d+1) * gamma_s

                elif depth <= 0 and not feedback_s:
                    depth = depth # shallow cut, depth is already negative
                    self.E_center_s = self.E_center_s + (1+self.d*depth)/(self.d+1) * gamma_s
                elif depth > 0 and feedback_s:
                    depth = -depth # shallow cut, depth becomes negative
                    self.E_center_s = self.E_center_s - (1+self.d*depth)/(self.d+1) * gamma_s
                
                elif depth > 0 and not feedback_s:
                    depth = depth # deep cut, depth is already positive
                    self.E_center_s = self.E_center_s + (1+self.d*depth)/(self.d+1) * gamma_s

                # Update the matrix
                self.E_matrix_s = self.d**2/(self.d**2 - 1) * (1-depth**2) *\
                    (self.E_matrix_s - (2*(1+self.d*depth))/((self.d + 1)*(1+depth)) * np.outer(gamma_s, gamma_s))
            

        if self.b_ceil - self.b_floor >= self.epsilon and q < self.b_ceil and q > self.b_floor:
            # Update the ellipsoid for the buyer's valuation
            gamma_b = self.E_matrix_b @ self.x / self.beta_b
            # Calculate depth of the cut
            delta = (self.b_ceil + self.b_floor)/2 - q
            depth = - delta / self.beta_b

            if abs(depth) < 0.5: # We limit the depth of the cuts for numerical stability
                        
                if depth <= 0 and not feedback_b:
                    depth = -depth # deep cut, depth becomes positive
                    self.E_center_b = self.E_center_b - (1+self.d*depth)/(self.d+1) * gamma_b
                elif depth <= 0 and feedback_b:
                    depth = depth # shallow cut, depth is already negative
                    self.E_center_b = self.E_center_b + (1+self.d*depth)/(self.d+1) * gamma_b
                elif depth > 0 and not feedback_b:
                    depth = -depth # shallow cut, depth becomes negative
                    self.E_center_b = self.E_center_b - (1+self.d*depth)/(self.d+1) * gamma_b
                elif depth > 0 and feedback_b:
                    depth = depth # deep cut, depth is already positive
                    self.E_center_b = self.E_center_b + (1+self.d*depth)/(self.d+1) * gamma_b

                # Update the matrix
                self.E_matrix_b = self.d**2/(self.d**2 - 1) * (1-depth**2) *\
                    (self.E_matrix_b - (2*(1+self.d*depth))/((self.d + 1)*(1+depth)) * np.outer(gamma_b, gamma_b))
                

class EllipsoidPricingWBBCuts(EllipsoidPricingWBB, EllipsoidPricingSBBCuts):
    """
    This class combines the two previous classes and implements the WBB algorithm with cuts.
    """
    def __init__(self, epsilon, d):
        super().__init__(epsilon, d)