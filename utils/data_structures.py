import numpy as np
from typing_extensions import Self
class TwoDimensionalNode:
    """
    This class is used to represent a node in a two-dimensional tree.
    It stores the x and y coordinates of the node, the criterion used to split the tree,
    and the left and right children of the node.
    """
    def __init__(self, x:float, y:float, valuations:np.ndarray[float, float], left=None, right=None, criterion=None)->None:
        self.x = x
        self.y = y
        self.valuations = valuations
        self.left = left
        self.right = right
        self.criterion = criterion
        self.temporary_distance = dict() # contains current_iteration, distance


class TwoDimensionalTree:
    def __init__(self, root:TwoDimensionalNode=None)->None:
        """
        Implementation of a k-d tree for two dimensions.
        """
        self.root = root
        if self.root:
            self.root.criterion = "x" # The root is split by x

    def insert(self, node:TwoDimensionalNode)->None:
        """
        This function inserts a node into the tree.
        """
        # If the tree is empty, the node becomes the root
        # and the root is split by x
        if self.root is None:
            self.root = node
            self.root.criterion = "x"
            return
        
        # Otherwise, we traverse the tree to find the correct position
        current = self.root
        while True:
            if current.criterion == "x":
                if node.x <= current.x:
                    if current.left is None:
                        node.criterion = "y"
                        current.left = node
                        break
                    else:
                        current = current.left
                else: # node.x > current.x
                    if current.right is None:
                        node.criterion = "y"
                        current.right = node
                        break
                    else:
                        current = current.right
            else: # current.criterion == "y"
                if node.y <= current.y:
                    if current.left is None:
                        node.criterion = "x"
                        current.left = node
                        break
                    else:
                        current = current.left
                else: # node.y > current.y
                    if current.right is None:
                        node.criterion = "x"
                        current.right = node
                        break
                    else:
                        current = current.right

    def query(self, x:float, y:float, radius:float, current_iteration:int) -> list[TwoDimensionalNode]:
        """
        This function queries the tree for all nodes within a certain radius of a point (x, y).
        """
        result:list = [] # we pass a list to the recursive function since it is mutable
        self._query_recursive(self.root, x, y, radius, result, current_iteration)
        return result

    def _query_recursive(self, node, x, y, radius, result, current_iteration):
        if node is None:
            return

        # Compute squared distance between node and query point
        dx = node.x - x
        dy = node.y - y
        dist = abs(dx) + abs(dy)

        # If the node is within the radius, 
        # update its information and add it to the result
        if dist <= radius:
            node.temporary_distance["current_iteration"] = current_iteration
            node.temporary_distance["distance"] = dist
            result.append(node)

        # Arguments for the recursive calls
        args = [x, y, radius, result, current_iteration]

        # Determine whether to search left and/or right subtrees
        if node.criterion == 'x':
            plane_dist_abs = abs(node.x - x)
            # First, decide which side to explore first
            if x <= node.x:
                # Query point is to the left of the splitting plane
                self._query_recursive(node.left, *args)
                if plane_dist_abs <= radius:
                    self._query_recursive(node.right, *args)
            else:
                # Query point is to the right of the splitting plane
                self._query_recursive(node.right, *args)
                if plane_dist_abs <= radius:
                    self._query_recursive(node.left, *args)
        else:  # node.criterion == 'y'
            plane_dist_abs = abs(node.y - y)
            # First, decide which side to explore first
            if y <= node.y:
                # Query point is below the splitting plane
                self._query_recursive(node.left, *args)
                if plane_dist_abs <= radius:
                    self._query_recursive(node.right, *args)
            else:
                # Query point is above the splitting plane
                self._query_recursive(node.right, *args)
                if plane_dist_abs <= radius:
                    self._query_recursive(node.left, *args)

    def find_nearest_neighbor(self, x:float, y:float) -> tuple[TwoDimensionalNode, float]:
            """
            This function finds the nearest neighbor to a given point (x, y).
            """
            if self.root is None:
                return None

            best = {'node': None, 'distance': float('inf')}
            self._find_nearest_recursive(self.root, x, y, best)
            return best['node'], best['distance']
    
    def _find_nearest_recursive(self, node:TwoDimensionalNode, x:float, y:float, best:dict) -> None:
        """
        A recursive function to find the nearest neighbor.
        Args:
            node: The current node being visited.
            x, y: The query point coordinates.
            best: A dictionary containing the best node and the distance.
        """
        if node is None:
            return

        # Compute distance from current node to the target point
        dx = node.x - x
        dy = node.y - y
        dist = abs(dx) + abs(dy)

        # Update the best node if this one is closer
        if dist < best['distance']:
            best['node'] = node
            best['distance'] = dist

        # Determine which side to search first
        if node.criterion == 'x':
            direction = 'left' if x <= node.x else 'right'
        else:
            direction = 'left' if y <= node.y else 'right'

        # Recursively search the side of the splitting plane where the point lies
        next_branch = node.left if direction == 'left' else node.right
        opposite_branch = node.right if direction == 'left' else node.left

        self._find_nearest_recursive(next_branch, x, y, best)

        # Check if we need to search the other side of the splitting plane
        if node.criterion == 'x':
            plane_dist = abs(node.x - x)
        else:
            plane_dist = abs(node.y - y)

        if plane_dist < best['distance']:
            self._find_nearest_recursive(opposite_branch, x, y, best)

"""
Children are assumed to be organized in the following order:
[[2, 3],
[0, 1]].

The origin is on the bottom left.
"""
class Quadrant:
    def __init__(self, coordinates:list[float])->None:
        """
        Coordinates are x, y, w, h.
        """
        self._coordinates = coordinates

    @property
    def coordinates(self):
        return self._coordinates
    
    @property
    def x(self):
        return self._coordinates[0]
    
    @property
    def y(self):
        return self._coordinates[1]
    
    @property
    def w(self):
        return self._coordinates[2]
    
    @property
    def h(self):
        return self._coordinates[3]
    
    def create_subquadrant(self, child_id:int)->Self:
        """
        This function creates a quadrant object, given the child id.
        """

        if child_id == 0:
            child_coordinates = [self.x, self.y, self.w/2, self.h/2]
            
        elif child_id == 1:
            child_coordinates = [self.x + self.w/2, self.y, self.w/2, self.h/2]
        
        elif child_id == 2:
            child_coordinates = [self.x, self.y + self.h/2, self.w/2, self.h/2]
        
        elif child_id == 3:
            child_coordinates = [self.x + self.w/2, self.y + self.h/2, self.w/2, self.h/2]
        
        else:
            raise ValueError("The child_id value is not within the admissible range.")

        child_quadrant = Quadrant(child_coordinates)
        return child_quadrant

class QuadNode:
    def __init__(self, value, quadrant:Quadrant, parent)->None:
        self._value = value
        self._quadrant = quadrant
        self._parent = parent
        self._children = dict()

    @property
    def children(self):
        return self._children
    
    @property
    def parent(self):
        return self._parent
    
    @property
    def value(self):
        return self._value

    def get_siblings(self)->list[Self]:
        children_list = list(self.parent.children.values())
        return [child for child in children_list if child is not self]
    
class QuadTree:
    def __init__(self)->None:
        self.root = None

    def insert(self, value:np.ndarray)->None:
        assert isinstance(value, np.ndarray)
        if self.root is None:
            root_quadrant = Quadrant([0., 0., 1., 1.])
            self.root = QuadNode(value=value, quadrant=root_quadrant, parent=None)
        else:
            _aux_insert(self.root, value)

        def _aux_insert(parent, value):
            # First find in what quadrant the value should be
            def find_subquadrant_id(quadrant, value:np.ndarray):
                mid_x = (quadrant.x + quadrant.w/2)
                mid_y = (quadrant.y + quadrant.h/2)
                if value[0] < mid_x:
                    if value[1] < mid_y:
                        return 0
                    else:
                        return 2
                else:
                    if value[1] < mid_y:
                        return 1
                    else:
                        return 3
                    
            subquadrant_id = find_subquadrant_id(parent.quadrant, value)
            
            # If the quadrant is free, assign and return
            if subquadrant_id not in parent.children:
                subquadrant = parent.quadrant.create_subquadrant(subquadrant_id)
                parent.children[subquadrant_id] = QuadNode(value, subquadrant, parent)
                return

            # If the quadrant is occupied, do recursive call
            else:
                _aux_insert(parent.children[subquadrant_id], value)


"""
Let's generalize this based on the number of dimensions d.
We ought to build a tree that has 2^d branching factor, with d >= 1.
Each node corresponds to a space partition of its parent node. 
We split each dimension into two parts.
We want the data structure to have insertion properties -> traverse the tree until you find a free node. 
Each node has certain attributes or states that must be stored (override the class later).
"""

class NNode:
    def __init__(self, d: int, parent: Self | None, boundaries: np.ndarray) -> None:
        self.d = d
        self.parent = parent
        self.boundaries = boundaries
        self.children: dict[int, Self] = {}

    def get_child(self, coordinates: np.ndarray) -> Self:
        d = coordinates.shape[0]
        assert self.boundaries.shape == (d, 2)

        pivots = np.mean(self.boundaries, axis=1) # shape (d,)
        child_index = coordinates >= pivots # shape (d,)
        child_index_bits = int(''.join(map(str, child_index.astype(int))), 2)

        if child_index_bits in self.children:
            return self.children[child_index_bits]
        
        else:
            # Create node and add it to the dictionary 
            lower = np.where(child_index, pivots, self.boundaries[:, 0])
            upper = np.where(child_index, self.boundaries[:, 1], pivots)
            child_boundaries = np.stack([lower, upper], axis=1) # shape (d, 2)

            child_node = NNode(d=self.d, parent=self, boundaries=child_boundaries)
            self.children[child_index_bits] = child_node
            return child_node

class NTree:
    def __init__(self, d: int) -> None:
        self.d = d
        unit_boundaries = np.stack([np.zeros(self.d), np.ones(self.d)], axis=1) # shape (d, 2)
        self.root = NNode(d=self.d, parent=None, boundaries=unit_boundaries)


"""
Modified node for the algorithm "Conquer and Divide"
"""

class NNodeConquerDivide(NNode):
    def __init__(self, d: int, parent: Self | None, boundaries: np.ndarray) -> None:
        super().__init__(d=d, parent=parent, boundaries=boundaries)
        # Adding other attributes for keeping track of the two subroutines
        self.p_solution: float | None = None
        self.is_routine_one_on: bool = True
        self.price_boundary_index_a: int | None = None
        self.price_boundary_index_b: int | None = None
        self.current_index_routine_one: int | None = None

class NTreeConquerDivide(NTree):
    def __init__(self, d: int, K: int) -> None:
        self.d = d
        self.K = K
        unit_boundaries = np.stack([np.zeros(self.d), np.ones(self.d)], axis=1) # shape (d, 2)
        self.root = NNodeConquerDivide(d=self.d, parent=None, boundaries=unit_boundaries)
        self.root.price_boundary_index_a = 0
        self.root.price_boundary_index_b = self.K
        self.root.current_index_routine_one = 0

    def get_node(self, context: np.ndarray) -> NNodeConquerDivide:
        node = self.root
        while node.p_solution:
            node = node.get_child(coordinates=context)
        return node