import numpy as np

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

    def query(self, x: float, y: float, radius: float, current_iteration:int) -> list[TwoDimensionalNode]:
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

    def find_nearest_neighbor(self, x: float, y: float) -> tuple[TwoDimensionalNode, float]:
            """
            This function finds the nearest neighbor to a given point (x, y).
            """
            if self.root is None:
                return None

            best = {'node': None, 'distance': float('inf')}
            self._find_nearest_recursive(self.root, x, y, best)
            return best['node'], best['distance']
    
    def _find_nearest_recursive(self, node: TwoDimensionalNode, x: float, y: float, best: dict) -> None:
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