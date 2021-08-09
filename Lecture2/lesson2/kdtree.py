# Implement a kdtree，including building tree and search nodes

import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet

# Node is the basic element of a tree
class Node:
    """
    axis: splitting position, the axis to split a subtree
    value:  the value in a node/subtree. leaf node has None value
    left: left subtree/children
    right: right subtree/children
    point_indices: container to store points that belongs to this partition
    """
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


"""
    Before building a tree, we need to sort the values and corresponding keys. why?
    
    input: 
            key: the node value
            value: the value in that subtree
    return:
            key_sorted: the keys after sorting
            value_sorted: the values after sorting    
"""

def sort_key_by_value(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted

# This method is used to determine the split axis, it is not optimal way to split the axis
# a better method is to consider the variance in the axes. Select the axis with larger variance,
# and split along the axis which is perpendicular to the axis with the larger variance.
# In this case, more points can be achieved with less partition.
def axis_round_robin(axis, dim):
    if axis == dim - 1:
        return 0
    else:
        return axis + 1


"""
    function: building a kdtree recursively
    
    input:
          root: root of the kdtree
          db: point cloud database
          point_indices: sorted keys of points
          axis: scalar, the axis to split a tree
          leaf_size: scalar
    
    return:
          root: a built kdtree
"""
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    # initialization
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # each sorted point index refers to a specific node value
        # so basically we sort the node values
        point_indices_sorted, _ = sort_key_by_value(point_indices, db[point_indices, axis])  # how axis work here?

        # hw1
        # start the code

        """ divide and conquer to build the tree """

        # get left subtree index and value
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        # get right subtree index and value
        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5

        # build the left subtree recursively
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)

        root.right = kdtree_recursive_build(root.right,
                                            db,
                                            point_indices_sorted[middle_right_idx:],
                                            axis_round_robin(axis, dim=db.shape[1]),
                                            leaf_size)

        # end the code

    return root

"""
    Traverse a kd_tree
    
    input:
            root: root node of a kd_tree
            depth: current depth
            max_depth: maximum depth
            
"""
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

"""
    build kd_tree (use kd_tree_recursive_build() as an interface)
    
    input:
            db_np: raw point cloud data
            leaf_size: scalar
    
    return:
            root: root node of a built kd_tree
"""

def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root

"""
    search for k nearest neighbors of a query value through kd_tree
    
    input:
            root: kd_tree root node
            db: raw point cloud data
            result_set: searched results
            query: query info
    
    return:
            boolean false if search fails
"""

def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # reach the leaf node that covers the query point
    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # hw2
    # start the code

    # q[axis] inside the partition
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    # q[axis] outside the partition
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    # end the code

    return False

"""
    search for k nearest neighbors inside specific radius area through kd_tree
    
    input:
            root: kd_tree root node
            db: raw point cloud data
            result_set: search results
            query: query info
    
    return: 
            boolean false if search fails
"""

def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # hw3
    # start the code

    # q[axis] inside the partition
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    # q[axis] outside the partition
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)

    # end the code

    return False

def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    db_np = np.random.rand(db_size, dim)
    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # kdtree_knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # kdtree_radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()