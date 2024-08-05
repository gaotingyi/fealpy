import numpy as np

init_mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "NN": 4,
        "NE": 4,
        "NF": 4,
        "NC": 1,
        "face2cell": np.array([[0, 0, 0, 0],
                               [0, 0, 3, 3],
                               [0, 0, 1, 1],
                               [0, 0, 2, 2]], dtype=np.int32),
    }, ]
box_data = [
    {
        "box": [0, 1, 0, 1],
        "nx": 1,
        "ny": 1,
        "threshold": None,
        "node": np.array([[0., 0.],
                          [0., 1.],
                          [1., 0.],
                          [1., 1.]], dtype=np.float64),
        "edge": np.array([[1, 0],
                          [0, 2],
                          [3, 1],
                          [2, 3]], dtype=np.int32),
        "cell": np.array([[0, 2, 3, 1]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 3, 3],
                               [0, 0, 0, 0],
                               [0, 0, 2, 2],
                               [0, 0, 1, 1]], dtype=np.int32),
    },
    {
        "box": [0, 1, 0, 1],
        "nx": 2,
        "ny": 2,
        "threshold": lambda p: (p[..., 0] < 0.5) & (p[..., 1] < 0.5),
        "node": np.array([[0., 0.5],
                          [0., 1.],
                          [0.5, 0.],
                          [0.5, 0.5],
                          [0.5, 1.],
                          [1., 0.],
                          [1., 0.5],
                          [1., 1.]], dtype=np.float64),
        "edge": np.array([[1, 0],
                          [0, 3],
                          [4, 1],
                          [3, 2],
                          [2, 5],
                          [3, 4],
                          [6, 3],
                          [7, 4],
                          [5, 6],
                          [6, 7]], dtype=np.int32),
        "cell": np.array([[0, 3, 4, 1],
                          [2, 5, 6, 3],
                          [3, 6, 7, 4]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 3, 3],
                               [0, 0, 0, 0],
                               [0, 0, 2, 2],
                               [1, 1, 3, 3],
                               [1, 1, 0, 0],
                               [0, 2, 1, 3],
                               [1, 2, 2, 0],
                               [2, 2, 2, 2],
                               [1, 1, 1, 1],
                               [2, 2, 1, 1]], dtype=np.int32),
    }, ]
entity_data = [
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "q": 2,
        "entity_measure": (np.array([0.0], dtype=np.float64),
                           np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
                           np.array([1.0], dtype=np.float64)),
        "edge_barycenter": np.array([[0.5, 0.],
                                     [0., 0.5],
                                     [1., 0.5],
                                     [0.5, 1.]], dtype=np.float64),
        "boundary_node_index": np.array([0, 1, 2, 3], dtype=np.int_),
        "boundary_edge_index": np.array([0, 1, 2, 3], dtype=np.int_),
        "boundary_cell_index": np.array([0], dtype=np.int_),
        "cell_barycenter": np.array([[0.5, 0.5]], dtype=np.float64),
        "bcs": (np.array([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=np.float64),
                np.array([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=np.float64)),
        "ws": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
        "point": np.array([[[0.21132487, 0.21132487]],
                           [[0.21132487, 0.78867513]],
                           [[0.78867513, 0.21132487]],
                           [[0.78867513, 0.78867513]]], dtype=np.float64),
    }, ]
geo_data = [
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "edge_frame": (np.array([[0., -1.], [-1., 0.], [1., 0.], [0., 1.]], dtype=np.float64),
                       np.array([[1., 0.], [0., -1.], [0., 1.], [-1., 0.]], dtype=np.float64)),
        "edge_unit_normal": np.array([[0., -1.], [-1., 0.], [1., 0.], [0., 1.]], dtype=np.float64),
    }, ]
cal_data = [
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "bcs": (np.array([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=np.float64),
                np.array([[0.78867513, 0.21132487], [0.21132487, 0.78867513]], dtype=np.float64)),
        "shape_function": np.array([[0.62200847, 0.16666667, 0.16666667, 0.0446582],
                                    [0.16666667, 0.62200847, 0.0446582, 0.16666667],
                                    [0.16666667, 0.0446582, 0.62200847, 0.16666667],
                                    [0.0446582, 0.16666667, 0.16666667, 0.62200847]],
                                   dtype=np.float64),
        "grad_shape_function": np.array(
            [[[-0.78867513, -0.78867513], [-0.21132487, 0.78867513], [0.78867513, -0.21132487],
              [0.21132487, 0.21132487]],
             [[-0.21132487, -0.78867513], [-0.78867513, 0.78867513], [0.21132487, -0.21132487],
              [0.78867513, 0.21132487]],
             [[-0.78867513, -0.21132487], [-0.21132487, 0.21132487], [0.78867513, -0.78867513],
              [0.21132487, 0.78867513]],
             [[-0.21132487, -0.21132487], [-0.78867513, 0.21132487], [0.21132487, -0.78867513],
              [0.78867513, 0.78867513]]],
            dtype=np.float64),
        "grad_shape_function_x": np.array(
            [[[[-0.78867513, - 0.78867513], [-0.21132487, 0.78867513], [0.78867513, - 0.21132487],
               [0.21132487, 0.21132487]]],
             [[[-0.21132487, - 0.78867513], [-0.78867513, 0.78867513], [0.21132487, -0.21132487],
               [0.78867513, 0.21132487]]],
             [[[-0.78867513, - 0.21132487], [-0.21132487, 0.21132487], [0.78867513, - 0.78867513],
               [0.21132487, 0.78867513]]],
             [[[-0.21132487, - 0.21132487], [-0.78867513, 0.21132487], [0.21132487, - 0.78867513],
               [0.78867513, 0.78867513]]]],
            dtype=np.float64),
        "jacobi_matrix": np.array(
            [[[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]]],
            dtype=np.float64),
        "first_fundamental_form": np.array(
            [[[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]],
             [[[1., 0.], [0., 1.]]]],
            dtype=np.float64),
    }, ]
extend_data = [
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "p": 2,
        "number_of_local_ipoints": 9,
        "number_of_global_ipoints": 9,
        "number_of_corner_nodes": 4,
        "interpolation_points": np.array([[0., 0.],
                                          [1., 0.],
                                          [1., 1.],
                                          [0., 1.],
                                          [0.5, 0.],
                                          [0., 0.5],
                                          [1., 0.5],
                                          [0.5, 1.],
                                          [0.5, 0.5]], dtype=np.float64),
        "p0": 1,
        "p1": 3,
        "prolongation_matrix": np.array([[1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.],
                                         [0.66666667, 0.33333333, 0., 0.],
                                         [0.33333333, 0.66666667, 0., 0.],
                                         [0.33333333, 0., 0., 0.66666667],
                                         [0.66666667, 0., 0., 0.33333333],
                                         [0., 0.66666667, 0.33333333, 0.],
                                         [0., 0.33333333, 0.66666667, 0.],
                                         [0., 0., 0.66666667, 0.33333333],
                                         [0., 0., 0.33333333, 0.66666667],
                                         [0.44444444, 0.22222222, 0.11111111, 0.22222222],
                                         [0.22222222, 0.11111111, 0.22222222, 0.44444444],
                                         [0.22222222, 0.44444444, 0.22222222, 0.11111111],
                                         [0.11111111, 0.22222222, 0.44444444, 0.22222222]], dtype=np.float64),
        "cell_to_ipoint": np.array([[0, 5, 3, 4, 8, 7, 1, 6, 2]], dtype=np.int32),
        "jacobi_at_corner": np.array([[1., 1., 1., 1.]], dtype=np.float64),
        "angle": np.array([[1.57079633, 1.57079633, 1.57079633, 1.57079633]], dtype=np.float64),
        "cell_quality": np.array([1.], np.float64),
    }, ]
refine_data = [{
    "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
    "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
    "n": 2,
    "refine_node": np.array([[0., 0.],
                             [1., 0.],
                             [1., 1.],
                             [0., 1.],
                             [0.5, 0.],
                             [0., 0.5],
                             [1., 0.5],
                             [0.5, 1.],
                             [0.5, 0.5],
                             [0.25, 0.],
                             [0., 0.25],
                             [0.75, 0.],
                             [1., 0.25],
                             [1., 0.75],
                             [0.75, 1.],
                             [0., 0.75],
                             [0.25, 1.],
                             [0.5, 0.25],
                             [0.25, 0.5],
                             [0.75, 0.5],
                             [0.5, 0.75],
                             [0.25, 0.25],
                             [0.75, 0.25],
                             [0.75, 0.75],
                             [0.25, 0.75]], dtype=np.float64),
    "refine_cell": np.array([[0, 9, 21, 10],
                             [9, 4, 17, 21],
                             [21, 17, 8, 18],
                             [10, 21, 18, 5],
                             [4, 11, 22, 17],
                             [11, 1, 12, 22],
                             [22, 12, 6, 19],
                             [17, 22, 19, 8],
                             [8, 19, 23, 20],
                             [19, 6, 13, 23],
                             [23, 13, 2, 14],
                             [20, 23, 14, 7],
                             [5, 18, 24, 15],
                             [18, 8, 20, 24],
                             [24, 20, 7, 16],
                             [15, 24, 16, 3]], dtype=np.int_),
    "refine_edge": np.array([[0, 9],
                             [10, 0],
                             [11, 1],
                             [1, 12],
                             [13, 2],
                             [2, 14],
                             [3, 15],
                             [16, 3],
                             [9, 4],
                             [4, 11],
                             [4, 17],
                             [5, 10],
                             [15, 5],
                             [18, 5],
                             [12, 6],
                             [6, 13],
                             [6, 19],
                             [14, 7],
                             [7, 16],
                             [7, 20],
                             [17, 8],
                             [8, 18],
                             [19, 8],
                             [20, 8],
                             [9, 21],
                             [21, 10],
                             [11, 22],
                             [12, 22],
                             [13, 23],
                             [14, 23],
                             [24, 15],
                             [16, 24],
                             [17, 21],
                             [22, 17],
                             [18, 21],
                             [18, 24],
                             [19, 22],
                             [19, 23],
                             [23, 20],
                             [20, 24]], dtype=np.int_),
    "refine_face_to_cell": np.array([[0, 0, 0, 0],
                                     [0, 0, 3, 3],
                                     [5, 5, 0, 0],
                                     [5, 5, 1, 1],
                                     [10, 10, 1, 1],
                                     [10, 10, 2, 2],
                                     [15, 15, 3, 3],
                                     [15, 15, 2, 2],
                                     [1, 1, 0, 0],
                                     [4, 4, 0, 0],
                                     [1, 4, 1, 3],
                                     [3, 3, 3, 3],
                                     [12, 12, 3, 3],
                                     [3, 12, 2, 0],
                                     [6, 6, 1, 1],
                                     [9, 9, 1, 1],
                                     [6, 9, 2, 0],
                                     [11, 11, 2, 2],
                                     [14, 14, 2, 2],
                                     [11, 14, 3, 1],
                                     [2, 7, 1, 3],
                                     [2, 13, 2, 0],
                                     [7, 8, 2, 0],
                                     [8, 13, 3, 1],
                                     [0, 1, 1, 3],
                                     [0, 3, 2, 0],
                                     [4, 5, 1, 3],
                                     [5, 6, 2, 0],
                                     [9, 10, 2, 0],
                                     [10, 11, 3, 1],
                                     [12, 15, 2, 0],
                                     [14, 15, 3, 1],
                                     [1, 2, 2, 0],
                                     [4, 7, 2, 0],
                                     [2, 3, 3, 1],
                                     [12, 13, 1, 3],
                                     [6, 7, 3, 1],
                                     [8, 9, 1, 3],
                                     [8, 11, 2, 0],
                                     [13, 14, 2, 0]], dtype=np.int_),
}, ]
