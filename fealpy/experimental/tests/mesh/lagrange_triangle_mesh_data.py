import numpy as np
from fealpy.geometry import SphereSurface

# 定义多个典型的 LagrangeTriangleMesh 对象
surface = SphereSurface() #以原点为球心，1 为半径的球
#node, cell = surface.init_mesh(meshtype='tri', returnnc=True)
init_data = [
        {
            "p": 3,
            "node": np.array([[ 0.        ,  0.85065081,  0.52573111],
                   [ 0.        ,  0.85065081, -0.52573111],
                   [ 0.85065081,  0.52573111,  0.        ],
                   [ 0.85065081, -0.52573111,  0.        ],
                   [ 0.        , -0.85065081, -0.52573111],
                   [ 0.        , -0.85065081,  0.52573111],
                   [ 0.52573111,  0.        ,  0.85065081],
                   [-0.52573111,  0.        ,  0.85065081],
                   [ 0.52573111,  0.        , -0.85065081],
                   [-0.52573111,  0.        , -0.85065081],
                   [-0.85065081,  0.52573111,  0.        ],
                   [-0.85065081, -0.52573111,  0.        ]], dtype=np.float64),
            "cell": np.array([[ 6,  2,  0],
                   [ 3,  2,  6],
                   [ 5,  3,  6],
                   [ 5,  6,  7],
                   [ 6,  0,  7],
                   [ 3,  8,  2],
                   [ 2,  8,  1],
                   [ 2,  1,  0],
                   [ 0,  1, 10],
                   [ 1,  9, 10],
                   [ 8,  9,  1],
                   [ 4,  8,  3],
                   [ 4,  3,  5],
                   [ 4,  5, 11],
                   [ 7, 10, 11],
                   [ 0, 10,  7],
                   [ 4, 11,  9],
                   [ 8,  4,  9],
                   [ 5,  7, 11],
                   [10,  9, 11]], dtype=np.int32),
            "surface": surface,
            "face2cell": 1,
            "NN": 12,
            "NC": 20
                }
]
