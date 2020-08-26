import numpy as np

from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
from .Mesh2d import Mesh2d, Mesh2dDataStructure

from .core import multi_index_matrix
from .core import lagrange_shape_function 
from .core import lagrange_grad_shape_function
from .core import LinearMeshDataStructure

class LinearQuadrangleMeshDataStructure(LinearMeshDataStructure):
    localEdge = np.array([(0, 2), (2, 3), (3, 1), (1, 0)])
    ccw = np.array([0, 2, 3, 1])
    V = 4
    E = 4
    F = 1
    EV = 2

    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype
        self.construct_edge()

class LagrangeQuadrangleMesh(Mesh2d):
    def __init__(self, node, cell, p=1, surface=None):

        self.p = p
        self.GD = node.shape[1]
        self.TD = 2
        self.ftype = node.dtype
        self.itype = cell.dtype
        self.meshtype = 'lquad'

        ds = LinearQuadrangleMeshDataStructure(node.shape[0], cell) # 线性网格的数据结构
        self.ds = LagrangeQuadrangleMeshDataStructure(ds, p)

        if p == 1:
            self.node = node
        else:
            NN = self.number_of_nodes()
            self.node = np.zeros((NN, self.GD), dtype=self.ftype)
            bc = multi_index_matrix[1](p)/p
            bc = np.einsum('im, jn->ijmn', bc, bc).reshape(-1, 4)
            self.node[self.ds.cell] = np.einsum('ijn, kj->ikn', node[cell], bc)

        if surface is not None:
            self.node, _ = surface.project(self.node)
   

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}

    def number_of_corner_nodes(self):
        """
        Notes
        -----

        拉格朗日三角形网格中的节点分为单元角点节点，边内部节节点和单元内部节点。

        这些节点默认的编号顺序也是：角点节点，边内部节点，单元内部节点。

        该函数返回角点节点的个数。
        """
        return self.ds.NCN

    def lagrange_dof(self, p):
        return LagrangeQuadrangleDof2d(self, p)

    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk 类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_QUADRILATERAL = 70 
            return VTK_LAGRANGE_QUADRILATERAL
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        index = vtk_cell_index(self.p, cellType) # TODO: 转化为 vtk 编号顺序
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, index]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def integrator(self, q, etype='cell'):
        qf = GaussLegendreQuadrature(q)
        if etype in {'cell', 2}:
            return TensorProductQuadrature(qf, TD=2) 
        elif etype in {'edge', 'face', 1}:
            return TensorProductQuadrature(qf, TD=1) 

    def cell_area(self, q=None, index=np.s_[:]):
        """

        Notes
        -----
        计算单元的面积。
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        l = np.sqrt(np.sum(n**2, axis=-1))
        a = np.einsum('i, ij->j', ws, l)/2.0
        return a

    def edge_length(self, q=None, index=np.s_[:]):
        """

        Note
        ----
        计算边的长度
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        J = self.jacobi_matrix(bcs, index=index)
        l = np.sqrt(np.sum(J**2, axis=(-1, -2)))
        a = np.einsum('i, ij->j', ws, l)
        return a

    def bc_to_point(self, bc, index=np.s_[:], etype='cell'):
        """

        Notes
        -----
        etype 这个参数实际上是不需要的，为了兼容，所以这里先保留。

        bc 是一个 tuple 数组

        Examples
        --------
        >>> bc = TensorProductQuadrature(3, TD=2) # 第三个张量积分公式
        >>> points = mesh.bc_to_point(bc)

        """
        node = self.node
        TD = len(bc) 
        entity = self.entity(etype=TD)[index] #
        phi = self.shape_function(bc) # (NQ, 1, ldof)
        p = np.einsum('ijk, jkn->ijn', phi, node[entity])
        return p


    def shape_function(self, bc, p=None):
        """

        Notes
        -----

        bc 是一个长度为 TD 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1]

        """
        p = self.p if p is None else p
        TD = len(bc)
        phi = lagrange_shape_function(bc[0]) 
        if TD == 2:
            # i 是积分点
            # j 是单元
            # m 是基函数
            phi = np.einsum('im, kn->ikmn', phi, phi)
            shape = phi.shape[:-2] + (-1, )
            phi = phi.reshape(shape) 
        return phi[..., None, :]  # (..., 1, p+1*p+1) 增加一个单元轴，方便广播运算

    def grad_shape_function(self, bc, p=None, index=np.s_[:], variables='u'):
        """

        Notes
        -----
        计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        bc 是一个长度为 TD 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1]

        """
        p = self.p if p is None else p
        TD = len(bc)
        Dlambda = np.array([[-1], [1]], dtype=self.ftype)

        # 一维基函数值
        # (NQ, p+1)
        phi = lagrange_shape_function(bc[0], p)  

        # 关于**一维变量重心坐标**的导数
        # lambda_0 = 1 - xi
        # lambda_1 = xi
        # (NQ, ldof, 2) 
        R = lagrange_grad_shape_function(bc[0], p)  

        # 关于**一维变量**的导数
        gphi = np.einsum('...ij, jn->...in', R, Dlambda) # (..., ldof, 1)

        if TD == 2:
            gphi0 = np.einsum('imt, kn->ikmn', gphi, phi)
            shape = gphi.shape[:-2] + ((p+1)*(p+1), TD)
            gphi = np.zeros(shape, dtype=self.ftype)
            gphi[..., 0].flat = gphi0.flat
            gphi[..., 1].flat = gphi0.swapaxes(-1, -2).flat

        if variables == 'u':
            return gphi[..., None, :, :] #(..., 1, ldof, TD) 增加一个单元轴
        elif variables == 'x':
            entity = self.entity(etype=TD)
            J = np.einsum(
                    'ijn, ...jk->...ink',
                    self.node[entity[index], :], gphi)
            shape = J.shape[0:-2] + (TD, TD)
            G = np.zeros(shape, dtype=self.ftype)
            for i in range(TD):
                G[..., i, i] = np.sum(J[..., i]**2, axis=-1)
                for j in range(i+1, TD):
                    G[..., i, j] = np.sum(J[..., i]*J[..., j], axis=-1)
                    G[..., j, i] = G[..., i, j]
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi)
            return gphi

    def jacobi_matrix(self, bc, index=np.s_[:], p=None, return_grad=False):
        """
        Notes
        -----
        计算参考单元 （xi, eta) 到实际 Lagrange 三角形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """

        TD = len(bc)
        entity = self.entity(etype=TD)
        gphi = self.grad_shape_function(bc, p=p)
        J = np.einsum(
                'ijn, ...ijk->...ink',
                self.node[entity[index], :], gphi)
        if return_grad is False:
            return J
        else:
            return J, gphi

    def first_fundamental_form(self, bc, index=np.s_[:], 
            return_jacobi=False, return_grad=False):
        """
        Notes
        -----
            计算拉格朗日网格在积分点处的第一基本形式。
        """

        TD = len(bc) 
        J = self.jacobi_matrix(bc, index=index,
                return_grad=return_grad)
        
        if return_grad:
            J, gphi = J

        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = np.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False): 
            return G, J
        elif (return_jacobi is False) & (return_grad is True): 
            return G, gphi 
        else:
            return G, J, gphi

    def print(self):

        """

        Notes
        -----
            打印网格信息， 用于调试。
        """

        node = self.entity('node')
        print('node:')
        for i, v in enumerate(node):
            print(i, ": ", v)

        edge = self.entity('edge')
        print('edge:')
        for i, e in enumerate(edge):
            print(i, ": ", e)

        cell = self.entity('cell')
        print('cell:')
        for i, c in enumerate(cell):
            print(i, ": ", c)

        edge2cell = self.ds.edge_to_cell()
        print('edge2cell:')
        for i, ec in enumerate(edge2cell):
            print(i, ": ", ec)

class LagrangeQuadrangleMeshDataStructure(Mesh2dDataStructure):
    def __init__(self, ds, p):

        self.itype = ds.itype

        self.p = p
        self.V = (p+1)*(p+1) 
        self.E = ds.E 
        self.F = ds.F

        self.NCN = ds.NN  # 角点的个数
        self.NN = ds.NN 
        self.NE = ds.NE 
        self.NC = ds.NC 

        self.edge2cell = ds.edge2cell 

        if p == 1:
            self.cell = ds.cell
            self.edge = ds.edge
        else:
            NE = ds.NE
            edge = ds.edge
            self.edge = np.zeros((NE, p+1), dtype=self.itype)
            self.edge[:, [0, -1]] = edge
            self.edge[:, 1:-1] = self.NN + np.arange(NE*(p-1)).reshape(NE, p-1)
            self.NN += NE*(p-1)

            NC = ds.NC
            self.cell = np.zeros((NC, (p+1)*(p+1)), dtype=self.itype)
            cell = self.cell.reshape((NC, p+1, p+1))

            edge2cell = ds.edge2cell

            flag = edge2cell[:, 2] == 0
            cell[edge2cell[flag, 0], :, 0] = self.edge[flag]
            flag = edge2cell[:, 2] == 1
            cell[edge2cell[flag, 0], -1, :] = self.edge[flag]
            flag = edge2cell[:, 2] == 2
            cell[edge2cell[flag, 0], :, -1] = self.edge[flag, -1::-1]
            flag = edge2cell[:, 2] == 3
            cell[edge2cell[flag, 0], 0, :] = self.edge[flag, -1::-1]

            flag = (edge2cell[:, 3] == 0) & (edge2cell[:, 0] != edge2cell[:, 1])
            cell[edge2cell[flag, 1], :, 0] = self.edge[flag, -1::-1]
            flag = (edge2cell[:, 3] == 1) & (edge2cell[:, 0] != edge2cell[:, 1])
            cell[edge2cell[flag, 1], -1, :] = self.edge[flag, -1::-1]
            flag = (edge2cell[:, 3] == 2) & (edge2cell[:, 0] != edge2cell[:, 1])
            cell[edge2cell[flag, 1], :, -1] = self.edge[flag]
            flag = (edge2cell[:, 3] == 3) & (edge2cell[:, 0] != edge2cell[:, 1])
            cell[edge2cell[flag, 1], 0, :] = self.edge[flag]

            cell[:, 1:-1, 1:-1] = self.NN + np.arange(NC*(p-1)*(p-1)).reshape(NC, p-1, p-1)
            self.NN += NC*(p-1)*(p-1)
            


class LagrangeQuadrangleDof2d():
    """

    Notes
    -----
    拉格朗日四边形网格上的自由度管理类。
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        return self.edge_to_dof()

    @property
    def edge2dof(self):
        return self.edge_to_dof()

    def edge_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分边上的自由度
        """
        p = self.p
        mesh = self.mesh
        edge = mesh.entity('edge')

        if p == mesh.p:
            return edge

        NN = mesh.number_of_corner_nodes()
        NE = mesh.number_of_edges()
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, -1]] = edge[:, [0, -1]] # edge 可以是高次曲线
        if p > 1:
            NN = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
            NE = mesh.number_of_edges()
            edge2dof[:, 1:-1] = NN + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    @property
    def cell2dof(self):
        """
        
        Notes
        -----
            把这个方法属性化，保证程序接口兼容性
        """
        return self.cell_to_dof()


    def cell_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分单元上的自由度。
        2. 用更高效的方式来生成单元自由度数组。
        """

        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell') # cell 可以是高次单元
        if p == mesh.p:
            return cell 

        # 空间自由度和网格的自由度不一致时，重新构造单元自由度矩阵
        edge2cell = self.mesh.ds.edge_to_cell()
        NN = self.mesh.number_of_corner_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells() 

        cell2dof = np.zeros((NC, (p+1)*(p+1)), dtype=self.itype)
        c2d = cell2dof.reshape((NC, p+1, p+1))

        e2d = self.edge_to_dof()
        flag = edge2cell[:, 2] == 0
        c2d[edge2cell[flag, 0], :, 0] = e2d[flag]
        flag = edge2cell[:, 2] == 1
        c2d[edge2cell[flag, 0], -1, :] = e2d[flag]
        flag = edge2cell[:, 2] == 2
        c2d[edge2cell[flag, 0], :, -1] = e2d[flag, -1::-1]
        flag = edge2cell[:, 2] == 3
        cell[edge2cell[flag, 0], 0, :] = e2d[flag, -1::-1]

        flag = (edge2cell[:, 3] == 0) & (edge2cell[:, 0] != edge2cell[:, 1])
        c2d[edge2cell[flag, 1], :, 0] = e2d[flag, -1::-1]
        flag = (edge2cell[:, 3] == 1) & (edge2cell[:, 0] != edge2cell[:, 1])
        c2d[edge2cell[flag, 1], -1, :] = e2d[flag, -1::-1]
        flag = (edge2cell[:, 3] == 2) & (edge2cell[:, 0] != edge2cell[:, 1])
        c2d[edge2cell[flag, 1], :, -1] = e2d[flag]
        flag = (edge2cell[:, 3] == 3) & (edge2cell[:, 0] != edge2cell[:, 1])
        c2d[edge2cell[flag, 1], 0, :] = e2d[flag]

        c2d[:, 1:-1, 1:-1] = NN + NE*(p-1) + np.arange(NC*(p-1)*(p-1)).reshape(NC, p-1, p-1)

        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        if p == mesh.p:
            return node

        cell2dof = self.cell_to_dof()
        GD = mesh.geo_dimension()
        gdof = self.number_of_global_dofs()
        ipoint = np.zeros((gdof, GD), dtype=np.float64)
        bc = multi_index_matrix[1](p)/p
        ipoint[cell2dof] = mesh.bc_to_point((bc, bc)).swapaxes(0, 1)
        return ipoint


    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh

        if p == mesh.p:
            return mesh.number_of_nodes()

        gdof = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
        if p > 1:
            NE = mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            NC = mesh.number_of_cells()
            gdof += (p - 1)*(p - 1)*NC
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 2}:
            return (p+1)*(p+1) 
        elif doftype in {'face', 'edge',  1}:
            return p + 1
        elif doftype in {'node', 0}:
            return 1