import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.experimental.tests.mesh.uniform_mesh_2d_data import *

#@pytest.fixture(params=meshing_data)
#def meshing(request):
#    meshdata = request.param
#    extent = meshdata['extent']
#    h = meshdata['h']
#    origin = meshdata['origin']
#    mesh = UniformMesh2d(extent, h, origin)
#    return mesh

class TestUniformMesh2dInterfaces:

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        node = bm.to_numpy(mesh.node)
        node_old = meshdata['node']
        np.testing.assert_almost_equal(node, node_old, decimal=7)
        edge = bm.to_numpy(mesh.edge)
        edge_old = meshdata['edge']
        np.testing.assert_almost_equal(edge, edge_old, decimal=7)
        face = bm.to_numpy(mesh.face)
        face_old = meshdata['face']
        np.testing.assert_almost_equal(face, face_old, decimal=7)
        cell = bm.to_numpy(mesh.cell)
        cell_old = meshdata['cell']
        np.testing.assert_almost_equal(cell, cell_old, decimal=7)

        assert mesh.node.shape == meshdata['node'].shape, "Node shapes do not match."
        assert mesh.edge.shape == meshdata['edge'].shape, "Edge shapes do not match."
        assert mesh.face.shape == meshdata['face'].shape, "Face shapes do not match."
        assert mesh.cell.shape == meshdata['cell'].shape, "Cell shapes do not match."

        assert mesh.number_of_nodes() == meshdata['NN'], "Number of nodes do not match."
        assert mesh.number_of_edges() == meshdata['NE'], "Number of edges do not match."
        assert mesh.number_of_faces() == meshdata['NF'], "Number of faces do not match."
        assert mesh.number_of_cells() == meshdata['NC'], "Number of cells do not match."

    @pytest.mark.parametrize("meshdata", entity_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_entity(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        node = bm.to_numpy(mesh.entity('node'))
        node_true = meshdata['entity_node']
        np.testing.assert_almost_equal(node, node_true, decimal=7)

    @pytest.mark.parametrize("meshdata", entity_measure_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_entity_measure(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        assert len(mesh.entity_measure('edge')) == len(meshdata['edge_length']), "Edge lengths must have the same length."
        for a, b in zip(mesh.entity_measure('edge'), meshdata['edge_length']):
            assert abs(a - b) < 1e-7, f"Difference between {a} and {b} is greater than 1e-7"

        assert (mesh.entity_measure('cell') - meshdata['cell_area']) < 1e-7, "Cell areas are not as expected."

    @pytest.mark.parametrize("meshdata", interpolation_points_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_interpolation_points(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        ipoints_p1 = mesh.interpolation_points(p=1)
        ipoints_p2 = mesh.interpolation_points(p=2)

        ipoints_p1_true = meshdata['ipoints_p1']
        ipoints_p2_true = meshdata['ipoints_p2']

        np.testing.assert_almost_equal(ipoints_p1, ipoints_p1_true, decimal=7)
        np.testing.assert_almost_equal(ipoints_p2, ipoints_p2_true, decimal=7)

    @pytest.mark.parametrize("meshdata", quadrature_formula_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_quadrature_formula(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        qf1 = mesh.quadrature_formula(q=1)
        bcs_qf1, ws_qf1 = qf1.get_quadrature_points_and_weights()
        qf2 = mesh.quadrature_formula(q=2)
        bcs_qf2, ws_qf2 = qf2.get_quadrature_points_and_weights()

        assert len(bcs_qf1) == len(meshdata['bcs_q1']), "The tuples for qf1 must have the same length."
        for a, b in zip(bcs_qf1, meshdata['bcs_q1']):
            assert np.all(np.abs(bm.to_numpy(a) - b) < 1e-7), f"Difference in quadrature points for qf1 between {a} and {b} is greater than 1e-7"

        assert len(bcs_qf2) == len(meshdata['bcs_q2']), "The tuples for qf2 must have the same length."
        for a, b in zip(bcs_qf2, meshdata['bcs_q2']):
            assert np.all(np.abs(bm.to_numpy(a) - b) < 1e-7), f"Difference in quadrature points for qf2 between {a} and {b} is greater than 1e-7"

    @pytest.mark.parametrize("meshdata", shape_function_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_shape_function(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        qf = mesh.quadrature_formula(q=1)
        bcs, ws = qf.get_quadrature_points_and_weights()

        shape_function_p1 = bm.to_numpy(mesh.shape_function(bcs=bcs, p=1))
        shape_function_p2 = bm.to_numpy(mesh.shape_function(bcs=bcs, p=2))

        shape_function_p1_true = meshdata['shape_function_p1']
        shape_function_p2_true = meshdata['shape_function_p2']

        np.testing.assert_allclose(shape_function_p1, shape_function_p1_true, atol=1e-8)
        np.testing.assert_allclose(shape_function_p2, shape_function_p2_true, atol=1e-8)

    # @pytest.mark.parametrize("meshdata", grad_shape_function_data)
    # @pytest.mark.parametrize("backend", ['pytorch'])
    # def test_grad_shape_function(self, meshing, meshdata, backend):
    #     bm.set_backend(backend)
    #     mesh = meshing
    #
    #     qf = mesh.quadrature_formula(q=1)
    #     bcs, ws = qf.get_quadrature_points_and_weights()
    #
    #     print("------------------")
    #     grad_shape_function_u = mesh.grad_shape_function(bcs=bcs, p=1, variables='u')
    #     print("sss:", grad_shape_function_u)
    #     #grad_shape_function_u = bm.to_numpy(grad_shape_function_u, dtype=np.float64)
    #     #print(type(grad_shape_function_u))
    #     #print(grad_shape_function_u)
    #     #print("-------------------")
    #     #grad_shape_function_u_true = meshdata['grad_shape_function_u']
    #     #print(type(grad_shape_function_u_true))
    #     #print(grad_shape_function_u_true)
    #     #grad_shape_function_x = mesh.grad_shape_function(bcs=bcs, p=1, variables='x')
    #
    #     #a = grad_shape_function_u - grad_shape_function_u_true
    #     #print(a)
    #     #np.testing.assert_almost_equal(grad_shape_function_u, grad_shape_function_u_true, decimal=7)
    #
    #     #assert all((grad_shape_function_u.reshape(-1) - bm.from_numpy(meshdata['grad_shape_function_u']).reshape(-1)) < 1e-7), \
    #     #     "Grad shape function are not as expected."
    #     #assert all((grad_shape_function_x.reshape(-1) - bm.from_numpy(meshdata['grad_shape_function_x']).reshape(-1)) < 1e-7), \
    #     #     "Grad shape function are not as expected."
