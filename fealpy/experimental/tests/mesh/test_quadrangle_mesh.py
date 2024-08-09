import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.quadrangle_mesh import QuadrangleMesh
from fealpy.experimental.tests.mesh.quadrangle_mesh_data import *


class TestQuadrangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = QuadrangleMesh(node, cell)

        assert mesh.number_of_nodes() == meshdata["NN"]
        assert mesh.number_of_edges() == meshdata["NE"]
        assert mesh.number_of_faces() == meshdata["NF"]
        assert mesh.number_of_cells() == meshdata["NC"]
        face2cell = mesh.face2cell
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        mesh_from_one_quadrangle_1 = QuadrangleMesh.from_one_quadrangle()
        mesh_from_one_quadrangle_2 = QuadrangleMesh.from_one_quadrangle(meshtype='rectangle')
        mesh_from_one_quadrangle_3 = QuadrangleMesh.from_one_quadrangle(meshtype='rhombus')

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", box_data)
    def test_from_box(self, meshdata, backend):
        box = bm.from_numpy(meshdata['box'])
        nx = bm.from_numpy(meshdata['nx'])
        ny = bm.from_numpy(meshdata['ny'])
        threshold = bm.from_numpy(meshdata['threshold'])
        mesh = QuadrangleMesh.from_box(box, nx, ny, threshold)

        node = mesh.node
        np.testing.assert_array_equal(bm.to_numpy(node), meshdata["node"])
        cell = mesh.cell
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])
        edge = mesh.edge
        np.testing.assert_array_equal(bm.to_numpy(edge), meshdata["edge"])
        face2cell = mesh.face2cell
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", entity_data)
    def test_entity(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)
        q = bm.from_numpy(meshdata['q'])

        assert mesh.entity_measure(0) == meshdata["entity_measure"][0]
        assert all(mesh.entity_measure(1) == meshdata["entity_measure"][1])
        assert all(mesh.entity_measure('cell') == meshdata["entity_measure"][2])

        edge_barycenter = mesh.entity_barycenter('edge')
        cell_barycenter = mesh.entity_barycenter('cell')
        np.testing.assert_allclose(bm.to_numpy(edge_barycenter), meshdata["edge_barycenter"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(cell_barycenter), meshdata["cell_barycenter"], atol=1e-7)

        # TODO: have no boundary_edge_index
        boundary_node_index = mesh.boundary_node_index()
        boundary_cell_index = mesh.boundary_cell_index()
        boundary_face_index = mesh.boundary_face_index()
        # boundary_edge_index = mesh.boundary_edge_index()
        np.testing.assert_array_equal(bm.to_numpy(boundary_node_index), meshdata["boundary_node_index"])
        np.testing.assert_array_equal(bm.to_numpy(boundary_face_index), meshdata["boundary_node_index"])
        np.testing.assert_array_equal(bm.to_numpy(boundary_cell_index), meshdata["boundary_cell_index"])
        # np.testing.assert_array_equal(bm.to_numpy(boundary_edge_index), meshdata["boundary_edge_index"])

        integrator = mesh.quadrature_formula(q)
        bcs, ws = integrator.get_quadrature_points_and_weights()

        np.testing.assert_allclose(bm.to_numpy(bcs), meshdata["bcs"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(ws), meshdata["ws"], atol=1e-7)

        point = mesh.bc_to_point(bcs)
        np.testing.assert_allclose(bm.to_numpy(point), meshdata["point"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", geo_data)
    def test_geo(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)

        edge_frame = mesh.edge_frame()
        edge_unit_normal = mesh.edge_unit_normal()

        np.testing.assert_allclose(bm.to_numpy(edge_frame), meshdata["edge_frame"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(edge_unit_normal), meshdata["edge_unit_normal"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", cal_data)
    def test_cal_data(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        bcs = bm.from_numpy(meshdata['bcs'])
        mesh = QuadrangleMesh(node, cell)

        shape_function = mesh.shape_function(bcs)
        np.testing.assert_allclose(bm.to_numpy(shape_function), meshdata["shape_function"], atol=1e-7)
        grad_shape_function = mesh.grad_shape_function(bcs)
        np.testing.assert_allclose(bm.to_numpy(grad_shape_function), meshdata["grad_shape_function"], atol=1e-7)
        grad_shape_function_x = mesh.grad_shape_function(bcs, variables='x')
        np.testing.assert_allclose(bm.to_numpy(grad_shape_function_x), meshdata["grad_shape_function_x"], atol=1e-7)

        jacobi_matrix = mesh.jacobi_matrix(bcs)
        np.testing.assert_allclose(bm.to_numpy(jacobi_matrix), meshdata["jacobi_matrix"], atol=1e-7)
        first_fundamental_form = mesh.first_fundamental_form(jacobi_matrix)
        np.testing.assert_allclose(bm.to_numpy(first_fundamental_form), meshdata["first_fundamental_form"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", extend_data)
    def test_extend_data(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)
        p = meshdata["p"]

        assert mesh.number_of_global_ipoints(p) == meshdata["number_of_global_ipoints"]
        assert mesh.number_of_local_ipoints(p) == meshdata["number_of_local_ipoints"]
        assert mesh.number_of_corner_nodes() == meshdata["number_of_corner_nodes"]

        cell_to_ipoint = mesh.cell_to_ipoint(p)
        np.testing.assert_allclose(bm.to_numpy(cell_to_ipoint), meshdata["cell_to_ipoint"], atol=1e-7)

        interpolation_points = mesh.interpolation_points(p)
        np.testing.assert_allclose(bm.to_numpy(interpolation_points), meshdata["interpolation_points"], atol=1e-7)

        jacobi_at_corner = mesh.jacobi_at_corner()
        np.testing.assert_allclose(bm.to_numpy(jacobi_at_corner), meshdata["jacobi_at_corner"], atol=1e-7)

        angle = mesh.angle()
        np.testing.assert_allclose(bm.to_numpy(angle), meshdata["angle"], atol=1e-7)

        cell_quality = mesh.cell_quality()
        np.testing.assert_allclose(bm.to_numpy(cell_quality), meshdata["cell_quality"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", refine_data)
    def test_refine(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)
        n = meshdata["n"]

        mesh.uniform_refine(n)
        refine_node = mesh.node
        np.testing.assert_allclose(bm.to_numpy(refine_node), meshdata["refine_node"], atol=1e-7)
        refine_cell = mesh.cell
        np.testing.assert_allclose(bm.to_numpy(refine_cell), meshdata["refine_cell"], atol=1e-7)
        refine_edge = mesh.edge
        np.testing.assert_allclose(bm.to_numpy(refine_edge), meshdata["refine_edge"], atol=1e-7)
        refine_face_to_cell = mesh.face2cell
        np.testing.assert_allclose(bm.to_numpy(refine_face_to_cell),meshdata["refine_face_to_cell"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_from_polygon_gmsh_data)
    def test_mesh_from_polygon_gmsh(self, meshdata, backend):
        vertices = bm.from_numpy(meshdata['vertices'])
        h = bm.from_numpy(meshdata['h'])
        mesh = QuadrangleMesh.from_polygon_gmsh(vertices, h)

        node = mesh.node
        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"], atol=1e-7)
        edge = mesh.edge
        np.testing.assert_allclose(bm.to_numpy(edge), meshdata["edge"], atol=1e-7)
        cell = mesh.cell
        np.testing.assert_allclose(bm.to_numpy(cell), meshdata["cell"], atol=1e-7)
        face2cell = mesh.face2cell
        np.testing.assert_allclose(bm.to_numpy(face2cell), meshdata["face2cell"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_from_triangle_data)
    def test_mesh_from_triangle(self, meshdata, backend):
        from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
        tri_node = bm.from_numpy(meshdata['tri_node'])
        tri_cell = bm.from_numpy(meshdata['tri_cell'])
        tri_mesh = TriangleMesh(tri_node, tri_cell)

        mesh = QuadrangleMesh.from_triangle_mesh(tri_mesh)

        node = mesh.node
        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"], atol=1e-7)
        edge = mesh.edge
        np.testing.assert_allclose(bm.to_numpy(edge), meshdata["edge"], atol=1e-7)
        cell = mesh.cell
        np.testing.assert_allclose(bm.to_numpy(cell), meshdata["cell"], atol=1e-7)
        face2cell = mesh.face2cell
        np.testing.assert_allclose(bm.to_numpy(face2cell), meshdata["face2cell"], atol=1e-7)


if __name__ == "__main__":
    pytest.main(["./test_quadrangle_mesh.py", "-k", "test_init"])