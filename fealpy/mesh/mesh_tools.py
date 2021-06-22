import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as colors
from matplotlib.tri import Triangulation

from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
import matplotlib.cm as cm
from matplotlib.patches import Polygon


def find_node(
        axes, node, index=None,
        showindex=False, color='r',
        markersize=20, fontsize=24, fontcolor='k', multiindex=None):

    if node.shape[1] == 1:
        node = np.r_['1', node, np.zeros_like(node)]
    if index is None:
        index = range(node.shape[0])
    elif (type(index) is np.int_):
        index = np.array([index], dtype=np.int_)
    elif (type(index) is np.ndarray) and (index.dtype == np.bool):
        index, = np.nonzero(index)
    elif (type(index) is list) and (type(index[0]) is np.bool):
        index, = np.nonzero(index)
    else:
        pass
        #TODO: raise a error

    if (type(color) is np.ndarray) and (np.isreal(color[0])):
        umax = color.max()
        umin = color.min()
        norm = colors.Normalize(vmin=umin, vmax=umax)
        mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
        color = mapper.to_rgba(color)

    bc = node[index]
    dim = node.shape[1]
    if dim == 2:
        axes.scatter(bc[..., 0], bc[..., 1], c=color, s=markersize)
        if showindex:
            if multiindex is not None:
                if (type(multiindex) is np.ndarray) and (len(multiindex.shape) > 1):
                    for i, idx in enumerate(multiindex):
                        s = str(idx).replace('[', '(')
                        s = s.replace(']', ')')
                        s = s.replace(' ', ',')
                        axes.text(bc[i, 0], bc[i, 1], s,
                                multialignment='center',
                                fontsize=fontsize, 
                                color=fontcolor)
                else:
                    for i, idx in enumerate(multiindex):
                        axes.text(bc[i, 0], bc[i, 1], str(idx),
                                multialignment='center',
                                fontsize=fontsize, 
                                color=fontcolor) 
            else:
                for i in range(len(index)):
                    axes.text(bc[i, 0], bc[i, 1], str(index[i]),
                            multialignment='center', fontsize=fontsize, 
                            color=fontcolor) 
    else:
        axes.scatter(bc[..., 0], bc[..., 1], bc[..., 2], c=color, s=markersize)
        if showindex:
            if multiindex is not None:
                if (type(multiindex) is np.ndarray) and (len(multiindex.shape) > 1):
                    for i, idx in enumerate(multiindex):
                        s = str(idx).replace('[', '(')
                        s = s.replace(']', ')')
                        s = s.replace(' ', ',')
                        axes.text(bc[i, 0], bc[i, 1], bc[i, 2], s,
                                multialignment='center',
                                fontsize=fontsize, 
                                color=fontcolor)
                else:
                    for i, idx in enumerate(multiindex):
                        axes.text(bc[i, 0], bc[i, 1], bc[i, 2], str(idx),
                                multialignment='center',
                                fontsize=fontsize, 
                                color=fontcolor) 
            else:
                for i in range(len(index)):
                    axes.text(bc[i, 0], bc[i, 1], bc[i, 2], str(index[i]),
                             multialignment='center', fontsize=fontsize, color=fontcolor) 


def find_entity(
        axes, mesh, entity='node',
        index=None, showindex=False,
        color='r', markersize=20, ecolor='r',
        fontsize=24, fontcolor='k', multiindex=None):

    bc = mesh.entity_barycenter(entity)
    if index is None:
        if entity == 'node':
            N = mesh.number_of_nodes()
            index = range(N)
        elif entity == 'edge':
            NE = mesh.number_of_edges()
            index = range(NE)
        elif entity == 'face':
            NF = mesh.number_of_faces()
            index = range(NF)
        elif entity == 'cell':
            NC = mesh.number_of_cells()
            index = range(NC)
        else:
            pass #TODO: raise a error
    elif (type(index) is np.ndarray) :
        if index.dtype == np.bool:
            index, = np.nonzero(index)
    elif (type(index) is list) & (type(index[0]) is np.bool):
        index, = np.nonzero(index)
    else:
        pass #TODO: raise a error

    if (type(color) is np.ndarray) & (np.isreal(color[0])):
        umax = color.max()
        umin = color.min()
        norm = colors.Normalize(vmin=umin, vmax=umax)
        mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
        color = mapper.to_rgba(color)

    if entity == 'edge':
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        e = mesh.entity(entity)
        vts = node[e[index], :]
        if GD == 2:
            lines = LineCollection(vts, linewidths=2, colors=ecolor)
        elif GD == 3:
            lines = Line3DCollection(vts, linewidth=2, colors=ecolor)
        axes.add_collection(lines)

    dim = mesh.geo_dimension()
    bc = bc[index]
    if dim == 2:
        axes.scatter(bc[:, 0], bc[:, 1], c=color, s=markersize)
        if showindex:
            if multiindex is not None:
                if (type(multiindex) is np.ndarray):
                    for i,idx in enumerate(multiindex):
                        s = str(idx).replace('[', '(')
                        s = s.replace(']', ')')
                        s = s.replace(' ', ',')
                        axes.text(bc[i, 0], bc[i, 1], s,
                                multialignment='center',
                                fontsize=fontsize, 
                                color=fontcolor) 
                else:
                    for i,idx in enumerate(multiindex):
                        axes.text(bc[i, 0], bc[i, 1], idx,
                                multialignment='center',
                                fontsize=fontsize, 
                                color=fontcolor) 
            else:
                for i in range(len(index)):
                    axes.text(bc[i, 0], bc[i, 1], str(index[i]),
                            multialignment='center', fontsize=fontsize, 
                            color=fontcolor) 
    else:
        axes.scatter(bc[:, 0], bc[:, 1], bc[:, 2], c=color, s=markersize)
        if showindex:
            for i in range(len(index)):
                axes.text(
                        bc[i, 0], bc[i, 1], bc[i, 2],
                        str(index[i]),
                        multialignment='center',
                        fontsize=fontsize, color=fontcolor)

def show_halfedge_mesh(axes, mesh,
        index=None, showindex=False, aspect='equal',
        nodecolor='r', edgecolor=['r', 'k'], 
        markersize=20, fontsize=5, fontcolor='k', 
        linewidth=1, multiindex=None, showaxis=False):

    try:
        axes.set_aspect(aspect)
    except NotImplementedError:
        pass

    if showaxis is False:
        axes.set_axis_off()
    else:
        axes.set_axis_on()

    node = mesh.entity('node')
    halfedge = mesh.entity('halfedge')

    p1 = node[halfedge[:, 0]]
    p0 = node[halfedge[halfedge[:, 4], 0]]

    w = np.array([(0,1),(-1,0)])
    v = p1 - p0
    p0 += 0.005*v@w
    p1 += 0.005*v@w
    ec = (p0 + p1)/2.0

    v *= 0.9
    p0 = ec - v/2.0
    p1 = ec + v/2.0
    
    ec += 0.04*v@w
    h = np.sqrt(np.sum(v**2, axis=-1))

    axes.scatter(node[:, 0], node[:, 1], c=nodecolor, s=markersize)

    NE = p0.shape[0]
    isMainHEdge = mesh.ds.main_halfedge_flag()
    for i in range(NE):
        if isMainHEdge[i]:
            axes.arrow(
                p0[i, 0], p0[i, 1], v[i, 0], v[i, 1], 
                shape='right', linewidth=h[i]*linewidth, 
                color=edgecolor[0], width=0.01*h[i], head_length=0.02*h[i])
        else:
            axes.arrow(
                p0[i, 0], p0[i, 1], v[i, 0], v[i, 1], 
                shape='right', linewidth=h[i]*linewidth,
                color=edgecolor[1], width=0.01*h[i], head_length=0.02*h[i])

    if showindex:
        for i in range(NE):
            if  isMainHEdge[i]:
                axes.text(
                        ec[i, 0], ec[i, 1],
                        str(i),
                        multialignment='center',
                        fontsize=fontsize, color=edgecolor[0])
            else:
                axes.text(
                        ec[i, 0], ec[i, 1],
                        str(i),
                        multialignment='center',
                        fontsize=fontsize, color=edgecolor[1])


def show_mesh_1d(
        axes, mesh,
        nodecolor='k',
        cellcolor='k',
        aspect='equal',
        linewidths=1, markersize=20,
        showaxis=False):
    axes.set_aspect(aspect)
    if showaxis == False:
        axes.set_axis_off()
    else:
        axes.set_axis_on()

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    if node.shape[1] == 1:
        node = np.r_['1', node, np.zeros_like(node)]

    axes.scatter(node[:, 0], node[:, 1], color=nodecolor, s=markersize)
    vts = node[cell, :]

    GD = mesh.geo_dimension()
    if GD < 3:
        lines = LineCollection(vts, linewidths=linewidths, colors=cellcolor)
        return axes.add_collection(lines)
    else:
        lines = Line3DCollection(vts, linewidths=linewidths, colors=cellcolor)
        return axes.add_collection3d(vts)


def show_mesh_2d(
        axes, mesh,
        nodecolor='k', edgecolor='k',
        cellcolor='grey', aspect='equal',
        linewidths=1, markersize=20,
        showaxis=False, showcolorbar=False, cmap='gnuplot2', box=None):

    try:
        axes.set_aspect(aspect)
    except NotImplementedError:
        pass

    if showaxis == False:
        axes.set_axis_off()
    else:
        axes.set_axis_on()

    if (type(nodecolor) is np.ndarray) & np.isreal(nodecolor[0]):
        cmax = nodecolor.max()
        cmin = nodecolor.min()
        norm = colors.Normalize(vmin=cmin, vmax=cmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        nodecolor = mapper.to_rgba(nodecolor)

    if isinstance(cellcolor, np.ndarray) & np.isreal(cellcolor[0]):
        cmax = cellcolor.max()
        cmin = cellcolor.min()
        norm = colors.Normalize(vmin=cmin, vmax=cmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        mapper.set_array(cellcolor)
        cellcolor = mapper.to_rgba(cellcolor)
        if showcolorbar:
            f = axes.get_figure()
            f.colorbar(mapper, shrink=0.5, ax=axes)
    node = mesh.entity('node')
    cell = mesh.entity('cell')

    if mesh.meshtype not in {'polygon', 'hepolygon', 'halfedge', 'halfedge2d'}:
        if mesh.geo_dimension() == 2:
            poly = PolyCollection(node[cell[:, mesh.ds.ccw], :])
        else:
            poly = a3.art3d.Poly3DCollection(node[cell, :])
    else:
        cell, cellLocation = cell
        NC = mesh.number_of_cells()
        patches = [
                Polygon(node[cell[cellLocation[i]:cellLocation[i+1]], :], True)
                for i in range(NC)]
        poly = PatchCollection(patches)

    poly.set_edgecolor(edgecolor)
    poly.set_linewidth(linewidths)
    poly.set_facecolors(cellcolor)

    if box is None:
        if mesh.geo_dimension() == 2:
            box = np.zeros(4, dtype=np.float64)
        else:
            box = np.zeros(6, dtype=np.float64)

        box[0::2] = np.min(node, axis=0)
        box[1::2] = np.max(node, axis=0)

    axes.set_xlim(box[0:2])
    axes.set_ylim(box[2:4])

    if mesh.geo_dimension() == 3:
        axes.set_zlim(box[4:6])

    return axes.add_collection(poly)


def show_mesh_3d(
        axes, mesh,
        nodecolor='k', edgecolor='k', facecolor='w', cellcolor='w',
        aspect='equal',
        linewidths=0.5, markersize=0,
        showaxis=False, alpha=0.8, shownode=False, showedge=False, threshold=None):

    try:
        axes.set_aspect(aspect)
    except NotImplementedError:
        pass
    if showaxis == False:
        axes.set_axis_off()
    else:
        axes.set_axis_on()

    if (type(nodecolor) is np.ndarray) & np.isreal(nodecolor[0]):
        cmax = nodecolor.max()
        cmin = nodecolor.min()
        norm = colors.Normalize(vmin=cmin, vmax=cmax)
        mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
        nodecolor = mapper.to_rgba(nodecolor)

    node = mesh.node
    if shownode:
        axes.scatter(
                node[:, 0], node[:, 1], node[:, 2],
                color=nodecolor, s=markersize)

    if showedge:
        edge = mesh.ds.edge
        vts = node[edge]
        edges = a3.art3d.Line3DCollection(
               vts,
               linewidths=linewidths,
               color=edgecolor)
        return axes.add_collection3d(edges)

    face = mesh.boundary_face(threshold=threshold)
    faces = a3.art3d.Poly3DCollection(
            node[face],
            facecolor=facecolor,
            linewidths=linewidths,
            edgecolor=edgecolor,
            alpha=alpha)
    h = axes.add_collection3d(faces)
    box = np.zeros((2, 3), dtype=np.float)
    box[0, :] = np.min(node, axis=0)
    box[1, :] = np.max(node, axis=0)
    axes.scatter(box[:, 0], box[:, 1], box[:, 2], s=0)
    return h


def unique_row(a):
    tmp = np.ascontiguousarray(a).view(np.dtype((np.void,
        a.dtype.itemsize * a.shape[1])))
    _, i, j = np.unique(tmp, return_index=True,
            return_inverse=True)
    b = a[i]
    return (b, i, j)


def show_point(axes, point):
    axes.plot(point[:, 0], point[:, 1], 'ro')


def show_mesh_quality(axes, mesh, quality=None):
    if quality is None:
        quality = mesh.cell_quality() 
    minq = np.min(quality)
    maxq = np.max(quality)
    meanq = np.mean(quality)
    hist, bins = np.histogram(quality, bins=50, range=(0, 1))
    center = (bins[:-1] + bins[1:]) / 2
    axes.bar(center, hist, align='center', width=0.02)
    axes.set_xlim(0, 1)
    axes.annotate('Min quality: {:.6}'.format(minq), xy=(0.1, 0.5),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0.1, 0.45),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0.1, 0.40),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    return minq, maxq, meanq

def show_mesh_angle(axes, mesh, angle=None):
    if angle is None:
        angle = mesh.angle() 
    hist, bins = np.histogram(angle.flatten('F')*180/np.pi, bins=50, range=(0, 180))
    center = (bins[:-1] + bins[1:])/2
    axes.bar(center, hist, align='center', width=180/50.0)
    axes.set_xlim(0, 180)
    mina = np.min(angle.flatten('F')*180/np.pi)
    maxa = np.max(angle.flatten('F')*180/np.pi)
    meana = np.mean(angle.flatten('F')*180/np.pi)
    axes.annotate('Min angle: {:.4}'.format(mina), xy=(0.41, 0.5),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    axes.annotate('Max angle: {:.4}'.format(maxa), xy=(0.41, 0.45),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    axes.annotate('Average angle: {:.4}'.format(meana), xy=(0.41, 0.40),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    return mina, maxa, meana

def show_solution(axes, mesh, u):
    points = mesh.points
    cells = mesh.cells
    tri = Triangulation(points[:,0], points[:,1], cells)
    axes.set_aspect('equal')
    axes.tricontourf(tri, u)
    return


