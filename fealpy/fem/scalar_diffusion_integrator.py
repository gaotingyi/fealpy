import numpy as np

from fealpy.fem.precomp_data import data

class ScalarDiffusionIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """    
    def __init__(self, c=None, q=None):
        self.coef = c
        self.q = q
        self.type = "BL0"

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @note 没有参考单元的组装方式
        """
        p = space.p
        q = self.q if self.q is not None else p+1 

        coef = self.coef
        mesh = space.mesh
        GD = mesh.geo_dimension()

        if cellmeasure is None:
            if mesh.meshtype == 'UniformMesh2d':
                 NC = mesh.number_of_cells()
                 cellmeasure = np.broadcast_to(mesh.entity_measure('cell', index=index), (NC,))
            else:
                cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        if out is None:
            D = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            D = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi0 = space.grad_basis(bcs, index=index) # (NQ, NC, ldof, GD)
        phi1 = phi0

        if coef is None:
            D += np.einsum('q, qcid, qcjd, c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
                        coef = coef(ps)
                    elif coef.coordtype == 'barycentric':
                        coef = coef(bcs, index=index)
                else:
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            if np.isscalar(coef):
                D += coef*np.einsum('q, qcid, qcjd, c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
            elif isinstance(coef, np.ndarray): 
                if coef.shape == (NC, ): 
                    D += np.einsum('q, c, qcid, qcjd, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (NQ, NC):
                    D += np.einsum('q, qc, qcid, qcjd, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (GD, GD):
                    D += np.einsum('q, dn, qcin, qcjd, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (NC, GD, GD):
                    D += np.einsum('q, cdn, qcin, qcjd, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (NQ, NC, GD, GD):
                    D += np.einsum('q, qcdn, qcin, qcjd, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                else:
                    raise ValueError(f"coef with shape {coef.shape}! Now we just support shape: (NC, ), (NQ, NC), (GD, GD), (NC, GD, GD) or NQ, NC, GD, GD)")
            else:
                raise ValueError("coef 不支持该类型")

        if out is None:
            return D


    def assembly_cell_matrix_fast(self, trialspace, testspace=None, coefspace=None,
            index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 基于无数值积分的组装方式
        """
        coef = self.coef

        mesh = trialspace.mesh 
        meshtype = mesh.type

        TAFtype = trialspace.btype
        TAFdegree = trialspace.p
        TAFldof = trialspace.number_of_local_dofs()  
        TSFtype = TAFtype
        TSFdegree = TAFdegree
        TSFldof = TAFldof
        if testspace is not None:
            TSFtype = testspace.btype
            TSFdegree = testspace.p 
            TSFldof = testspace.number_of_local_dofs()
        COFtype = TAFtype
        COFdegree = TAFdegree
        COFldof = TAFldof
        if coefspace is not None:
            COFtype = coefspace.btype
            COFdegree = coefspace.p 
            COFldof = coefspace.number_of_local_dofs()
        Itype = self.type 
        dataindex = Itype + "_" + meshtype + "_TAF_" + TAFtype + "_" + \
                str(TAFdegree) + "_TSF_" + TSFtype + "_" + str(TSFdegree)

        if cellmeasure is None:
            if mesh.meshtype == 'UniformMesh2d':
                 NC = mesh.number_of_cells()
                 cellmeasure = np.broadcast_to(mesh.entity_measure('cell', index=index), (NC,))
            else:
                 cellmeasure = mesh.entity_measure('cell', index=index)
        
        NC = len(cellmeasure)

        if out is None:
            D = np.zeros((NC, TSFldof, TAFldof), dtype=trialspace.ftype)
        else:
            D = out
        
        glambda = mesh.grad_lambda()
        if coef is None:
            D += np.einsum('ijkl, c, ck, cl -> cij', data[dataindex], cellmeasure, glambda[..., 0], glambda[..., 0], optimize=True)
            D += np.einsum('ijkl, c, ck, cl -> cij', data[dataindex], cellmeasure, glambda[..., 1], glambda[..., 1], optimize=True)
        else:
            if callable(coef):
                u = coefspace.interpolate(coef)
                cell2dof = coefspace.cell_to_dof()
                coef = u[cell2dof]
            if np.isscalar(coef):
                D += np.einsum('ijkl, c, ck, cl -> cij', data[dataindex], cellmeasure, glambda[..., 0], glambda[..., 0], optimize=True)
                D += np.einsum('ijkl, c, ck, cl -> cij', data[dataindex], cellmeasure, glambda[..., 1], glambda[..., 1], optimize=True)
                D *= coef
            elif coef.shape == (NC, COFldof):
                dataindex += "_COF_" + COFtype + "_" + str(COFdegree)
                D += np.einsum('ijkmn, c, cm, cn, ck -> cij', data[dataindex], cellmeasure, glambda[..., 0], glambda[..., 0], coef, optimize=True)
                D += np.einsum('ijkmn, c, cm, cn, ck -> cij', data[dataindex], cellmeasure, glambda[..., 1], glambda[..., 1], coef, optimize=True)
            else:
                raise ValueError("coef is not correct!")

        if out is None:
            return D


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元矩阵组装方式
        """
