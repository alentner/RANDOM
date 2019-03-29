import numpy, scipy
from scipy import sparse
from scipy.sparse import linalg

def constCoeff(res, bnd = [0, 0, 0, 0], xbnd = [0, 1], ybnd = [0, 1]):

    # boundary conditions where the following:
    #        type: 0 = diriclet, 1 = neumann
    #         loc: left, right, bottom, top
    # coordinate domain and resolution w/:
    #                  ?bnd  = {min, max}
    #                  npnts = {Nx, Ny}


    # set up solution parameters - grid
    npts = [res, res]
    dx = (xbnd[1] - xbnd[0]) / npts[0]
    dy = (ybnd[1] - ybnd[0]) / npts[1]


    # set up solution parameters - constants
    cObnd = [3, 1]

    ### build sparse coefficient matrix

    # form main diagonal coefficents and/or vectors
    vOcc = numpy.ones(npts[0] - 2) * -2 * (dx**2 + dy**2)
    vOcl = -(2 * dx**2 + cObnd[bnd[0]] * dy**2) 
    vOcr = -(2 * dx**2 + cObnd[bnd[1]] * dy**2) 
    vObc = numpy.ones(npts[0] - 2) * (-2 * dy**2 + cObnd[bnd[2]] * dx**2)
    vOtc = numpy.ones(npts[0] - 2) * (-2 * dy**2 + cObnd[bnd[3]] * dx**2)
    vObl = -(cObnd[bnd[0]] * dy**2 + cObnd[bnd[2]] * dx**2)
    vObr = -(cObnd[bnd[2]] * dy**2 + cObnd[bnd[2]] * dx**2)
    vOtl = -(cObnd[bnd[0]] * dy**2 + cObnd[bnd[3]] * dx**2)
    vOtr = -(cObnd[bnd[2]] * dy**2 + cObnd[bnd[3]] * dx**2)
    dO = numpy.r_[numpy.r_[vObl, vObc, vObr], numpy.tile(numpy.r_[vOcl, vOcc, vOcr], npts[1] - 2), numpy.r_[vOtl, vOtc, vOtr]]
    del vOcc, vOcl, vOcr, vObc, vOtc, vObl, vObr, vOtl, vOtr

    # form east diagonal coefficents and/or vectors
    vEcc = numpy.ones(npts[0] - 2) * dy**2
    vEcl = dy**2
    vEcr = 0
    vEbc = numpy.ones(npts[0] - 2) * dy**2
    vEtc = numpy.ones(npts[0] - 2) * dy**2
    vEbl = dy**2
    vEbr = 0
    vEtl = dy**2
    dE = numpy.r_[numpy.r_[vEbl, vEbc, vEbr], numpy.tile(numpy.r_[vEcl, vEcc, vEcr], npts[1] - 2), numpy.r_[vEtl, vEtc]]
    del vEcc, vEcl, vEcr, vEbc, vEtc, vEbl, vEbr, vEtl

    # form east diagonal coefficents and/or vectors
    vWcc = numpy.ones(npts[0] - 2) * dy**2
    vWcl = 0
    vWcr = dy**2
    vWbc = numpy.ones(npts[0] - 2) * dy**2
    vWtc = numpy.ones(npts[0] - 2) * dy**2
    vWbr = dy**2
    vWtl = 0
    vWtr = dy**2
    dW = numpy.r_[numpy.r_[vWbc, vWbr], numpy.tile(numpy.r_[vWcl, vWcc, vWcr], npts[1] - 2), numpy.r_[vWtl, vWtc, vWtr]]
    del vWcc, vWcl, vWcr, vWbc, vWtc, vWbr, vWtl, vWtr

    # form north diagonal coefficents and/or vectors
    vNcc = numpy.ones(npts[0] - 2) * dx**2
    vNcl = dx**2 
    vNcr = dx**2
    vNbc = numpy.ones(npts[0] - 2) * dx**2
    vNbl = dx**2
    vNbr = dx**2
    dN = numpy.r_[numpy.r_[vNbl, vNbc, vNbr], numpy.tile(numpy.r_[vNcl, vNcc, vNcr], npts[1] - 2)]
    del vNcc, vNcl, vNcr, vNbc, vNbl, vNbr

    # form south diagonal coefficents and/or vectors
    vScc = numpy.ones(npts[0] - 2) * dx**2
    vScl = dx**2 
    vScr = dx**2
    vStc = numpy.ones(npts[0] - 2) * dx**2
    vStl = dx**2
    vStr = dx**2
    dS = numpy.r_[numpy.tile(numpy.r_[vScl, vScc, vScr], npts[1] - 2), numpy.r_[vStl, vStc, vStr]]
    del vScc, vScl, vScr, vStc, vStl, vStr

    A = sparse.diags([dO, dE, dW, dN, dS], [0, 1, -1, npts[0], -npts[0]], shape=(npts[0] * npts[1], npts[0] * npts[1]), format="csc")
    del dO, dE, dW, dN, dS

    return A
def constRHS(res, bnd = [0, 0, 0, 0], xbnd = [0, 1], ybnd = [0, 1], f = 0, g = [0.0, 0.0, 0.0, 0.0]):
    # boundary conditions where the following:
    #        type: 0 = diriclet, 1 = neumann
    #         loc: left, right, bottom, top
    # coordinate domain and resolution w/:
    #                  ?bnd  = {min, max}
    #                  npnts = {Nx, Ny}


    # set up solution parameters - grid
    npts = [res, res]
    dx = (xbnd[1] - xbnd[0]) / npts[0]
    dy = (ybnd[1] - ybnd[0]) / npts[1]

    # set up solution parameters - constants
    cEbnd = [2, -dx]
    cNbnd = [2, -dy]

    if f is 0:
        f = numpy.zeros((res, res))
    elif f is 1:
        x = numpy.linspace(xbnd[0] + dx / 2, xbnd[1] - dx / 2, npts[0])
        y = numpy.linspace(ybnd[0] + dy / 2, ybnd[1] - dy / 2, npts[1])
        X, Y = numpy.meshgrid(x, y)
        f = numpy.sin(numpy.pi * X / xbnd[1]) * numpy.sin(numpy.pi * Y / ybnd[1])

    ### build right hand side vector
    b = dx**2 * dy**2 * f
    b[:,0]  += cEbnd[bnd[0]] * g[0]
    b[:,-1] += cEbnd[bnd[1]] * g[1]
    b[0,:]  += cNbnd[bnd[2]] * g[2]
    b[-1,:] += cNbnd[bnd[3]] * g[3]
    b = numpy.reshape(b, npts[0] * npts[1])

    return b
def constLU(res, bnd = [0, 0, 0, 0], xbnd = [0, 1], ybnd = [0, 1], f = 0, g = [0.0, 0.0, 0.0, 0.0]):
    a = constCoeff(res, bnd, xbnd, ybnd)
    return linalg.splu(a)
def solve(res, bnd = [0, 0, 0, 0], xbnd = [0, 1], ybnd = [0, 1], f = 0, g = [0.0, 0.0, 0.0, 0.0]):
    a = constCoeff(res, bnd, xbnd, ybnd)
    b = constRHS(res, bnd, xbnd, ybnd, f, g)
    return linalg.spsolve(a, b)
def solveLU(lu, res, bnd = [0, 0, 0, 0], xbnd = [0, 1], ybnd = [0, 1], f = 0, g = [0.0, 0.0, 0.0, 0.0]):
    b = constRHS(res, bnd, xbnd, ybnd, f, g)
    return lu.solve(b)
