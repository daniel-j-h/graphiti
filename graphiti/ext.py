"""Extensions for interacting with the nvgraph bindings.

   This module exposes numpy and scipy specific abstractions.
"""

# Todo:
# - do we have to depend on the nvgraph bindings here

import ctypes

import numpy
import scipy.sparse

from .nvgraph import libnvgraph
from .api import CudaDataType, CSRTopology, CSCTopology, setGraphStructure, raiseOnError


def _isContiguousMemory(a):
    return a.flags['C_CONTIGUOUS']

def _isQuadratic2dMatrix(m):
    return len(m.shape) == 2 and m.shape[0] == m.shape[1]


def topoViewfromSciPySparseMatrix(m):
    _sparseMatrixTypeMap = {scipy.sparse.csr_matrix: CSRTopology,
                            scipy.sparse.csc_matrix: CSCTopology}

    assert type(m) in _sparseMatrixTypeMap

    TopoTy = _sparseMatrixTypeMap[type(m)]
    topo = TopoTy()

    assert _isContiguousMemory(m.indices)
    assert _isContiguousMemory(m.indptr)
    assert _isQuadratic2dMatrix(m)

    if TopoTy == CSRTopology:
        topo.nvertices = m.shape[0]
        topo.nedges = m.nnz
        topo.source_offsets = m.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        topo.destination_indices = m.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    elif TopoTy == CSCTopology:
        topo.nvertices = m.shape[0]
        topo.nedges = m.nnz
        topo.destination_offsets = m.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        topo.source_indices = m.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    else:
        assert "unknown topology type"

    return topo


def setGraphStructureFromSciPySparseMatrix(handle, desc, m):
    return setGraphStructure(handle, desc, topoViewfromSciPySparseMatrix(m))


_cudaTypeMap = {numpy.float16: CudaDataType.R_16F,
                numpy.float32: CudaDataType.R_32F,
                numpy.float64: CudaDataType.R_64F,
                numpy.int8: CudaDataType.R_8I,
                numpy.uint8: CudaDataType.R_8U,
                numpy.int32: CudaDataType.R_32I,
                numpy.uint32: CudaDataType.R_32U}


def allocateGraphVertexData(handle, desc, dtype, n=1):
    assert dtype in _cudaTypeMap
    assert n >= 0

    ty = _cudaTypeMap[dtype]
    tys = numpy.array([ty] * n)
    tyview = tys.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    raiseOnError(libnvgraph.nvgraphAllocateVertexData(handle, desc, n, tyview))


def allocateGraphEdgeData(handle, desc, dtype, n=1):
    assert dtype in _cudaTypeMap
    assert n >= 0

    ty = _cudaTypeMap[dtype]
    tys = numpy.array([ty] * n)
    tyview = tys.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    raiseOnError(libnvgraph.nvgraphAllocateEdgeData(handle, desc, n, tyview))


def setGraphVertexData(handle, desc, a, i=0):
    assert a.dtype.type in _cudaTypeMap
    assert _isContiguousMemory(a)
    assert i >= 0

    view = a.ctypes.data_as(ctypes.c_void_p)

    raiseOnError(libnvgraph.nvgraphSetVertexData(handle, desc, view, i))


def getGraphVertexData(handle, desc, dtype, n, i=0):
    assert dtype in _cudaTypeMap
    assert n >= 0
    assert i >= 0

    a = numpy.empty(n, dtype)
    view = a.ctypes.data_as(ctypes.c_void_p)

    raiseOnError(libnvgraph.nvgraphGetVertexData(handle, desc, view, i))

    return a


def setGraphEdgeData(handle, desc, a, i=0):
    assert a.dtype.type in _cudaTypeMap
    assert _isContiguousMemory(a)
    assert i >= 0

    view = a.ctypes.data_as(ctypes.c_void_p)

    raiseOnError(libnvgraph.nvgraphSetEdgeData(handle, desc, view, i))


def getGraphEdgeData(handle, desc, dtype, n, i=0):
    assert dtype in _cudaTypeMap
    assert n >= 0
    assert i >= 0

    a = numpy.empty(n, dtype)
    view = a.ctypes.data_as(ctypes.c_void_p)

    raiseOnError(libnvgraph.nvgraphGetEdgeData(handle, desc, view, i))

    return a
