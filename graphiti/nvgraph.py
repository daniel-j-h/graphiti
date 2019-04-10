"""Low level ctypes bindings for nvgraph.

   This module exposes a low-level Python interface for nvgraph.

   Users should not have to use this module directly; instead
   we provide higher-level and idiomatic Python abstractions.
"""

# Todo:
# - expose remaining api, right now mainly targeting sssp

import enum
import ctypes
import ctypes.util


libnvgraph = ctypes.cdll.LoadLibrary(ctypes.util.find_library("nvgraph"))


class LibraryProperty(enum.IntEnum):
    MAJOR_VERSION = 0
    MINOR_VERSION = 1
    PATCH_LEVEL = 2


class CudaDataType(enum.IntEnum):
    R_16F = 2
    C_16F = 6
    R_32F = 0
    C_32F = 4
    R_64F = 1
    C_64F = 5
    R_8I = 3
    C_8I = 7
    R_8U = 8
    C_8U = 9
    R_32I = 10
    C_32I = 11
    R_32U = 12
    C_32U = 13


class Status(enum.IntEnum):
    SUCCESS = 0
    NOT_INITIALIZED = 1
    ALLOC_FAILED = 2
    INVALID_VALUE = 3
    ARCH_MISMATCH = 4
    MAPPING_ERROR = 5
    EXECUTION_FAILED = 6
    INTERNAL_ERROR = 7
    TYPE_NOT_SUPPORTED = 8
    NOT_CONVERGED = 9
    GRAPH_TYPE_NOT_SUPPORTED = 10


libnvgraph.nvgraphStatusGetString.restype = ctypes.c_char_p
libnvgraph.nvgraphStatusGetString.argtypes = [ctypes.c_int]


class _GraphHandle(ctypes.Structure):
    pass

class _GraphDescr(ctypes.Structure):
    pass

GraphHandle = ctypes.POINTER(_GraphHandle)
GraphDescr = ctypes.POINTER(_GraphDescr)


class Topology(enum.IntEnum):
   CSR = 0
   CSC = 1
   COO = 2


class CSRTopology(ctypes.Structure):
    _fields_ = [("nvertices", ctypes.c_int),
                ("nedges", ctypes.c_int),
                ("source_offsets", ctypes.POINTER(ctypes.c_int)),
                ("destination_indices", ctypes.POINTER(ctypes.c_int))]

class CSCTopology(ctypes.Structure):
    _fields_ = [("nvertices", ctypes.c_int),
                ("nedges", ctypes.c_int),
                ("destination_offsets", ctypes.POINTER(ctypes.c_int)),
                ("source_indices", ctypes.POINTER(ctypes.c_int))]


libnvgraph.nvgraphGetProperty.restype = ctypes.c_int
libnvgraph.nvgraphGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]


libnvgraph.nvgraphCreate.restype = ctypes.c_int
libnvgraph.nvgraphCreate.argtypes = [ctypes.POINTER(GraphHandle)]


libnvgraph.nvgraphDestroy.restype = ctypes.c_int
libnvgraph.nvgraphDestroy.argtypes = [GraphHandle]


libnvgraph.nvgraphCreateGraphDescr.restype = ctypes.c_int
libnvgraph.nvgraphCreateGraphDescr.argtypes = [GraphHandle, ctypes.POINTER(GraphDescr)]


libnvgraph.nvgraphDestroyGraphDescr.restype = ctypes.c_int
libnvgraph.nvgraphDestroyGraphDescr.argtypes = [GraphHandle, GraphDescr]


libnvgraph.nvgraphSetGraphStructure.restype = ctypes.c_int
libnvgraph.nvgraphSetGraphStructure.argtypes = [GraphHandle, GraphDescr, ctypes.c_void_p, ctypes.c_int]

libnvgraph.nvgraphGetGraphStructure.restype = ctypes.c_int
libnvgraph.nvgraphGetGraphStructure.argtypes = [GraphHandle, GraphDescr, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]


libnvgraph.nvgraphAllocateVertexData.restype = ctypes.c_int
libnvgraph.nvgraphAllocateVertexData.argtypes = [GraphHandle, GraphDescr, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]


libnvgraph.nvgraphAllocateEdgeData.restype = ctypes.c_int
libnvgraph.nvgraphAllocateEdgeData.argtypes = [GraphHandle, GraphDescr, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]


libnvgraph.nvgraphSetVertexData.restype = ctypes.c_int
libnvgraph.nvgraphSetVertexData.argtypes = [GraphHandle, GraphDescr, ctypes.c_void_p, ctypes.c_size_t]

libnvgraph.nvgraphGetVertexData.restype = ctypes.c_int
libnvgraph.nvgraphGetVertexData.argtypes = [GraphHandle, GraphDescr, ctypes.c_void_p, ctypes.c_size_t]


libnvgraph.nvgraphSetEdgeData.restype = ctypes.c_int
libnvgraph.nvgraphSetEdgeData.argtypes = [GraphHandle, GraphDescr, ctypes.c_void_p, ctypes.c_size_t]

libnvgraph.nvgraphGetEdgeData.restype = ctypes.c_int
libnvgraph.nvgraphGetEdgeData.argtypes = [GraphHandle, GraphDescr, ctypes.c_void_p, ctypes.c_size_t]


libnvgraph.nvgraphSssp.restype = ctypes.c_int
libnvgraph.nvgraphSssp.argtypes = [GraphHandle, GraphDescr, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]


libnvgraph.nvgraphWidestPath.restype = ctypes.c_int
libnvgraph.nvgraphWidestPath.argtypes = [GraphHandle, GraphDescr, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
