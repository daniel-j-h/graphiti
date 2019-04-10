"""High-level Python bindings on top of nvgraph.

   This module exposes abstractions on top of the nvgraph bindings.
"""

# Todo:
# - better abstract away stateful gpu graph interaction
# - investigate graphblas abstracting away nvgraph impl

import ctypes
import contextlib

from .nvgraph import libnvgraph, LibraryProperty, CudaDataType, Status, GraphHandle, GraphDescr, Topology, CSRTopology, CSCTopology


class StatusError(Exception):
    def __init__(self, status):
        self.status = status
        message = libnvgraph.nvgraphStatusGetString(status).decode("utf-8")
        super().__init__(message)


def raiseOnError(status):
    if status != Status.SUCCESS:
        raise StatusError(status)


def getVersion():
    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    patch = ctypes.c_int(0)

    raiseOnError(libnvgraph.nvgraphGetProperty(LibraryProperty.MAJOR_VERSION, ctypes.byref(major)))
    raiseOnError(libnvgraph.nvgraphGetProperty(LibraryProperty.MINOR_VERSION, ctypes.byref(minor)))
    raiseOnError(libnvgraph.nvgraphGetProperty(LibraryProperty.PATCH_LEVEL, ctypes.byref(patch)))

    return major.value, minor.value, patch.value


@contextlib.contextmanager
def scopedHandle():
    handle = GraphHandle()
    raiseOnError(libnvgraph.nvgraphCreate(ctypes.byref(handle)))
    yield handle
    raiseOnError(libnvgraph.nvgraphDestroy(handle))


@contextlib.contextmanager
def scopedGraph(handle):
    desc = GraphDescr()
    raiseOnError(libnvgraph.nvgraphCreateGraphDescr(handle, ctypes.byref(desc)))
    yield desc
    raiseOnError(libnvgraph.nvgraphDestroyGraphDescr(handle, desc))


def setGraphStructure(handle, desc, topo):
    _topoTypeMap = {CSRTopology: Topology.CSR,
                    CSCTopology: Topology.CSC}

    assert type(topo) in _topoTypeMap

    ty = ctypes.c_int(_topoTypeMap[type(topo)])
    raiseOnError(libnvgraph.nvgraphSetGraphStructure(handle, desc, ctypes.byref(topo), ty))


def getGraphStructure(handle, desc):
    ty = ctypes.c_int()

    raiseOnError(libnvgraph.nvgraphGetGraphStructure(handle, desc, None, ctypes.byref(ty)))

    return Topology(ty)


def singleSourceShortestPaths(handle, desc, source, es=0, vs=0):
    assert source >= 0
    assert es >= 0
    assert vs >= 0

    s = ctypes.c_int(source)

    raiseOnError(libnvgraph.nvgraphSssp(handle, desc, es, ctypes.byref(s), vs))


def widestPaths(handle, desc, source, es=0, vs=0):
    assert source >= 0
    assert es >= 0
    assert vs >= 0

    s = ctypes.c_int(source)

    raiseOnError(libnvgraph.nvgraphWidestPath(handle, desc, es, ctypes.byref(s), vs))
