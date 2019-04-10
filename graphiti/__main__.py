import numpy as np
import scipy.sparse

from .api import getVersion, scopedHandle, scopedGraph, singleSourceShortestPaths

from .ext import setGraphStructureFromSciPySparseMatrix
from .ext import allocateGraphVertexData, allocateGraphEdgeData
from .ext import setGraphEdgeData, getGraphVertexData


def main():
    major, minor, patch = getVersion()
    print("version {}.{}.{}".format(major, minor, patch))

    # adjacency matrix: adj(i, j) == edge(i, j) in the graph
    # represent matrix in compressed sparse column format

    adj = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [1, 0, 0]], dtype=np.float32)

    csc = scipy.sparse.csc_matrix(adj)

    # context managers for library handles and graph handles
    # manages the lifetime or all allocated memory, on host and gpus

    with scopedHandle() as handle, scopedGraph(handle) as graph:

        # initializes gpu graph topology from a sparse matrix (csc or csr)
        setGraphStructureFromSciPySparseMatrix(handle, graph, csc)

        # vertex storage for the shortest path values per vertex
        allocateGraphVertexData(handle, graph, np.float32)

        # edge storage for the edge weights
        allocateGraphEdgeData(handle, graph, np.float32)

        # initialize gpu graph edge weights from sparse matrix
        setGraphEdgeData(handle, graph, csc.data)

        # runs single source shortest path on the gpu, starting at vertex 1
        singleSourceShortestPaths(handle, graph, 1)

        # shortest path values are attached to all three vertices in the gpu graph
        etas = getGraphVertexData(handle, graph, n=adj.shape[0], dtype=np.float32)
        print("etas: {}".format(etas))


        # the stateful interaction with the gpu handles and graphs are
        # prone to errors and still a bit too low-level; ideally we'd
        # have a nice high-level abstraction to simplify all of this


main()
