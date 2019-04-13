import numpy as np
import scipy.sparse

from .api import getVersion
from .ext import scopedSimpleGraph


def main():
    major, minor, patch = getVersion()
    print("version {}.{}.{}".format(major, minor, patch))

    # adjacency matrix: adj(i, j) == edge(i, j) in the graph
    # represent matrix in compressed sparse column format

    adj = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [1, 0, 0]], dtype=np.float32)

    csc = scipy.sparse.csc_matrix(adj)

    with scopedSimpleGraph(csc) as graph:
        etas = graph.singleSourceShortestPaths(source=1)
        print("etas {}".format(etas))


main()
