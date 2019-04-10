# Graphiti

Making use of linear algebra on GPUs for graph computations: sparse matrix vector multiplications.


## Background

We can represent a graph as adjacency matrix where `A[i][j]` represents an edge from vertex `i` to vertex `j`.

```
    [0, 1, 0]
A = [1, 0, 1]
    [1, 0, 0]
```

Breadth-first search (BFS) can now be implemented as a matrix vector product between the transposed adjacency matrix and a vector for all vertices to start the search from.
Start the search from the single vertex with index `1`.

```
                       [0, 1, 1]   [0]   [1]
y = transpose(A) . v = [1, 0, 0] . [1] = [0]
                       [0, 1, 0]   [0]   [1]
```

We can reach vertices with index `0` and `2` in a single step starting from vertex with index `1`.

The idea behind this project is exploring highly-efficient graph computations on GPUs using sparse matrices.


## Example

See [the main example](./graphiti/__main__.py) and run it via

    python3 -m graphiti


## Docker

Build and run

    docker build -t danieljh/graphiti .
    docker run --rm danieljh/graphiti

And away we go

    docker image save danieljh/graphiti | ssh djh@rig docker image load
    docker run --runtime=nvidia --rm danieljh/graphiti

Requires nvidia-docker and cuda drivers on the gpu rig.


## Headers

    docker run -it -v "${PWD}:/out" --rm nvidia/cuda:10.1-devel cp /usr/local/cuda/include/nvgraph.h /out/


## References

- [Graph Algorithms in the Language of Linear Algebra - by Jeremy Kepner, John Gilbert](https://www.goodreads.com/book/show/11768822-graph-algorithms-in-the-language-of-linear-algebra)
- [nvGraph documentation](https://docs.nvidia.com/cuda/nvgraph/index.html)
- [nvGraph overview](https://developer.nvidia.com/nvgraph)
- [GraphBLAS](graphblas.org)
- [SuiteSparse::GraphBLAS](http://faculty.cse.tamu.edu/davis/suitesparse.html)
- [ctypes](https://docs.python.org/3/library/ctypes.html)


## License

Copyright © 2019 Daniel J. Hofmann

Distributed under the MIT License (MIT).
