"""
Find the d-cores of a graph.

The d-core is found by recursively pruning nodes with degrees less than k.

See the following references for details:

An O(m) Algorithm for Cores Decomposition of Networks
Vladimir Batagelj and Matjaz Zaversnik, 2003.
https://arxiv.org/abs/cs.DS/0310049

Generalized Cores
Vladimir Batagelj and Matjaz Zaversnik, 2002.
https://arxiv.org/pdf/cs/0202039

For directed graphs a more general notion is that of D-cores which
looks at (k, l) restrictions on (in, out) degree. The (k, k) D-core
is the k-core.

D-cores: Measuring Collaboration of Directed Graphs Based on Degeneracy
Christos Giatsidis, Dimitrios M. Thilikos, Michalis Vazirgiannis, ICDM 2011.
http://www.graphdegeneracy.org/dcores_ICDM_2011.pdf

Multi-scale structure and topological anomaly detection via a new network \
statistic: The onion decomposition
L. Hébert-Dufresne, J. A. Grochow, and A. Allard
Scientific Reports 6, 31708 (2016)
http://doi.org/10.1038/srep31708

"""
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for

__all__ = [
    "d_core_number",
    "find_d_cores",
    "d_core"
]


@not_implemented_for("multigraph")


#######################################
# DIRECTED CORE DECOMPOSITION BABYYYY #
#######################################


def d_core_number(G):
    """Returns the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    d_core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The d-core is not implemented for graphs with self loops
        or parallel edges or undirected graphs.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    Only node out-degree is considered. For k-cores use core_number(G) node degree is defined to be the
    in-degree + out-degree.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       https://arxiv.org/abs/cs.DS/0310049
    """
    if nx.number_of_selfloops(G) > 0 or nx.is_directed(G) == False:
        msg = (
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise NetworkXError(msg)

    out_degrees = dict(G.out_degree())
    # Sort nodes by degree.
    nodes = sorted(out_degrees, key=out_degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if out_degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (out_degrees[v] - curr_degree))
            curr_degree = out_degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = out_degrees
    nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


find_d_cores = d_core_number


def _d_core_subgraph(G, d_filter, d=None, core=None):
    """Returns the subgraph induced by nodes passing filter `d_filter`.

    Parameters
    ----------
    G : NetworkX graph
       The directed graph to process
    d_filter : filter function
       This function filters the nodes chosen. It takes three inputs:
       A node of G, the filter's cutoff, and the core dict of the graph.
       The function should return a Boolean value.
    d : int, optional
      The order of the directed core. If not specified use the max core number.
      This value is used as the cutoff for the filter.
    core : dict, optional
      Precomputed core numbers keyed by node for the graph `G`.
      If not specified, the core numbers will be computed from `G`.

    """
    if core is None:
        core = d_core_number(G)
    if d is None:
        d = max(core.values())
    nodes = (v for v in core if d_filter(v, d, core))
    return G.subgraph(nodes).copy()


def d_core(G, d=None, d_core_number=None):
    """Returns the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree d or more.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph
    d : int, optional
      The order of the core.  If not specified return the main core.
    d_core_number : dictionary, optional
      Precomputed directed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
      The k-core subgraph

    Raises
    ------
    NetworkXError
      The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    The main core is the core with the largest degree.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    d_core_number

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik,  2003.
       https://arxiv.org/abs/cs.DS/0310049
    """

    def d_filter(v, d, c):
        return c[v] >= d

    return _d_core_subgraph(G, d_filter, d, d_core_number)

####################################
# DIRECTED CORE DECOMPOSITION OVER #
####################################