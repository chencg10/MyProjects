# import itertools
# import networkx as nx
#
#
# def get_n_from_user():
#     while True:
#         try:
#             n = int(input('Please enter a number: '))
#             if n < 0:
#                 print('Please enter a positive number')
#                 continue
#             return n
#         except ValueError:
#             print('Please enter a number')
#
#
# def create_clique_graph(n):
#     # create a directed graph object
#     clique = nx.DiGraph()
#
#     # Add nodes to the graph
#     clique.add_nodes_from(range(1, n + 1))
#
#     # Add edges to form a clique
#     for i in range(1, n + 1):
#         for j in range(1, n + 1):
#             if i != j:
#                 clique.add_edge(i, j)
#
#     # print the edges
#     print(clique.edges())
#
#     return clique
#
#
# def search_SCC_in_graph(graph):
#     # find the strongly connected components in the graph
#     SCC = nx.connected_components(graph)
#     # print the SCC
#     for i, c in enumerate(SCC):
#         print(f"SCC #{i + 1}: {c}")
#
#
# def DFS_G(graph, i, isVisited, stack):
#     # mark the current node as visited
#     isVisited[i] = True
#     # iterate over all the neighbors of the current node
#     for j in graph.neighbors(i):
#         if isVisited[j] == False:
#             DFS_G(graph, j, isVisited, stack)
#     # push the current node to the stack
#     stack.append(i)
#
#
# def DFS_G_T(transpose_graph, i, isVisited, stack, SCC_list):
#     isVisited[i] = True
#     SCC_list.add(i)
#
#
#
# def SCC(graph):
#     # find the strongly connected components in the graph
#     numOfNodes = graph.number_of_nodes()
#     isVisited = [False] * numOfNodes
#     stack = []
#     SCC_list = []
#
#     # iterate over all the nodes
#     for i in range(numOfNodes):
#         if isVisited[i] == False:
#             # find the SCC of the current node
#             SCC_list.append(DFS_G(graph, i, isVisited, stack))
#
#     # create a transpose graph
#     transpose_graph = graph.reverse()
#
#     # mark all the nodes as not visited
#     isVisited = [False] * numOfNodes
#     # iterate over all the nodes using the stack
#     while stack:
#         # pop the top node from the stack
#         i = stack.pop()
#         if isVisited[i] == False:
#             # find the SCC of the current node
#             SCC_list.append(DFS_G_T(transpose_graph, i, isVisited, stack, SCC_list))
#
#
#
#
# def main():
#     # get the number of nodes from the user
#     n = get_n_from_user()
#     # create the clique graph
#     clique = create_clique_graph(n)
#     # search for the strongly connected components in the graph
#     search_SCC_in_graph(clique)
#
#
# main()
import timeit

import networkx as nx
import itertools
import time
import os
import matplotlib.pyplot as plt
import numpy as np


def mask_list(n, mask):
    # return a list that gives res[i]==True if the i-th edge is in the graph
    res = [False] * (n ** 2 - n)
    for i in mask:
        res[i] = True
    return res


def k_digraphs(n, k):
    '''
    generate all the directed graphs with exactly k edges
    '''
    possible_edges = [
        (i, j) for i, j in itertools.product(range(n), repeat=2) if i != j
    ]

    # go over all the possibilities of k edges out of all the n*(n-1) edges:
    for edge_mask in itertools.combinations(range(n * n - n), k):
        # The result is already sorted
        yield tuple(edge for include, edge in zip(mask_list(n, edge_mask), possible_edges) if include)


def unique_motifs(n, k):
    '''
    generate all the unique graphs with exactly k edges (up to isomorphism)
    '''
    already_seen = set()
    for graph in k_digraphs(n, k):
        if graph not in already_seen:
            # add all permutation of the current graph to the set of graphs we have already seen
            # (all permutations=all graphs isomorphic to the current one)
            already_seen |= {
                tuple(sorted((perm[i], perm[j]) for i, j in graph))
                for perm in itertools.permutations(range(n))
            }
            yield graph


def k_motifs(n, k):
    '''
    return all directed graphs with exactly k edges which keep the graph with n nodes connected
    '''
    k_graphs = map(nx.DiGraph, unique_motifs(n, k))
    connected_graphs = filter(nx.is_weakly_connected,
                              filter(lambda g: len(g) == n,
                                     k_graphs)
                              )
    return connected_graphs


def all_motifs(n, format='list', verbose=False):
    '''
    return all graphs of n nodes which are connected, that are unique up to isomorphism
    list them and the sum of how many are there
    '''
    sum = 0
    str_all_motifs = ''
    for k in range(n - 1, n ** 2 - n + 1):
        # Go over all the graphs of size k \in [n-1,n**2-n] (if k<n-1 the graph cannot be connected)
        if format == 'list':
            cur_str, cur_count = k_motifs_to_str(n, k, verbose)
        else:
            motifs = k_motifs(n, k)
            cur_str, cur_count = sum_k_motifs(motifs, verbose=verbose)
        str_all_motifs += cur_str
        sum += cur_count
    return str_all_motifs, sum


def sum_k_motifs(motifs, verbose=False):
    count = 0
    str_k_motifs = ''
    for motif in motifs:
        str_k_motifs += motif_to_str(motif, verbose)
        count += 1
    return str_k_motifs, count


def motif_to_str(motif, verbose=False):
    motif_str = f'#k={motif.number_of_edges()}\n'
    for u, v, d in motif.edges.data():
        motif_str += f'{u} {v}\n'
    if verbose:
        print(motif_str)
    return motif_str


def k_motifs_to_str(n, k, verbose=False):
    if n <= 1:
        return '', 0
    motifs = k_motifs(n, k)
    res = f'#{k}\n'
    count = 0
    for motif in motifs:
        edges = list(motif.edges())
        res += f'{edges}\n'
        count += 1
    return res, count


def main_n(n, format='list', verbose=False):
    '''
    give the result for the question for graph of size n
    '''
    res_str, count = all_motifs(n, format='list', verbose=verbose)
    res_str = f'n={n}\ncount={count}\n' + res_str
    if verbose:
        print(f'n={n}\ncount={count}')
    return res_str


def save_motifs(n, path='.', verbose=False):
    '''
    save a file with the details of graph of n nodes
    '''
    format = 'D-%d-%m-T-%H-%M-%S'
    stamp = time.strftime(format, time.localtime())
    file_name = f'M-{n}-{stamp}.txt'
    full_path = os.path.join(path, file_name)
    with open(full_path, 'w') as f:
        f.write(main_n(n, verbose))


def save_motifs_range(start, end=None, path='.', verbose=False):
    '''
    save a file with details on all graphs in the range [start,end]
    or [1,start] if end is None
    '''
    if end is None:
        start, end = 1, start
    assert end >= start
    format = 'D-%d-%m-T-%H-%M-%S'  # D-{day}-{month}-T-{hour}-{minute}-{second}
    stamp = time.strftime(format, time.localtime())
    file_name = f'M-{start}to{end}-{stamp}.txt'  # M-{motif range}-D-{day}-T-{time of day}
    full_path = os.path.join(path, file_name)
    res = ''
    for n in range(start, end + 1):
        with open(full_path, 'a') as f:
            res = main_n(n, verbose)
            f.write(res)
            f.write('\n')

save_motifs_range(5)
