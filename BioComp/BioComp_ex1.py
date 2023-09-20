import timeit
import networkx as nx
import itertools


def get_n_from_user():
    n = None
    while type(n) != int:
        # get the number of nodes from the user
        n = int(input("Please enter the number of nodes: "))
        return n


def maskList(n, mask):
    # return a list that gives res[i]==True if the i-th edge is in the graph
    res = [False] * (n ** 2 - n)
    for i in mask:
        res[i] = True
    return res


def kDigraphs(n, k):
    # this part takes subsets of the nodes and create a container of all the possible edges
    # for this permutation of nodes and return it
    possible_edges = [(u, v) for u, v in itertools.product(range(n), repeat=2) if u != v]

    # go over all the possibilities of k edges out of all the n*(n-1) edges:
    for edge_mask in itertools.combinations(range(n * n - n), k):
        # The result is already sorted
        yield tuple(edge for include, edge in zip(maskList(n, edge_mask), possible_edges) if include)


def uniqueMotifs(n, k):
    # generate all the possible edges for a graph with k nodes
    # filter isomorphic graphs
    already_seen = set()
    for graph in kDigraphs(n, k):
        if graph not in already_seen:
            # add all permutation of the current graph to the set of graphs we have already seen
            # (all permutations=all graphs isomorphic to the current one)
            # this part takes a permutation of the nodes and creates a new graph
            # and checks if it is isomorphic to the current one we are checking
            already_seen |= {
                tuple(sorted((perm[i], perm[j]) for i, j in graph))
                for perm in itertools.permutations(range(n))
            }
            yield graph


def k_motifs_to_str(n, k):
    # create a directed graph object
    k_graph = map(nx.DiGraph, uniqueMotifs(n, k))
    # filter the graphs that are not weakly connected
    weakly_conn = filter(nx.is_weakly_connected,
                         filter(lambda g: len(g) == n, k_graph))
    # return the graph
    return weakly_conn


def calcMotifs_k(n, k):
    # special case for n = 1
    if n <= 1:
        return '', 0
    # create a directed graph object
    motifs = k_motifs_to_str(n, k)
    # write the motif to a string
    motif_str = f'#{k}\n'
    count_of_motifs_curK = 0
    # iterate over all the motifs
    for motif in motifs:
        # get edges list:
        edges = list(motif.edges())
        # update the count of motifs and the string
        count_of_motifs_curK += 1
        motif_str += f'{edges}\n'

    return motif_str, count_of_motifs_curK


def createAllmotifsOfSizeN(n):
    # create a directed graph object
    sumOfMotifs = 0
    # an str to save all the motifs and write it to a file
    motifs = ""
    # iterate over all the possible edges number for a graph with n nodes
    for i in range(n - 1, n ** 2 - n + 1):
        cur_str, cur_count = calcMotifs_k(n, i)
        # add the current motifs to the str
        motifs += cur_str
        # add the current count to the sum
        sumOfMotifs += cur_count
    return motifs, sumOfMotifs


def main():
    # get the number of nodes from the user
    n = get_n_from_user()

    # measure the time it takes to run the program
    start = timeit.default_timer()
    motifs = ''
    for i in range(1, n + 1):
        with open("motifs.txt", 'a') as f:
            f.write(f'#n={i}\n')
            motifs, sumOfMotifs = createAllmotifsOfSizeN(i)
            # write the motifs to a file and the motifs count
            f.write(motifs)
    # measure the time it took to run the program
    stop = timeit.default_timer()
    # print the time it took to run the program
    print('Running time in seconds: ', stop - start)
    # print the sum of all the motifs
    print(sumOfMotifs)


if __name__ == '__main__':
    main()
