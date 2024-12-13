"""
Lab 2 template
"""
from collections import deque
import time
import matplotlib.pyplot as plt

def read_incidence_matrix(filename: str) -> list[list]:
    """
    :param str filename: path to file
    :returns list[list]: the incidence matrix of a given graph
    """
    edges_vert = {}
    vertices = set()

    with open(filename, mode = 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines[1:-1]:
            line = line.strip().replace(' ','').replace(';','').split('->')
            edges_vert[tuple(line)] = line
            vertices.update(line)
    
    vertices = sorted(vertices)
    edges_vert = dict(sorted(edges_vert.items(), key=lambda x: x[0]))
    incid_matrx = [[1 if vert in edge else 0 for edge in edges_vert] for vert in vertices]
    return incid_matrx

def read_adjacency_matrix(filename: str) -> list[list]:
    """
    :param str filename: path to file
    :returns list[list]: the adjacency matrix of a given graph
    """
    vertices = set()
    edges = set()
    with open(filename, mode = 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:-1]
        for line in lines:
            line = line.strip().replace(' ','').replace(';','').split('->')
            edges.add(tuple(line))
            vertices.update(line)
    adj_matrx = [[1 if (vert, vert_pin) in edges else 0 for vert in vertices] for vert_pin in vertices]
    return adj_matrx

def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict: the adjacency dict of a given graph
    """
    final = {}
    with open(filename, mode = 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:-1]
        for line in lines:
            line = list(map(int, line.strip().replace(' ','').replace(';','').split('->')))
            if line[0] not in final:
                final[line[0]] = [line[1]]
            else:
                final[line[0]].append(line[1])
    return final

def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param list[list] graph: the adjacency dict of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    def dive_dfs(graph: dict[int, list[int]], stack = None, visited = None):
        if not stack:
            stack = [start]
        if not visited:
            visited = set()
        while stack:
            vert = stack.pop(0)
            if vert not in visited:
                visited.add(vert)
                stack+=graph[vert]
        return list(visited)
    
    res = dive_dfs(graph)
    return res

def iterative_adjacency_matrix_dfs(graph: list[list], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = [start]
    stack = [start]

    while stack:
        found = False
        for ref, val in enumerate(graph[stack[-1]]):
            if val == 1 and ref not in stack + visited:
                visited.append(ref)
                stack.append(ref)
                found = True
                break
        if not found:
            stack.pop(-1)
    return visited


def recursive_adjacency_dict_dfs(graph: dict[int, list[int]], start: int, \
                                 visited=None, stack=None) -> list[int]:
    """
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = [start] if visited is None else visited
    stack = [start] if stack is None else stack

    found = False
    for val in graph[stack[-1]]:
        if val not in stack + visited:
            visited.append(val)
            stack.append(val)
            found = True
            break
    if not found:
        stack.pop(-1)
    
    if stack:
        recursive_adjacency_dict_dfs(graph, stack[-1], visited, stack)

    return visited

def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    def dive_dfs(graph: list[list[int]], node: int, visited = None):
        if not visited:
            visited = set()
        visited.add(node)
        for index, vert in enumerate(graph[node]):
            if vert == 1 and (index not in visited):
                dive_dfs(graph, index, visited)
        return list(visited)
    res = dive_dfs(graph, start)
    return res


def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    result = [start]
    visited = set([start])
    queue = []
    queue.extend(sorted(graph[start]))
    while len(queue) >= 1:
        if queue[0] not in visited:
            result.append(queue[0])
            visited.add(queue[0])
            queue.extend(graph[queue[0]])
        queue.pop(0)
    return result


def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) ->list[int]:
    """
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    quee = deque()
    quee.append(start)
    visited = {start}
    while quee:
        vert = quee.popleft()
        for ind, el in enumerate(graph[vert]):
            if el == 1 and ind not in visited:
                visited.add(ind)
                quee.append(ind)
    return list(visited)

def custom_matrix_dfs(graph: list[list], start:int):
    """
    dfs that looks for the longest path
    param: graph - graph's adjacency matrix
           start - initial vertex
    return: int - length of the longest path.
    """
    n = len(graph)
    distance = [-1] * n
    queue = [start]
    distance[start] = 0

    while queue:
        node = queue.pop(0)

        for neighbor in range(n):
            if graph[node][neighbor] == 1 and distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return max(distance)

def custom_dict_dfs(graph: dict[list[int]], start:int):
    """
    dfs that looks for the longest path
    param: graph - graph's adjacency dictionary
           start - initial vertex
    return: int - length of the longest path.
    """
    visited = {v: False for v in graph}
    distance = {v: float('inf') for v in graph}
    queue = [start]
    visited[start] = True
    distance[start] = 0

    while queue:
        node = queue.pop(0)

        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return max(distance.values())

def adjacency_matrix_radius(graph: list[list]) -> int:
    """
    :param list[list] graph: the adjacency matrix of a given graph
    :returns int: the radius of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    """
    radius = []
    for ver in range(len(graph)):
        radius.append(custom_matrix_dfs(graph, ver))
    return min(radius)


def adjacency_dict_radius(graph: dict[int: list[int]]) -> int:
    """
    :param dict graph: the adjacency list of a given graph
    :returns int: the radius of the graph
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [1]})
    2
    """
    radius = []
    for ver in range(len(graph)):
        radius.append(custom_dict_dfs(graph, ver))
    return min(radius)

def comparative_analysis(filename: str) -> float:
    """
    This function calculates time complexety for each function-
    algorithm in laboratory work.
    -----------------------------------------------------------
    param: filename - path to a file to write algorithms'
    time complexities.
    return: dict[list[float]] - algorithms' time complexeties,
    where keys - algorithms names, values - time complexety on
    different data.
    """
    def record_time(algorithm, graph, start):
        """
        This function calculates time complexity for
        and algorithm represented in a form of lambda-func.
        ---------------------------------------------------
        param: algorithm - lambda function of the algorithm
               graph - data in adjacency matrix or dict form.
               start - number of intial vertex or 'None'
        return: float - time complexity of an algorithm
        on a particular data
        """
        start_time = time.perf_counter()
        res = algorithm(graph, start)
        end_time = time.perf_counter()
        return format((end_time - start_time) * (10 ** 6), '.4f')

    all_algo_matr = [
        ('iterative_adjacency_matrix_dfs', lambda graph, start : iterative_adjacency_matrix_dfs(graph, start)),
        ('recursive_adjacency_matrix_dfs', lambda graph, start : recursive_adjacency_matrix_dfs(graph, start)),
        ('iterative_adjacency_matrix_bfs', lambda graph, start : iterative_adjacency_matrix_bfs(graph, start)),
        ('adjacency_matrix_radius', lambda graph, start: adjacency_matrix_radius(graph))]

    all_algo_dict = [
        ('iterative_adjacency_dict_dfs', lambda graph, start : iterative_adjacency_dict_dfs(graph, start)),
        ('recursive_adjacency_dict_dfs', lambda graph, start : recursive_adjacency_dict_dfs(graph, start)),
        ('iterative_adjacency_dict_bfs', lambda graph, start : iterative_adjacency_dict_bfs(graph, start)),
        ('adjacency_dict_radius', lambda graph, start: adjacency_dict_radius(graph))]

    filenames = ['input.dot', 'input2.dot']

    all_adj_matrix = [read_adjacency_matrix(filename) for filename in filenames]
    all_adj_dict = [read_adjacency_dict(filename) for filename in filenames]

    results = {algo[0]: [] for algo in all_algo_matr + all_algo_dict}

    for algo in all_algo_matr:
        if 'radius' in algo[0]:
            for file in all_adj_matrix:
                res = record_time(algo[1], file, 'None')
                results[algo[0]].append(res)
        else:
            for file in all_adj_matrix:
                res = record_time(algo[1], file, 0)
                results[algo[0]].append(res)

    for algo in all_algo_dict:
        if 'radius' in algo[0]:
            for index, file in enumerate(all_adj_dict):
                res = record_time(algo[1], file, 'None')
                results[algo[0]].append(res)
        else:
            for index, file in enumerate(all_adj_dict):
                res = record_time(algo[1], file, 0)
                results[algo[0]].append(res)

    with open(filename, mode='w', encoding='utf-8') as file:
        for key in results:
            file.write(f'{key}: {results[key]}\n\n')
    return results

def visulize_results(results: dict):
    """
    This function takes time complexeties
    of algorithms and depicts them on
    diagram.
    ------------------------------------
    param: results - time complexeties of
    algorithms.
    return: None
    """
    datasets = ['3V graph', '25V graph']
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
    
    if len(datasets) == 1:
        axes = [axes]
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple', 'pink', 'lightcoral', 'gold']
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        times = [float(result[i]) for result in results.values()]
        algorithms = list(results.keys())

        if len(algorithms) > len(colors):
            colors = colors * ((len(algorithms) // len(colors)) + 1)

        ax.barh(algorithms, times, color=colors[:len(algorithms)])
        
        ax.set_title(f'Час для {dataset}')
        ax.set_xlabel('Час (мікросек.)')
        ax.set_xlim(0, max(times) * 1.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
