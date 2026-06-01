import numpy as np

def remove_last_row_and_column(adjacency_matrix):
    new_adjacency_matrix = [row[:-1] for row in adjacency_matrix[:-1]]

    return new_adjacency_matrix

def print_degrees_and_connections(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    adjacency_matrix = remove_last_row_and_column(adjacency_matrix)
    num_nodes -= 1 
    x = {}
    for i in range(num_nodes):
        all_weight = []

        for j in range(num_nodes): 
            if abs(adjacency_matrix[i][j]) > 1e-3:
                all_weight.append(abs(adjacency_matrix[i][j]))
        for j in range(num_nodes):
            if abs(adjacency_matrix[j][i]) > 1e-3:
                all_weight.append(abs(adjacency_matrix[j][i]))

        x[i] = sum(all_weight)

    return x

def get_adjacency_matrix(self):
    nodes = self.get_nodes()
    node_count = len(nodes)
    adjacency_matrix = np.zeros((node_count, node_count), dtype=int)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    for edge in self.get_graph_edges():
        idx1 = node_index[edge.node1]
        idx2 = node_index[edge.node2]
        adjacency_matrix[idx1, idx2] = 1
        
    return adjacency_matrix

