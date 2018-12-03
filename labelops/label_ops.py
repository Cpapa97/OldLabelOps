# label_ops.py

import numpy as np

class LabelOps: #labelops:
    @classmethod
    def generate_neighborhood(self, triangles, num_vertices=None):
        """

        """
        neighborhood = dict()
        if num_vertices is None:
            for i in np.unique(triangles):
                neighborhood[i] = set()
        else:
            for i in range(num_vertices):
                neighborhood[i] = set()
        # Faster than iterating through row by row it turns out.
        row1, row2, row3 = triangles.T
        for i in range(len(triangles)):
            node1 = row1[i]
            node2 = row2[i]
            node3 = row3[i]
            neighborhood[node1].update((node2, node3))
            neighborhood[node2].update((node1, node3))
            neighborhood[node3].update((node1, node2))
            
        return neighborhood
        
    @classmethod
    def compress_labels(self, neighborhood, vertex_labels, as_dict=False):
        """
        
        """
        boundaries = dict()
        visited_nodes = set()
        temp_stack = set()
        neighbor_set = set(neighborhood.keys())
        
        while len(neighbor_set) > 0:
            starting_node = next(iter(neighbor_set))
            found_boundaries = False
            temp_stack.add(starting_node)
            while len(temp_stack) > 0:
                next_node = temp_stack.pop()
                if next_node not in visited_nodes:
                    node_label = vertex_labels[next_node]
                    for neighboring_node in neighborhood[next_node]:
                        if node_label != vertex_labels[neighboring_node]:
                            found_boundaries = True
                            boundaries[next_node] = node_label
                    neighbor_set.remove(next_node)
                    visited_nodes.add(next_node)
                    temp_stack.update(neighborhood[next_node])
            # This takes care of cases where there are disconnected portions of the mesh that are made up of entirely one label.
            if not found_boundaries:
                boundaries[starting_node] = vertex_labels[starting_node]
        
        if as_dict:
            return boundaries
        else:
            return self.dok_as_array(boundaries)
    
    @classmethod
    def decompress_labels(self, neighborhood, boundaries, as_dict=False):
        """
        
        """
        if type(boundaries) is np.ndarray:
            temp = boundaries.T
            boundaries = dict()
            for i, node in enumerate(temp[0]):
                boundaries[node] = temp[1][i]
                
        visited_nodes_with_labels = dict()
        temp_stack = set()
        boundary_points_remaining = boundaries.copy()
        while len(boundary_points_remaining) > 0:
            starting_node = next(iter(boundary_points_remaining))
            temp_stack.add(starting_node)
            while len(temp_stack) > 0:
                next_node = temp_stack.pop()
                if next_node not in visited_nodes_with_labels:
                    if next_node in boundaries:
                        visited_nodes_with_labels[next_node] = boundaries[next_node]
                        del boundary_points_remaining[next_node]
                    else:
                        for neighboring_node in neighborhood[next_node]:
                            if neighboring_node in visited_nodes_with_labels:
                                visited_nodes_with_labels[next_node] = visited_nodes_with_labels[neighboring_node]
                                break
                    temp_stack.update(neighborhood[next_node])
        
        if as_dict:
            return visited_nodes_with_labels
        else:
            return self.dok_as_array(visited_nodes_with_labels)
    
    @classmethod
    def reconstruct_labels(self, neighborhood, boundaries, num_vertices):
        """
        
        """
        return self.reconstruct_from_dict(self.decompress_labels(neighborhood, boundaries, as_dict=True), num_vertices)
    
    @classmethod
    def decompress_targets(self, target_nodes, neighborhood, boundaries):
        """
        
        """
        if type(boundaries) is np.ndarray:
            boundaries = self.array_as_dok(boundaries)
        full_visited_nodes = set()
        temp_stack = set()
        targeted_decompressed_components = dict()
        for target_node in target_nodes:
            visited_nodes = set()
            if target_node not in full_visited_nodes:
                temp_stack.add(target_node)
                label_type = None
                while len(temp_stack) > 0:
                    next_node = temp_stack.pop()
                    if next_node not in visited_nodes:
                        visited_nodes.add(next_node)
                        if next_node in boundaries:
                            label_type = boundaries[next_node]
                        else:
                            temp_stack.update(neighborhood[next_node])
            full_visited_nodes.update(visited_nodes)
            for node in visited_nodes:
                targeted_decompressed_components[node] = label_type
        
        target_node_labels = dict()
        for target_node in target_nodes:
            target_node_labels[target_node] = targeted_decompressed_components[target_node]
        
        return target_node_labels
    
    @classmethod
    def decompress_component(self, target_nodes, neighborhood, boundaries):
        """
        
        """
        if type(boundaries) is np.ndarray:
            boundaries = self.array_as_dok(boundaries)
        full_visited_nodes = set()
        temp_stack = set()
        targeted_decompressed_components = dict()
        for target_node in target_nodes:
            visited_nodes = set()
            if target_node not in full_visited_nodes:
                temp_stack.add(target_node)
                label_type = None
                while len(temp_stack) > 0:
                    next_node = temp_stack.pop()
                    if next_node not in visited_nodes:
                        visited_nodes.add(next_node)
                        if next_node in boundaries:
                            if label_type is None:
                                label_type = boundaries[next_node]
                            boundary_neighbors = [node for node in neighborhood[next_node] if node in boundaries]
                            temp_stack.update([node for node in boundary_neighbors if boundaries[node] == label_type])
                        else:
                            temp_stack.update(neighborhood[next_node])
            full_visited_nodes.update(visited_nodes)
            for node in visited_nodes:
                targeted_decompressed_components[node] = label_type
        
        return targeted_decompressed_components
    
    @classmethod
    def decompress_compartment_type(self, label_type, neighborhood, boundaries):
        """
        
        """
        if type(boundaries) is np.ndarray:
            boundaries = self.array_as_dok(boundaries)
        
        targeted_boundary_nodes = []
        for node, label in boundaries.items():
            if label == label_type:
                targeted_boundary_nodes.append(node)
        targeted_boundary_nodes_set = set(targeted_boundary_nodes)
        
        full_visited_nodes = set()
        temp_stack = set()
        targeted_decompressed_compartments = dict()
        for target_node in targeted_boundary_nodes:
            visited_nodes = set()
            if target_node not in full_visited_nodes:
                temp_stack.add(target_node)
                label_type = None
                while len(temp_stack) > 0:
                    next_node = temp_stack.pop()
                    if next_node not in visited_nodes:
                        visited_nodes.add(next_node)
                        if next_node in boundaries:
                            same_label_nodes = list()
                            for node in neighborhood[next_node]:
                                if node in boundaries:
                                    if node in targeted_boundary_nodes_set:
                                        same_label_nodes.append(node)
                                else:
                                    same_label_nodes.append(node)
                            temp_stack.update(same_label_nodes)
                        else:
                            temp_stack.update(neighborhood[next_node])
            full_visited_nodes.update(visited_nodes)
            for node in visited_nodes:
                targeted_decompressed_compartments[node] = label_type
        
        return targeted_decompressed_compartments
    
    @classmethod
    def dok_as_array(self, label_dictionary, dtype=np.uint32):
        """
        Dictionary of keys with the label as values.
        """
        return np.array(list(label_dictionary.items()), dtype=dtype)
    
    @classmethod
    def array_as_dok(self, label_array):
        """
        
        """
        temp = label_array.T
        label_dict = dict()
        for i, node in enumerate(temp[0]):
            label_dict[node] = temp[1][i]
        return label_dict
    
    @classmethod
    def reconstruct_from_dict(self, label_dict, num_vertices):
        """
        
        """
        reconstructed_array = np.zeros(num_vertices, dtype=np.uint8)
        for i in range(num_vertices):
            if i in label_dict:
                reconstructed_array[i] = label_dict[i]
        return reconstructed_array
    
    @classmethod
    def reconstruct_from_array(self, label_array, num_vertices):
        """
        
        """
        self.dok_as_array(label_array)
