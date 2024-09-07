# kg/knowledge_graph.py

import networkx as nx
from typing import List, Tuple, Dict

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_triple(self, head: str, relation: str, tail: str):
        """Add a triple to the knowledge graph."""
        self.graph.add_edge(head, tail, relation=relation)

    def get_relations(self, entity: str) -> List[str]:
        """Get all relations connected to an entity."""
        relations = set()
        for _, _, data in self.graph.out_edges(entity, data=True):
            relations.add(data['relation'])
        for _, _, data in self.graph.in_edges(entity, data=True):
            relations.add(f"inverse_{data['relation']}")
        return list(relations)

    def get_neighbors(self, entity: str, relation: str) -> List[str]:
        """Get all neighboring entities connected by a specific relation."""
        neighbors = []
        if relation.startswith("inverse_"):
            relation = relation[8:]  # Remove "inverse_" prefix
            for src, _ in self.graph.in_edges(entity):
                if self.graph[src][entity].get('relation') == relation:
                    neighbors.append(src)
        else:
            for _, dst in self.graph.out_edges(entity):
                if self.graph[entity][dst].get('relation') == relation:
                    neighbors.append(dst)
        return neighbors

    def get_path(self, start: str, end: str, max_depth: int = 3) -> List[Tuple[str, str, str]]:
        """Find a path between two entities, returning a list of (head, relation, tail) triples."""
        path = nx.shortest_path(self.graph, start, end, weight=None)
        if len(path) > max_depth + 1:
            return []
        
        result = []
        for i in range(len(path) - 1):
            head, tail = path[i], path[i+1]
            relation = self.graph[head][tail]['relation']
            result.append((head, relation, tail))
        return result

    def load_from_file(self, filename: str):
        """Load knowledge graph from a file (assuming tab-separated triples)."""
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                self.add_triple(head, relation, tail)

    def save_to_file(self, filename: str):
        """Save knowledge graph to a file (tab-separated triples)."""
        with open(filename, 'w', encoding='utf-8') as f:
            for head, tail, data in self.graph.edges(data=True):
                f.write(f"{head}\t{data['relation']}\t{tail}\n")