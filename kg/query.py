# kg/query.py

from typing import List, Dict
from kg.knowledge_graph import KnowledgeGraph

class KGQuery:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def get_possible_actions(self, entity: str) -> List[str]:
        """Get all possible actions (relations) from a given entity."""
        return self.kg.get_relations(entity)

    def execute_action(self, entity: str, action: str) -> List[str]:
        """Execute an action (follow a relation) from a given entity."""
        return self.kg.get_neighbors(entity, action)

    def check_answer(self, entity: str, question: Dict) -> bool:
        """Check if the current entity is a valid answer to the question."""
        # This is a simplified implementation. In practice, you might need more sophisticated
        # answer checking logic, possibly involving the critical model.
        return entity in question.get('answer_entities', [])

    def get_context(self, entity: str) -> Dict[str, List[str]]:
        """Get the context (neighboring entities and relations) for a given entity."""
        context = {"relations": [], "entities": []}
        for relation in self.kg.get_relations(entity):
            context["relations"].append(relation)
            neighbors = self.kg.get_neighbors(entity, relation)
            context["entities"].extend(neighbors)
        return context

    def find_path(self, start: str, end: str, max_depth: int = 3) -> List[Dict[str, str]]:
        """Find a path between two entities, returning a list of steps."""
        path = self.kg.get_path(start, end, max_depth)
        return [{"from": h, "relation": r, "to": t} for h, r, t in path]