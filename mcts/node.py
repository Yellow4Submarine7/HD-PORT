# mcts/node.py

import math
from typing import List, Optional

class MCTSNode:
    def __init__(self, state, parent=None, action=None, hop_level=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0
        self.prior = 0
        self.hop_level = hop_level

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            child = MCTSNode(state=None, parent=self, action=action, hop_level=self.hop_level + 1)
            child.prior = prior
            self.children.append(child)

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select_child(self, c_puct=1.0, hop_weight=0.1):
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            uct_score = child.value + c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            hop_preference = hop_weight * (1 / (child.hop_level + 1))
            total_score = uct_score + hop_preference
            if total_score > best_score:
                best_score = total_score
                best_child = child

        return best_child

    def update(self, value, max_hop_level):
        self.visits += 1
        hop_weight = (max_hop_level - self.hop_level + 1) / max_hop_level
        self.value += hop_weight * (value - self.value) / self.visits

    def __repr__(self):
        return f"MCTSNode(action={self.action}, visits={self.visits}, value={self.value:.2f}, hop_level={self.hop_level})"