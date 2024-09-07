# mcts/mcts.py

import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select_child(self):
        c_puct = 1.0
        return max(self.children, key=lambda c: c.value / (c.visits + 1) + c_puct * c.prior * (self.visits ** 0.5) / (c.visits + 1))

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            child = MCTSNode(state=None, parent=self, action=action)
            child.prior = prior
            self.children.append(child)

    def update(self, value):
        self.visits += 1
        self.value += value

class MCTS:
    def __init__(self, llm_model, num_simulations: int = 100, max_depth: int = 5):
        self.llm_model = llm_model
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.preferences = []

    def search(self, root_state):
        root = MCTSNode(state=root_state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_fully_expanded() and len(search_path) < self.max_depth:
                node = node.select_child()
                search_path.append(node)

            # Expansion
            if len(search_path) < self.max_depth:
                actions, priors = self.llm_model.get_action_probabilities(node.state)
                node.expand(actions, priors)
                node = random.choice(node.children)
                search_path.append(node)

            # Evaluation
            value = self.llm_model.evaluate(node.state)

            # Backpropagation
            for node in reversed(search_path):
                node.update(value)

        return root

    def get_action_probabilities(self, state, temperature=1):
        root = self.search(state)
        visits = [child.visits for child in root.children]
        actions = [child.action for child in root.children]
        
        if temperature == 0:
            best_index = visits.index(max(visits))
            probs = [0] * len(actions)
            probs[best_index] = 1
            return actions, probs
        
        visits = [v ** (1 / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]
        
        return actions, probs

    def collect_preferences(self, root):
        def traverse(node):
            if node.children:
                best_child = max(node.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
                worst_child = min(node.children, key=lambda c: c.value / c.visits if c.visits > 0 else float('inf'))
                self.preferences.append((node.state, best_child.action, 1))
                self.preferences.append((node.state, worst_child.action, 0))
                for child in node.children:
                    traverse(child)
        
        traverse(root)

    def hdpo_loss(self, logits, values):
        preferred_logits = logits[:, 0]
        dispreferred_logits = logits[:, 1]
        preferred_values = values[:, 0]
        dispreferred_values = values[:, 1]

        policy_loss = -torch.log(torch.sigmoid(preferred_logits - dispreferred_logits))
        value_loss = torch.max(torch.zeros_like(preferred_values), 0.1 - (preferred_values - dispreferred_values))
        reg_loss = sum(p.pow(2.0).sum() for p in self.llm_model.parameters())

        return policy_loss.mean() + value_loss.mean() + 0.01 * reg_loss

    def fine_tune_llm(self):
        optimizer = optim.Adam(self.llm_model.parameters(), lr=1e-5)
        self.llm_model.train()

        for preference in self.preferences:
            state, action, value = preference
            
            input_text = f"State: {state}, Action: {action}"
            inputs = self.llm_model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            
            policy_output, value_output = self.llm_model(inputs.input_ids, inputs.attention_mask)
            
            loss = self.hdpo_loss(policy_output, value_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.preferences = []