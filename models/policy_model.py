# models/policy_model.py

import openai
from typing import List, Dict
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class PolicyModel:
    def __init__(self, model_name: str = "meta-llama/Llama-3-8b-base"):
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

    def set_llm(self, llm):
        self.model = llm

    def get_action_probabilities(self, state: Dict, possible_actions: List[str]) -> Dict[str, float]:
        input_text = f"State: {state}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        action_probs = {action: probs[0][self.tokenizer.encode(action, add_special_tokens=False)[0]].item() 
                        for action in possible_actions}
        return action_probs