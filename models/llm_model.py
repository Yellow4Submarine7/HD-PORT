import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class LLMModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.policy_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.value_head = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        policy_output = self.policy_head(last_hidden_state[:, -1, :])
        value_output = self.value_head(last_hidden_state[:, -1, :])
        return policy_output, value_output

    def get_action_probabilities(self, state, actions):
        input_text = f"State: {state}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            policy_output, _ = self(inputs.input_ids, inputs.attention_mask)
        probs = torch.softmax(policy_output, dim=-1)
        action_probs = {action: probs[0][self.tokenizer.encode(action, add_special_tokens=False)[0]].item() 
                        for action in actions}
        return action_probs

    def evaluate(self, state):
        input_text = f"State: {state}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            _, value_output = self(inputs.input_ids, inputs.attention_mask)
        return value_output.item()