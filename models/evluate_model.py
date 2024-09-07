# models/rollout_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

class RolloutModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", hidden_size=768):
        super(RolloutModel, self).__init__()
        self.llm = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.value_head = nn.Linear(self.llm.config.hidden_size, 1)
        
    def forward(self, state):
        inputs = self.tokenizer(state, return_tensors="pt", padding=True, truncation=True)
        outputs = self.llm(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token representation
        value = self.value_head(pooled_output)
        return value.squeeze()

    def train_step(self, state, target_value, optimizer):
        optimizer.zero_grad()
        predicted_value = self(state)
        loss = nn.MSELoss()(predicted_value, target_value)
        loss.backward()
        optimizer.step()
        return loss.item()

def train_rollout_model(model, train_data, epochs=10, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for state, target_value in train_data:
            loss = model.train_step(state, target_value, optimizer)
            total_loss += loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}")