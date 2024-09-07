# train.py

import torch
from models.llm_model import LLMModel
from mcts.mcts import MCTS
from kg.knowledge_graph import KnowledgeGraph
from kg.query import KGQuery
from utils.data_loader import load_dataset
from utils.evaluation import evaluate_performance
from tqdm import tqdm

def train(config):
    # Initialize model
    llm_model = LLMModel(config.llm_model_name)
    
    # Load knowledge graph and create query interface
    kg = KnowledgeGraph()
    kg.load_from_file(config.kg_file)
    kg_query = KGQuery(kg)
    
    # Load dataset
    train_questions = load_dataset(config.train_file)
    test_questions = load_dataset(config.test_file)
    
    # Initialize MCTS
    mcts = MCTS(llm_model, num_simulations=config.num_simulations, max_depth=config.max_depth)
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        for question in tqdm(train_questions):
            root_state = {"current_entity": question["seed_entity"], "question": question["question"]}
            root = mcts.search(root_state)
            mcts.collect_preferences(root)
        
        # HDPO optimization
        mcts.fine_tune_llm()
        
        # Evaluate and record performance
        performance = evaluate_performance(mcts, test_questions, kg_query)
        print(f"Performance after epoch {epoch+1}: {performance}")
    
    # Save trained model
    torch.save(llm_model.state_dict(), config.model_save_path)

if __name__ == "__main__":
    from config import TrainingConfig
    train(TrainingConfig())