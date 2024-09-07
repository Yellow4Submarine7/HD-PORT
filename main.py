# main.py

from models.llm_model import LLMModel
from mcts.mcts import MCTS
from kg.knowledge_graph import KnowledgeGraph
from kg.query import KGQuery
from utils.data_loader import load_dataset
from utils.evaluation import evaluate_performance
from config import InferenceConfig

def main(config):
    # 加载模型
    llm_model = LLMModel(config.llm_model_name)
    llm_model.load_state_dict(torch.load(config.llm_model_save_path))
    
    # Load knowledge graph and create query interface
    kg = KnowledgeGraph()
    kg.load_from_file(config.kg_file)
    kg_query = KGQuery(kg)
    
    # Load test dataset
    test_questions = load_dataset(config.test_file)
    
    # Initialize MCTS
    def policy_fn(state):
        actions = kg_query.get_possible_actions(state["current_entity"])
        probs = llm_model.get_action_probabilities(state, actions)
        return actions, list(probs.values())
    
    def value_fn(state):
        return llm_model.evaluate(state)
    
    mcts = MCTS(llm_model, num_simulations=config.num_simulations)
    
    # Inference
    results = []
    for question in test_questions:
        root_state = {"current_entity": question["seed_entity"], "question": question["question"]}
        mcts.search(root_state)
        
        # Select the best action based on visit counts
        best_action = max(mcts.root.children, key=lambda c: c.visits).action
        final_entity = kg_query.execute_action(root_state["current_entity"], best_action)[0]
        
        results.append({
            "question": question["question"],
            "predicted_answer": final_entity,
            "true_answer": question["answer_entities"]
        })
    
    # Evaluate performance
    performance = evaluate_performance(results)
    print("Performance:", performance)

if __name__ == "__main__":
    main(InferenceConfig())