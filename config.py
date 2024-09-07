# config.py

class TrainingConfig:
    llm_model_name = "meta-llama/Llama-3-8b-base"
    kg_file = "path/to/knowledge_graph.txt"
    train_file = "path/to/train_data.json"
    num_epochs = 10
    num_simulations = 100
    max_depth = 5
    model_save_path = "path/to/save/llm_model.pth"
    hdpo_learning_rate = 1e-5

class InferenceConfig:
    model_name = "meta-llama/Llama-3-8b-base"
    model_path = "path/to/saved/llm_model.pth"
    kg_file = "path/to/knowledge_graph.txt"
    test_file = "path/to/test_data.json"
    num_simulations = 100
    max_depth = 5