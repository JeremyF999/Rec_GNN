# rec_inference.py

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from rec_gnn import HeteroGNN
import sys
import logging

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Load the trained model
def load_model(model_path, metadata, device):
    model = HeteroGNN(metadata).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model

# Prepare the graph data for inference
def prepare_graph_data(test_df, vectorizer):
    logger.info("Preparing graph data for inference...")
    data = HeteroData()

    # Nodes
    author_ids = test_df['author_id'].unique()
    paper_ids = test_df['paper_id'].unique()
    unit_ids = test_df['unit_id'].unique()

    author_id_map = {id_: idx for idx, id_ in enumerate(author_ids)}
    paper_id_map = {id_: idx for idx, id_ in enumerate(paper_ids)}
    unit_id_map = {id_: idx for idx, id_ in enumerate(unit_ids)}

    data['author'].num_nodes = len(author_ids)
    data['paper'].num_nodes = len(paper_ids)
    data['unit'].num_nodes = len(unit_ids)

    # Node Features
    paper_features = vectorizer.transform(test_df['abstract'].fillna('')).toarray()
    data['paper'].x = torch.tensor(paper_features, dtype=torch.float)
    data['author'].x = torch.randn(data['author'].num_nodes, 128)
    data['unit'].x = torch.randn(data['unit'].num_nodes, 128)

    # Edges
    src = test_df['author_id'].map(author_id_map).values
    dst = test_df['paper_id'].map(paper_id_map).values
    data['author', 'writes', 'paper'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    src = test_df['paper_id'].map(paper_id_map).values
    dst = test_df['unit_id'].map(unit_id_map).values
    data['paper', 'published_in', 'unit'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    data['paper', 'rev_writes', 'author'].edge_index = data['author', 'writes', 'paper'].edge_index.flip(0)
    data['unit', 'rev_published_in', 'paper'].edge_index = data['paper', 'published_in', 'unit'].edge_index.flip(0)

    logger.info("Graph data prepared for inference.")
    return data, author_id_map, unit_id_map

# Perform inference
def recommend_units(test_df, model, data, author_id_map, unit_ids, top_n=10):
    logger.info("Generating recommendations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    x_dict = model(data.x_dict, data.edge_index_dict)

    author_emb = x_dict['author']
    unit_emb = x_dict['unit']

    test_author_ids = test_df['author_id'].unique()
    test_author_indices = [author_id_map.get(id_) for id_ in test_author_ids if id_ in author_id_map]

    if not test_author_indices:
        logger.warning("No test authors found in the training data.")
        return {}

    test_author_indices = torch.tensor(test_author_indices, dtype=torch.long).to(device)
    test_author_emb = author_emb[test_author_indices]
    scores = torch.matmul(test_author_emb, unit_emb.t())

    # Get top-N recommendations
    top_n_units = torch.topk(scores, top_n, dim=1).indices.cpu().numpy()

    recommendations = {}
    for idx, author_id in enumerate(test_author_ids):
        recommended_unit_indices = top_n_units[idx]
        recommended_unit_ids = [unit_ids[i] for i in recommended_unit_indices]
        recommendations[author_id] = recommended_unit_ids

    logger.info("Recommendations generated.")
    return recommendations

# Load the test data and generate recommendations
def main():
    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_csv('./data/test.csv')
    logger.info("Test data loaded.")

    # Load vectorizer
    vectorizer = torch.load('./src/model/vectorizer.pth')
    logger.info("Vectorizer loaded.")

    # Prepare graph data for inference
    data, author_id_map, unit_id_map = prepare_graph_data(test_df, vectorizer)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('./src/model/
                       ', data.metadata(), device)

    # Generate recommendations
    recommendations = recommend_units(test_df, model, data, author_id_map, list(unit_id_map.keys()), top_n=10)
    logger.info(f"Generated recommendations: {recommendations}")

if __name__ == "__main__":
    main()
