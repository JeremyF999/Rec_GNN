import pandas as pd
import numpy as np
import torch
from torch.nn import Module
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
import random
import chardet
import sys
import traceback
from tqdm import tqdm
import math
import time
import os
import joblib

class HeteroGNN(Module):
    def __init__(self, metadata, hidden_dim=32, dropout=0.1):
        super().__init__()
        # Define the first convolution layer for each relation type
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('paper', 'published_in', 'unit'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('paper', 'rev_writes', 'author'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('unit', 'rev_published_in', 'paper'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
        }, aggr='mean')

        # Define a linear transformation layer after the graph convolution
        self.lin = Linear(hidden_dim, 32)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        # Perform graph convolution and activation
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(self.lin(x)) for key, x in x_dict.items()}
        return x_dict

# Function to set up logging
def setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more detailed output
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Function to evaluate recommendations
def evaluate_recommendations(test_df, author_emb, unit_emb, author_id_map, unit_ids, top_n=10):
    logger.info("Evaluating recommendations...")

    # Compute scores
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

    # Evaluation Metrics
    test_author_units = test_df.groupby('author_id')['unit_id'].apply(list).to_dict()

    y_true, y_scores = [], []
    hit, mrr_hit, ndgg_i = 0, 0, 0
    rec_count, test_count = 0, 0
    hr_count = len(test_author_units)

    for idx, author_idx in enumerate(test_author_indices):
        author_id = test_author_ids[idx]
        true_units = test_author_units.get(author_id, [])
        recommended_unit_indices = top_n_units[idx]
        recommended_unit_ids = [unit_ids[i] for i in recommended_unit_indices]
        relevance = [1 if unit in true_units else 0 for unit in recommended_unit_ids]
        y_true.append(relevance)
        y_scores.append([1] * top_n)

        # MRR & NDGG Calculation
        for i, unit in enumerate(recommended_unit_ids):
            if unit in true_units:
                mrr_hit += 1
                ndgg_i += 1 / (math.log2(1 + i + 1))
                break

        # Precision & Recall Calculation
        hit += sum([1 for unit in recommended_unit_ids if unit in true_units])
        rec_count += top_n
        test_count += len(true_units)

    # Metrics Calculations
    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    hr = hit / (1.0 * hr_count)
    mrr = mrr_hit / hr_count
    ndgg = ndgg_i / hr_count
    f1 = (2 * precision * recall) / (recall + precision) if (precision + recall) > 0 else 0

    results = {
        'Precision': precision,
        'Recall': recall,
        'Hit Rate': hr,
        'MRR': mrr,
        'NDGG': ndgg,
        'F1-score': f1
    }

    logger.info(f"Evaluation Metrics: {results}")
    return results

def get_train_test_set():
    global train_df, test_df, train_pairs, trainSet_author_abstract, trainSet_unit_abstract
    trainSet_author_paperID_abstract = dict()
    trainSet_unit_paperID_abstract = dict()
    trainSet_author_abstract = dict()
    trainSet_unit_abstract = dict()

    # Detect encoding for train.csv
    logger.info("Detecting encoding for train.csv...")
    with open('./data/train.csv', 'rb') as f:
        result = chardet.detect(f.read())
        train_encoding = result['encoding']
        logger.info(f"Detected encoding for train.csv: {train_encoding}")

    # Detect encoding for test.csv
    logger.info("Detecting encoding for test.csv...")
    with open('./data/test.csv', 'rb') as f:
        result = chardet.detect(f.read())
        test_encoding = result['encoding']
        logger.info(f"Detected encoding for test.csv: {test_encoding}")

    # Load train and test datasets using detected encodings
    logger.info("Loading train and test datasets...")
    try:
        train_df = pd.read_csv("./data/train.csv", encoding=train_encoding, low_memory=False)
        test_df = pd.read_csv("./data/test.csv", encoding=test_encoding, low_memory=False)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Prepare author and unit abstracts
    logger.info("Processing training data for author and unit abstracts...")
    
    # Collect abstracts by author
    for _, row in train_df.iterrows():
        author_id, paper_id, abstract = row['author_id'], row['paper_id'], row['abstract']
        trainSet_author_paperID_abstract.setdefault(author_id, {})
        trainSet_author_paperID_abstract[author_id][paper_id] = abstract

    # Aggregate abstracts per author
    for author, papers in trainSet_author_paperID_abstract.items():
        abstract_list = [abstract for _, abstract in papers.items()]
        trainSet_author_abstract[author] = " ".join(abstract_list)
    
    # Collect abstracts by unit
    for _, row in train_df.iterrows():
        unit_id, paper_id, abstract = row['unit_id'], row['paper_id'], row['abstract']
        trainSet_unit_paperID_abstract.setdefault(unit_id, {})
        trainSet_unit_paperID_abstract[unit_id][paper_id] = abstract

    # Aggregate abstracts per unit
    for unit, papers in trainSet_unit_paperID_abstract.items():
        abstract_list = [abstract for _, abstract in papers.items()]
        trainSet_unit_abstract[unit] = " ".join(abstract_list)

    # Create mappings for author_id and unit_id to indices
    logger.info("Creating author and unit ID mappings...")
    author_ids = pd.concat([train_df['author_id'], test_df['author_id']]).unique()
    unit_ids = pd.concat([train_df['unit_id'], test_df['unit_id']]).unique()

    # Generate mappings from IDs to indices
    author_id_map = {id_: idx for idx, id_ in enumerate(author_ids)}
    unit_id_map = {id_: idx for idx, id_ in enumerate(unit_ids)}

    # Prepare training pairs for recommendation
    logger.info("Creating training pairs...")
    train_positive_pairs = train_df[['author_id', 'unit_id']].drop_duplicates()
    train_positive_pairs['label'] = 1

    # Create indices for author and unit using the mapping
    logger.info("Mapping author_id and unit_id to indices...")
    train_positive_pairs['author_idx'] = train_positive_pairs['author_id'].map(author_id_map)
    train_positive_pairs['unit_idx'] = train_positive_pairs['unit_id'].map(unit_id_map)

    # Combine samples
    train_pairs = train_positive_pairs.reset_index(drop=True)
    logger.info(f"Total training pairs: {len(train_pairs)}")

    # Verify the index columns are created correctly
    logger.info(f"Training pairs sample:\n{train_pairs[['author_id', 'author_idx', 'unit_id', 'unit_idx']].head()}")

    logger.info("Training and test set preparation completed.")

# Function for automatic hyperparameter tuning
def hyperparameter_tuning(train_df, test_df, param_grid, data):
    best_metrics = None
    best_params = None
    all_results = []
    best_model_path = "./src/model/best_model.pth"  # Path to save the best model

    for params in ParameterGrid(param_grid):
        logger.info(f"Evaluating with parameters: {params}")
        model = HeteroGNN(data.metadata(), hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # Training loop
        for epoch in range(params['num_epochs']):
            model.train()

            # Reset losses for this epoch
            losses = []

            author_indices = torch.tensor(train_pairs['author_idx'].values, dtype=torch.long).to(device)
            unit_indices = torch.tensor(train_pairs['unit_idx'].values, dtype=torch.long).to(device)
            labels = torch.tensor(train_pairs['label'].values, dtype=torch.float).to(device)

            for i in range(0, len(author_indices), params['batch_size']):
                # Zero the gradients for the current batch
                optimizer.zero_grad()

                # Compute embeddings within each batch to avoid graph reuse
                x_dict = model(data.x_dict, data.edge_index_dict)
                author_emb = x_dict['author']
                unit_emb = x_dict['unit']

                # Get the batch data
                batch_author_indices = author_indices[i:i+params['batch_size']]
                batch_unit_indices = unit_indices[i:i+params['batch_size']]
                batch_labels = labels[i:i+params['batch_size']]

                # Retrieve embeddings for the current batch
                author_vecs = author_emb[batch_author_indices]
                unit_vecs = unit_emb[batch_unit_indices]
                scores = (author_vecs * unit_vecs).sum(dim=1)

                # Compute the loss
                loss = torch.nn.BCEWithLogitsLoss()(scores, batch_labels)
                
                # Compute gradients
                loss.backward()  # Compute gradients
                
                # Perform optimization step
                optimizer.step()

                # Store the loss
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            # Save current model after each epoch
            torch.save(model.state_dict(), './src/model/tmp.pth')
            logger.info(f"Temporary model saved as tmp.pth at epoch {epoch+1}")

        # Evaluate after training
        metrics = evaluate_recommendations(test_df, author_emb, unit_emb, author_id_map, unit_ids, top_n=10)
        all_results.append({'params': params, 'metrics': metrics})

        # Update the best model if this one is better
        if best_metrics is None or metrics['F1-score'] > best_metrics['F1-score']:
            best_metrics = metrics
            best_params = params
            # Save the best model's parameters
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved the best model with parameters: {params} to {best_model_path}")

    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Metrics: {best_metrics}")

    return best_params, all_results

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.0001],
    'num_epochs': [100],
    'hidden_dim': [32, 64, 128],
    'dropout': [0.1],
    'batch_size': [512]  # Include different batch sizes here
}


# Load data, build graph, and split data
get_train_test_set()

# Generate Embeddings
logger.info("Generating text embeddings...")
vectorizer = TfidfVectorizer(max_features=5000)

# Combine training and test abstracts to fit the vectorizer
combined_abstracts = pd.concat([train_df['abstract'].fillna(''), test_df['abstract'].fillna('')])
vectorizer.fit(combined_abstracts)

# Transform the training and testing data separately
train_embeddings = vectorizer.transform(train_df['abstract'].fillna(''))
test_embeddings = vectorizer.transform(test_df['abstract'].fillna(''))
logger.info("Embeddings generated.")

# Save the vectorizer after fitting it
joblib.dump(vectorizer, './src/model/vectorizer.pth')
logger.info("Vectorizer saved as vectorizer.pth")

# Initialize the graph data (data) after loading train and test datasets
logger.info("Constructing heterogeneous graph...")
data = HeteroData()

# Nodes
logger.info("Adding nodes to the graph...")
author_ids = pd.concat([train_df['author_id'], test_df['author_id']]).unique()
paper_ids = pd.concat([train_df['paper_id'], test_df['paper_id']]).unique()
unit_ids = pd.concat([train_df['unit_id'], test_df['unit_id']]).unique()

author_id_map = {id_: idx for idx, id_ in enumerate(author_ids)}
paper_id_map = {id_: idx for idx, id_ in enumerate(paper_ids)}
unit_id_map = {id_: idx for idx, id_ in enumerate(unit_ids)}

data['author'].num_nodes = len(author_ids)
data['paper'].num_nodes = len(paper_ids)
data['unit'].num_nodes = len(unit_ids)
logger.info(f"Number of authors: {data['author'].num_nodes}")
logger.info(f"Number of papers: {data['paper'].num_nodes}")
logger.info(f"Number of units: {data['unit'].num_nodes}")

# Node Features
logger.info("Adding node features...")
data['author'].x = torch.randn(data['author'].num_nodes, 128)
paper_features = np.vstack([train_embeddings.toarray(), test_embeddings.toarray()])
data['paper'].x = torch.tensor(paper_features, dtype=torch.float)
data['unit'].x = torch.randn(data['unit'].num_nodes, 128)
logger.info("Node features added.")

# Edges
logger.info("Adding edges to the graph...")
# Author-Paper
src = train_df['author_id'].map(author_id_map).values
dst = train_df['paper_id'].map(paper_id_map).values
data['author', 'writes', 'paper'].edge_index = torch.tensor([src, dst], dtype=torch.long)
logger.info(f"Author-Paper edges: {data['author', 'writes', 'paper'].edge_index.shape}")

# Paper-Unit
src = train_df['paper_id'].map(paper_id_map).values
dst = train_df['unit_id'].map(unit_id_map).values
data['paper', 'published_in', 'unit'].edge_index = torch.tensor([src, dst], dtype=torch.long)
logger.info(f"Paper-Unit edges: {data['paper', 'published_in', 'unit'].edge_index.shape}")

# Reverse Edges
data['paper', 'rev_writes', 'author'].edge_index = data['author', 'writes', 'paper'].edge_index.flip(0)
data['unit', 'rev_published_in', 'paper'].edge_index = data['paper', 'published_in', 'unit'].edge_index.flip(0)
logger.info("Edges added to the graph.")

# Define the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Perform hyperparameter tuning
best_params, tuning_results = hyperparameter_tuning(train_df, test_df, param_grid, data)
