# Rec_GNN.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import chardet
import sys
import traceback
from tqdm import tqdm

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

try:
    # Step 1: Detect encoding of train.csv
    logger.info("Detecting file encoding...")
    with open('train.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        confidence = result['confidence']
    logger.info(f"Detected encoding for train.csv: {encoding} with confidence {confidence}")

    # Step 2: Load Data
    logger.info("Loading data...")
    try:
        train_df = pd.read_csv('train.csv', encoding=encoding)
        test_df = pd.read_csv('test.csv', encoding=encoding)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Verify data
    logger.info("Verifying data integrity...")
    logger.info(f"Training data sample:\n{train_df.head()}")
    logger.info(f"Testing data sample:\n{test_df.head()}")

    # Step 3: Preprocess Text Data
    logger.info("Preprocessing text data...")
    train_df['text'] = train_df['abstract'].fillna('')
    test_df['text'] = test_df['abstract'].fillna('')
    logger.info("Text data preprocessed.")

    # Step 4: Generate Embeddings
    logger.info("Generating text embeddings...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(train_df['text'])
    train_embeddings = vectorizer.transform(train_df['text'])
    test_embeddings = vectorizer.transform(test_df['text'])
    logger.info("Embeddings generated.")

    # Step 5: Construct Heterogeneous Graph
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

    # Step 6: Define the GNN Model
    logger.info("Defining the GNN model...")
    from torch.nn import Module

    class HeteroGNN(Module):
        def __init__(self, metadata):
            super().__init__()
            self.conv1 = HeteroConv({
                ('author', 'writes', 'paper'): GATConv((-1, -1), 64, add_self_loops=False),
                ('paper', 'published_in', 'unit'): GATConv((-1, -1), 64, add_self_loops=False),
                ('paper', 'rev_writes', 'author'): GATConv((-1, -1), 64, add_self_loops=False),
                ('unit', 'rev_published_in', 'paper'): GATConv((-1, -1), 64, add_self_loops=False),
            }, aggr='mean')

            self.lin = Linear(64, 32)

        def forward(self, x_dict, edge_index_dict):
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.lin(x) for key, x in x_dict.items()}
            return x_dict

    logger.info("GNN model defined.")

    # Step 7: Prepare Training Data
    logger.info("Preparing training data...")
    # Positive samples
    train_positive_pairs = train_df[['author_id', 'unit_id']].drop_duplicates()
    train_positive_pairs['label'] = 1

    # Negative sampling
    logger.info("Performing negative sampling...")
    all_units = set(unit_ids)
    negative_samples = []
    for author in tqdm(train_positive_pairs['author_id'].unique(), desc="Negative Sampling"):
        positive_units = set(train_positive_pairs[train_positive_pairs['author_id'] == author]['unit_id'])
        negative_units = all_units - positive_units
        num_samples = min(len(positive_units), len(negative_units))
        if num_samples == 0:
            continue  # Skip authors with no negative units
        negative_units_sampled = random.sample(list(negative_units), num_samples)
        for unit in negative_units_sampled:
            negative_samples.append({'author_id': author, 'unit_id': unit, 'label': 0})

    train_negative_pairs = pd.DataFrame(negative_samples)

    # Combine samples
    train_pairs = pd.concat([train_positive_pairs, train_negative_pairs]).reset_index(drop=True)
    train_pairs['author_idx'] = train_pairs['author_id'].map(author_id_map)
    train_pairs['unit_idx'] = train_pairs['unit_id'].map(unit_id_map)
    labels = torch.tensor(train_pairs['label'].values, dtype=torch.float)
    logger.info(f"Total training pairs: {len(train_pairs)}")

    # Step 8: Training the Model
    logger.info("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = HeteroGNN(data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    data = data.to(device)
    labels = labels.to(device)
    batch_size = 1024  # Adjust based on your memory capacity

    for epoch in range(10):
        model.train()

        author_indices = torch.tensor(train_pairs['author_idx'].values, dtype=torch.long).to(device)
        unit_indices = torch.tensor(train_pairs['unit_idx'].values, dtype=torch.long).to(device)

        losses = []
        for i in range(0, len(author_indices), batch_size):
            # Zero the gradients for the current batch
            optimizer.zero_grad()

            # Get the batch data
            batch_author_indices = author_indices[i:i+batch_size]
            batch_unit_indices = unit_indices[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Forward pass within the batch
            x_dict = model(data.x_dict, data.edge_index_dict)
            author_emb = x_dict['author']
            unit_emb = x_dict['unit']

            # Retrieve embeddings for the current batch
            author_vecs = author_emb[batch_author_indices]
            unit_vecs = unit_emb[batch_unit_indices]
            scores = (author_vecs * unit_vecs).sum(dim=1)

            # Compute the loss
            loss = torch.nn.BCEWithLogitsLoss()(scores, batch_labels)
            loss.backward()  # Compute gradients

            # Perform optimization step
            optimizer.step()

            # Store the loss
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    logger.info("Training completed.")

    # Step 9: Generating Recommendations
    logger.info("Generating recommendations...")
    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)
        author_emb = x_dict['author']
        unit_emb = x_dict['unit']

    # Test authors
    logger.info("Preparing test data...")
    test_author_ids = test_df['author_id'].unique()
    test_author_indices = [author_id_map.get(id_) for id_ in test_author_ids if id_ in author_id_map]
    if not test_author_indices:
        logger.warning("No test authors found in the training data.")
    else:
        test_author_indices = torch.tensor(test_author_indices, dtype=torch.long).to(device)

        # Compute scores
        logger.info("Computing scores for test authors...")
        test_author_emb = author_emb[test_author_indices]
        scores = torch.matmul(test_author_emb, unit_emb.t())

        # Get top-N recommendations
        N = 10
        logger.info(f"Selecting top {N} recommendations for each test author...")
        topN_units = torch.topk(scores, N, dim=1).indices.cpu().numpy()

        # Step 10: Evaluation
        logger.info("Evaluating recommendations...")
        # Ground truth
        test_author_units = test_df.groupby('author_id')['unit_id'].apply(list).to_dict()

        # Evaluation metrics
        y_true = []
        y_scores = []

        for idx, author_idx in enumerate(test_author_indices):
            author_id = author_ids[author_idx]
            true_units = test_author_units.get(author_id, [])
            recommended_unit_indices = topN_units[idx]
            recommended_unit_ids = [unit_ids[i] for i in recommended_unit_indices]
            relevance = [1 if unit in true_units else 0 for unit in recommended_unit_ids]
            y_true.append(relevance)
            y_scores.append([1]*N)

        # NDCG
        from sklearn.metrics import ndcg_score
        ndcg = ndcg_score(y_true, y_scores)
        logger.info(f"NDCG: {ndcg:.4f}")

        # MRR
        def mean_reciprocal_rank(y_true):
            rr_sum = 0.0
            for relevance in y_true:
                for idx, rel in enumerate(relevance):
                    if rel == 1:
                        rr_sum += 1.0 / (idx + 1)
                        break
            return rr_sum / len(y_true)

        mrr = mean_reciprocal_rank(y_true)
        logger.info(f"MRR: {mrr:.4f}")

        # Recall@N
        def recall_at_k(y_true, k):
            total = 0
            hits = 0
            for relevance in y_true:
                total += 1
                hits += 1 if sum(relevance[:k]) > 0 else 0
            return hits / total

        recall = recall_at_k(y_true, N)
        logger.info(f"Recall@{N}: {recall:.4f}")

        logger.info("Evaluation completed.")
        print(f"NDCG: {ndcg:.4f}, MRR: {mrr:.4f}, Recall@{N}: {recall:.4f}")

except Exception as e:
    logger.error("An error occurred during execution:")
    logger.error(traceback.format_exc())
