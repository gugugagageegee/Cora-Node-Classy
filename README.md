# Cora Node Classification: MLP vs GCN vs GAT vs Spectral Clustering

A comparative study of four approaches for node classification and clustering
on the Cora citation network using PyTorch and PyTorch Geometric.

## 1. Project Overview

This project investigates how node features and graph structure contribute
to node classification and clustering.

Four approaches are implemented:

1. **MLP** – feature-based neural network baseline.
2. **GCN** – Graph Convolutional Network using node features and graph structure.
3. **GAT** – Graph Attention Network using attention-based neighborhood aggregation.
4. **Feature-Augmented Normalized Spectral Clustering (NSC)** – an unsupervised
   spectral clustering approach using both graph connectivity and node-feature
   similarity.

All methods use the complete Cora graph.

## 2. Dataset

The project uses the Cora citation network from PyTorch Geometric's Planetoid
dataset.

| Property | Value |
|---|---:|
| Nodes | 2,708 |
| Undirected edges | 5,429 |
| Node features | 1,433 |
| Classes | 7 |
| Training nodes | 140 |
| Validation nodes | 500 |
| Test nodes | 1,000 |

Each node represents a scientific publication.

Each edge represents a citation relationship.

Node features are bag-of-words representations, and node labels correspond
to seven publication categories.

### Dataset Split

The public split provided by Planetoid is used.

- Training set: 140 nodes.
- Validation set: 500 nodes.
- Test set: 1,000 nodes.

The task follows the transductive node classification setting.

The complete graph structure and node features are available during training,
but supervised loss is calculated only on training nodes.

Validation labels are used for model selection, while test labels are reserved
for final evaluation.

## 3. Methods

### 3.1 MLP

The MLP uses node features without graph connectivity.

Architecture:

```text
Input: 1433
    |
Linear(1433, 64)
    |
ReLU
    |
Dropout
    |
Linear(64, 7)
    |
Output: 7 classes