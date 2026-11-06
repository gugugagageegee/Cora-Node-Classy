import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.config import DATA_DIR, DATASET_NAME


def load_cora():
    dataset = Planetoid(
        root=str(DATA_DIR),
        name=DATASET_NAME,
        split="public",
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    return dataset, data


def get_graph_components(data):
    edge_index = data.edge_index.cpu().numpy()

    adjacency = coo_matrix(
        (
            np.ones(edge_index.shape[1]),
            (edge_index[0], edge_index[1])
        ),
        shape=(data.num_nodes, data.num_nodes)
    ).tocsr()

    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    num_components, component_labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True
    )

    component_sizes = np.bincount(
        component_labels,
        minlength=num_components
    )

    return num_components, component_labels, component_sizes


def get_dataset_info(dataset, data):
    edge_index = data.edge_index.cpu().numpy()

    source = edge_index[0]
    target = edge_index[1]

    pairs = np.stack(
        [
            np.minimum(source, target),
            np.maximum(source, target)
        ],
        axis=1
    )

    unique_edges = np.unique(
        pairs,
        axis=0
    ).shape[0]

    num_components, _, component_sizes = get_graph_components(data)

    info = {
        "num_nodes": data.num_nodes,
        "edge_index_entries": data.edge_index.size(1),
        "undirected_edges": unique_edges,
        "num_features": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
        "num_components": num_components,
        "smallest_component_size": int(component_sizes.min()),
        "largest_component_size": int(component_sizes.max())
    }

    return info