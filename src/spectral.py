import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import eigsh, LinearOperator

from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)


def build_affinity(
    data,
    gamma=5.0,
    feature_weight=0.5
):
    num_nodes = data.num_nodes

    # Build adjacency matrix
    edge_index = data.edge_index.cpu().numpy()

    source = edge_index[0]
    target = edge_index[1]

    adjacency = sp.coo_matrix(
        (
            np.ones(len(source)),
            (source, target)
        ),
        shape=(num_nodes, num_nodes)
    ).tocsr()

    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0

    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Convert node features to sparse matrix
    feature_array = (
        data.x.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    features = sp.csr_matrix(feature_array)

    # L2-normalize node features
    feature_norms = np.sqrt(
        np.asarray(
            features.multiply(features).sum(axis=1)
        ).ravel()
    )

    inverse_norms = np.zeros_like(feature_norms)

    nonzero = feature_norms > 0
    inverse_norms[nonzero] = 1.0 / feature_norms[nonzero]

    normalized_features = (
        sp.diags(inverse_norms) @ features
    )

    # Calculate cosine similarity
    cosine_similarity = (
        normalized_features @ normalized_features.T
    ).toarray()

    cosine_similarity = np.clip(
        cosine_similarity,
        0.0,
        1.0
    )

    cosine_similarity = (
        cosine_similarity + cosine_similarity.T
    ) / 2

    # RBF feature affinity
    feature_affinity = np.exp(
        -gamma * (1.0 - cosine_similarity)
    )

    np.fill_diagonal(feature_affinity, 0.0)

    # Scale feature affinity
    graph_mean_degree = float(
        np.asarray(adjacency.sum(axis=1)).mean()
    )

    feature_mean_degree = float(
        feature_affinity.sum(axis=1).mean()
    )

    if graph_mean_degree <= 0:
        raise ValueError("Graph has no edges.")

    if feature_mean_degree <= 0:
        raise ValueError("Invalid feature affinity.")

    feature_affinity *= (
        feature_weight
        * graph_mean_degree
        / feature_mean_degree
    )

    diagnostics = {
        "graph_mean_degree": graph_mean_degree,
        "feature_mean_degree": float(
            feature_affinity.sum(axis=1).mean()
        ),
        "gamma": gamma,
        "feature_weight": feature_weight
    }

    return adjacency, feature_affinity, diagnostics


def normalized_spectral_clustering(
    data,
    num_clusters,
    seed=42,
    gamma=5.0,
    feature_weight=0.5
):
    num_nodes = data.num_nodes

    if num_clusters < 2 or num_clusters >= num_nodes:
        raise ValueError("Invalid number of clusters.")

    if gamma <= 0 or feature_weight <= 0:
        raise ValueError("gamma and feature_weight must be positive.")

    # Construct combined affinity
    adjacency, feature_affinity, diagnostics = build_affinity(
        data=data,
        gamma=gamma,
        feature_weight=feature_weight
    )

    # Degree of combined affinity matrix
    combined_degree = (
        np.asarray(adjacency.sum(axis=1)).ravel()
        + feature_affinity.sum(axis=1)
    )

    inverse_sqrt_degree = 1.0 / np.sqrt(combined_degree)

    # Normalized Laplacian:
    # L = I - D^(-1/2) W D^(-1/2)
    #
    # W = A + feature_affinity

    def laplacian_matvec(vector):
        scaled_vector = inverse_sqrt_degree * vector

        graph_product = adjacency @ scaled_vector

        feature_product = feature_affinity @ scaled_vector

        result = (
            vector
            - inverse_sqrt_degree
            * (graph_product + feature_product)
        )

        return result

    laplacian = LinearOperator(
        shape=(num_nodes, num_nodes),
        matvec=laplacian_matvec,
        rmatvec=laplacian_matvec,
        dtype=np.float64
    )

    # Calculate smallest eigenvalues and eigenvectors
    rng = np.random.default_rng(seed)

    initial_vector = rng.normal(size=num_nodes)

    eigenvalues, eigenvectors = eigsh(
        laplacian,
        k=num_clusters,
        which="SA",
        v0=initial_vector,
        tol=1e-5,
        maxiter=10000
    )

    # Sort eigenvectors by ascending eigenvalues
    order = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Row normalization
    row_norms = np.linalg.norm(
        eigenvectors,
        axis=1,
        keepdims=True
    )

    embedding = eigenvectors / np.maximum(
        row_norms,
        1e-12
    )

    # KMeans clustering
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=20
    )

    cluster_labels = kmeans.fit_predict(embedding)

    return (
        cluster_labels,
        eigenvalues,
        embedding,
        diagnostics
    )


def map_clusters_to_classes(
    cluster_labels,
    true_labels,
    train_mask,
    embedding,
    num_classes
):
    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    embedding = np.asarray(embedding)

    num_clusters = int(cluster_labels.max()) + 1

    if not train_mask.any():
        raise ValueError("Training mask is empty.")

    if not (
        len(cluster_labels)
        == len(true_labels)
        == len(train_mask)
        == embedding.shape[0]
    ):
        raise ValueError("Input dimensions do not match.")

    # Count training labels in each cluster
    contingency = np.zeros(
        (num_clusters, num_classes),
        dtype=np.int64
    )

    train_indices = np.where(train_mask)[0]

    for node in train_indices:
        cluster_id = cluster_labels[node]
        class_id = true_labels[node]

        contingency[cluster_id, class_id] += 1

    # Calculate class centroids using training nodes only
    class_centroids = []

    for class_id in range(num_classes):

        class_nodes = np.where(
            train_mask & (true_labels == class_id)
        )[0]

        if len(class_nodes) == 0:
            raise ValueError(
                f"No training nodes for class {class_id}."
            )

        centroid = embedding[class_nodes].mean(axis=0)

        norm = np.linalg.norm(centroid)

        centroid = centroid / max(norm, 1e-12)

        class_centroids.append(centroid)

    class_centroids = np.asarray(class_centroids)

    mapping = np.zeros(num_clusters, dtype=int)

    for cluster_id in range(num_clusters):

        counts = contingency[cluster_id]

        # Cluster contains training nodes
        if counts.sum() > 0:

            candidates = np.where(
                counts == counts.max()
            )[0]

            # Majority vote
            if len(candidates) == 1:
                mapping[cluster_id] = candidates[0]

            else:
                # Resolve ties using spectral embedding
                cluster_nodes = np.where(
                    cluster_labels == cluster_id
                )[0]

                cluster_centroid = embedding[cluster_nodes].mean(axis=0)

                norm = np.linalg.norm(cluster_centroid)

                cluster_centroid = (
                    cluster_centroid / max(norm, 1e-12)
                )

                distances = np.linalg.norm(
                    class_centroids[candidates]
                    - cluster_centroid,
                    axis=1
                )

                mapping[cluster_id] = candidates[
                    np.argmin(distances)
                ]

        # Cluster has no training nodes
        else:

            cluster_nodes = np.where(
                cluster_labels == cluster_id
            )[0]

            cluster_centroid = embedding[cluster_nodes].mean(axis=0)

            norm = np.linalg.norm(cluster_centroid)

            cluster_centroid = (
                cluster_centroid / max(norm, 1e-12)
            )

            distances = np.linalg.norm(
                class_centroids - cluster_centroid,
                axis=1
            )

            mapping[cluster_id] = np.argmin(distances)

    return mapping, contingency


def evaluate_spectral_clustering(
    cluster_labels,
    true_labels,
    train_mask,
    test_mask,
    embedding,
    num_classes
):
    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)

    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)

    # Map clusters to classes using training labels only
    mapping, contingency = map_clusters_to_classes(
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        train_mask=train_mask,
        embedding=embedding,
        num_classes=num_classes
    )

    # Evaluate on test nodes
    test_clusters = cluster_labels[test_mask]
    test_true = true_labels[test_mask]

    test_predictions = mapping[test_clusters]

    labels = np.arange(num_classes)

    metrics = {
        "accuracy": accuracy_score(
            test_true,
            test_predictions
        ),

        "precision_macro": precision_score(
            test_true,
            test_predictions,
            labels=labels,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            test_true,
            test_predictions,
            labels=labels,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            test_true,
            test_predictions,
            labels=labels,
            average="macro",
            zero_division=0
        ),

        "nmi": normalized_mutual_info_score(
            test_true,
            test_clusters
        ),

        "ari": adjusted_rand_score(
            test_true,
            test_clusters
        )
    }

    return (
        metrics,
        mapping,
        test_predictions,
        contingency
    )