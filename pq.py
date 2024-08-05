import numpy as np
import faiss

# Create some sample data
d = 256  # Dimensionality of the vectors
nb = 10000  # Number of database vectors
nq = 100  # Number of query vectors

np.random.seed(1234)  # Set random seed for reproducibility

# Generate random vectors for the database and queries
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Normalize the vectors (FAISS expects normalized vectors for IndexPQ)
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

# Exact Nearest Neighbor Search using IndexFlatL2
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D_exact, I_exact = index_flat.search(xq, 1)

# Approximate Nearest Neighbor Search using IndexPQ
m = 256  # Number of subquantizers
n_bits = 8  # Number of bits per subquantizer

index_pq = faiss.IndexPQ(d, m, n_bits)

index_pq.train(xb)
index_pq.add(xb)

import time
st = time.time()
D_pq, I_pq = index_pq.search(xq, 1)
et = time.time()
print(et-st)

# Calculate the recall
def calculate_recall(I_true, I_pred):
    recall_at_k = []
    for i in range(I_true.shape[0]):
        recall_at_k.append(len(set(I_true[i]) & set(I_pred[i])) / len(set(I_true[i])))
    return np.mean(recall_at_k)

recall = calculate_recall(I_exact, I_pq)

# Print the results
# print("Exact Nearest Neighbors Indices:\n", I_exact)
# print("PQ Nearest Neighbors Indices:\n", I_pq)
print("Recall at 1: ", recall)