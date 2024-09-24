from eva import *
from eva.ckks import *
from eva.seal import *
import numpy as np
import numpy as np
import random
import faiss
import time
from compress_pca import CA_PCAer
from tqdm import tqdm

""" Configuration """
dim = 256
cdim = 8
nv = 10000
nq = 1000
m = 8 # number of subquantizers (slices)
n_bits = 4 # nbits for each subquantizer
k = 5 # nearest neighbor

""" Create vector database & queries """
data = np.random.rand(nv, dim)
data = data / np.linalg.norm(data, axis=1, keepdims=True)
random_indices = [random.randint(0, len(data) - 1) for _ in range(nq)]
query = data[random_indices]

""" Generate crypto components """
compiler = CKKSCompiler()
homomorphic_test = EvaProgram('HomomorphicTest', vec_size=dim)
with homomorphic_test:
    a = Input('a')
    b = Input('b')
    product = a * b
    Output('product', product)

homomorphic_test.set_output_ranges(30)
homomorphic_test.set_input_scales(30)

compiled, params_test, signature = compiler.compile(homomorphic_test)
public_ctx, secret_ctx = generate_keys(params_test)

""" Compress Vectors """
ca = CA_PCAer(cdim)
cdata = ca.train(data)

""" Generate PQ indices """
index_pq = faiss.IndexPQ(cdim, m, n_bits)
index_pq.train(cdata)
index_pq.add(cdata)

""" Homomorphic retrieval w/ idx """
total_time = 0
pq_hit = 0
result = []
for idx_q, q in enumerate(tqdm(query)):
    """ PQ search index """
    cq = ca.forward(q)
    _, I_pq = index_pq.search(np.expand_dims(cq, axis=0), k)
    """ Record PQ search hit """
    if random_indices[idx_q] in I_pq:
        pq_hit += 1
    """ Search on crypto space """
    sim_score = []
    for idx_d, d in enumerate(data):
        if idx_d not in I_pq:
            sim_score.append(0)
            continue
        input = {
            'a': [q[i] for i in range(len(q))],
            'b': [d[i] for i in range(len(d))]
        }
        enc_input = public_ctx.encrypt(input, signature)
        st = time.time()
        enc_output = public_ctx.execute(compiled, enc_input)
        dec_output = secret_ctx.decrypt(enc_output, signature)
        sim_score.append(np.sum(dec_output['product']))
        et = time.time()
        total_time += et - st
    result.append(np.argmax(sim_score))
print(f'Homomorphic retrieval w/ idx: {total_time}, Accuracy: {sum(np.equal(result, random_indices)) / nq}')

# """ Homomorphic retrieval w/o idx"""
# total_time = 0
# result = []
# for idx_q, q in enumerate(query):
#     sim_score = []
#     for d in tqdm(data, desc=f"process for query {idx_q}"):
#         input = {
#             'a': [q[i] for i in range(len(q))],
#             'b': [d[i] for i in range(len(d))]
#         }
#         enc_input = public_ctx.encrypt(input, signature)
#         st = time.time()
#         enc_output = public_ctx.execute(compiled, enc_input)
#         dec_output = secret_ctx.decrypt(enc_output, signature)
#         sim_score.append(np.sum(dec_output['product']))
#         et = time.time()
#         total_time += et - st
#     result.append(np.argmax(sim_score))
# print(f'Homomorphic retrieval w/o idx: {et - st}, Accuracy: {sum(np.equal(result, random_indices)) / nq}')