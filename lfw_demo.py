from eva import *
from eva.ckks import *
from eva.seal import *
import numpy as np
import numpy as np
import faiss
import time
from compress_pca import CA_PCAer
from tqdm import tqdm


def test_with_local_vector(d_emb, d_lab, q_emb, q_lab, cdim, m, n_bits, k, logfile=None):
    """ Load local embedding vectors and test pqhe performance

    Args:
        d_emb (str): path to data embedding (.npy)
        d_lab (str): path to data labels (.txt)
        q_emb (str): path to query embedding (.npy)
        q_lab (str): path to query labels (.txt)
        cdim (int): dimension after compression
        m (int): number of subquantizers (slices)
        n_bits (int): nbits for each subquantizer
        k (int): number of nearest neighbor
    """

    """ Create vector database & queries """
    data = np.load(d_emb)
    query = np.load(q_emb)
    labels_query = [line.strip().split()[0] for line in open(q_lab)]
    idx_query = [int(line.strip().split()[1]) for line in open(q_lab)]
    data = np.delete(data, idx_query, axis=0) # Eliminate queries from database
    labels_data = [line.strip() for idx, line in enumerate(open(d_lab)) if idx not in idx_query]
    dim = data.shape[-1]

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
    st = time.time()
    ca = CA_PCAer(cdim)
    cdata = ca.train(data)
    et = time.time()
    compression_time = et - st

    """ Generate PQ indices """
    st = time.time()
    index_pq = faiss.IndexPQ(cdim, m, n_bits)
    index_pq.train(cdata)
    index_pq.add(cdata)
    et = time.time()
    pq_train_time = et - st

    """ Homomorphic retrieval w/ idx """
    pq_time, retrieve_time, total_time, pq_hit_cnt, pq_hit_idx, result, result_hit = 0, 0, 0, 0, [], [], 0
    for idx_q, q in enumerate(tqdm(query)):
        """ PQ search index """
        cq = ca.forward(q)
        I_pq_all = index_pq.search(np.expand_dims(cq, axis=0), len(data))[1][0]
        st = time.time()
        I_pq = index_pq.search(np.expand_dims(cq, axis=0), k)[1][0]
        et = time.time()
        pq_time += et - st
        """ Record PQ search hit """
        gt_name = labels_query[idx_q]
        ret_name = [labels_data[ipq] for ipq in I_pq_all]
        hit_idx = ret_name.index(gt_name)
        if hit_idx < k:
            pq_hit_cnt += 1
        pq_hit_idx.append(hit_idx)
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
            retrieve_time += et - st
        # sim_score[np.argmax(sim_score)] = 0    # find the second best sim_score (prevent q from retrieving itself)
        result.append(np.argmax(sim_score))
        if labels_data[np.argmax(sim_score)] == labels_query[idx_q]:
            result_hit += 1
    
    acc = result_hit / len(query)
    total_time += (pq_time + retrieve_time)

    """ print all results """
    print(f'{"Homomorphic retrieval w/ idx:":<20}')
    print(f'{"dim":<20}{"dbsize":<20}{"num_queries":<20}{"cdim":<20}{"m":<20}{"qbits":<20}{"k":<20}{"acc":<20}'
            f'{"pq_time/query":<20}{"ret_time/query":<20}'
            f'{"total_time/query":<20}{"compression_time":<20}{"pq_training_time":<20}')

    print(f'{data.shape[-1]:<20}{data.shape[0]:<20}{query.shape[0]:<20}{cdim:<20}{m:<20}{n_bits:<20}{k:<20}{acc:<20}'
            f'{pq_time / len(query):<20.5f}{retrieve_time / len(query):<20.5f}'
            f'{total_time / len(query):<20.5f}{compression_time:<20.5f}{pq_train_time:<20.5f}')
    
    if logfile:
        with open(logfile, 'a') as f:
            f.write(f'{data.shape[-1]:<20}{data.shape[0]:<20}{query.shape[0]:<20}{cdim:<20}{m:<20}{n_bits:<20}{k:<20}{acc:<20}'
                    f'{pq_time / len(query):<20.5f}{retrieve_time / len(query):<20.5f}'
                    f'{total_time / len(query):<20.5f}{compression_time:<20.5f}{pq_train_time:<20.5f}''\n')
    
    return acc

    # """ Homomorphic retrieval w/o idx (time approximation) """
    # total_time = 0
    # result, result_hit = [], 0
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
    #     sim_score[np.argmax(sim_score)] = 0    # find the second best sim_score (prevent q from retrieving itself)
    #     result.append(np.argmax(sim_score))
    #     if labels_data[np.argmax(sim_score)] == labels_query[idx_q]:
    #         print("hit")
    #         result_hit += 1
    #     else:
    #         print("miss")
    # print(f'Homomorphic retrieval w/o idx: {total_time / len(query)}, Accuracy: {result_hit / len(query)}')


if __name__ == '__main__':
    d_emb="/root/lfw_vectors/all.npy"
    d_lab="/root/lfw_vectors/all.txt"
    q_emb="/root/lfw_vectors/1680.npy"
    q_lab="/root/lfw_vectors/1680.txt"

    grid = {
        'cdim': [64],
        'cdim2m': [1, 2, 4, 8, 16, 32, 64],
        'qbits': range(13, 0, -1),
        'k': range(10, 0, -1)
    }
    state = {
        'cdim': 0,
        'cdim2m': 0,
        'qbits': 0,
        'k': 0,
    }
    plateau_record = None
    for key in state.keys():
        while state[key] < len(grid[key]) and (acc := test_with_local_vector(
            d_emb=d_emb,
            d_lab=d_lab,
            q_emb=q_emb,
            q_lab=q_lab,
            cdim=grid['cdim'][state['cdim']],
            m=grid['cdim'][state['cdim']] // grid['cdim2m'][state['cdim2m']],
            n_bits=grid['qbits'][state['qbits']],
            k=grid['k'][state['k']],
            logfile='log.txt'
        )) > 0.9:
            if not plateau_record or acc == plateau_record[0]:
                plateau_record = (acc, state.copy())
            state[key] += 1
        state = plateau_record[1]


    # acc = test_with_local_vector(
    #     d_emb=d_emb,
    #     d_lab=d_lab,
    #     q_emb=q_emb,
    #     q_lab=q_lab,
    #     cdim=cdim[state['cdim']],
    #     m=8,
    #     n_bits=8,
    #     k=5,
    #     logfile='log.txt'
    # )
