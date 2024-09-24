""" Create a subset of LFW
"""

import numpy as np
import random
import os


def create_subset(npyfile, labfile, savedir, num_samples):
    data = np.load(npyfile)
    names = [line.split()[0] for line in open(labfile)]

    """ split the names by identity """
    identities, current_name_entries, current_name = [], [], ""
    for idx, name in enumerate(names):
        if name == current_name:
            current_name_entries.append((idx, name))
        else:
            if current_name_entries:
                identities.append(current_name_entries)
            current_name_entries = [(idx, name)]
            current_name = name
    if current_name_entries:  # Add the last identity to the list
        identities.append(current_name_entries)

    """ random pick one from each identity with more than one entry """
    queries = []
    for identity in identities:
        if len(identity) > 1:
            queries.append(identity[random.randint(0, len(identity) - 1)])
    
    """ randomly sample queries by num_samples """
    assert num_samples <= len(queries)
    random_idx = random.sample(range(len(queries)), num_samples)
    queries = [queries[i] for i in random_idx]

    """ construct new numpy array """
    qvecs, ids, labels = [], [q[0] for q in queries], [q[1] for q in queries]
    for id in ids:
        qvecs.append(data[id])
    qvecs_np = np.stack(qvecs, axis=0)

    """ write to file """
    np.save(os.path.join(savedir, f"{num_samples}.npy"), qvecs_np)
    
    # Save the labels and their corresponding original indices
    with open(os.path.join(savedir, f"{num_samples}.txt"), "w") as file:
        for id, label in zip(ids, labels):
            file.write(f"{label} {id}\n")  # Save both label and original index
    

if __name__ == '__main__':
    for n in range(100, 1680, 100):
        create_subset("/root/lfw_vectors/all.npy", "/root/lfw_vectors/all.txt", "/root/lfw_vectors/", n)
