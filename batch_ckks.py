from eva import *
from eva.ckks import *
from eva.seal import *
from eva.metric import valuation_mse
import numpy as np

compiler = CKKSCompiler()

def dot_product(a, b):
    result = 0.
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Additional code to test homomorphic properties
# Homomorphic addition and multiplication
homomorphic_test = EvaProgram('HomomorphicTest', vec_size=256)
with homomorphic_test:
    a = Input('a')
    b = Input('b')
    product = a * b
    Output('product', product)

homomorphic_test.set_output_ranges(30)
homomorphic_test.set_input_scales(30)

compiled_test, params_test, signature_test = compiler.compile(homomorphic_test)
public_ctx, secret_ctx = generate_keys(params_test)

""" Build vectors and query """
vectors = []
for _ in range(1000):
    v = np.random.rand(256)
    uv = v / np.linalg.norm(v)
    vectors.append(uv)

query = np.random.rand(256)
query = query / np.linalg.norm(query)

""" Build inputs """
inputs = []
for vector in vectors:
    inputs.append({
        'a': [query[i] for i in range(len(query))],
        'b': [vector[i] for i in range(len(vector))]
    })

""" Encode inputs """
enc_inputs = []
for input in inputs:
    enc_inputs.append(public_ctx.encrypt(input, signature_test))

""" Execute homomorphic expressions """
enc_outputs = []
for enc_input in enc_inputs:
    enc_outputs.append(public_ctx.execute(compiled_test, enc_input))

""" Decrypt """
decrypt_outputs = []
for enc_output in enc_outputs:
    decrypt_outputs.append(secret_ctx.decrypt(enc_output, signature_test))

""" Evaluate """
mse_values = []
for idx, decrypt_output in enumerate(decrypt_outputs):
    reference = query @ vectors[idx]
    mse_values.append(valuation_mse({'sim': np.sum(decrypt_output['product'])}, {'sim': reference}))

print(np.mean(mse_values))