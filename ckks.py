from eva import *
from eva.ckks import *
from eva.seal import *
from eva.metric import valuation_mse

compiler = CKKSCompiler()

# Additional code to test homomorphic properties
# Homomorphic addition and multiplication
homomorphic_test = EvaProgram('HomomorphicTest', vec_size=4096)
with homomorphic_test:
    a = Input('a')
    b = Input('b')
    c = Input('c')
    d = Input('d')
    sum_ab = a + b
    product_cd = c * d
    Output('sum_ab', sum_ab)
    Output('product_cd', product_cd)

homomorphic_test.set_output_ranges(30)
homomorphic_test.set_input_scales(30)

compiled_test, params_test, signature_test = compiler.compile(homomorphic_test)
public_ctx, secret_ctx = generate_keys(params_test)
print(compiled_test.to_DOT())

# Prepare inputs for homomorphic test
inputs_test = {
    'a': [i for i in range(compiled_test.vec_size)],
    'b': [2*i for i in range(compiled_test.vec_size)],
    'c': [3*i for i in range(compiled_test.vec_size)],
    'd': [4*i for i in range(compiled_test.vec_size)]
}

encInputs_test = public_ctx.encrypt(inputs_test, signature_test)

# Execute the homomorphic test operations on encrypted inputs
encOutputs_test = public_ctx.execute(compiled_test, encInputs_test)
outputs_test = secret_ctx.decrypt(encOutputs_test, signature_test)

# Reference computations for comparison
reference_test = evaluate(compiled_test, inputs_test)
sum_ab_reference = [inputs_test['a'][i] + inputs_test['b'][i] for i in range(compiled_test.vec_size)]
product_cd_reference = [inputs_test['c'][i] * inputs_test['d'][i] for i in range(compiled_test.vec_size)]

# Print results
print('Homomorphic sum (encrypted then decrypted):', outputs_test['sum_ab'][:10])  # Print first 10 values for brevity
print('Reference sum (plaintext computation):', sum_ab_reference[:10])
print('Homomorphic product (encrypted then decrypted):', outputs_test['product_cd'][:10])
print('Reference product (plaintext computation):', product_cd_reference[:10])

# Calculate and print MSE for homomorphic test
mse_sum = valuation_mse({'sum_ab': outputs_test['sum_ab']}, {'sum_ab': sum_ab_reference})
mse_product = valuation_mse({'product_cd': outputs_test['product_cd']}, {'product_cd': product_cd_reference})
print('MSE for sum:', mse_sum)
print('MSE for product:', mse_product)