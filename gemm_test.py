import tvm
import numpy
import timeit

M = tvm.const(32) * tvm.var('M')
K = tvm.const(32) * tvm.var('K')
N = tvm.const(32) * tvm.var('N')
M_num = 1024
K_num = 1024
N_num = 1024

# The default tensor type in tvm
dtype = "float32"

target = 'llvm'
ctx = tvm.context(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M_num, K_num).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K_num, N_num).astype(dtype), ctx)

# Algorithm
k = tvm.reduce_axis((0, K), 'k')
A = tvm.placeholder((M, K), name='A')
B = tvm.placeholder((K, N), name='B')
C = tvm.compute(
           (M, N),
           lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
           name='C')

bn = 32
s = tvm.create_schedule(C.op)

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, ko, ki, xi, yi)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# func = tvm.build(s, [A, B, C], target=target, name='mmult')
# assert func

# c = tvm.nd.array(numpy.zeros((M_num, N_num), dtype = dtype), ctx)
# func(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

# evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
# print('Opt1: %f' % evaluator(a, b, c).mean)
