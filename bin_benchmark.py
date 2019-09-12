import tvm
import numpy as _np

target = 'llvm'
ctx = tvm.context(target, 0)

ishape = [128, 128, 128]
ishape_num = [128, 128, 128]
dtype = 'float32'
A = tvm.placeholder(ishape, name='A', dtype=dtype)
B = tvm.placeholder(ishape, name='B', dtype=dtype)
C = tvm.compute(ishape, lambda *idx: A[idx] + B[idx], name='C')
s = tvm.create_schedule(C.op)
s[C].parallel(C.op.axis[0])

func = tvm.build(s, [A, B, C], target=target, name="add")
a = tvm.nd.array(_np.array(_np.random.uniform(-2.0, 2.0, size=ishape), dtype=dtype))
b = tvm.nd.array(_np.array(_np.random.uniform(-2.0, 2.0, size=ishape), dtype=dtype))
c = tvm.nd.array(_np.zeros(ishape, dtype=dtype))

evaluator = func.time_evaluator(func.entry_name, ctx, number=50)
res = evaluator(a, b, c).mean
print("time = {}".format(res))
