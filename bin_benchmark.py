import tvm

ishape = [128, 128, 128]
dtype = 'float32'
A = tvm.placeholder(ishape, name='A', dtype=dtype)
B = tvm.placeholder(ishape, name='B', dtype=dtype)
C = tvm.compute(ishape, lambda *idx: A[idx] + B[idx], name='C')
s = tvm.create_schedule(C.op)
s[C].parallel(C.op.axis[0])
