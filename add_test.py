import tvm
import numpy as _np

def compute_add():
    n = tvm.var('n')
    m = tvm.var('m')
    X = tvm.placeholder((n, m), dtype='float32', name='X')
    # Y = tvm.placeholder((n, m), dtype='float32', name='Y')
    Y = tvm.compute((n, m), lambda i, j: X[i, j] + 1)
    s = tvm.create_schedule(Y.op)
    return s, X, Y


s, X, Y = compute_add()
f = tvm.build(s, [X, Y])
data_np = _np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
data = tvm.nd.array(data_np)
f(data, data)
print(data)
