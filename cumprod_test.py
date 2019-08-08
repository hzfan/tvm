import tvm
import numpy as _np


def backward_compute_cumprod():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.compute((m, n), lambda i, j: (i < j))
    s = tvm.create_schedule(X.op)
    return s, X


s, ret = backward_compute_cumprod()
f = tvm.build(s, [ret])
n = 2
m = 3
ctx = tvm.cpu()
a = tvm.nd.array(_np.zeros((m, n), dtype=ret.dtype), ctx)
f(a)
print(a)