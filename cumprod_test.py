import tvm
import numpy as _np


def compute_backward_cumprod(dtype, ndim, axis):
    def swapaxis(idx, axis1, axis2):
        ret = list(idx)
        if axis1 != axis2:
            ret[axis1], ret[axis2] = ret[axis2], ret[axis1]
        return ret if isinstance(idx, list) else tuple(ret)

    ishape = [tvm.var() for _ in range(ndim)]
    sshape = swapaxis(ishape, 0, axis) + [ishape[axis]]
    X = tvm.placeholder(ishape, dtype=dtype)  # input data
    out_grad = tvm.placeholder(ishape, dtype=dtype)  # output grad
    s_state = tvm.placeholder(sshape, dtype=dtype)
    s_init = tvm.compute([1] + sshape[1:], 
                         lambda *idx: tvm.expr.Select(idx[-1] > 0,
                                                      tvm.const(0, dtype),
                                                      tvm.const(1, dtype)))
    s_update = tvm.compute(sshape,
                           lambda *idx: tvm.expr.Select(idx[0] < idx[-1], 
                                                        tvm.const(0, dtype),
                                                        tvm.expr.Select(idx[0] == idx[-1],
                                                                        s_state[(idx[0] - 1, ) + idx[1:-1] + (idx[-1] - 1, )]
                                                                        * X[swapaxis((idx[0] - 1, ) + idx[:-1], 0, axis)],
                                                                        s_state[(idx[0] - 1,) + idx[1:]]
                                                                        * X[swapaxis(idx[:-1], 0, axis)])))
    s_scan = tvm.scan(s_init, s_update, s_state)   
    A = tvm.compute(sshape, lambda *idx: s_scan[idx] * out_grad[swapaxis(idx[:-1], 0, axis)])
    k = tvm.reduce_axis((sshape[0],) + (0,) * ndim, name="k")
    ret = tvm.compute(ishape,
                      lambda* idx: tvm.sum(A[(k,) + idx[:axis] + idx[axis + 1:] + (idx[axis],)],
                                           axis=k), name="ret")
    s = tvm.create_schedule(ret.op)
    return s, out_grad, X, ret


def test():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.compute((m, n), lambda i, j: tvm.expr.Select(i <= j, 5, 10))
    s = tvm.create_schedule(X.op)
    return s, X


s, out_grad, X, ret = compute_backward_cumprod('int32', 2, 0)
f = tvm.build(s, [out_grad, X, ret])
n = 2
m = 3
ctx = tvm.cpu()
out_grad = tvm.nd.array(_np.ones((m, n), dtype=out_grad.dtype), ctx)
X = tvm.nd.array(_np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=X.dtype), ctx)
ret = tvm.nd.array(_np.zeros((m, n), dtype=ret.dtype), ctx)
f(out_grad, X, ret)
print(ret)