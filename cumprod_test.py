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
                                                                        * X[swapaxis((idx[0] - 1, ) + idx[1:-1], 0, axis)],
                                                                        s_state[(idx[0] - 1, ) + idx[1:]]
                                                                        * X[swapaxis(idx[:-1], 0, axis)])))
    s_scan = tvm.scan(s_init, s_update, s_state)
    # A = tvm.compute(sshape, lambda *idx: s_scan[idx])
    ret = s_scan
    # k = tvm.reduce_axis((0, sshape[0]), name="k")
    # ret = tvm.compute(ishape,
    #                   lambda* idx: tvm.sum(A[(k,) + idx[:axis] + idx[axis + 1:] + (idx[axis],)],
    #                                        axis=k), name="ret")
    s = tvm.create_schedule(ret.op)
    return s, out_grad, X, ret


def replay():
    m = tvm.var("m")
    n = tvm.var("n")
    s_state = tvm.placeholder((n, m, n), dtype="int32")
    s_init = tvm.compute((1, m, n), lambda *idx: 1)
    s_update = tvm.compute((n, m, n), lambda *idx: s_state[idx] + 1)
    s_scan = tvm.scan(s_init, s_update, s_state)
    ret = s_scan
    s = tvm.create_schedule(ret.op)
    return s, ret

def test():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.compute((m, n), lambda i, j: tvm.expr.Select(i <= j, 5, 10))
    s = tvm.create_schedule(X.op)
    return s, X


s, ret = replay()
f = tvm.build(s, [ret])
ctx = tvm.cpu()
a = tvm.nd.array(_np.zeros(4, 6, 4, dtype="int32"), ctx)
f(a)
print(a)

s, out_grad, X, ret = compute_backward_cumprod('int32', 2, 1)
f = tvm.build(s, [out_grad, X, ret])
m = 6
n = 4
ctx = tvm.cpu()
a = tvm.nd.array(_np.ones((m, n), dtype=out_grad.dtype), ctx)
b = tvm.nd.array(_np.array([[-9, 4, 7, -6],
                            [ 4, 5, 1, 5],
                            [ 8, 1, 0, 0],
                            [ 0, 8, -2, 4],
                            [-9, 7, -8, 9],
                            [ 0, 4, 2, -1]], dtype=X.dtype), ctx)
c = tvm.nd.array(_np.zeros((n, m, n), dtype=ret.dtype), ctx)
f(a, b, c)
print(c)
