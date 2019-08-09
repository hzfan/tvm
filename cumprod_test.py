import tvm
import numpy as _np


def compute_backward_cumprod(dtype, ndim, axis):
    def swapaxis(idx, axis1, axis2):
        ret = list(idx)
        if axis1 != axis2:
            ret[axis1], ret[axis2] = ret[axis2], ret[axis1]
        return ret if isinstance(idx, list) else tuple(ret)

    ishape = [tvm.var("shape" + str(i)) for i in range(ndim)]
    sshape = swapaxis(ishape, 0, axis) + [ishape[axis]]
    X = tvm.placeholder(ishape, dtype=dtype, name="idata")  # input data
    out_grad = tvm.placeholder(ishape, dtype=dtype, name="ograd")  # output grad
    s_state = tvm.placeholder(sshape, dtype=dtype, name="state")
    s_init = tvm.compute([1] + sshape[1:], 
                         lambda *idx: 1)
    print("sshape = {}".format(sshape))
    s_update = tvm.compute(sshape,
                           lambda *idx: s_state[(idx[0] - 1,) + idx[1:]] + X[swapaxis(idx[:-1], 0, axis)])
    s_scan = tvm.scan(s_init, s_update, s_state)
    A = tvm.compute(sshape, lambda *idx: s_scan[idx])
    ret = A
    # k = tvm.reduce_axis((0, sshape[0]), name="k")
    # ret = tvm.compute(ishape,
    #                   lambda* idx: tvm.sum(A[(k,) + idx[:axis] + idx[axis + 1:] + (idx[axis],)],
    #                                        axis=k), name="ret")
    s = tvm.create_schedule(ret.op)
    return s, out_grad, X, ret


def replay():
    b = tvm.var("b")
    c = tvm.var("c")
    d = tvm.var("d")
    a = d
    s_state = tvm.placeholder((a, b, c, d), dtype="int32")
    s_init = tvm.compute((1, b, c, d), lambda *idx: 1)
    s_update = tvm.compute((a, b, c, d), lambda i, j, k, l: s_state[i - 1, j, k, l] + 1)
    s_scan = tvm.scan(s_init, s_update, s_state)
    A = tvm.compute((a, b, c, d), lambda *idx: s_scan[idx])
    k = tvm.reduce_axis((0, a), name="k")
    ret = tvm.compute((b, c, d), lambda *idx: tvm.sum(A[(k, b, c, d)], axis=k), name="sum")
    s = tvm.create_schedule(ret.op)
    return s, ret

def test():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.compute((m, n), lambda i, j: tvm.expr.Select(i <= j, 5, 10))
    s = tvm.create_schedule(X.op)
    return s, X


s, ret = replay()
print(tvm.lower(s, [ret], simple_mode=True))
f = tvm.build(s, [ret])
ctx = tvm.cpu()
a = tvm.nd.array(_np.zeros((5, 1, 5), dtype="int32"), ctx)
f(a)
print(a)

s, out_grad, X, ret = compute_backward_cumprod('int32', 3, 2)
print(tvm.lower(s, [out_grad, X, ret], simple_mode=True))
f = tvm.build(s, [out_grad, X, ret])
ctx = tvm.cpu()
a = tvm.nd.array(_np.ones((5, 1, 5), dtype=out_grad.dtype), ctx)
b = tvm.nd.array(_np.array([[[7, 0, 0, -6, -9]],
                            [[-7, 2, -1, 1, -7]],
                            [[5, -6, -6, 0, 7]],
                            [[1, -5, 7, -3, -3]],
                            [[2, -8, 0, 0, 0]]], dtype=X.dtype), ctx)
c = tvm.nd.array(_np.zeros((5, 1, 5, 5), dtype=ret.dtype), ctx)
f(a, b, c)
print(c)
