import tvm
import numpy as _np


def compute_cumprod(dtype, ndim, axis):
    def swapaxis(idx, axis1, axis2):
        ret = list(idx)
        if axis1 != axis2:
            ret[axis1], ret[axis2] = ret[axis2], ret[axis1]
        return ret

    ishape = [tvm.var() for _ in range(ndim)]
    oshape = swapaxis(ishape, 0, axis)
    X = tvm.placeholder(ishape, name='X', dtype=dtype)
    s_state = tvm.placeholder(oshape, dtype=dtype)
    s_init = tvm.compute([1] + oshape[1:], lambda *idx: X[tuple(swapaxis(idx, 0, axis))])
    s_update = tvm.compute(oshape, lambda *idx: s_state[(idx[0] - 1, ) + idx[1:]] * X[tuple(swapaxis(idx, 0, axis))])
    s_scan = tvm.scan(s_init, s_update, s_state)
    ret = tvm.compute(ishape, lambda *idx: s_scan[tuple(swapaxis(idx, 0, axis))])
    s = tvm.create_schedule(ret.op)
    return s, X, ret, s_init, s_update


def cuda_vcumprod(dtype, ndim, axis):
    s, X, ret, s_init, s_update = compute_cumprod(dtype, ndim, axis)
    num_thread = 64
    s_list = [s_init, s_update]
    c_list = [ret]
    for (i, t) in enumerate(s_list):
        axes = [axis for axis in t.op.axis[1:]]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        block_x = tvm.thread_axis("blockIdx.x_s" + str(i))
        thread_x = tvm.thread_axis("threadIdx.x_s" + str(i))
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    for (i, t) in enumerate(c_list):
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        block_x = tvm.thread_axis("blockIdx.x_c" + str(i))
        thread_x = tvm.thread_axis("threadIdx.x_c" + str(i))
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [X, ret]


def vcumprod(dtype, ndim, axis):
    s, X, ret, s_init, s_update = compute_cumprod(dtype, ndim, axis)
    axes = [axis for axis in s_init.op.axis[1:]]
    fused = s[s_init].fuse(*axes)
    s[s_init].parallel(fused)
    axes = [axis for axis in s_update.op.axis[1:]]
    fused = s[s_update].fuse(*axes)
    s[s_update].parallel(fused)
    axes = [axis for axis in ret.op.axis]
    fused = s[ret].fuse(*axes)
    s[ret].parallel(fused)
    axes = [axis for axis in s_update.op.axis]
    fused = s[s_update].fuse(*axes)
    return s, [X, ret]


# s, [X, ret] = cuda_vcumprod('int32', 3, 2)
s, [X, ret] = vcumprod('int32', 3, 2)
print(tvm.lower(s, [X, ret], simple_mode=True))
lowered = tvm.lower(s, [X, ret], name="cumprod")
# f = tvm.build(lowered, target="cuda")
# ctx = tvm.gpu(0)
f = tvm.build(lowered, target="llvm")
ctx = tvm.cpu()
a = tvm.nd.array(_np.array([[[7, 0, 0, -6, -9]],
                            [[-7, 2, -1, 1, -7]],
                            [[5, -6, -6, 0, 7]],
                            [[1, -5, 7, -3, -3]],
                            [[2, -8, 0, 0, 0]]], dtype=X.dtype), ctx)
b = tvm.nd.array(_np.zeros((5, 1, 5), dtype=ret.dtype), ctx)
f(a, b)
print(b)
