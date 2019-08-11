import tvm
import topi
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
    num_thread = 4
    s_list = [s_init, s_update]
    c_list = [ret]
    for (i, t) in enumerate(s_list):
        axes = [axis for axis in t.op.axis[1:]]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    for (i, t) in enumerate(c_list):
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
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
    return s, [X, ret]


def unravel(dtype, ndim):
    ishape = [tvm.var() for _ in range(ndim)]
    X = tvm.placeholder(ishape, name='X', dtype=dtype)
    ret = topi.reshape(X, (tvm.var(),))
    s = tvm.create_schedule(ret.op)
    axes = [axis for axis in ret.op.axis[:]]
    fused = s[ret].fuse(*axes)
    bx, tx = s[tret.split(fused, factor=64)
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    s[ret].bind(bx, block_x)
    s[ret].bind(tx, thread_x)
    return s, [X, ret]


s, [X, ret] = unravel('int32', 3)
print(tvm.lower(s, [X, ret], simple_mode=True))
lowered = tvm.lower(s, [X, ret], name="unravel")
f = tvm.build(lowered, target="cuda")
ctx = tvm.gpu(0)
a = tvm.nd.array(_np.ones((3, 3, 3), dtype=X.dtype), ctx)
b = tvm.nd.array(_np.zeros((3 * 3 * 3, ), dtype=ret.dtype), ctx)
f(a, b)
print(b)

# # AllTypes = ["float32", "float64", "float16", "uint8", "int8", "int32", "int64"]
# s, [X, ret] = cuda_vcumprod('int32', 2, 1)
# # s, [X, ret] = vcumprod('int32', 1, 0)
# print(tvm.lower(s, [X, ret], simple_mode=True))
# lowered = tvm.lower(s, [X, ret], name="cumprod")
# f = tvm.build(lowered, target="cuda")
# ctx = tvm.gpu(0)
# # f = tvm.build(lowered, target="llvm")
# # ctx = tvm.cpu()
# # a = tvm.nd.array(_np.empty((0,), dtype=X.dtype), ctx)
# # b = tvm.nd.array(_np.empty((0,), dtype=ret.dtype), ctx)
# a = tvm.nd.array(_np.array([1], dtype=X.dtype), ctx)
# b = tvm.nd.array(_np.empty((1,), dtype=ret.dtype), ctx)
# f(a, b)
# print(b)
