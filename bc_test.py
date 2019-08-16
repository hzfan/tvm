import tvm

def compute_backward_vadd(dtype, ndim, reduce1st):
    ishape = [tvm.var() for _ in range(ndim)]
    odim = (len(ishape) + 1 - axes[0]) // 2
    oshape = [tvm.var() for _ in range(odim)]
    X = tvm.placeholder(ishape, name='X', dtype=dtype)
    ret = tvm.compute(oshape, lambda *idx: tvm.const(1, dtype=dtype), name='ret')
    s = tvm.create_schedule(ret.op)
    return s, X, ret, [ret]

def backward_vadd(dtype, ndim, reduce1st):
    s, X, ret, c_list = compute_backward_vadd(dtype, ndim, reduce1st)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [X, ret]


s, [X, ret] = backward_vadd('float32', 2, 1)
print(tvm.lower(s, [X, ret], simple_mode=True))
f = tvm.build(s, [X, ret])
data = tvm.nd.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
ret = tvm.nd.array([[0, 0, 0]], dtype='float32')
f(data, ret)
print(ret)