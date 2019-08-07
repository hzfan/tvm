import tvm
import numpy as _np


def compute_cumsum1():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.placeholder((m, n), name="X")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    s_scan = tvm.scan(s_init, s_update, s_state, inputs=[X])
    ret = s_scan
    s = tvm.create_schedule(ret.op)
    return s, X, ret


def compute_cumsum2():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.placeholder((m, n), name="X")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    s_scan = tvm.scan(s_init, s_update, s_state, inputs=[X])
    ret = tvm.compute((m, n), lambda i, j: s_scan[(i, j)])
    s = tvm.create_schedule(ret.op)
    return s, X, ret


s, A, ret = compute_cumsum1()
print(tvm.lower(s, [A, ret], simple_mode=True))
# fscan = tvm.build(s, [A, ret])
print("build cumsum1 successfully")
s, A, ret = compute_cumsum2()
print(tvm.lower(s, [A, ret], simple_mode=True))
# fscan = tvm.build(s, [A, ret])
print("build cumsum2 successfully")
