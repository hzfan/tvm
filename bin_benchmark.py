import tvm
import numpy as _np


def run_tvm_tests(times, func_name, *args):
    costs = []
    for i in range(times):
        wrapped_args = [tvm.nd.array(arg) for arg in args]
        cost = measure_tvm_cost(1, func_name, *wrapped_args)
        costs.append(cost)
    return costs


def measure_tvm_cost(repeat, func_name, *args, **kwargs):
    """Measure time cost of running a function
    """
    start = time.time()
    for _ in range(repeat):
        func_name(*args, **kwargs)
    end = time.time()
    diff = end - start
    return diff / repeat


def stabalize(x):
    # warm up
    x = x[1:]
    return x


def stat(name, nbytes, costs):
    costs = stabalize(costs)
    print("{}:".format(name))
    mean = _np.mean(costs)
    print("mean(s):         {}".format(mean))
    print("std/mean:        {}".format(_np.std(costs) / mean))
    print("bandwidth(GBps): {}".format(nbytes / mean / 2 ** 30))
    print("")


target = 'cuda'
ctx = tvm.context(target, 0)

ishape = [128, 128, 128]
dtype = 'float32'
A = tvm.placeholder(ishape, name='A', dtype=dtype)
B = tvm.placeholder(ishape, name='B', dtype=dtype)
C = tvm.compute(ishape, lambda *idx: A[idx] + B[idx], name='C')
s = tvm.create_schedule(C.op)
bx, tx = C.op.axis[0], C.op.axis[1]



# test
dsize = 4
ishape_num = [128, 128, 128]
func = tvm.build(s, [A, B, C], target=target, name="add")
a_np = _np.array(_np.random.uniform(-2.0, 2.0, size=ishape_num), dtype=dtype)
b_np = _np.array(_np.random.uniform(-2.0, 2.0, size=ishape_num), dtype=dtype)
c_np = _np.zeros(ishape, dtype=dtype)

costs = run_tvm_tests(100, func, a_np, b_np, c_np)
stat("tvm", (a_np.size + b_np.size + c_np.size) * dsize, costs)
