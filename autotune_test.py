import logging
import sys

import numpy as _np
import tvm

from tvm import autotvm

@autotvm.template  # 1. use a decorator
def matmul(N, L, M, dtype):
# def matmul(dtype):
    # N = tvm.var('N')
    # L = tvm.var('L')
    # M = tvm.var('M')
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()
    cfg.flop = 1

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


N, L, M = 512, 512, 512
# N, L, M = tvm.var('N'), tvm.var('L'), tvm.var('M')
task = autotvm.task.create(matmul, args=(N, L, M, 'float32',), target='llvm')
# task = autotvm.task.create(matmul, args=('float32',), target='llvm')
print("type of task = {}".format(type(task)))
print("task = {}".format(task))
print("type of config_space = {}".format(type(task.config_space)))
print("config_space = {}".format(task.config_space))
print("task.flop = {}".format(task.flop))
print("config entity = {}".format(task.config_space.get(0)))

# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `matmul.log`
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

# apply history best from log file
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create("llvm"):
        # N = tvm.var('N')
        # L = tvm.var('L')
        # M = tvm.var('M')
        s, arg_bufs = matmul(N, L, M, 'float32')
        func = tvm.build(s, arg_bufs)

# check correctness
# N = 2
# L = 2
# M = 2
a_np = _np.random.uniform(size=(N, L)).astype(_np.float32)
b_np = _np.random.uniform(size=(L, M)).astype(_np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

print("c_np = {}".format(c_np))
print("c_tvm.asnumpy = {}".format(c_tvm.asnumpy()))
tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)