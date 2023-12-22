
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtffbgyzpmndrtmnkndyeuy4hjabdw3dj3vb5g6wurfwxudbafx.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax, aten.nll_loss_forward]
# loss => amax, convert_element_type, div, exp, full_default_1, ne_1, ne_2, neg, sub, sum_1, sum_2, sum_3, where_1
triton_per_fused__log_softmax_nll_loss_forward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_nll_loss_forward_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.full([1, 1], -100, tl.int64)
    tmp14 = tmp12 != tmp13
    tmp15 = tl.full([1, 1], 0, tl.int64)
    tmp16 = tl.where(tmp14, tmp12, tmp15)
    tmp17 = tmp16 + 2
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert((0 <= tmp19) & (tmp19 < 2), "index out of bounds: 0 <= tmp19 < 2")
    tmp20 = tl.load(in_ptr0 + (tmp19), None, eviction_policy='evict_last')
    tmp21 = tmp20 - tmp4
    tmp22 = tl.log(tmp10)
    tmp23 = tmp21 - tmp22
    tmp24 = -tmp23
    tmp25 = 0.0
    tmp26 = tl.where(tmp14, tmp24, tmp25)
    tmp27 = tmp14.to(tl.int64)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp29, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2), (2, 1))
    assert_size_stride(arg1_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty((1, 1), device='cuda', dtype=torch.float32)
        buf2 = reinterpret_tensor(buf1, (), (), 0); del buf1  # reuse
        # Source Nodes: [loss], Original ATen: [aten._log_softmax, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_per_fused__log_softmax_nll_loss_forward_0.run(buf2, arg0_1, arg1_1, 1, 2, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
