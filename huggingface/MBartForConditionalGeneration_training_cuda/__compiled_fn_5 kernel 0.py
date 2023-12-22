
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


# kernel path: /tmp/torchinductor_youkaichao/zr/czrpwu56hhhrkoi3pjhn3jzyb5cosyf4bnuuh5yfvur6cpvaj7wf.py
# Source Nodes: [clone_1, eq, masked_fill_, ne, setitem, setitem_1, sum_1], Original ATen: [aten.clone, aten.copy, aten.eq, aten.masked_fill, aten.ne, aten.select_scatter, aten.slice_scatter, aten.sum]
# clone_1 => clone_1
# eq => eq
# masked_fill_ => full_default, where
# ne => ne
# setitem => copy, slice_scatter
# setitem_1 => copy_1, select_scatter
# sum_1 => sum_1
triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_scatter_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_scatter_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = r0
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = tmp11 == tmp12
    tmp14 = tmp10 - tmp3
    tmp15 = tmp14 + 1024
    tmp16 = tmp14 < 0
    tmp17 = tl.where(tmp16, tmp15, tmp14)
    tl.device_assert((0 <= tmp17) & (tmp17 < 1024), "index out of bounds: 0 <= tmp17 < 1024")
    tmp18 = tl.load(in_ptr0 + (tmp17), None, eviction_policy='evict_last')
    tmp19 = tmp18 == tmp1
    tmp20 = tl.where(tmp19, tmp3, tmp18)
    tmp21 = tmp11 >= tmp3
    tmp22 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + r0, [RBLOCK])), rmask & tmp21, other=0.0)
    tmp23 = tmp22 == tmp1
    tmp24 = tl.where(tmp23, tmp3, tmp22)
    tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tl.where(tmp21, tmp26, tmp4)
    tmp28 = tl.where(tmp13, tmp20, tmp27)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp28, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty((1, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [clone_1, eq, masked_fill_, ne, setitem, setitem_1, sum_1], Original ATen: [aten.clone, aten.copy, aten.eq, aten.masked_fill, aten.ne, aten.select_scatter, aten.slice_scatter, aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_clone_copy_eq_masked_fill_ne_select_scatter_slice_scatter_sum_0.run(arg0_1, buf1, 1, 1024, grid=grid(1), stream=stream0)
        del arg0_1
        return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForConditionalGeneration', benchmark_compiled_module)
