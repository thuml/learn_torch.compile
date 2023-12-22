
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


# kernel path: /tmp/torchinductor_youkaichao/zp/czpncyotmjg7wwq4qun54m4afaie7zkiffkr277viexv26zxalbs.py
# Source Nodes: [any_1, is_index_global_attn, is_index_masked], Original ATen: [aten.any, aten.gt, aten.lt]
# any_1 => any_1
# is_index_global_attn => gt
# is_index_masked => lt
triton_per_fused_any_gt_lt_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*i1', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_gt_lt_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 < tmp1
    tmp4 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp6, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp2, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp7, None)
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
        buf0 = empty((1, 1024), device='cuda', dtype=torch.bool)
        buf2 = empty((1, 1024), device='cuda', dtype=torch.bool)
        buf1 = empty((), device='cuda', dtype=torch.bool)
        # Source Nodes: [any_1, is_index_global_attn, is_index_masked], Original ATen: [aten.any, aten.gt, aten.lt]
        stream0 = get_cuda_stream(0)
        triton_per_fused_any_gt_lt_0.run(arg0_1, buf0, buf2, buf1, 1, 1024, grid=grid(1), stream=stream0)
        del arg0_1
        return (buf1, buf2, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
