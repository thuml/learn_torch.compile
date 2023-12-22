
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


# kernel path: /tmp/torchinductor_youkaichao/pz/cpz72zxttxwb7jggloyedllj2ghy5r2baqwot23qr3ftm75mu5ct.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax]
# loss => amax, exp, log, sub, sub_1, sum_1
triton_red_fused__log_softmax_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2047
    rnumel = 50272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 - tmp2
        tmp12 = tl.log(tmp8)
        tmp13 = tmp11 - tmp12
        tl.store(out_ptr2 + (r1 + (50272*x0)), tmp13, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ie/cie2xavnbhjwjakqhj7te5nuya7dlvbgub5ypxxhkqhombyanqwy.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# loss => convert_element_type, div, full_default_1, ne, neg, sum_2, sum_3, where_1
triton_red_fused_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2047
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tl.where(tmp2, tmp0, tmp7)
        tmp9 = tmp8 + 50272
        tmp10 = tmp8 < 0
        tmp11 = tl.where(tmp10, tmp9, tmp8)
        tl.device_assert((0 <= tmp11) & (tmp11 < 50272), "index out of bounds: 0 <= tmp11 < 50272")
        tmp12 = tl.load(in_ptr1 + (tmp11 + (50272*r0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = -tmp12
        tmp14 = 0.0
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tmp5.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp19, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp20, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (50272, 768), (768, 1))
    assert_size_stride(primals_2, (1, 2048, 768), (1572864, 768, 1))
    assert_size_stride(primals_3, (1, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((2048, 50272), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__self___lm_head], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_2, (2048, 768), (768, 1), 0), reinterpret_tensor(primals_1, (768, 50272), (1, 768), 0), out=buf0)
        buf3 = empty((2047, 50272), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax]
        stream0 = get_cuda_stream(0)
        triton_red_fused__log_softmax_0.run(buf0, buf3, 2047, 50272, grid=grid(2047), stream=stream0)
        buf6 = empty((), device='cuda', dtype=torch.float32)
        buf5 = empty((), device='cuda', dtype=torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_1.run(buf7, primals_3, buf3, buf5, 1, 2047, grid=grid(1), stream=stream0)
        return (buf7, reinterpret_tensor(buf0, (1, 2048, 50272), (102957056, 50272, 1), 0), primals_3, reinterpret_tensor(primals_2, (2048, 768), (768, 1), 0), buf3, buf5, reinterpret_tensor(primals_1, (50272, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
