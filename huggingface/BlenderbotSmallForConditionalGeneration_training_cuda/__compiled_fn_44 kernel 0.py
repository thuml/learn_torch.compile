
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


# kernel path: /tmp/torchinductor_youkaichao/ly/clyuy4nqdva2eqmep2xdclp4funf6327lv6suftbg3gejwgizsue.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# masked_lm_loss => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6433920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/oj/cojib5cjtva2qww365umgv3zoxwow4fqvapujgxd7ejhr7zbxrys.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# masked_lm_loss => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cabuolj3tdioc4su5gfhliu5ql3a3yw7kecuzliizbahfgozhyyx.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# masked_lm_loss => full_default_1
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 7181
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7181*x0)
        tmp1 = tl.full([1, 1], 50265, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7181*x0) + (50265*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.full([1, 1], -100, tl.int64)
        tmp6 = tmp4 != tmp5
        tmp11 = tmp8 / tmp10
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp3 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cydxjnyyy7lgxlposqor7gs3yqjmha722asuhzdhkbeyb3a4j77i.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# masked_lm_loss => full_default_1
triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7frdaiwuwc3uqnszupdx65vypuswaoi3d3dyjeh5pzeookfwfb.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6433920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 50265)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr4 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.load(in_ptr5 + (x2), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp9 = tmp6 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tl.exp(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 - tmp16
    tmp18 = tmp0 + tmp17
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (1, 128), (128, 1))
    assert_size_stride(view, (128, 512), (512, 1))
    assert_size_stride(sub_1, (128, 50265), (50265, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_3, (50265, 512), (512, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 50265), (6433920, 50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 6433920, grid=grid(6433920), stream=stream0)
        buf1 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_4, buf1, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf4 = empty_strided((128, 1, 7), (7, 896, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_4, tangents_1, convert_element_type, buf4, 896, 7181, grid=grid(896), stream=stream0)
        buf5 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3.run(buf4, buf5, 128, 7, grid=grid(128), stream=stream0)
        del buf4
        buf3 = empty((128, 50265), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf3, (1, 128, 50265), (6433920, 50265, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf6, tangents_2, buf0, primals_4, tangents_1, convert_element_type, sub_1, buf5, 6433920, grid=grid(6433920), stream=stream0)
        del buf0
        del buf5
        del convert_element_type
        del primals_4
        del sub_1
        del tangents_1
        del tangents_2
        buf7 = empty((50265, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (50265, 128), (1, 50265), 0), view, out=buf7)
        del view
        buf8 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 50265), (50265, 1), 0), permute_3, out=buf8)
        del buf6
        del permute_3
        return (reinterpret_tensor(buf7, (50265, 512), (512, 1), 0), None, reinterpret_tensor(buf8, (1, 128, 512), (65536, 512, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    sub_1 = rand_strided((128, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((50265, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 50265), (6433920, 50265, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotSmallForConditionalGeneration', benchmark_compiled_module)
