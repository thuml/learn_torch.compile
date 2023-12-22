
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uxgglo3ujolhg5q263dujw3i7hkt3pw3x3gvw3dl2ui3ci7pqe.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102906784
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


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxaipkumvjupvw2u7hl26733xandeldynmuvgbuj72haerqiwhl.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2047
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/op/copucobl6cgogpswb5mbsxsbkwkyrqm5bubo4fdjdjl7a4366fhk.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2047
    rnumel = 50272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/66/c6675kz36yyrmy344wdpbfy4lcqwmpg5gn6secwwd6usfcq5xft2.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102957056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 50272)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 2047, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x2), tmp3, other=0.0)
    tmp5 = tl.load(in_ptr2 + (1 + x1), tmp3, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], -100, tl.int64)
    tmp7 = tmp5 != tmp6
    tmp12 = tmp9 / tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tl.load(in_ptr5 + (x2), tmp3, other=0.0)
    tmp17 = tl.exp(tmp16)
    tmp18 = tl.load(in_ptr6 + (x1), tmp3, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 - tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp3, tmp20, tmp21)
    tmp23 = tl.where(tmp3, tmp22, tmp13)
    tmp24 = tmp0 + tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_3, (1, 2048), (2048, 1))
    assert_size_stride(view, (2048, 768), (768, 1))
    assert_size_stride(sub_1, (2047, 50272), (50272, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_3, (50272, 768), (768, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 2048, 50272), (102957056, 50272, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((2047, 50272), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 102906784, grid=grid(102906784), stream=stream0)
        buf1 = empty_strided((2047, 1), (1, 2047), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_3, buf1, 2047, grid=grid(2047), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf4 = empty_strided((2047, 1), (1, 2047), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_3, tangents_1, convert_element_type, buf4, 2047, 50272, grid=grid(2047), stream=stream0)
        buf5 = empty((1, 2048, 50272), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_3.run(tangents_2, buf0, primals_3, tangents_1, convert_element_type, sub_1, buf4, buf5, 102957056, grid=grid(102957056), stream=stream0)
        del buf0
        del buf4
        del convert_element_type
        del primals_3
        del sub_1
        del tangents_1
        del tangents_2
        buf6 = empty((50272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (50272, 2048), (1, 50272), 0), view, out=buf6)
        del view
        buf7 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (2048, 50272), (50272, 1), 0), permute_3, out=buf7)
        del buf5
        del permute_3
        return (reinterpret_tensor(buf6, (50272, 768), (768, 1), 0), reinterpret_tensor(buf7, (1, 2048, 768), (1572864, 768, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_1 = rand_strided((2047, 50272), (50272, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((50272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 2048, 50272), (102957056, 50272, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
