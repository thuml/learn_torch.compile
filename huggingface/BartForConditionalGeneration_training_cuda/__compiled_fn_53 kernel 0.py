
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


# kernel path: /tmp/torchinductor_youkaichao/aa/caaxgv73rfw7opceuebf3llee36pg3jmzxubleuxrda3xppbei2p.py
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
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51471360
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


# kernel path: /tmp/torchinductor_youkaichao/24/c24ehugbhq4mk4kqjd5l3f7rgbwpj6e3pixv2bkokp455tpd7tk7.py
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
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/bs/cbskgr7ek2xlvwu72q7ouaulfrpktfvg36d6ss6vzn4e35vnyxrn.py
# Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# masked_lm_loss => full_default_1
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 50265
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr4 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp26 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(out_ptr1 + (r1 + (50265*x0)), tmp31, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (1, 1024), (1024, 1))
    assert_size_stride(view, (1024, 1024), (1024, 1))
    assert_size_stride(sub_1, (1024, 50265), (50265, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_3, (50265, 1024), (1024, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 1024, 50265), (51471360, 50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1024, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 51471360, grid=grid(51471360), stream=stream0)
        buf1 = empty_strided((1024, 1), (1, 1024), device='cuda', dtype=torch.int64)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_4, buf1, 1024, grid=grid(1024), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 1024, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_lm_loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_4, tangents_1, convert_element_type, tangents_2, sub_1, buf5, 1024, 50265, grid=grid(1024), stream=stream0)
        del convert_element_type
        del primals_4
        del sub_1
        del tangents_1
        del tangents_2
        buf6 = reinterpret_tensor(buf0, (50265, 1024), (1024, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (50265, 1024), (1, 50265), 0), view, out=buf6)
        del view
        buf7 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 50265), (50265, 1), 0), permute_3, out=buf7)
        del buf5
        del permute_3
        return (reinterpret_tensor(buf6, (50265, 1024), (1024, 1), 0), None, reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    sub_1 = rand_strided((1024, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 1024, 50265), (51471360, 50265, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
