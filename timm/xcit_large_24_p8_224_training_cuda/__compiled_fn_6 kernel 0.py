
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtcbt4g57hrviadj7jcv3toxxh4xcnjikjw4nylbesjf4s5onol.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/lw/clwvyoprruvtbt6i4i3rqekjoyuu72nqnlh76eb4hw7bcgkzjolv.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 785
    x1 = (xindex // 785)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp7 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = x0
        tmp18 = tl.full([1, 1], 0, tl.int32)
        tmp19 = tmp17 == tmp18
        tmp21 = 0.0
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = 768.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 - tmp9
        tmp29 = tmp28 * tmp14
        tmp30 = tmp27 - tmp29
        tmp31 = tmp16 * tmp30
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/cayzpqg7r6kzz2l7ypopz3wyqfydcfwtwx5dvo575sgl3kku2sad.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (126*x1)) % 785
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (768*(((r2 + (126*x1)) // 785) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmrdgp2mqluk26ulictuyxhxz2nsnfv7c425yspzp5gqn4pwpgl.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldliznolcpnd7yfi2w6gw6b7wuvtgvejojx344ca4m3rwgociz7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 6280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 785
        r2 = (rindex // 785)
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r1
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4o/c4oqm32le2arxuy3atqralhe26axw7ym6n7bl4cysxqd2twxgmzf.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cyge4hyha5fyaavadeaxuaa6xgk7oafyonfonfxejsymrjkwju2s.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rm/crmuk7plu3v7kqos3lwojjp35ai43r2bwzxavyvgndvquh3bvetr.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbmkq5ns372bujjtjz6evqoxrc7pead7wcwpfutbq7ybtsh7dlu.py
# Source Nodes: [x_475, x_476], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
# x_475 => add_444
# x_476 => add_445, erf_51, mul_600
triton_per_fused_add_gelu_gelu_backward_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp0 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce527g6rwrs6i7e6t3ecau7b4nqlu6x2gtsgo4c4vw2x3zgz3cuo.py
# Source Nodes: [x_475, x_476], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
# x_475 => add_444
# x_476 => add_445, erf_51, mul_600
triton_poi_fused_add_gelu_gelu_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp0 * tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cytqhxv3xnoi3qlwjf3ktj3jp6qjdlu62bemqlsauschgndjc6by.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]

triton_per_fused_add_mul_native_layer_norm_backward_select_backward_slice_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_backward_select_backward_slice_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp3 & xmask, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = 0.0
    tmp8 = tl.where(tmp3, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp10 = tmp1 < tmp2
    tmp11 = tl.load(in_ptr1 + (r2 + (768*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp7)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp34, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cma7qfok2ukuyxgj2uvbqc2lwycddwxtzanl6jyrzzbjbdmgdh43.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]

triton_red_fused_add_native_layer_norm_backward_select_backward_slice_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_select_backward_slice_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = (r2 + (126*x1)) % 785
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tmp6 & tmp2
        tmp8 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp7 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp11 = 0.0
        tmp12 = tl.where(tmp6, tmp10, tmp11)
        tmp13 = tmp3 + tmp12
        tmp14 = tmp4 < tmp5
        tmp15 = tmp14 & tmp2
        tmp16 = tl.load(in_ptr1 + (x0 + (768*(((r2 + (126*x1)) // 785) % 8))), rmask & tmp15 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp14, tmp18, tmp11)
        tmp20 = tmp13 + tmp19
        tmp21 = tl.load(in_ptr2 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp28 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp29 = tl.where(tmp2, tmp20, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbutn5edxenuzmrps2u53dudazwqruj7rfj3zycnwxxm4676bwjy.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftaohgm2qleqoeqc472piolitrlzxxmuetat5xphbvcuvjcru4w.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (602880*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwjlziuxnhy5qqkmvybdym5zwpgp2e3fsao6mn6wnuqb34vxqsg.py
# Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]

triton_poi_fused__scaled_dot_product_efficient_attention_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 48
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (48*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbuag4qv3c7bfz6g7aojcndca6ovcw2owxh5oq3e7ropb7dqg475.py
# Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]

triton_poi_fused__scaled_dot_product_efficient_attention_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 37680
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (602880*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (37680*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5isbxhfgkattteynekjaabgvgsryhhszb3hwlgeqyzaz4gmbkri.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2) + (96768*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhwes6rzre5cd2uqjbfco2ldk4l336z2e2p26bcyvjogy2ovpdr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]

triton_per_fused_add_native_layer_norm_backward_select_backward_slice_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_select_backward_slice_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 785)
    tmp10 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp0 == tmp14
    tmp17 = tl.where(tmp15, tmp16, tmp8)
    tmp18 = tmp13 + tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tmp20 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp33 = 768.0
    tmp34 = tmp20 * tmp33
    tmp35 = tmp34 - tmp24
    tmp36 = tmp25 * tmp30
    tmp37 = tmp35 - tmp36
    tmp38 = tmp32 * tmp37
    tmp39 = tmp31 + tmp38
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp18, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3keorx63eew4fqhmtqxg66sldklshcg254jhkdjroglmkyvv2dv.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp12 = tl.where(tmp2, tmp3, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/66/c66k24qwi27g6sxmsmzreij53e3ba34efhf34ruxtvpln3xubnci.py
# Source Nodes: [x_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# x_norm1 => mul_585, sub_123
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 785)
    tmp10 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp0 == tmp14
    tmp17 = tl.where(tmp15, tmp16, tmp8)
    tmp18 = tmp13 + tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp27 = tmp25 - tmp26
    tmp29 = tmp27 * tmp28
    tmp30 = tmp20 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = 768.0
    tmp37 = tmp28 / tmp36
    tmp38 = tmp20 * tmp36
    tmp39 = tmp38 - tmp24
    tmp40 = tmp29 * tmp34
    tmp41 = tmp39 - tmp40
    tmp42 = tmp37 * tmp41
    tmp43 = tmp35 + tmp42
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp18, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp43, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t6/ct67y6b6m436xunqyq5um37svsjj2fvfbgiynjnulmafkd3sviti.py
# Source Nodes: [x_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_norm1 => mul_585, sub_123
triton_red_fused_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 38400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (126*x1)) % 6280), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (126*x1)) % 6280), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp16 = tl.where(tmp2, tmp3, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crgseurem6dvw4s3q3eoeyddnaebt5zi63nm64xpdrlpp5ov7jgk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (768 + x0 + (768*((r2 + (128*x1)) % 784)) + (602880*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pxqkxidpdns5wpew35s3olvhyimmj6f6cupcfepwvo6xc65dx6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_per_fused_mul_sum_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cba75p55qrma4xzhbnrsy3os27ziwoqf7axun6xnxvdacjevi33v.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 602112)
    x3 = xindex % 602112
    x0 = xindex % 768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x3 + (602880*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xy/cxys7y7jzurouklwrjkh4maaxqmknqrhtxxge2a5dv5raor2gidj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtxt4niydnmixjksjfuj2b3y6c7fkf3q4mjlysqbh6bcnx5qzbs.py
# Source Nodes: [x_454], Original ATen: [aten.gelu, aten.gelu_backward]
# x_454 => add_429, erf_49, mul_582
triton_poi_fused_gelu_gelu_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2omag6c7oeuvzy3rfb6ba3a4lbb3jbehcemmrj42c3qvvajiid.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[262144, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 150528
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nf/cnfyrav4zpn3xmytmhl5bb6sfhgkf2vorqtyzcveizbplfk3tuwx.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmm2zast3xi66kpkykojlywgptkh3ozogyyn3uf5obglnsj2o2cu.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]

triton_per_fused_add_mul_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 784
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (768 + r1 + (768*x2) + (602880*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpckfko4ifyyrmxtfyvwdcozllbbjoq7gtoftsfvx34arrwyzqd.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kyo4g5b6hq5cbvbmkb7lq2c325dm3to3xb5bc35vlfhsmzgatd.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6u/c6uifjcar5wnnxpcn5hi2b76p3nzafewqqulvc2phajzyw7gtzbp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtz4oer6gwquesyfkxkoc56moutcathmy5eborl4sixw55npx4o.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgs2gtwc2mv2rfoa6kjqp3wgygvz4w2gscyb57gt7zdo4kty4wty.py
# Source Nodes: [x_448], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
# x_448 => add_420, erf_48, mul_568, mul_569, mul_570
triton_red_fused_gelu_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp10 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (602112*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (768*r2) + (98304*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 * tmp2
        tmp4 = 0.7071067811865476
        tmp5 = tmp1 * tmp4
        tmp6 = tl.math.erf(tmp5)
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp9 = tmp3 * tmp8
        tmp11 = tmp9 - tmp10
        tmp12 = tmp0 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckciiqs42ukvd2y42j4lkocq4vofnqyzemoqulqai2ijp2xbwtde.py
# Source Nodes: [x_448], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
# x_448 => add_420, erf_48, mul_568, mul_569, mul_570
triton_per_fused_gelu_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjfunqpcsyocedi6xgc524a6izl3k7r3uxvx7xxsene43k6jtni.py
# Source Nodes: [x_448], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
# x_448 => add_420, erf_48, mul_568, mul_569, mul_570
triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (768*x2) + (602112*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp0 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp8 * tmp2
    tmp27 = tmp1 * tmp1
    tmp28 = -0.5
    tmp29 = tmp27 * tmp28
    tmp30 = tl.exp(tmp29)
    tmp31 = 0.3989422804014327
    tmp32 = tmp30 * tmp31
    tmp33 = tmp1 * tmp32
    tmp34 = tmp26 + tmp33
    tmp35 = tmp25 * tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nygbgvyea5i4ukz4fhwy44gdxshbuqss6ngajtbez7e4ekjpld.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 6
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (100352*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckboj32f4y56fzofasudzbn2c2tkvouf3jjdtne4g6cr6n6wvzr6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (4704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfofm7kfwptu73ol6pkkfixttvkarse4zaf5uompwolgjjwsvmz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 784
    x2 = (xindex // 4704)
    x5 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (784*r3) + (100352*x0) + (602112*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymvudhjinjwcj6nua4pdua6k2l64ckt3lyuh4mnf7x6jxq2pzib.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3ji55clio32knxfpvzakm3fssa5vpoy53ysobljkwf3miiu6a6r.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (602112*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (768*r2) + (98304*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyr5hkth47fam63gytuu2yndnyy6evvvty5oh2ru7qsdxvwbo72l.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73yebe3okddkdr6d5s7dodkr64pjplfwdiwa7l4p6ojophf5k5n.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]

triton_poi_fused_add_mul_native_layer_norm_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_layer_norm_backward_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0 + (784*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 768.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tmp16 = tmp14 * tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (768*y3)), tmp14, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosvsrvpvmy5tuva5nsymr4s3gchykzixakzmh4ukl2nqbjbwkm5.py
# Source Nodes: [x_443], Original ATen: [aten.add, aten.mul, aten.sum]
# x_443 => add_416
triton_red_fused_add_mul_sum_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cquky75epyt2i4wcc4ndsoy2nu2hohpb2sd5xpav7ob755ozehvy.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (602112*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/crajvb7gdlptp47vatb5szxce6lffinvbdw5oqwh3cokc3vepjuw.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 48
    x1 = (xindex // 48) % 16
    x2 = (xindex // 768)
    tmp0 = tl.load(in_ptr0 + (r3 + (48*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1 + (16*r3) + (768*x0) + (36864*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlrwvqsovljw4trakzoztz4qagqzddrs2nmlobpagrrzyvfjyhy.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_red_fused__softmax_backward_data_mul_sum_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_mul_sum_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((48*(((r2 + (6144*x1)) // 48) % 48)) + (2304*x0) + (36864*((r2 + (6144*x1)) // 2304)) + (r2 % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (16*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((48*x0) + (768*((r2 + (6144*x1)) // 2304)) + (((r2 + (6144*x1)) // 48) % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((48*(((r2 + (6144*x1)) // 48) % 48)) + (2304*x0) + (36864*((r2 + (6144*x1)) // 2304)) + (r2 % 48)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp1 * tmp3
        tmp5 = tmp2 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cibabz2y4mmkcytdg7c7yl7jzr74kxfg7x745svbi4whiswiz6lb.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2ihqxakkan4tlhnfkkbtuins4ruchgkkxkp7i2zzjqg5gdoyj6.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_poi_fused__softmax_backward_data_mul_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_backward_data_mul_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    x3 = (xindex // 48)
    tmp0 = tl.load(in_out_ptr0 + (x5 + (2304*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (16*x5) + (36864*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3 + (48*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp7 = tmp5 * tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (2304*y4)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqmigq5u2vmzodw4d7ldgrcqzqtswbmyb7fmwojdppdeltuz2tv.py
# Source Nodes: [k_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
# k_47 => div_76
triton_red_fused_div_mul_neg_sum_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_neg_sum_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x5 = (xindex // 48)
    x1 = (xindex // 48) % 7
    x2 = (xindex // 336) % 16
    x3 = (xindex // 5376)
    tmp3 = tl.load(in_ptr2 + (x2 + (16*x0) + (768*x3)), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r4) + (5376*x5)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr1 + (x0 + (48*x2) + (2304*r4) + (258048*x1) + (1806336*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -tmp0
        tmp4 = 1e-12
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = tmp2 / tmp5
        tmp7 = tmp6 / tmp5
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caemnmh4uwe7nhwbpwuqvthtrxfwrs237htlghkesi3eicijevmw.py
# Source Nodes: [k_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
# k_47 => div_76
triton_per_fused_div_mul_neg_sum_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_neg_sum_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 48
    x1 = (xindex // 48)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (336*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkebjg5o4kd2mf4cchnniatpzgnwm766v4fnnaqmllmlf5cy5eq.py
# Source Nodes: [q_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
# q_47 => div_75
triton_red_fused_div_mul_neg_sum_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_neg_sum_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 43008
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    x0 = xindex % 7
    x3 = (xindex // 5376)
    x6 = (xindex // 7) % 768
    x1 = (xindex // 7) % 48
    x2 = (xindex // 336) % 16
    tmp3 = tl.load(in_ptr2 + (x2 + (16*x1) + (768*x3)), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = tl.load(in_ptr0 + (r4 + (112*x5)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr1 + (x6 + (2304*r4) + (258048*x0) + (1806336*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -tmp0
        tmp4 = 1e-12
        tmp5 = triton_helpers.maximum(tmp3, tmp4)
        tmp6 = tmp2 / tmp5
        tmp7 = tmp6 / tmp5
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfrq4trxq7glfu52c6lbuig32rtn3ouxqs5yvhwphbdekpqbu65.py
# Source Nodes: [q_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
# q_47 => div_75
triton_per_fused_div_mul_neg_sum_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_neg_sum_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl24yn7fkfmbd7nggmfh5z6os3rpq7yd6lqyee77i4qeopsijfrb.py
# Source Nodes: [], Original ATen: [aten.stack]

triton_poi_fused_stack_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 768)
    x3 = xindex
    y4 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48) % 16
    y5 = yindex % 768
    y6 = (yindex // 48)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (784*y4)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y1 + (16*y0) + (768*y2), [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = 1e-12
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp5 / tmp8
    tmp10 = tmp6 >= tmp7
    tmp11 = tl.load(in_ptr2 + (tl.broadcast_to(y4, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp6 == tmp12
    tmp15 = tl.load(in_ptr3 + (y5 + (2304*x3) + (1806336*y2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15 / tmp6
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tmp13 * tmp17
    tmp19 = tmp9 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = tl.full([1, 1], 16, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tmp22 & tmp24
    tmp26 = tl.load(in_ptr4 + ((-4816896) + y0 + (48*x3) + (37632*y6)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr5 + (tl.broadcast_to((-6144) + y1 + (16*y0) + (768*y2), [XBLOCK, YBLOCK])), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = triton_helpers.maximum(tmp27, tmp7)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp27 >= tmp7
    tmp31 = tl.load(in_ptr6 + (tl.broadcast_to((-6144) + y4, [XBLOCK, YBLOCK])), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp30, tmp31, tmp12)
    tmp33 = tmp27 == tmp12
    tmp34 = tl.load(in_ptr7 + ((-14450688) + y5 + (2304*x3) + (1806336*y2)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 / tmp27
    tmp36 = tl.where(tmp33, tmp12, tmp35)
    tmp37 = tmp32 * tmp36
    tmp38 = tmp29 + tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp25, tmp38, tmp39)
    tmp41 = tmp0 >= tmp23
    tmp42 = tl.full([1, 1], 24, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr8 + ((-9633792) + x3 + (784*y4)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp25, tmp40, tmp46)
    tmp48 = tl.where(tmp4, tmp21, tmp47)
    tl.store(out_ptr0 + (x3 + (784*y4)), tmp48, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznxdo5sdvtu5qxxqqgkns3e7vvxkuu2rpujhhoay75dkt5xptms.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((784*(x1 % 768)) + (602112*(y0 // 784)) + (4816896*(x1 // 768)) + (y0 % 784)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (2304*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesh7r2fxzuleddslznaadyyl46uhn4fxfjtfcwqcoqcrem2g33i.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112896
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2304
    x1 = (xindex // 2304)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (294912*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwssfk4ca3x6uvidxg2uptmhout4p7vzg62t7xhhpdqjtaqxzfai.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2304*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6ar7rv6cynrkfj7fo73oovjrbopghvxcdbwdoajrjukwr27xwut.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]

triton_per_fused_add_mul_native_layer_norm_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_backward_57', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqkeqjwbnc74tmhgfgaz6zgcwwydwdny7cigq2xsormzbiftjzme.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctka3majuoe7guoxeuzlcdugxhstsvneas7jortwjalgxsrv2ss2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1048576, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + (768*x0) + (602112*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5vkglg4eu5rn236npw6j7vf5bn3itvptvzsbl5ttjd3klpv2ci.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5lungsmhbeqwgfdemclmow7m6v2u4sijvg622yagjdtsydr2ue.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuj62yr7qsvui4ji34u47kifcaf6gc4jyq4r3aorvc2yyfuhpdz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjosciuixbwrbegki2qdzcbsvmngnbwnfedpsiq3xfed5zj5owhi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6haiikfzun4sneubs5go56ud47k63iixsaosxf6f46uxhyl5iyb.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_red_fused_gelu_backward_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_backward_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (1204224*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (384*r2) + (49152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmljslqxwqhy2dbzssrg6su25q2pt2wngr66h6tmmtcaxksdgdbc.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_per_fused_gelu_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_backward_native_batch_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazh42jrz5ieabpqw5wtdxuaa33mr5liprklcjmkdkxhaqtu3koe.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_red_fused_gelu_backward_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_backward_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (1204224*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgihmmeak5kx2yw6ppzi463tose4aed74jvxdcbjo2qwmwfpxve.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_red_fused_gelu_backward_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_backward_native_batch_norm_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbenx6bfucxsxuw5ui7juq2t6imgzmej7dib7fr4x5xbltc4oyf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.gelu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_gelu_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_gelu_backward_native_batch_norm_backward_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (384*x2) + (1204224*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (384*x2) + (1204224*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.985969387755102e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4wa6ejqdrrr3elymnrqz4p5suhpngne5xp5x6wpt4peqrchvfh.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_red_fused_gelu_backward_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_backward_native_batch_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (2408448*((r2 + (196*x0)) // 12544)) + ((r2 + (196*x0)) % 12544)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (37632*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzoqhqbxgk4homxlgf3p3wqoanbipibqpw53ubvn5gisgpgvqsf.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_per_fused_gelu_backward_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_backward_native_batch_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 192
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcrrsndnhxeuhy5gtkp3mdp77cdvkxm6pay2vxlptyvuqeugzel.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_red_fused_gelu_backward_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_backward_native_batch_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (2408448*((r2 + (196*x1)) // 12544)) + ((r2 + (196*x1)) % 12544)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (37632*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (192*r2) + (37632*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2gk2oojxyw7kjh4wzp47gkadihpbe25i2ud4a6nsc2z7wqxefz.py
# Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]

triton_red_fused_gelu_backward_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_backward_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pvjpez3hmv3qcdblptfdhaauw2niqncve6lwiej5ekwwijyws2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.gelu_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_gelu_backward_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_gelu_backward_native_batch_norm_backward_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (192*x2) + (2408448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (192*x2) + (2408448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp19, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_113, primals_118, primals_119, primals_121, primals_123, primals_125, primals_127, primals_133, primals_138, primals_139, primals_141, primals_143, primals_145, primals_147, primals_153, primals_158, primals_159, primals_161, primals_163, primals_165, primals_167, primals_173, primals_178, primals_179, primals_181, primals_183, primals_185, primals_187, primals_193, primals_198, primals_199, primals_201, primals_203, primals_205, primals_207, primals_213, primals_218, primals_219, primals_221, primals_223, primals_225, primals_227, primals_233, primals_238, primals_239, primals_241, primals_243, primals_245, primals_247, primals_253, primals_258, primals_259, primals_261, primals_263, primals_265, primals_267, primals_273, primals_278, primals_279, primals_281, primals_283, primals_285, primals_287, primals_293, primals_298, primals_299, primals_301, primals_303, primals_305, primals_307, primals_313, primals_318, primals_319, primals_321, primals_323, primals_325, primals_327, primals_333, primals_338, primals_339, primals_341, primals_343, primals_345, primals_347, primals_353, primals_358, primals_359, primals_361, primals_363, primals_365, primals_367, primals_373, primals_378, primals_379, primals_381, primals_383, primals_385, primals_387, primals_393, primals_398, primals_399, primals_401, primals_403, primals_405, primals_407, primals_413, primals_418, primals_419, primals_421, primals_423, primals_425, primals_427, primals_433, primals_438, primals_439, primals_441, primals_443, primals_445, primals_447, primals_453, primals_458, primals_459, primals_461, primals_463, primals_465, primals_467, primals_473, primals_478, primals_479, primals_481, primals_483, primals_485, primals_487, primals_493, primals_498, primals_499, primals_501, primals_503, primals_505, primals_507, primals_513, primals_518, primals_519, primals_521, primals_523, primals_525, primals_527, primals_533, primals_538, primals_539, primals_541, primals_543, primals_545, primals_547, primals_553, primals_558, primals_559, primals_561, primals_563, primals_565, primals_567, primals_573, primals_578, primals_579, primals_581, primals_583, primals_585, primals_587, primals_593, primals_603, primals_606, primals_609, primals_619, primals_622, primals_625, primals_710, convolution, squeeze_1, mul_9, convolution_1, squeeze_4, mul_19, convolution_2, squeeze_7, permute_1, mul_33, view_4, getitem_8, getitem_9, pow_3, pow_5, bmm, view_14, mm, mul_37, view_16, convolution_4, squeeze_10, add_34, convolution_5, mul_50, view_18, addmm_1, view_20, addmm_2, mul_56, view_22, getitem_19, getitem_20, pow_7, pow_9, bmm_2, view_32, mm_1, mul_60, view_34, convolution_6, squeeze_13, add_51, convolution_7, mul_73, view_36, addmm_4, view_38, addmm_5, mul_79, view_40, getitem_30, getitem_31, pow_11, pow_13, bmm_4, view_50, mm_2, mul_83, view_52, convolution_8, squeeze_16, add_68, convolution_9, mul_96, view_54, addmm_7, view_56, addmm_8, mul_102, view_58, getitem_41, getitem_42, pow_15, pow_17, bmm_6, view_68, mm_3, mul_106, view_70, convolution_10, squeeze_19, add_85, convolution_11, mul_119, view_72, addmm_10, view_74, addmm_11, mul_125, view_76, getitem_52, getitem_53, pow_19, pow_21, bmm_8, view_86, mm_4, mul_129, view_88, convolution_12, squeeze_22, add_102, convolution_13, mul_142, view_90, addmm_13, view_92, addmm_14, mul_148, view_94, getitem_63, getitem_64, pow_23, pow_25, bmm_10, view_104, mm_5, mul_152, view_106, convolution_14, squeeze_25, add_119, convolution_15, mul_165, view_108, addmm_16, view_110, addmm_17, mul_171, view_112, getitem_74, getitem_75, pow_27, pow_29, bmm_12, view_122, mm_6, mul_175, view_124, convolution_16, squeeze_28, add_136, convolution_17, mul_188, view_126, addmm_19, view_128, addmm_20, mul_194, view_130, getitem_85, getitem_86, pow_31, pow_33, bmm_14, view_140, mm_7, mul_198, view_142, convolution_18, squeeze_31, add_153, convolution_19, mul_211, view_144, addmm_22, view_146, addmm_23, mul_217, view_148, getitem_96, getitem_97, pow_35, pow_37, bmm_16, view_158, mm_8, mul_221, view_160, convolution_20, squeeze_34, add_170, convolution_21, mul_234, view_162, addmm_25, view_164, addmm_26, mul_240, view_166, getitem_107, getitem_108, pow_39, pow_41, bmm_18, view_176, mm_9, mul_244, view_178, convolution_22, squeeze_37, add_187, convolution_23, mul_257, view_180, addmm_28, view_182, addmm_29, mul_263, view_184, getitem_118, getitem_119, pow_43, pow_45, bmm_20, view_194, mm_10, mul_267, view_196, convolution_24, squeeze_40, add_204, convolution_25, mul_280, view_198, addmm_31, view_200, addmm_32, mul_286, view_202, getitem_129, getitem_130, pow_47, pow_49, bmm_22, view_212, mm_11, mul_290, view_214, convolution_26, squeeze_43, add_221, convolution_27, mul_303, view_216, addmm_34, view_218, addmm_35, mul_309, view_220, getitem_140, getitem_141, pow_51, pow_53, bmm_24, view_230, mm_12, mul_313, view_232, convolution_28, squeeze_46, add_238, convolution_29, mul_326, view_234, addmm_37, view_236, addmm_38, mul_332, view_238, getitem_151, getitem_152, pow_55, pow_57, bmm_26, view_248, mm_13, mul_336, view_250, convolution_30, squeeze_49, add_255, convolution_31, mul_349, view_252, addmm_40, view_254, addmm_41, mul_355, view_256, getitem_162, getitem_163, pow_59, pow_61, bmm_28, view_266, mm_14, mul_359, view_268, convolution_32, squeeze_52, add_272, convolution_33, mul_372, view_270, addmm_43, view_272, addmm_44, mul_378, view_274, getitem_173, getitem_174, pow_63, pow_65, bmm_30, view_284, mm_15, mul_382, view_286, convolution_34, squeeze_55, add_289, convolution_35, mul_395, view_288, addmm_46, view_290, addmm_47, mul_401, view_292, getitem_184, getitem_185, pow_67, pow_69, bmm_32, view_302, mm_16, mul_405, view_304, convolution_36, squeeze_58, add_306, convolution_37, mul_418, view_306, addmm_49, view_308, addmm_50, mul_424, view_310, getitem_195, getitem_196, pow_71, pow_73, bmm_34, view_320, mm_17, mul_428, view_322, convolution_38, squeeze_61, add_323, convolution_39, mul_441, view_324, addmm_52, view_326, addmm_53, mul_447, view_328, getitem_206, getitem_207, pow_75, pow_77, bmm_36, view_338, mm_18, mul_451, view_340, convolution_40, squeeze_64, add_340, convolution_41, mul_464, view_342, addmm_55, view_344, addmm_56, mul_470, view_346, getitem_217, getitem_218, pow_79, pow_81, bmm_38, view_356, mm_19, mul_474, view_358, convolution_42, squeeze_67, add_357, convolution_43, mul_487, view_360, addmm_58, view_362, addmm_59, mul_493, view_364, getitem_228, getitem_229, pow_83, pow_85, bmm_40, view_374, mm_20, mul_497, view_376, convolution_44, squeeze_70, add_374, convolution_45, mul_510, view_378, addmm_61, view_380, addmm_62, mul_516, view_382, getitem_239, getitem_240, pow_87, pow_89, bmm_42, view_392, mm_21, mul_520, view_394, convolution_46, squeeze_73, add_391, convolution_47, mul_533, view_396, addmm_64, view_398, addmm_65, mul_539, view_400, getitem_250, getitem_251, pow_91, pow_93, bmm_44, view_410, mm_22, mul_543, view_412, convolution_48, squeeze_76, add_408, convolution_49, mul_556, view_414, addmm_67, view_416, addmm_68, mul_562, view_418, getitem_261, getitem_262, pow_95, pow_97, bmm_46, view_428, mm_23, mul_566, view_430, convolution_50, squeeze_79, add_425, convolution_51, mul_579, view_432, addmm_70, view_434, addmm_71, cat_3, getitem_271, rsqrt_99, select, permute_220, view_437, permute_222, permute_224, getitem_273, getitem_274, getitem_275, view_444, cat_4, mul_588, view_446, mm_24, view_448, addmm_76, mul_594, select_1, permute_230, view_451, permute_232, permute_234, getitem_281, getitem_282, getitem_283, view_458, cat_6, mul_597, view_460, mm_25, view_462, addmm_81, mul_603, clone_271, permute_240, div_78, permute_244, permute_250, div_79, permute_252, alias_74, permute_258, permute_263, permute_268, div_80, permute_272, permute_278, div_81, permute_280, alias_75, permute_286, permute_291, permute_296, permute_300, permute_304, div_83, unsqueeze_119, div_84, permute_312, permute_315, permute_316, alias_76, permute_317, permute_318, permute_321, div_93, permute_325, permute_329, div_94, unsqueeze_131, div_95, permute_337, permute_340, permute_341, alias_79, permute_342, permute_343, permute_346, div_104, permute_350, permute_354, div_105, unsqueeze_143, div_106, permute_362, permute_365, permute_366, alias_82, permute_367, permute_368, permute_371, div_115, permute_375, permute_379, div_116, unsqueeze_155, div_117, permute_387, permute_390, permute_391, alias_85, permute_392, permute_393, permute_396, div_126, permute_400, permute_404, div_127, unsqueeze_167, div_128, permute_412, permute_415, permute_416, alias_88, permute_417, permute_418, permute_421, div_137, permute_425, permute_429, div_138, unsqueeze_179, div_139, permute_437, permute_440, permute_441, alias_91, permute_442, permute_443, permute_446, div_148, permute_450, permute_454, div_149, unsqueeze_191, div_150, permute_462, permute_465, permute_466, alias_94, permute_467, permute_468, permute_471, div_159, permute_475, permute_479, div_160, unsqueeze_203, div_161, permute_487, permute_490, permute_491, alias_97, permute_492, permute_493, permute_496, div_170, permute_500, permute_504, div_171, unsqueeze_215, div_172, permute_512, permute_515, permute_516, alias_100, permute_517, permute_518, permute_521, div_181, permute_525, permute_529, div_182, unsqueeze_227, div_183, permute_537, permute_540, permute_541, alias_103, permute_542, permute_543, permute_546, div_192, permute_550, permute_554, div_193, unsqueeze_239, div_194, permute_562, permute_565, permute_566, alias_106, permute_567, permute_568, permute_571, div_203, permute_575, permute_579, div_204, unsqueeze_251, div_205, permute_587, permute_590, permute_591, alias_109, permute_592, permute_593, permute_596, div_214, permute_600, permute_604, div_215, unsqueeze_263, div_216, permute_612, permute_615, permute_616, alias_112, permute_617, permute_618, permute_621, div_225, permute_625, permute_629, div_226, unsqueeze_275, div_227, permute_637, permute_640, permute_641, alias_115, permute_642, permute_643, permute_646, div_236, permute_650, permute_654, div_237, unsqueeze_287, div_238, permute_662, permute_665, permute_666, alias_118, permute_667, permute_668, permute_671, div_247, permute_675, permute_679, div_248, unsqueeze_299, div_249, permute_687, permute_690, permute_691, alias_121, permute_692, permute_693, permute_696, div_258, permute_700, permute_704, div_259, unsqueeze_311, div_260, permute_712, permute_715, permute_716, alias_124, permute_717, permute_718, permute_721, div_269, permute_725, permute_729, div_270, unsqueeze_323, div_271, permute_737, permute_740, permute_741, alias_127, permute_742, permute_743, permute_746, div_280, permute_750, permute_754, div_281, unsqueeze_335, div_282, permute_762, permute_765, permute_766, alias_130, permute_767, permute_768, permute_771, div_291, permute_775, permute_779, div_292, unsqueeze_347, div_293, permute_787, permute_790, permute_791, alias_133, permute_792, permute_793, permute_796, div_302, permute_800, permute_804, div_303, unsqueeze_359, div_304, permute_812, permute_815, permute_816, alias_136, permute_817, permute_818, permute_821, div_313, permute_825, permute_829, div_314, unsqueeze_371, div_315, permute_837, permute_840, permute_841, alias_139, permute_842, permute_843, permute_846, div_324, permute_850, permute_854, div_325, unsqueeze_383, div_326, permute_862, permute_865, permute_866, alias_142, permute_867, permute_868, permute_871, div_335, permute_875, permute_879, div_336, unsqueeze_395, div_337, permute_887, permute_890, permute_891, alias_145, permute_892, permute_893, permute_896, div_346, unsqueeze_407, add_682, unsqueeze_419, add_684, unsqueeze_431, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_2, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (192, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_103, (192, ), (1, ))
    assert_size_stride(primals_105, (384, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_108, (768, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_111, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_201, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_205, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_261, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_263, (768, ), (1, ))
    assert_size_stride(primals_265, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_278, (768, ), (1, ))
    assert_size_stride(primals_279, (768, ), (1, ))
    assert_size_stride(primals_281, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_283, (768, ), (1, ))
    assert_size_stride(primals_285, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_293, (768, ), (1, ))
    assert_size_stride(primals_298, (768, ), (1, ))
    assert_size_stride(primals_299, (768, ), (1, ))
    assert_size_stride(primals_301, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_305, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_307, (768, ), (1, ))
    assert_size_stride(primals_313, (768, ), (1, ))
    assert_size_stride(primals_318, (768, ), (1, ))
    assert_size_stride(primals_319, (768, ), (1, ))
    assert_size_stride(primals_321, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_323, (768, ), (1, ))
    assert_size_stride(primals_325, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_327, (768, ), (1, ))
    assert_size_stride(primals_333, (768, ), (1, ))
    assert_size_stride(primals_338, (768, ), (1, ))
    assert_size_stride(primals_339, (768, ), (1, ))
    assert_size_stride(primals_341, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_343, (768, ), (1, ))
    assert_size_stride(primals_345, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_347, (768, ), (1, ))
    assert_size_stride(primals_353, (768, ), (1, ))
    assert_size_stride(primals_358, (768, ), (1, ))
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_361, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_363, (768, ), (1, ))
    assert_size_stride(primals_365, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_373, (768, ), (1, ))
    assert_size_stride(primals_378, (768, ), (1, ))
    assert_size_stride(primals_379, (768, ), (1, ))
    assert_size_stride(primals_381, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_383, (768, ), (1, ))
    assert_size_stride(primals_385, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_387, (768, ), (1, ))
    assert_size_stride(primals_393, (768, ), (1, ))
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (768, ), (1, ))
    assert_size_stride(primals_401, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_403, (768, ), (1, ))
    assert_size_stride(primals_405, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_407, (768, ), (1, ))
    assert_size_stride(primals_413, (768, ), (1, ))
    assert_size_stride(primals_418, (768, ), (1, ))
    assert_size_stride(primals_419, (768, ), (1, ))
    assert_size_stride(primals_421, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_423, (768, ), (1, ))
    assert_size_stride(primals_425, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_427, (768, ), (1, ))
    assert_size_stride(primals_433, (768, ), (1, ))
    assert_size_stride(primals_438, (768, ), (1, ))
    assert_size_stride(primals_439, (768, ), (1, ))
    assert_size_stride(primals_441, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_443, (768, ), (1, ))
    assert_size_stride(primals_445, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_447, (768, ), (1, ))
    assert_size_stride(primals_453, (768, ), (1, ))
    assert_size_stride(primals_458, (768, ), (1, ))
    assert_size_stride(primals_459, (768, ), (1, ))
    assert_size_stride(primals_461, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_463, (768, ), (1, ))
    assert_size_stride(primals_465, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_467, (768, ), (1, ))
    assert_size_stride(primals_473, (768, ), (1, ))
    assert_size_stride(primals_478, (768, ), (1, ))
    assert_size_stride(primals_479, (768, ), (1, ))
    assert_size_stride(primals_481, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_483, (768, ), (1, ))
    assert_size_stride(primals_485, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_487, (768, ), (1, ))
    assert_size_stride(primals_493, (768, ), (1, ))
    assert_size_stride(primals_498, (768, ), (1, ))
    assert_size_stride(primals_499, (768, ), (1, ))
    assert_size_stride(primals_501, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_503, (768, ), (1, ))
    assert_size_stride(primals_505, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_507, (768, ), (1, ))
    assert_size_stride(primals_513, (768, ), (1, ))
    assert_size_stride(primals_518, (768, ), (1, ))
    assert_size_stride(primals_519, (768, ), (1, ))
    assert_size_stride(primals_521, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_523, (768, ), (1, ))
    assert_size_stride(primals_525, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_527, (768, ), (1, ))
    assert_size_stride(primals_533, (768, ), (1, ))
    assert_size_stride(primals_538, (768, ), (1, ))
    assert_size_stride(primals_539, (768, ), (1, ))
    assert_size_stride(primals_541, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_543, (768, ), (1, ))
    assert_size_stride(primals_545, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_547, (768, ), (1, ))
    assert_size_stride(primals_553, (768, ), (1, ))
    assert_size_stride(primals_558, (768, ), (1, ))
    assert_size_stride(primals_559, (768, ), (1, ))
    assert_size_stride(primals_561, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_563, (768, ), (1, ))
    assert_size_stride(primals_565, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_567, (768, ), (1, ))
    assert_size_stride(primals_573, (768, ), (1, ))
    assert_size_stride(primals_578, (768, ), (1, ))
    assert_size_stride(primals_579, (768, ), (1, ))
    assert_size_stride(primals_581, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_583, (768, ), (1, ))
    assert_size_stride(primals_585, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_587, (768, ), (1, ))
    assert_size_stride(primals_593, (768, ), (1, ))
    assert_size_stride(primals_603, (768, ), (1, ))
    assert_size_stride(primals_606, (3072, ), (1, ))
    assert_size_stride(primals_609, (768, ), (1, ))
    assert_size_stride(primals_619, (768, ), (1, ))
    assert_size_stride(primals_622, (3072, ), (1, ))
    assert_size_stride(primals_625, (768, ), (1, ))
    assert_size_stride(primals_710, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(squeeze_1, (192, ), (1, ))
    assert_size_stride(mul_9, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(convolution_1, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(squeeze_4, (384, ), (1, ))
    assert_size_stride(mul_19, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(convolution_2, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_7, (768, ), (1, ))
    assert_size_stride(permute_1, (1, 64, 28, 28), (50176, 1, 1792, 64))
    assert_size_stride(mul_33, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_4, (6272, 768), (768, 1))
    assert_size_stride(getitem_8, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_9, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_3, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_5, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_14, (6272, 768), (768, 1))
    assert_size_stride(mm, (6272, 768), (768, 1))
    assert_size_stride(mul_37, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_16, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_4, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_10, (768, ), (1, ))
    assert_size_stride(add_34, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_5, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_50, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_18, (6272, 768), (768, 1))
    assert_size_stride(addmm_1, (6272, 3072), (3072, 1))
    assert_size_stride(view_20, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_2, (6272, 768), (768, 1))
    assert_size_stride(mul_56, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_22, (6272, 768), (768, 1))
    assert_size_stride(getitem_19, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_20, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_7, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_9, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_2, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_32, (6272, 768), (768, 1))
    assert_size_stride(mm_1, (6272, 768), (768, 1))
    assert_size_stride(mul_60, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_34, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_6, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_13, (768, ), (1, ))
    assert_size_stride(add_51, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_7, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_73, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_36, (6272, 768), (768, 1))
    assert_size_stride(addmm_4, (6272, 3072), (3072, 1))
    assert_size_stride(view_38, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_5, (6272, 768), (768, 1))
    assert_size_stride(mul_79, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_40, (6272, 768), (768, 1))
    assert_size_stride(getitem_30, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_31, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_11, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_13, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_4, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_50, (6272, 768), (768, 1))
    assert_size_stride(mm_2, (6272, 768), (768, 1))
    assert_size_stride(mul_83, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_52, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_8, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_16, (768, ), (1, ))
    assert_size_stride(add_68, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_9, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_96, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_54, (6272, 768), (768, 1))
    assert_size_stride(addmm_7, (6272, 3072), (3072, 1))
    assert_size_stride(view_56, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_8, (6272, 768), (768, 1))
    assert_size_stride(mul_102, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_58, (6272, 768), (768, 1))
    assert_size_stride(getitem_41, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_42, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_15, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_17, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_6, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_68, (6272, 768), (768, 1))
    assert_size_stride(mm_3, (6272, 768), (768, 1))
    assert_size_stride(mul_106, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_70, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_10, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_19, (768, ), (1, ))
    assert_size_stride(add_85, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_11, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_119, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_72, (6272, 768), (768, 1))
    assert_size_stride(addmm_10, (6272, 3072), (3072, 1))
    assert_size_stride(view_74, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_11, (6272, 768), (768, 1))
    assert_size_stride(mul_125, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_76, (6272, 768), (768, 1))
    assert_size_stride(getitem_52, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_53, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_19, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_21, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_8, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_86, (6272, 768), (768, 1))
    assert_size_stride(mm_4, (6272, 768), (768, 1))
    assert_size_stride(mul_129, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_88, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_12, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_22, (768, ), (1, ))
    assert_size_stride(add_102, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_13, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_142, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_90, (6272, 768), (768, 1))
    assert_size_stride(addmm_13, (6272, 3072), (3072, 1))
    assert_size_stride(view_92, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_14, (6272, 768), (768, 1))
    assert_size_stride(mul_148, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_94, (6272, 768), (768, 1))
    assert_size_stride(getitem_63, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_64, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_23, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_25, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_10, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_104, (6272, 768), (768, 1))
    assert_size_stride(mm_5, (6272, 768), (768, 1))
    assert_size_stride(mul_152, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_106, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_14, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_25, (768, ), (1, ))
    assert_size_stride(add_119, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_15, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_165, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_108, (6272, 768), (768, 1))
    assert_size_stride(addmm_16, (6272, 3072), (3072, 1))
    assert_size_stride(view_110, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_17, (6272, 768), (768, 1))
    assert_size_stride(mul_171, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_112, (6272, 768), (768, 1))
    assert_size_stride(getitem_74, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_75, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_27, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_29, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_12, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_122, (6272, 768), (768, 1))
    assert_size_stride(mm_6, (6272, 768), (768, 1))
    assert_size_stride(mul_175, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_124, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_16, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_28, (768, ), (1, ))
    assert_size_stride(add_136, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_17, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_188, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_126, (6272, 768), (768, 1))
    assert_size_stride(addmm_19, (6272, 3072), (3072, 1))
    assert_size_stride(view_128, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_20, (6272, 768), (768, 1))
    assert_size_stride(mul_194, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_130, (6272, 768), (768, 1))
    assert_size_stride(getitem_85, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_86, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_31, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_33, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_14, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_140, (6272, 768), (768, 1))
    assert_size_stride(mm_7, (6272, 768), (768, 1))
    assert_size_stride(mul_198, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_142, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_18, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_31, (768, ), (1, ))
    assert_size_stride(add_153, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_19, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_211, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_144, (6272, 768), (768, 1))
    assert_size_stride(addmm_22, (6272, 3072), (3072, 1))
    assert_size_stride(view_146, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_23, (6272, 768), (768, 1))
    assert_size_stride(mul_217, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_148, (6272, 768), (768, 1))
    assert_size_stride(getitem_96, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_97, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_35, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_37, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_16, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_158, (6272, 768), (768, 1))
    assert_size_stride(mm_8, (6272, 768), (768, 1))
    assert_size_stride(mul_221, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_160, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_20, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_34, (768, ), (1, ))
    assert_size_stride(add_170, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_21, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_234, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_162, (6272, 768), (768, 1))
    assert_size_stride(addmm_25, (6272, 3072), (3072, 1))
    assert_size_stride(view_164, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_26, (6272, 768), (768, 1))
    assert_size_stride(mul_240, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_166, (6272, 768), (768, 1))
    assert_size_stride(getitem_107, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_108, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_39, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_41, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_18, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_176, (6272, 768), (768, 1))
    assert_size_stride(mm_9, (6272, 768), (768, 1))
    assert_size_stride(mul_244, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_178, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_22, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_37, (768, ), (1, ))
    assert_size_stride(add_187, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_23, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_257, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_180, (6272, 768), (768, 1))
    assert_size_stride(addmm_28, (6272, 3072), (3072, 1))
    assert_size_stride(view_182, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_29, (6272, 768), (768, 1))
    assert_size_stride(mul_263, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_184, (6272, 768), (768, 1))
    assert_size_stride(getitem_118, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_119, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_43, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_45, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_20, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_194, (6272, 768), (768, 1))
    assert_size_stride(mm_10, (6272, 768), (768, 1))
    assert_size_stride(mul_267, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_196, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_24, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_40, (768, ), (1, ))
    assert_size_stride(add_204, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_25, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_280, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_198, (6272, 768), (768, 1))
    assert_size_stride(addmm_31, (6272, 3072), (3072, 1))
    assert_size_stride(view_200, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_32, (6272, 768), (768, 1))
    assert_size_stride(mul_286, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_202, (6272, 768), (768, 1))
    assert_size_stride(getitem_129, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_130, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_47, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_49, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_22, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_212, (6272, 768), (768, 1))
    assert_size_stride(mm_11, (6272, 768), (768, 1))
    assert_size_stride(mul_290, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_214, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_26, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_43, (768, ), (1, ))
    assert_size_stride(add_221, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_27, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_303, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_216, (6272, 768), (768, 1))
    assert_size_stride(addmm_34, (6272, 3072), (3072, 1))
    assert_size_stride(view_218, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_35, (6272, 768), (768, 1))
    assert_size_stride(mul_309, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_220, (6272, 768), (768, 1))
    assert_size_stride(getitem_140, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_141, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_51, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_53, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_24, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_230, (6272, 768), (768, 1))
    assert_size_stride(mm_12, (6272, 768), (768, 1))
    assert_size_stride(mul_313, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_232, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_28, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_46, (768, ), (1, ))
    assert_size_stride(add_238, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_29, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_326, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_234, (6272, 768), (768, 1))
    assert_size_stride(addmm_37, (6272, 3072), (3072, 1))
    assert_size_stride(view_236, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_38, (6272, 768), (768, 1))
    assert_size_stride(mul_332, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_238, (6272, 768), (768, 1))
    assert_size_stride(getitem_151, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_152, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_55, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_57, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_26, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_248, (6272, 768), (768, 1))
    assert_size_stride(mm_13, (6272, 768), (768, 1))
    assert_size_stride(mul_336, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_250, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_30, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_49, (768, ), (1, ))
    assert_size_stride(add_255, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_31, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_349, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_252, (6272, 768), (768, 1))
    assert_size_stride(addmm_40, (6272, 3072), (3072, 1))
    assert_size_stride(view_254, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_41, (6272, 768), (768, 1))
    assert_size_stride(mul_355, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_256, (6272, 768), (768, 1))
    assert_size_stride(getitem_162, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_163, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_59, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_61, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_28, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_266, (6272, 768), (768, 1))
    assert_size_stride(mm_14, (6272, 768), (768, 1))
    assert_size_stride(mul_359, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_268, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_32, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_52, (768, ), (1, ))
    assert_size_stride(add_272, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_33, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_372, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_270, (6272, 768), (768, 1))
    assert_size_stride(addmm_43, (6272, 3072), (3072, 1))
    assert_size_stride(view_272, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_44, (6272, 768), (768, 1))
    assert_size_stride(mul_378, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_274, (6272, 768), (768, 1))
    assert_size_stride(getitem_173, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_174, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_63, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_65, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_30, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_284, (6272, 768), (768, 1))
    assert_size_stride(mm_15, (6272, 768), (768, 1))
    assert_size_stride(mul_382, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_286, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_34, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(add_289, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_35, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_395, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_288, (6272, 768), (768, 1))
    assert_size_stride(addmm_46, (6272, 3072), (3072, 1))
    assert_size_stride(view_290, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_47, (6272, 768), (768, 1))
    assert_size_stride(mul_401, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_292, (6272, 768), (768, 1))
    assert_size_stride(getitem_184, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_185, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_67, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_69, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_32, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_302, (6272, 768), (768, 1))
    assert_size_stride(mm_16, (6272, 768), (768, 1))
    assert_size_stride(mul_405, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_304, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_36, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_58, (768, ), (1, ))
    assert_size_stride(add_306, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_37, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_418, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_306, (6272, 768), (768, 1))
    assert_size_stride(addmm_49, (6272, 3072), (3072, 1))
    assert_size_stride(view_308, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_50, (6272, 768), (768, 1))
    assert_size_stride(mul_424, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_310, (6272, 768), (768, 1))
    assert_size_stride(getitem_195, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_196, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_71, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_73, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_34, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_320, (6272, 768), (768, 1))
    assert_size_stride(mm_17, (6272, 768), (768, 1))
    assert_size_stride(mul_428, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_322, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_38, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_61, (768, ), (1, ))
    assert_size_stride(add_323, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_39, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_441, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_324, (6272, 768), (768, 1))
    assert_size_stride(addmm_52, (6272, 3072), (3072, 1))
    assert_size_stride(view_326, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_53, (6272, 768), (768, 1))
    assert_size_stride(mul_447, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_328, (6272, 768), (768, 1))
    assert_size_stride(getitem_206, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_207, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_75, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_77, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_36, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_338, (6272, 768), (768, 1))
    assert_size_stride(mm_18, (6272, 768), (768, 1))
    assert_size_stride(mul_451, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_340, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_40, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(add_340, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_41, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_464, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_342, (6272, 768), (768, 1))
    assert_size_stride(addmm_55, (6272, 3072), (3072, 1))
    assert_size_stride(view_344, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_56, (6272, 768), (768, 1))
    assert_size_stride(mul_470, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_346, (6272, 768), (768, 1))
    assert_size_stride(getitem_217, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_218, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_79, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_81, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_38, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_356, (6272, 768), (768, 1))
    assert_size_stride(mm_19, (6272, 768), (768, 1))
    assert_size_stride(mul_474, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_358, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_42, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(add_357, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_43, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_487, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_360, (6272, 768), (768, 1))
    assert_size_stride(addmm_58, (6272, 3072), (3072, 1))
    assert_size_stride(view_362, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_59, (6272, 768), (768, 1))
    assert_size_stride(mul_493, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_364, (6272, 768), (768, 1))
    assert_size_stride(getitem_228, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_229, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_83, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_85, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_40, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_374, (6272, 768), (768, 1))
    assert_size_stride(mm_20, (6272, 768), (768, 1))
    assert_size_stride(mul_497, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_376, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_44, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_70, (768, ), (1, ))
    assert_size_stride(add_374, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_45, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_510, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_378, (6272, 768), (768, 1))
    assert_size_stride(addmm_61, (6272, 3072), (3072, 1))
    assert_size_stride(view_380, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_62, (6272, 768), (768, 1))
    assert_size_stride(mul_516, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_382, (6272, 768), (768, 1))
    assert_size_stride(getitem_239, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_240, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_87, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_89, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_42, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_392, (6272, 768), (768, 1))
    assert_size_stride(mm_21, (6272, 768), (768, 1))
    assert_size_stride(mul_520, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_394, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_46, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(add_391, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_47, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_533, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_396, (6272, 768), (768, 1))
    assert_size_stride(addmm_64, (6272, 3072), (3072, 1))
    assert_size_stride(view_398, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_65, (6272, 768), (768, 1))
    assert_size_stride(mul_539, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_400, (6272, 768), (768, 1))
    assert_size_stride(getitem_250, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_251, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_91, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_93, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_44, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_410, (6272, 768), (768, 1))
    assert_size_stride(mm_22, (6272, 768), (768, 1))
    assert_size_stride(mul_543, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_412, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_48, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_76, (768, ), (1, ))
    assert_size_stride(add_408, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_49, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_556, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_414, (6272, 768), (768, 1))
    assert_size_stride(addmm_67, (6272, 3072), (3072, 1))
    assert_size_stride(view_416, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_68, (6272, 768), (768, 1))
    assert_size_stride(mul_562, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_418, (6272, 768), (768, 1))
    assert_size_stride(getitem_261, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(getitem_262, (8, 16, 48, 784), (1806336, 48, 1, 2304))
    assert_size_stride(pow_95, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(pow_97, (8, 16, 48, 1), (768, 1, 16, 16))
    assert_size_stride(bmm_46, (128, 48, 48), (2304, 48, 1))
    assert_size_stride(view_428, (6272, 768), (768, 1))
    assert_size_stride(mm_23, (6272, 768), (768, 1))
    assert_size_stride(mul_566, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_430, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_50, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(add_425, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(convolution_51, (8, 768, 28, 28), (602112, 1, 21504, 768))
    assert_size_stride(mul_579, (8, 784, 768), (602112, 768, 1))
    assert_size_stride(view_432, (6272, 768), (768, 1))
    assert_size_stride(addmm_70, (6272, 3072), (3072, 1))
    assert_size_stride(view_434, (6272, 3072), (3072, 1))
    assert_size_stride(addmm_71, (6272, 768), (768, 1))
    assert_size_stride(cat_3, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(getitem_271, (8, 785, 1), (785, 1, 1))
    assert_size_stride(rsqrt_99, (8, 785, 1), (785, 1, 1))
    assert_size_stride(select, (8, 768), (602880, 1))
    assert_size_stride(permute_220, (8, 16, 1, 48), (768, 1, 768, 16))
    assert_size_stride(view_437, (6280, 768), (768, 1))
    assert_size_stride(permute_222, (8, 16, 785, 48), (602880, 1, 768, 16))
    assert_size_stride(permute_224, (8, 16, 785, 48), (602880, 1, 768, 16))
    assert_size_stride(getitem_273, (8, 16, 32), (512, 32, 1))
    assert_size_stride(getitem_274, (), ())
    assert_size_stride(getitem_275, (), ())
    assert_size_stride(view_444, (8, 768), (768, 1))
    assert_size_stride(cat_4, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(mul_588, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(view_446, (8, 768), (602880, 1))
    assert_size_stride(mm_24, (8, 3072), (3072, 1))
    assert_size_stride(view_448, (8, 3072), (3072, 1))
    assert_size_stride(addmm_76, (8, 768), (768, 1))
    assert_size_stride(mul_594, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(select_1, (8, 768), (602880, 1))
    assert_size_stride(permute_230, (8, 16, 1, 48), (768, 1, 768, 16))
    assert_size_stride(view_451, (6280, 768), (768, 1))
    assert_size_stride(permute_232, (8, 16, 785, 48), (602880, 1, 768, 16))
    assert_size_stride(permute_234, (8, 16, 785, 48), (602880, 1, 768, 16))
    assert_size_stride(getitem_281, (8, 16, 32), (512, 32, 1))
    assert_size_stride(getitem_282, (), ())
    assert_size_stride(getitem_283, (), ())
    assert_size_stride(view_458, (8, 768), (768, 1))
    assert_size_stride(cat_6, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(mul_597, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(view_460, (8, 768), (602880, 1))
    assert_size_stride(mm_25, (8, 3072), (3072, 1))
    assert_size_stride(view_462, (8, 3072), (3072, 1))
    assert_size_stride(addmm_81, (8, 768), (768, 1))
    assert_size_stride(mul_603, (8, 785, 768), (602880, 768, 1))
    assert_size_stride(clone_271, (8, 768), (768, 1))
    assert_size_stride(permute_240, (1000, 768), (768, 1))
    assert_size_stride(div_78, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_244, (768, 3072), (3072, 1))
    assert_size_stride(permute_250, (3072, 768), (768, 1))
    assert_size_stride(div_79, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_252, (768, 768), (768, 1))
    assert_size_stride(alias_74, (8, 16, 1, 48), (768, 1, 768, 16))
    assert_size_stride(permute_258, (768, 768), (768, 1))
    assert_size_stride(permute_263, (768, 768), (768, 1))
    assert_size_stride(permute_268, (768, 768), (768, 1))
    assert_size_stride(div_80, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_272, (768, 3072), (3072, 1))
    assert_size_stride(permute_278, (3072, 768), (768, 1))
    assert_size_stride(div_81, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_280, (768, 768), (768, 1))
    assert_size_stride(alias_75, (8, 16, 1, 48), (768, 1, 768, 16))
    assert_size_stride(permute_286, (768, 768), (768, 1))
    assert_size_stride(permute_291, (768, 768), (768, 1))
    assert_size_stride(permute_296, (768, 768), (768, 1))
    assert_size_stride(permute_300, (768, 3072), (3072, 1))
    assert_size_stride(permute_304, (3072, 768), (768, 1))
    assert_size_stride(div_83, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_119, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_84, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_312, (768, 768), (768, 1))
    assert_size_stride(permute_315, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_316, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_76, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_317, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_318, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_321, (2304, 768), (768, 1))
    assert_size_stride(div_93, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_325, (768, 3072), (3072, 1))
    assert_size_stride(permute_329, (3072, 768), (768, 1))
    assert_size_stride(div_94, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_131, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_95, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_337, (768, 768), (768, 1))
    assert_size_stride(permute_340, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_341, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_79, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_342, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_343, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_346, (2304, 768), (768, 1))
    assert_size_stride(div_104, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_350, (768, 3072), (3072, 1))
    assert_size_stride(permute_354, (3072, 768), (768, 1))
    assert_size_stride(div_105, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_143, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_106, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_362, (768, 768), (768, 1))
    assert_size_stride(permute_365, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_366, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_82, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_367, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_368, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_371, (2304, 768), (768, 1))
    assert_size_stride(div_115, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_375, (768, 3072), (3072, 1))
    assert_size_stride(permute_379, (3072, 768), (768, 1))
    assert_size_stride(div_116, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_155, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_117, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_387, (768, 768), (768, 1))
    assert_size_stride(permute_390, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_391, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_85, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_392, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_393, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_396, (2304, 768), (768, 1))
    assert_size_stride(div_126, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_400, (768, 3072), (3072, 1))
    assert_size_stride(permute_404, (3072, 768), (768, 1))
    assert_size_stride(div_127, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_167, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_128, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_412, (768, 768), (768, 1))
    assert_size_stride(permute_415, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_416, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_88, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_417, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_418, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_421, (2304, 768), (768, 1))
    assert_size_stride(div_137, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_425, (768, 3072), (3072, 1))
    assert_size_stride(permute_429, (3072, 768), (768, 1))
    assert_size_stride(div_138, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_179, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_139, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_437, (768, 768), (768, 1))
    assert_size_stride(permute_440, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_441, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_91, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_442, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_443, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_446, (2304, 768), (768, 1))
    assert_size_stride(div_148, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_450, (768, 3072), (3072, 1))
    assert_size_stride(permute_454, (3072, 768), (768, 1))
    assert_size_stride(div_149, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_191, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_150, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_462, (768, 768), (768, 1))
    assert_size_stride(permute_465, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_466, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_94, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_467, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_468, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_471, (2304, 768), (768, 1))
    assert_size_stride(div_159, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_475, (768, 3072), (3072, 1))
    assert_size_stride(permute_479, (3072, 768), (768, 1))
    assert_size_stride(div_160, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_203, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_161, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_487, (768, 768), (768, 1))
    assert_size_stride(permute_490, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_491, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_97, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_492, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_493, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_496, (2304, 768), (768, 1))
    assert_size_stride(div_170, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_500, (768, 3072), (3072, 1))
    assert_size_stride(permute_504, (3072, 768), (768, 1))
    assert_size_stride(div_171, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_215, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_172, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_512, (768, 768), (768, 1))
    assert_size_stride(permute_515, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_516, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_100, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_517, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_518, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_521, (2304, 768), (768, 1))
    assert_size_stride(div_181, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_525, (768, 3072), (3072, 1))
    assert_size_stride(permute_529, (3072, 768), (768, 1))
    assert_size_stride(div_182, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_227, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_183, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_537, (768, 768), (768, 1))
    assert_size_stride(permute_540, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_541, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_103, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_542, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_543, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_546, (2304, 768), (768, 1))
    assert_size_stride(div_192, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_550, (768, 3072), (3072, 1))
    assert_size_stride(permute_554, (3072, 768), (768, 1))
    assert_size_stride(div_193, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_239, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_194, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_562, (768, 768), (768, 1))
    assert_size_stride(permute_565, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_566, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_106, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_567, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_568, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_571, (2304, 768), (768, 1))
    assert_size_stride(div_203, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_575, (768, 3072), (3072, 1))
    assert_size_stride(permute_579, (3072, 768), (768, 1))
    assert_size_stride(div_204, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_251, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_205, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_587, (768, 768), (768, 1))
    assert_size_stride(permute_590, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_591, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_109, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_592, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_593, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_596, (2304, 768), (768, 1))
    assert_size_stride(div_214, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_600, (768, 3072), (3072, 1))
    assert_size_stride(permute_604, (3072, 768), (768, 1))
    assert_size_stride(div_215, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_263, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_216, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_612, (768, 768), (768, 1))
    assert_size_stride(permute_615, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_616, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_112, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_617, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_618, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_621, (2304, 768), (768, 1))
    assert_size_stride(div_225, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_625, (768, 3072), (3072, 1))
    assert_size_stride(permute_629, (3072, 768), (768, 1))
    assert_size_stride(div_226, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_275, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_227, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_637, (768, 768), (768, 1))
    assert_size_stride(permute_640, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_641, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_115, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_642, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_643, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_646, (2304, 768), (768, 1))
    assert_size_stride(div_236, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_650, (768, 3072), (3072, 1))
    assert_size_stride(permute_654, (3072, 768), (768, 1))
    assert_size_stride(div_237, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_287, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_238, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_662, (768, 768), (768, 1))
    assert_size_stride(permute_665, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_666, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_118, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_667, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_668, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_671, (2304, 768), (768, 1))
    assert_size_stride(div_247, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_675, (768, 3072), (3072, 1))
    assert_size_stride(permute_679, (3072, 768), (768, 1))
    assert_size_stride(div_248, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_299, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_249, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_687, (768, 768), (768, 1))
    assert_size_stride(permute_690, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_691, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_121, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_692, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_693, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_696, (2304, 768), (768, 1))
    assert_size_stride(div_258, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_700, (768, 3072), (3072, 1))
    assert_size_stride(permute_704, (3072, 768), (768, 1))
    assert_size_stride(div_259, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_311, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_260, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_712, (768, 768), (768, 1))
    assert_size_stride(permute_715, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_716, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_124, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_717, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_718, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_721, (2304, 768), (768, 1))
    assert_size_stride(div_269, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_725, (768, 3072), (3072, 1))
    assert_size_stride(permute_729, (3072, 768), (768, 1))
    assert_size_stride(div_270, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_323, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_271, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_737, (768, 768), (768, 1))
    assert_size_stride(permute_740, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_741, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_127, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_742, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_743, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_746, (2304, 768), (768, 1))
    assert_size_stride(div_280, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_750, (768, 3072), (3072, 1))
    assert_size_stride(permute_754, (3072, 768), (768, 1))
    assert_size_stride(div_281, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_335, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_282, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_762, (768, 768), (768, 1))
    assert_size_stride(permute_765, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_766, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_130, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_767, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_768, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_771, (2304, 768), (768, 1))
    assert_size_stride(div_291, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_775, (768, 3072), (3072, 1))
    assert_size_stride(permute_779, (3072, 768), (768, 1))
    assert_size_stride(div_292, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_347, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_293, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_787, (768, 768), (768, 1))
    assert_size_stride(permute_790, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_791, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_133, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_792, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_793, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_796, (2304, 768), (768, 1))
    assert_size_stride(div_302, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_800, (768, 3072), (3072, 1))
    assert_size_stride(permute_804, (3072, 768), (768, 1))
    assert_size_stride(div_303, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_359, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_304, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_812, (768, 768), (768, 1))
    assert_size_stride(permute_815, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_816, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_136, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_817, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_818, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_821, (2304, 768), (768, 1))
    assert_size_stride(div_313, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_825, (768, 3072), (3072, 1))
    assert_size_stride(permute_829, (3072, 768), (768, 1))
    assert_size_stride(div_314, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_371, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_315, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_837, (768, 768), (768, 1))
    assert_size_stride(permute_840, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_841, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_139, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_842, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_843, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_846, (2304, 768), (768, 1))
    assert_size_stride(div_324, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_850, (768, 3072), (3072, 1))
    assert_size_stride(permute_854, (3072, 768), (768, 1))
    assert_size_stride(div_325, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_383, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_326, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_862, (768, 768), (768, 1))
    assert_size_stride(permute_865, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_866, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_142, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_867, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_868, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_871, (2304, 768), (768, 1))
    assert_size_stride(div_335, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_875, (768, 3072), (3072, 1))
    assert_size_stride(permute_879, (3072, 768), (768, 1))
    assert_size_stride(div_336, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_395, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(div_337, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_887, (768, 768), (768, 1))
    assert_size_stride(permute_890, (128, 48, 48), (2304, 1, 48))
    assert_size_stride(permute_891, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(alias_145, (8, 16, 48, 48), (36864, 1, 768, 16))
    assert_size_stride(permute_892, (128, 784, 48), (37632, 1, 784))
    assert_size_stride(permute_893, (128, 48, 784), (37632, 1, 48))
    assert_size_stride(permute_896, (2304, 768), (768, 1))
    assert_size_stride(div_346, (8, 784, 1), (784, 1, 1))
    assert_size_stride(unsqueeze_407, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(add_682, (8, 384, 56, 56), (1204224, 1, 21504, 384))
    assert_size_stride(unsqueeze_419, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(add_684, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(unsqueeze_431, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_240, out=buf0)
        del permute_240
        buf1 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_271, out=buf1)
        del clone_271
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_1.run(buf0, primals_625, mul_603, div_78, buf5, 6280, 768, grid=grid(6280), stream=stream0)
        del div_78
        del primals_625
        buf6 = empty_strided((768, 50), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_2.run(buf0, mul_603, buf6, 38400, 126, grid=grid(38400), stream=stream0)
        del mul_603
        buf7 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf6, buf7, 768, 50, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_4.run(buf0, buf8, 768, 6280, grid=grid(768), stream=stream0)
        buf9 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_5.run(buf5, addmm_81, buf9, 768, 8, grid=grid(768), stream=stream0)
        del addmm_81
        buf10 = reinterpret_tensor(buf0, (8, 1, 768), (768, 768, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf5, primals_101, buf10, 6144, grid=grid(6144), stream=stream0)
        del primals_101
        buf11 = empty((8, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (8, 768), (768, 1), 0), permute_244, out=buf11)
        del permute_244
        buf12 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (768, 8), (1, 768), 0), view_462, out=buf12)
        del view_462
        buf13 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf10, buf13, 768, 8, grid=grid(768), stream=stream0)
        buf14 = empty((1, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_475, x_476], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_8.run(buf11, mm_25, primals_622, buf14, 3072, 8, grid=grid(3072), stream=stream0)
        buf15 = reinterpret_tensor(buf11, (8, 1, 3072), (3072, 3072, 1), 0); del buf11  # reuse
        # Source Nodes: [x_475, x_476], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_9.run(buf15, mm_25, primals_622, 24576, grid=grid(24576), stream=stream0)
        del mm_25
        del primals_622
        buf16 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (3072, 8), (1, 3072), 0), view_460, out=buf16)
        del view_460
        buf17 = reinterpret_tensor(buf10, (8, 768), (768, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (8, 3072), (3072, 1), 0), permute_250, out=buf17)
        del permute_250
        buf20 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        buf27 = empty((8, 785, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_select_backward_slice_backward_10.run(buf5, buf17, primals_619, mul_597, div_79, primals_100, buf20, buf27, 6280, 768, grid=grid(6280), stream=stream0)
        del div_79
        del primals_619
        buf21 = buf6; del buf6  # reuse
        buf23 = empty_strided((768, 50), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_backward_select_backward_slice_backward_11.run(buf5, buf17, mul_597, buf21, buf23, 38400, 126, grid=grid(38400), stream=stream0)
        del mul_597
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf21, buf22, 768, 50, grid=grid(768), stream=stream0)
        buf24 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf23, buf24, 768, 50, grid=grid(768), stream=stream0)
        buf25 = reinterpret_tensor(buf23, (1, 1, 768, 50), (38400, 38400, 1, 768), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf20, cat_6, buf25, 38400, 126, grid=grid(38400), stream=stream0)
        del cat_6
        buf26 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf25, buf26, 768, 50, grid=grid(768), stream=stream0)
        buf28 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (8, 768), (602880, 1), 0), permute_252, out=buf28)
        del permute_252
        buf29 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (768, 8), (1, 602880), 0), view_458, out=buf29)
        del view_458
        buf30 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf27, buf30, 768, 8, grid=grid(768), stream=stream0)
        buf31 = empty((8, 16, 1, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_14.run(permute_230, buf31, 128, 48, grid=grid(128, 48), stream=stream0)
        del permute_230
        buf32 = reinterpret_tensor(buf27, (8, 16, 785, 48), (602880, 37680, 48, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_15.run(permute_232, buf32, 128, 37680, grid=grid(128, 37680), stream=stream0)
        del permute_232
        buf33 = reinterpret_tensor(buf5, (8, 16, 785, 48), (602880, 37680, 48, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_15.run(permute_234, buf33, 128, 37680, grid=grid(128, 37680), stream=stream0)
        del permute_234
        buf34 = empty((8, 16, 1, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_14.run(alias_74, buf34, 128, 48, grid=grid(128, 48), stream=stream0)
        del alias_74
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf35 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf28, (8, 16, 1, 48), (768, 48, 768, 1), 0), buf31, buf32, buf33, None, buf34, getitem_281, getitem_282, getitem_283, 0.0, [True, True, True, False])
        del buf28
        del buf32
        del getitem_281
        del getitem_282
        del getitem_283
        buf36 = buf35[0]
        buf37 = buf35[1]
        buf38 = buf35[2]
        del buf35
        buf39 = reinterpret_tensor(buf33, (6280, 768), (768, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (6280, 768), (768, 1), 0), permute_258, out=buf39)
        del permute_258
        buf40 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (768, 6280), (1, 768), 0), view_451, out=buf40)
        buf41 = reinterpret_tensor(buf25, (1, 768, 50), (38400, 1, 768), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf38, buf41, 38400, 126, grid=grid(38400), stream=stream0)
        buf42 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf41, buf42, 768, 50, grid=grid(768), stream=stream0)
        buf43 = reinterpret_tensor(buf38, (6280, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (6280, 768), (768, 1), 0), permute_263, out=buf43)
        del permute_263
        buf44 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (768, 6280), (1, 768), 0), view_451, out=buf44)
        del view_451
        buf45 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf37, buf45, 38400, 126, grid=grid(38400), stream=stream0)
        buf46 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf45, buf46, 768, 50, grid=grid(768), stream=stream0)
        buf47 = reinterpret_tensor(buf34, (8, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (8, 768), (768, 1), 0), permute_268, out=buf47)
        del permute_268
        buf48 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (768, 8), (1, 768), 0), select_1, out=buf48)
        del select_1
        buf49 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf36, buf49, 768, 8, grid=grid(768), stream=stream0)
        buf50 = reinterpret_tensor(buf39, (8, 785, 768), (602880, 768, 1), 0); del buf39  # reuse
        buf57 = reinterpret_tensor(buf37, (8, 785, 768), (602880, 768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_slice_backward_17.run(buf50, buf20, primals_100, buf43, buf47, primals_609, mul_594, div_80, buf57, 6280, 768, grid=grid(6280), stream=stream0)
        del buf20
        del div_80
        del primals_100
        del primals_609
        buf53 = reinterpret_tensor(buf45, (768, 50), (1, 768), 0); del buf45  # reuse
        buf55 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_18.run(buf50, mul_594, buf53, buf55, 38400, 126, grid=grid(38400), stream=stream0)
        del mul_594
        buf54 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf53, buf54, 768, 50, grid=grid(768), stream=stream0)
        buf56 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf55, buf56, 768, 50, grid=grid(768), stream=stream0)
        buf58 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_5.run(buf57, addmm_76, buf58, 768, 8, grid=grid(768), stream=stream0)
        del addmm_76
        buf59 = reinterpret_tensor(buf47, (8, 1, 768), (768, 768, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf57, primals_99, buf59, 6144, grid=grid(6144), stream=stream0)
        del primals_99
        buf60 = reinterpret_tensor(buf15, (8, 3072), (3072, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (8, 768), (768, 1), 0), permute_272, out=buf60)
        del permute_272
        buf61 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (768, 8), (1, 768), 0), view_448, out=buf61)
        del view_448
        buf62 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf59, buf62, 768, 8, grid=grid(768), stream=stream0)
        buf63 = empty((1, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_464, x_465], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_8.run(buf60, mm_24, primals_606, buf63, 3072, 8, grid=grid(3072), stream=stream0)
        buf64 = reinterpret_tensor(buf60, (8, 1, 3072), (3072, 3072, 1), 0); del buf60  # reuse
        # Source Nodes: [x_464, x_465], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_9.run(buf64, mm_24, primals_606, 24576, grid=grid(24576), stream=stream0)
        del mm_24
        del primals_606
        buf65 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (3072, 8), (1, 3072), 0), view_446, out=buf65)
        del view_446
        buf66 = reinterpret_tensor(buf59, (8, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (8, 3072), (3072, 1), 0), permute_278, out=buf66)
        del buf64
        del permute_278
        buf69 = buf50; del buf50  # reuse
        buf76 = reinterpret_tensor(buf43, (8, 785, 768), (602880, 768, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_select_backward_slice_backward_10.run(buf57, buf66, primals_603, mul_588, div_81, primals_98, buf69, buf76, 6280, 768, grid=grid(6280), stream=stream0)
        del div_81
        del primals_603
        buf70 = buf55; del buf55  # reuse
        buf72 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_backward_select_backward_slice_backward_11.run(buf57, buf66, mul_588, buf70, buf72, 38400, 126, grid=grid(38400), stream=stream0)
        del mul_588
        buf71 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf70, buf71, 768, 50, grid=grid(768), stream=stream0)
        buf73 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf72, buf73, 768, 50, grid=grid(768), stream=stream0)
        buf74 = reinterpret_tensor(buf72, (1, 1, 768, 50), (38400, 38400, 1, 768), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf69, cat_4, buf74, 38400, 126, grid=grid(38400), stream=stream0)
        del cat_4
        buf75 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf74, buf75, 768, 50, grid=grid(768), stream=stream0)
        buf77 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (8, 768), (602880, 1), 0), permute_280, out=buf77)
        del permute_280
        buf78 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (768, 8), (1, 602880), 0), view_444, out=buf78)
        del view_444
        buf79 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf76, buf79, 768, 8, grid=grid(768), stream=stream0)
        buf80 = reinterpret_tensor(buf36, (8, 16, 1, 48), (768, 48, 48, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_14.run(permute_220, buf80, 128, 48, grid=grid(128, 48), stream=stream0)
        del permute_220
        buf81 = reinterpret_tensor(buf76, (8, 16, 785, 48), (602880, 37680, 48, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_15.run(permute_222, buf81, 128, 37680, grid=grid(128, 37680), stream=stream0)
        del permute_222
        buf82 = reinterpret_tensor(buf57, (8, 16, 785, 48), (602880, 37680, 48, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_15.run(permute_224, buf82, 128, 37680, grid=grid(128, 37680), stream=stream0)
        del permute_224
        buf83 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_14.run(alias_75, buf83, 128, 48, grid=grid(128, 48), stream=stream0)
        del alias_75
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf84 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf77, (8, 16, 1, 48), (768, 48, 768, 1), 0), buf80, buf81, buf82, None, buf83, getitem_273, getitem_274, getitem_275, 0.0, [True, True, True, False])
        del buf77
        del buf80
        del buf81
        del getitem_273
        del getitem_274
        del getitem_275
        buf85 = buf84[0]
        buf86 = buf84[1]
        buf87 = buf84[2]
        del buf84
        buf88 = reinterpret_tensor(buf82, (6280, 768), (768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (6280, 768), (768, 1), 0), permute_286, out=buf88)
        del permute_286
        buf89 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (768, 6280), (1, 768), 0), view_437, out=buf89)
        buf90 = reinterpret_tensor(buf74, (1, 768, 50), (38400, 1, 768), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf87, buf90, 38400, 126, grid=grid(38400), stream=stream0)
        buf91 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf90, buf91, 768, 50, grid=grid(768), stream=stream0)
        buf92 = reinterpret_tensor(buf87, (6280, 768), (768, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (6280, 768), (768, 1), 0), permute_291, out=buf92)
        del permute_291
        buf93 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (768, 6280), (1, 768), 0), view_437, out=buf93)
        del view_437
        buf94 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf86, buf94, 38400, 126, grid=grid(38400), stream=stream0)
        buf95 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf94, buf95, 768, 50, grid=grid(768), stream=stream0)
        buf96 = reinterpret_tensor(buf83, (8, 768), (768, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (8, 768), (768, 1), 0), permute_296, out=buf96)
        del permute_296
        buf97 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (768, 8), (1, 768), 0), select, out=buf97)
        del select
        buf98 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf85, buf98, 768, 8, grid=grid(768), stream=stream0)
        buf99 = reinterpret_tensor(buf88, (8, 785, 768), (602880, 768, 1), 0); del buf88  # reuse
        buf106 = reinterpret_tensor(buf86, (8, 785, 768), (602880, 768, 1), 0); del buf86  # reuse
        # Source Nodes: [x_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_19.run(buf99, buf69, primals_98, buf92, buf96, primals_593, cat_3, getitem_271, rsqrt_99, buf106, 6280, 768, grid=grid(6280), stream=stream0)
        del buf69
        del buf92
        del primals_593
        del primals_98
        buf102 = reinterpret_tensor(buf94, (768, 50), (1, 768), 0); del buf94  # reuse
        buf104 = buf70; del buf70  # reuse
        # Source Nodes: [x_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_20.run(buf99, cat_3, getitem_271, rsqrt_99, buf102, buf104, 38400, 126, grid=grid(38400), stream=stream0)
        del buf99
        del cat_3
        del getitem_271
        del rsqrt_99
        buf103 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf102, buf103, 768, 50, grid=grid(768), stream=stream0)
        del buf102
        buf105 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf104, buf105, 768, 50, grid=grid(768), stream=stream0)
        del buf104
        buf107 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf106, buf107, 768, 8, grid=grid(768), stream=stream0)
        buf108 = empty_strided((1, 1, 768, 49), (37632, 37632, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_21.run(buf106, addmm_71, buf108, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_71
        buf109 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf108, buf109, 768, 49, grid=grid(768), stream=stream0)
        buf110 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_23.run(buf106, primals_96, buf110, 4816896, grid=grid(4816896), stream=stream0)
        del primals_96
        buf111 = empty((6272, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (6272, 768), (768, 1), 0), permute_300, out=buf111)
        del permute_300
        buf112 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (768, 6272), (1, 768), 0), view_434, out=buf112)
        del view_434
        buf113 = reinterpret_tensor(buf108, (1, 768, 49), (37632, 1, 768), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf110, buf113, 37632, 128, grid=grid(37632), stream=stream0)
        buf114 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf113, buf114, 768, 49, grid=grid(768), stream=stream0)
        buf115 = reinterpret_tensor(buf111, (8, 784, 3072), (2408448, 3072, 1), 0); del buf111  # reuse
        # Source Nodes: [x_454], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf115, addmm_70, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_70
        buf116 = reinterpret_tensor(buf110, (6272, 768), (768, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (6272, 3072), (3072, 1), 0), permute_304, out=buf116)
        del permute_304
        buf117 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (3072, 6272), (1, 3072), 0), view_432, out=buf117)
        del view_432
        buf118 = empty_strided((1, 3072, 49), (150528, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf115, buf118, 150528, 128, grid=grid(150528), stream=stream0)
        buf119 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf118, buf119, 3072, 49, grid=grid(3072), stream=stream0)
        buf126 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        buf129 = empty((8, 784, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_28.run(buf116, primals_587, mul_579, buf106, div_83, primals_95, buf126, buf129, 6272, 768, grid=grid(6272), stream=stream0)
        del buf106
        del div_83
        del primals_587
        del primals_95
        buf122 = reinterpret_tensor(buf113, (768, 49), (1, 768), 0); del buf113  # reuse
        buf124 = empty_strided((768, 49), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf116, mul_579, buf122, buf124, 37632, 128, grid=grid(37632), stream=stream0)
        del buf116
        del mul_579
        buf123 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf122, buf123, 768, 49, grid=grid(768), stream=stream0)
        buf125 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf124, buf125, 768, 49, grid=grid(768), stream=stream0)
        buf127 = reinterpret_tensor(buf124, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf126, convolution_51, buf127, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_51
        buf128 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf127, buf128, 768, 49, grid=grid(768), stream=stream0)
        buf130 = reinterpret_tensor(buf127, (768, 49), (1, 768), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf129, buf130, 37632, 128, grid=grid(37632), stream=stream0)
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf130, buf131, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf132 = aten.convolution_backward(reinterpret_tensor(buf129, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_425, primals_585, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_425
        del primals_585
        buf133 = buf132[0]
        buf134 = buf132[1]
        del buf132
        buf135 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf133, buf135, 768, 6272, grid=grid(768), stream=stream0)
        buf136 = reinterpret_tensor(buf130, (768, 49), (49, 1), 0); del buf130  # reuse
        # Source Nodes: [x_448], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf133, convolution_50, unsqueeze_119, buf136, 37632, 128, grid=grid(37632), stream=stream0)
        buf137 = empty((768, ), device='cuda', dtype=torch.float32)
        buf138 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_448], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf136, squeeze_79, buf137, buf138, 768, 49, grid=grid(768), stream=stream0)
        buf139 = buf133; del buf133  # reuse
        # Source Nodes: [x_448], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf139, convolution_50, unsqueeze_119, buf137, squeeze_79, buf135, primals_583, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_50
        del primals_583
        del squeeze_79
        del unsqueeze_119
        buf140 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf139, buf140, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf141 = aten.convolution_backward(buf139, view_430, primals_581, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_581
        del view_430
        buf142 = buf141[0]
        buf143 = buf141[1]
        del buf141
        buf144 = reinterpret_tensor(buf136, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf142, primals_579, buf144, 37632, 128, grid=grid(37632), stream=stream0)
        buf145 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf144, buf145, 6272, 6, grid=grid(6272), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf142, primals_579, mul_566, buf146, 37632, 128, grid=grid(37632), stream=stream0)
        buf147 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf146, buf147, 6272, 6, grid=grid(6272), stream=stream0)
        buf148 = reinterpret_tensor(buf146, (768, 49), (49, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf142, mul_566, buf148, 37632, 128, grid=grid(37632), stream=stream0)
        buf149 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf148, buf149, 768, 49, grid=grid(768), stream=stream0)
        buf150 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf142, buf150, 768, 6272, grid=grid(768), stream=stream0)
        buf151 = buf126; del buf126  # reuse
        buf156 = reinterpret_tensor(buf139, (8, 784, 768), (602112, 768, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf151, div_84, buf142, primals_579, buf145, mul_566, buf147, primals_93, buf156, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_84
        del mul_566
        del primals_579
        buf152 = reinterpret_tensor(buf148, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf148  # reuse
        buf154 = reinterpret_tensor(buf122, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf122  # reuse
        # Source Nodes: [x_443], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf151, mm_23, primals_578, primals_93, buf152, buf154, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_23
        del primals_578
        del primals_93
        buf153 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_443], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf152, buf153, 768, 49, grid=grid(768), stream=stream0)
        buf155 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf154, buf155, 768, 49, grid=grid(768), stream=stream0)
        buf157 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (768, 6272), (1, 768), 0), view_428, out=buf157)
        del view_428
        buf158 = reinterpret_tensor(buf142, (6272, 768), (768, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (6272, 768), (768, 1), 0), permute_312, out=buf158)
        del permute_312
        buf159 = reinterpret_tensor(buf156, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf158, buf159, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (128, 48, 784), (37632, 784, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_315, reinterpret_tensor(buf159, (128, 48, 784), (37632, 784, 1), 0), out=buf160)
        del permute_315
        buf161 = empty((128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf159, (128, 48, 784), (37632, 784, 1), 0), permute_316, out=buf161)
        del permute_316
        buf162 = reinterpret_tensor(buf96, (8, 16, 48, 1), (768, 48, 1, 6144), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf161, alias_76, buf162, 6144, 48, grid=grid(6144), stream=stream0)
        buf163 = empty_strided((1, 16, 1, 1, 3), (48, 1, 48, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf161, alias_76, buf162, bmm_46, buf163, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_46
        buf164 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf163, buf164, 16, 3, grid=grid(16), stream=stream0)
        buf165 = reinterpret_tensor(buf161, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf165, alias_76, buf162, primals_94, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_76
        del primals_94
        buf166 = reinterpret_tensor(buf159, (128, 784, 48), (37632, 48, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_317, reinterpret_tensor(buf165, (128, 48, 48), (2304, 48, 1), 0), out=buf166)
        del permute_317
        buf167 = reinterpret_tensor(buf129, (128, 48, 784), (37632, 784, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf165, (128, 48, 48), (2304, 48, 1), 0), permute_318, out=buf167)
        del permute_318
        buf168 = empty_strided((8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf166, getitem_262, pow_97, buf168, 43008, 112, grid=grid(43008), stream=stream0)
        buf169 = buf162; del buf162  # reuse
        # Source Nodes: [k_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf168, buf169, 6144, 7, grid=grid(6144), stream=stream0)
        buf170 = reinterpret_tensor(buf168, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf168  # reuse
        # Source Nodes: [q_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf167, getitem_261, pow_95, buf170, 43008, 112, grid=grid(43008), stream=stream0)
        buf171 = reinterpret_tensor(buf85, (8, 16, 48, 1), (768, 48, 1, 6144), 0); del buf85  # reuse
        # Source Nodes: [q_47], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf170, buf171, 6144, 7, grid=grid(6144), stream=stream0)
        buf172 = empty((24, 16, 48, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf167, pow_95, buf171, getitem_261, buf166, pow_97, buf169, getitem_262, buf160, buf172, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf160
        del getitem_261
        del getitem_262
        del pow_95
        del pow_97
        buf173 = empty((6272, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf172, buf173, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf174 = reinterpret_tensor(buf167, (6272, 768), (768, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf173, permute_321, out=buf174)
        del permute_321
        buf175 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (2304, 6272), (1, 2304), 0), view_418, out=buf175)
        del view_418
        buf176 = empty_strided((1, 2304, 49), (112896, 1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf173, buf176, 112896, 128, grid=grid(112896), stream=stream0)
        buf177 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf176, buf177, 2304, 49, grid=grid(2304), stream=stream0)
        buf184 = buf151; del buf151  # reuse
        buf187 = reinterpret_tensor(buf166, (8, 784, 768), (602112, 768, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf184, buf174, primals_573, mul_562, div_93, primals_92, buf187, 6272, 768, grid=grid(6272), stream=stream0)
        del div_93
        del primals_573
        del primals_92
        buf180 = reinterpret_tensor(buf154, (768, 49), (1, 768), 0); del buf154  # reuse
        buf182 = reinterpret_tensor(buf152, (768, 49), (1, 768), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf174, mul_562, buf180, buf182, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_562
        buf181 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf180, buf181, 768, 49, grid=grid(768), stream=stream0)
        buf183 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf182, buf183, 768, 49, grid=grid(768), stream=stream0)
        buf185 = reinterpret_tensor(buf182, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf184, addmm_68, buf185, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_68
        buf186 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf185, buf186, 768, 49, grid=grid(768), stream=stream0)
        buf188 = reinterpret_tensor(buf115, (6272, 3072), (3072, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (6272, 768), (768, 1), 0), permute_325, out=buf188)
        del permute_325
        buf189 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (768, 6272), (1, 768), 0), view_416, out=buf189)
        del view_416
        buf190 = reinterpret_tensor(buf185, (1, 768, 49), (37632, 1, 768), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf187, buf190, 37632, 128, grid=grid(37632), stream=stream0)
        buf191 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf190, buf191, 768, 49, grid=grid(768), stream=stream0)
        buf192 = reinterpret_tensor(buf188, (8, 784, 3072), (2408448, 3072, 1), 0); del buf188  # reuse
        # Source Nodes: [x_435], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf192, addmm_67, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_67
        buf193 = reinterpret_tensor(buf187, (6272, 768), (768, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (6272, 3072), (3072, 1), 0), permute_329, out=buf193)
        del permute_329
        buf194 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (3072, 6272), (1, 3072), 0), view_414, out=buf194)
        del view_414
        buf195 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf192, buf195, 150528, 128, grid=grid(150528), stream=stream0)
        buf196 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf195, buf196, 3072, 49, grid=grid(3072), stream=stream0)
        buf203 = buf184; del buf184  # reuse
        buf206 = reinterpret_tensor(buf174, (8, 784, 768), (602112, 768, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf203, buf193, primals_567, mul_556, div_94, primals_91, buf206, 6272, 768, grid=grid(6272), stream=stream0)
        del div_94
        del primals_567
        del primals_91
        buf199 = reinterpret_tensor(buf190, (768, 49), (1, 768), 0); del buf190  # reuse
        buf201 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf193, mul_556, buf199, buf201, 37632, 128, grid=grid(37632), stream=stream0)
        del buf193
        del mul_556
        buf200 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf199, buf200, 768, 49, grid=grid(768), stream=stream0)
        buf202 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf201, buf202, 768, 49, grid=grid(768), stream=stream0)
        buf204 = reinterpret_tensor(buf201, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf203, convolution_49, buf204, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_49
        buf205 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf204, buf205, 768, 49, grid=grid(768), stream=stream0)
        buf207 = reinterpret_tensor(buf204, (768, 49), (1, 768), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf206, buf207, 37632, 128, grid=grid(37632), stream=stream0)
        buf208 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf207, buf208, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf209 = aten.convolution_backward(reinterpret_tensor(buf206, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_408, primals_565, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_408
        del primals_565
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf212 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf210, buf212, 768, 6272, grid=grid(768), stream=stream0)
        buf213 = reinterpret_tensor(buf207, (768, 49), (49, 1), 0); del buf207  # reuse
        # Source Nodes: [x_429], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf210, convolution_48, unsqueeze_131, buf213, 37632, 128, grid=grid(37632), stream=stream0)
        buf214 = empty((768, ), device='cuda', dtype=torch.float32)
        buf215 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_429], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf213, squeeze_76, buf214, buf215, 768, 49, grid=grid(768), stream=stream0)
        buf216 = buf210; del buf210  # reuse
        # Source Nodes: [x_429], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf216, convolution_48, unsqueeze_131, buf214, squeeze_76, buf212, primals_563, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_48
        del primals_563
        del squeeze_76
        del unsqueeze_131
        buf217 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf216, buf217, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf218 = aten.convolution_backward(buf216, view_412, primals_561, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_561
        del view_412
        buf219 = buf218[0]
        buf220 = buf218[1]
        del buf218
        buf221 = reinterpret_tensor(buf213, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf219, primals_559, buf221, 37632, 128, grid=grid(37632), stream=stream0)
        buf222 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf221, buf222, 6272, 6, grid=grid(6272), stream=stream0)
        buf223 = reinterpret_tensor(buf221, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf219, primals_559, mul_543, buf223, 37632, 128, grid=grid(37632), stream=stream0)
        buf224 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf223, buf224, 6272, 6, grid=grid(6272), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (768, 49), (49, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf219, mul_543, buf225, 37632, 128, grid=grid(37632), stream=stream0)
        buf226 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf225, buf226, 768, 49, grid=grid(768), stream=stream0)
        buf227 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf219, buf227, 768, 6272, grid=grid(768), stream=stream0)
        buf228 = buf203; del buf203  # reuse
        buf233 = reinterpret_tensor(buf216, (8, 784, 768), (602112, 768, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf228, div_95, buf219, primals_559, buf222, mul_543, buf224, primals_89, buf233, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_95
        del mul_543
        del primals_559
        buf229 = reinterpret_tensor(buf225, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf225  # reuse
        buf231 = reinterpret_tensor(buf199, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf199  # reuse
        # Source Nodes: [x_424], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf228, mm_22, primals_558, primals_89, buf229, buf231, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_22
        del primals_558
        del primals_89
        buf230 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_424], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf229, buf230, 768, 49, grid=grid(768), stream=stream0)
        buf232 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf231, buf232, 768, 49, grid=grid(768), stream=stream0)
        buf234 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (768, 6272), (1, 768), 0), view_410, out=buf234)
        del view_410
        buf235 = reinterpret_tensor(buf219, (6272, 768), (768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (6272, 768), (768, 1), 0), permute_337, out=buf235)
        del permute_337
        buf236 = reinterpret_tensor(buf233, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf235, buf236, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf237 = reinterpret_tensor(buf235, (128, 48, 784), (37632, 784, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_340, reinterpret_tensor(buf236, (128, 48, 784), (37632, 784, 1), 0), out=buf237)
        del permute_340
        buf238 = reinterpret_tensor(buf165, (128, 48, 48), (2304, 48, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (128, 48, 784), (37632, 784, 1), 0), permute_341, out=buf238)
        del permute_341
        buf239 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf238, alias_79, buf239, 6144, 48, grid=grid(6144), stream=stream0)
        buf240 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf238, alias_79, buf239, bmm_44, buf240, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_44
        buf241 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf240, buf241, 16, 3, grid=grid(16), stream=stream0)
        buf242 = reinterpret_tensor(buf238, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf242, alias_79, buf239, primals_90, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_79
        del primals_90
        buf243 = reinterpret_tensor(buf236, (128, 784, 48), (37632, 48, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_342, reinterpret_tensor(buf242, (128, 48, 48), (2304, 48, 1), 0), out=buf243)
        del permute_342
        buf244 = reinterpret_tensor(buf206, (128, 48, 784), (37632, 784, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (128, 48, 48), (2304, 48, 1), 0), permute_343, out=buf244)
        del permute_343
        buf245 = reinterpret_tensor(buf170, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf170  # reuse
        # Source Nodes: [k_45], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf243, getitem_251, pow_93, buf245, 43008, 112, grid=grid(43008), stream=stream0)
        buf246 = buf239; del buf239  # reuse
        # Source Nodes: [k_45], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf245, buf246, 6144, 7, grid=grid(6144), stream=stream0)
        buf247 = reinterpret_tensor(buf245, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf245  # reuse
        # Source Nodes: [q_45], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf244, getitem_250, pow_91, buf247, 43008, 112, grid=grid(43008), stream=stream0)
        buf248 = buf169; del buf169  # reuse
        # Source Nodes: [q_45], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf247, buf248, 6144, 7, grid=grid(6144), stream=stream0)
        buf249 = reinterpret_tensor(buf173, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf244, pow_91, buf248, getitem_250, buf243, pow_93, buf246, getitem_251, buf237, buf249, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf237
        del getitem_250
        del getitem_251
        del pow_91
        del pow_93
        buf250 = reinterpret_tensor(buf172, (6272, 2304), (2304, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf249, buf250, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf251 = reinterpret_tensor(buf244, (6272, 768), (768, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf250, permute_346, out=buf251)
        del permute_346
        buf252 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (2304, 6272), (1, 2304), 0), view_400, out=buf252)
        del view_400
        buf253 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf250, buf253, 112896, 128, grid=grid(112896), stream=stream0)
        buf254 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf253, buf254, 2304, 49, grid=grid(2304), stream=stream0)
        buf261 = buf228; del buf228  # reuse
        buf264 = reinterpret_tensor(buf243, (8, 784, 768), (602112, 768, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf261, buf251, primals_553, mul_539, div_104, primals_88, buf264, 6272, 768, grid=grid(6272), stream=stream0)
        del div_104
        del primals_553
        del primals_88
        buf257 = reinterpret_tensor(buf231, (768, 49), (1, 768), 0); del buf231  # reuse
        buf259 = reinterpret_tensor(buf229, (768, 49), (1, 768), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf251, mul_539, buf257, buf259, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_539
        buf258 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf257, buf258, 768, 49, grid=grid(768), stream=stream0)
        buf260 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf259, buf260, 768, 49, grid=grid(768), stream=stream0)
        buf262 = reinterpret_tensor(buf259, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf261, addmm_65, buf262, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_65
        buf263 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf262, buf263, 768, 49, grid=grid(768), stream=stream0)
        buf265 = reinterpret_tensor(buf192, (6272, 3072), (3072, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (6272, 768), (768, 1), 0), permute_350, out=buf265)
        del permute_350
        buf266 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (768, 6272), (1, 768), 0), view_398, out=buf266)
        del view_398
        buf267 = reinterpret_tensor(buf262, (1, 768, 49), (37632, 1, 768), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf264, buf267, 37632, 128, grid=grid(37632), stream=stream0)
        buf268 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf267, buf268, 768, 49, grid=grid(768), stream=stream0)
        buf269 = reinterpret_tensor(buf265, (8, 784, 3072), (2408448, 3072, 1), 0); del buf265  # reuse
        # Source Nodes: [x_416], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf269, addmm_64, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_64
        buf270 = reinterpret_tensor(buf264, (6272, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (6272, 3072), (3072, 1), 0), permute_354, out=buf270)
        del permute_354
        buf271 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (3072, 6272), (1, 3072), 0), view_396, out=buf271)
        del view_396
        buf272 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf269, buf272, 150528, 128, grid=grid(150528), stream=stream0)
        buf273 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf272, buf273, 3072, 49, grid=grid(3072), stream=stream0)
        buf280 = buf261; del buf261  # reuse
        buf283 = reinterpret_tensor(buf251, (8, 784, 768), (602112, 768, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf280, buf270, primals_547, mul_533, div_105, primals_87, buf283, 6272, 768, grid=grid(6272), stream=stream0)
        del div_105
        del primals_547
        del primals_87
        buf276 = reinterpret_tensor(buf267, (768, 49), (1, 768), 0); del buf267  # reuse
        buf278 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf270, mul_533, buf276, buf278, 37632, 128, grid=grid(37632), stream=stream0)
        del buf270
        del mul_533
        buf277 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf276, buf277, 768, 49, grid=grid(768), stream=stream0)
        buf279 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf278, buf279, 768, 49, grid=grid(768), stream=stream0)
        buf281 = reinterpret_tensor(buf278, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf280, convolution_47, buf281, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_47
        buf282 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf281, buf282, 768, 49, grid=grid(768), stream=stream0)
        buf284 = reinterpret_tensor(buf281, (768, 49), (1, 768), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf283, buf284, 37632, 128, grid=grid(37632), stream=stream0)
        buf285 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf284, buf285, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf286 = aten.convolution_backward(reinterpret_tensor(buf283, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_391, primals_545, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_391
        del primals_545
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        buf289 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf287, buf289, 768, 6272, grid=grid(768), stream=stream0)
        buf290 = reinterpret_tensor(buf284, (768, 49), (49, 1), 0); del buf284  # reuse
        # Source Nodes: [x_410], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf287, convolution_46, unsqueeze_143, buf290, 37632, 128, grid=grid(37632), stream=stream0)
        buf291 = empty((768, ), device='cuda', dtype=torch.float32)
        buf292 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_410], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf290, squeeze_73, buf291, buf292, 768, 49, grid=grid(768), stream=stream0)
        buf293 = buf287; del buf287  # reuse
        # Source Nodes: [x_410], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf293, convolution_46, unsqueeze_143, buf291, squeeze_73, buf289, primals_543, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_46
        del primals_543
        del squeeze_73
        del unsqueeze_143
        buf294 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf293, buf294, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf295 = aten.convolution_backward(buf293, view_394, primals_541, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_541
        del view_394
        buf296 = buf295[0]
        buf297 = buf295[1]
        del buf295
        buf298 = reinterpret_tensor(buf290, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf296, primals_539, buf298, 37632, 128, grid=grid(37632), stream=stream0)
        buf299 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf298, buf299, 6272, 6, grid=grid(6272), stream=stream0)
        buf300 = reinterpret_tensor(buf298, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf296, primals_539, mul_520, buf300, 37632, 128, grid=grid(37632), stream=stream0)
        buf301 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf300, buf301, 6272, 6, grid=grid(6272), stream=stream0)
        buf302 = reinterpret_tensor(buf300, (768, 49), (49, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf296, mul_520, buf302, 37632, 128, grid=grid(37632), stream=stream0)
        buf303 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf302, buf303, 768, 49, grid=grid(768), stream=stream0)
        buf304 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf296, buf304, 768, 6272, grid=grid(768), stream=stream0)
        buf305 = buf280; del buf280  # reuse
        buf310 = reinterpret_tensor(buf293, (8, 784, 768), (602112, 768, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf305, div_106, buf296, primals_539, buf299, mul_520, buf301, primals_85, buf310, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_106
        del mul_520
        del primals_539
        buf306 = reinterpret_tensor(buf302, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf302  # reuse
        buf308 = reinterpret_tensor(buf276, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf276  # reuse
        # Source Nodes: [x_405], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf305, mm_21, primals_538, primals_85, buf306, buf308, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_21
        del primals_538
        del primals_85
        buf307 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_405], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf306, buf307, 768, 49, grid=grid(768), stream=stream0)
        buf309 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf308, buf309, 768, 49, grid=grid(768), stream=stream0)
        buf311 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (768, 6272), (1, 768), 0), view_392, out=buf311)
        del view_392
        buf312 = reinterpret_tensor(buf296, (6272, 768), (768, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (6272, 768), (768, 1), 0), permute_362, out=buf312)
        del permute_362
        buf313 = reinterpret_tensor(buf310, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf312, buf313, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf314 = reinterpret_tensor(buf312, (128, 48, 784), (37632, 784, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_365, reinterpret_tensor(buf313, (128, 48, 784), (37632, 784, 1), 0), out=buf314)
        del permute_365
        buf315 = reinterpret_tensor(buf242, (128, 48, 48), (2304, 48, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf313, (128, 48, 784), (37632, 784, 1), 0), permute_366, out=buf315)
        del permute_366
        buf316 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf315, alias_82, buf316, 6144, 48, grid=grid(6144), stream=stream0)
        buf317 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf315, alias_82, buf316, bmm_42, buf317, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_42
        buf318 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf317, buf318, 16, 3, grid=grid(16), stream=stream0)
        buf319 = reinterpret_tensor(buf315, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf319, alias_82, buf316, primals_86, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_82
        del primals_86
        buf320 = reinterpret_tensor(buf313, (128, 784, 48), (37632, 48, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_367, reinterpret_tensor(buf319, (128, 48, 48), (2304, 48, 1), 0), out=buf320)
        del permute_367
        buf321 = reinterpret_tensor(buf283, (128, 48, 784), (37632, 784, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (128, 48, 48), (2304, 48, 1), 0), permute_368, out=buf321)
        del permute_368
        buf322 = reinterpret_tensor(buf247, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf247  # reuse
        # Source Nodes: [k_43], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf320, getitem_240, pow_89, buf322, 43008, 112, grid=grid(43008), stream=stream0)
        buf323 = buf316; del buf316  # reuse
        # Source Nodes: [k_43], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf322, buf323, 6144, 7, grid=grid(6144), stream=stream0)
        buf324 = reinterpret_tensor(buf322, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf322  # reuse
        # Source Nodes: [q_43], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf321, getitem_239, pow_87, buf324, 43008, 112, grid=grid(43008), stream=stream0)
        buf325 = buf246; del buf246  # reuse
        # Source Nodes: [q_43], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf324, buf325, 6144, 7, grid=grid(6144), stream=stream0)
        buf326 = reinterpret_tensor(buf250, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf321, pow_87, buf325, getitem_239, buf320, pow_89, buf323, getitem_240, buf314, buf326, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf314
        del getitem_239
        del getitem_240
        del pow_87
        del pow_89
        buf327 = reinterpret_tensor(buf249, (6272, 2304), (2304, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf326, buf327, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf328 = reinterpret_tensor(buf321, (6272, 768), (768, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf327, permute_371, out=buf328)
        del permute_371
        buf329 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (2304, 6272), (1, 2304), 0), view_382, out=buf329)
        del view_382
        buf330 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf327, buf330, 112896, 128, grid=grid(112896), stream=stream0)
        buf331 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf330, buf331, 2304, 49, grid=grid(2304), stream=stream0)
        buf338 = buf305; del buf305  # reuse
        buf341 = reinterpret_tensor(buf320, (8, 784, 768), (602112, 768, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf338, buf328, primals_533, mul_516, div_115, primals_84, buf341, 6272, 768, grid=grid(6272), stream=stream0)
        del div_115
        del primals_533
        del primals_84
        buf334 = reinterpret_tensor(buf308, (768, 49), (1, 768), 0); del buf308  # reuse
        buf336 = reinterpret_tensor(buf306, (768, 49), (1, 768), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf328, mul_516, buf334, buf336, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_516
        buf335 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf334, buf335, 768, 49, grid=grid(768), stream=stream0)
        buf337 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf336, buf337, 768, 49, grid=grid(768), stream=stream0)
        buf339 = reinterpret_tensor(buf336, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf338, addmm_62, buf339, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_62
        buf340 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf339, buf340, 768, 49, grid=grid(768), stream=stream0)
        buf342 = reinterpret_tensor(buf269, (6272, 3072), (3072, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (6272, 768), (768, 1), 0), permute_375, out=buf342)
        del permute_375
        buf343 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (768, 6272), (1, 768), 0), view_380, out=buf343)
        del view_380
        buf344 = reinterpret_tensor(buf339, (1, 768, 49), (37632, 1, 768), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf341, buf344, 37632, 128, grid=grid(37632), stream=stream0)
        buf345 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf344, buf345, 768, 49, grid=grid(768), stream=stream0)
        buf346 = reinterpret_tensor(buf342, (8, 784, 3072), (2408448, 3072, 1), 0); del buf342  # reuse
        # Source Nodes: [x_397], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf346, addmm_61, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_61
        buf347 = reinterpret_tensor(buf341, (6272, 768), (768, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (6272, 3072), (3072, 1), 0), permute_379, out=buf347)
        del permute_379
        buf348 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (3072, 6272), (1, 3072), 0), view_378, out=buf348)
        del view_378
        buf349 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf346, buf349, 150528, 128, grid=grid(150528), stream=stream0)
        buf350 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf349, buf350, 3072, 49, grid=grid(3072), stream=stream0)
        buf357 = buf338; del buf338  # reuse
        buf360 = reinterpret_tensor(buf328, (8, 784, 768), (602112, 768, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf357, buf347, primals_527, mul_510, div_116, primals_83, buf360, 6272, 768, grid=grid(6272), stream=stream0)
        del div_116
        del primals_527
        del primals_83
        buf353 = reinterpret_tensor(buf344, (768, 49), (1, 768), 0); del buf344  # reuse
        buf355 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf347, mul_510, buf353, buf355, 37632, 128, grid=grid(37632), stream=stream0)
        del buf347
        del mul_510
        buf354 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf353, buf354, 768, 49, grid=grid(768), stream=stream0)
        buf356 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf355, buf356, 768, 49, grid=grid(768), stream=stream0)
        buf358 = reinterpret_tensor(buf355, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf357, convolution_45, buf358, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_45
        buf359 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf358, buf359, 768, 49, grid=grid(768), stream=stream0)
        buf361 = reinterpret_tensor(buf358, (768, 49), (1, 768), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf360, buf361, 37632, 128, grid=grid(37632), stream=stream0)
        buf362 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf361, buf362, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf363 = aten.convolution_backward(reinterpret_tensor(buf360, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_374, primals_525, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_374
        del primals_525
        buf364 = buf363[0]
        buf365 = buf363[1]
        del buf363
        buf366 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf364, buf366, 768, 6272, grid=grid(768), stream=stream0)
        buf367 = reinterpret_tensor(buf361, (768, 49), (49, 1), 0); del buf361  # reuse
        # Source Nodes: [x_391], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf364, convolution_44, unsqueeze_155, buf367, 37632, 128, grid=grid(37632), stream=stream0)
        buf368 = empty((768, ), device='cuda', dtype=torch.float32)
        buf369 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_391], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf367, squeeze_70, buf368, buf369, 768, 49, grid=grid(768), stream=stream0)
        buf370 = buf364; del buf364  # reuse
        # Source Nodes: [x_391], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf370, convolution_44, unsqueeze_155, buf368, squeeze_70, buf366, primals_523, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_44
        del primals_523
        del squeeze_70
        del unsqueeze_155
        buf371 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf370, buf371, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf372 = aten.convolution_backward(buf370, view_376, primals_521, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_521
        del view_376
        buf373 = buf372[0]
        buf374 = buf372[1]
        del buf372
        buf375 = reinterpret_tensor(buf367, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf373, primals_519, buf375, 37632, 128, grid=grid(37632), stream=stream0)
        buf376 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf375, buf376, 6272, 6, grid=grid(6272), stream=stream0)
        buf377 = reinterpret_tensor(buf375, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf373, primals_519, mul_497, buf377, 37632, 128, grid=grid(37632), stream=stream0)
        buf378 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf377, buf378, 6272, 6, grid=grid(6272), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (768, 49), (49, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf373, mul_497, buf379, 37632, 128, grid=grid(37632), stream=stream0)
        buf380 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf379, buf380, 768, 49, grid=grid(768), stream=stream0)
        buf381 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf373, buf381, 768, 6272, grid=grid(768), stream=stream0)
        buf382 = buf357; del buf357  # reuse
        buf387 = reinterpret_tensor(buf370, (8, 784, 768), (602112, 768, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf382, div_117, buf373, primals_519, buf376, mul_497, buf378, primals_81, buf387, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_117
        del mul_497
        del primals_519
        buf383 = reinterpret_tensor(buf379, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf379  # reuse
        buf385 = reinterpret_tensor(buf353, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf353  # reuse
        # Source Nodes: [x_386], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf382, mm_20, primals_518, primals_81, buf383, buf385, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_20
        del primals_518
        del primals_81
        buf384 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_386], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf383, buf384, 768, 49, grid=grid(768), stream=stream0)
        buf386 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf385, buf386, 768, 49, grid=grid(768), stream=stream0)
        buf388 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (768, 6272), (1, 768), 0), view_374, out=buf388)
        del view_374
        buf389 = reinterpret_tensor(buf373, (6272, 768), (768, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (6272, 768), (768, 1), 0), permute_387, out=buf389)
        del permute_387
        buf390 = reinterpret_tensor(buf387, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf389, buf390, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf391 = reinterpret_tensor(buf389, (128, 48, 784), (37632, 784, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_390, reinterpret_tensor(buf390, (128, 48, 784), (37632, 784, 1), 0), out=buf391)
        del permute_390
        buf392 = reinterpret_tensor(buf319, (128, 48, 48), (2304, 48, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf390, (128, 48, 784), (37632, 784, 1), 0), permute_391, out=buf392)
        del permute_391
        buf393 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf392, alias_85, buf393, 6144, 48, grid=grid(6144), stream=stream0)
        buf394 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf392, alias_85, buf393, bmm_40, buf394, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_40
        buf395 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf394, buf395, 16, 3, grid=grid(16), stream=stream0)
        buf396 = reinterpret_tensor(buf392, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf396, alias_85, buf393, primals_82, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_85
        del primals_82
        buf397 = reinterpret_tensor(buf390, (128, 784, 48), (37632, 48, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_392, reinterpret_tensor(buf396, (128, 48, 48), (2304, 48, 1), 0), out=buf397)
        del permute_392
        buf398 = reinterpret_tensor(buf360, (128, 48, 784), (37632, 784, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (128, 48, 48), (2304, 48, 1), 0), permute_393, out=buf398)
        del permute_393
        buf399 = reinterpret_tensor(buf324, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf324  # reuse
        # Source Nodes: [k_41], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf397, getitem_229, pow_85, buf399, 43008, 112, grid=grid(43008), stream=stream0)
        buf400 = buf393; del buf393  # reuse
        # Source Nodes: [k_41], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf399, buf400, 6144, 7, grid=grid(6144), stream=stream0)
        buf401 = reinterpret_tensor(buf399, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf399  # reuse
        # Source Nodes: [q_41], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf398, getitem_228, pow_83, buf401, 43008, 112, grid=grid(43008), stream=stream0)
        buf402 = buf323; del buf323  # reuse
        # Source Nodes: [q_41], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf401, buf402, 6144, 7, grid=grid(6144), stream=stream0)
        buf403 = reinterpret_tensor(buf327, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf398, pow_83, buf402, getitem_228, buf397, pow_85, buf400, getitem_229, buf391, buf403, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf391
        del getitem_228
        del getitem_229
        del pow_83
        del pow_85
        buf404 = reinterpret_tensor(buf326, (6272, 2304), (2304, 1), 0); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf403, buf404, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf405 = reinterpret_tensor(buf398, (6272, 768), (768, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf404, permute_396, out=buf405)
        del permute_396
        buf406 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (2304, 6272), (1, 2304), 0), view_364, out=buf406)
        del view_364
        buf407 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf404, buf407, 112896, 128, grid=grid(112896), stream=stream0)
        buf408 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf407, buf408, 2304, 49, grid=grid(2304), stream=stream0)
        buf415 = buf382; del buf382  # reuse
        buf418 = reinterpret_tensor(buf397, (8, 784, 768), (602112, 768, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf415, buf405, primals_513, mul_493, div_126, primals_80, buf418, 6272, 768, grid=grid(6272), stream=stream0)
        del div_126
        del primals_513
        del primals_80
        buf411 = reinterpret_tensor(buf385, (768, 49), (1, 768), 0); del buf385  # reuse
        buf413 = reinterpret_tensor(buf383, (768, 49), (1, 768), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf405, mul_493, buf411, buf413, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_493
        buf412 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf411, buf412, 768, 49, grid=grid(768), stream=stream0)
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf413, buf414, 768, 49, grid=grid(768), stream=stream0)
        buf416 = reinterpret_tensor(buf413, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf415, addmm_59, buf416, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_59
        buf417 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf416, buf417, 768, 49, grid=grid(768), stream=stream0)
        buf419 = reinterpret_tensor(buf346, (6272, 3072), (3072, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (6272, 768), (768, 1), 0), permute_400, out=buf419)
        del permute_400
        buf420 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (768, 6272), (1, 768), 0), view_362, out=buf420)
        del view_362
        buf421 = reinterpret_tensor(buf416, (1, 768, 49), (37632, 1, 768), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf418, buf421, 37632, 128, grid=grid(37632), stream=stream0)
        buf422 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf421, buf422, 768, 49, grid=grid(768), stream=stream0)
        buf423 = reinterpret_tensor(buf419, (8, 784, 3072), (2408448, 3072, 1), 0); del buf419  # reuse
        # Source Nodes: [x_378], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf423, addmm_58, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_58
        buf424 = reinterpret_tensor(buf418, (6272, 768), (768, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (6272, 3072), (3072, 1), 0), permute_404, out=buf424)
        del permute_404
        buf425 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (3072, 6272), (1, 3072), 0), view_360, out=buf425)
        del view_360
        buf426 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf423, buf426, 150528, 128, grid=grid(150528), stream=stream0)
        buf427 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf426, buf427, 3072, 49, grid=grid(3072), stream=stream0)
        buf434 = buf415; del buf415  # reuse
        buf437 = reinterpret_tensor(buf405, (8, 784, 768), (602112, 768, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf434, buf424, primals_507, mul_487, div_127, primals_79, buf437, 6272, 768, grid=grid(6272), stream=stream0)
        del div_127
        del primals_507
        del primals_79
        buf430 = reinterpret_tensor(buf421, (768, 49), (1, 768), 0); del buf421  # reuse
        buf432 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf424, mul_487, buf430, buf432, 37632, 128, grid=grid(37632), stream=stream0)
        del buf424
        del mul_487
        buf431 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf430, buf431, 768, 49, grid=grid(768), stream=stream0)
        buf433 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf432, buf433, 768, 49, grid=grid(768), stream=stream0)
        buf435 = reinterpret_tensor(buf432, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf434, convolution_43, buf435, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_43
        buf436 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf435, buf436, 768, 49, grid=grid(768), stream=stream0)
        buf438 = reinterpret_tensor(buf435, (768, 49), (1, 768), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf437, buf438, 37632, 128, grid=grid(37632), stream=stream0)
        buf439 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf438, buf439, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf440 = aten.convolution_backward(reinterpret_tensor(buf437, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_357, primals_505, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_357
        del primals_505
        buf441 = buf440[0]
        buf442 = buf440[1]
        del buf440
        buf443 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf441, buf443, 768, 6272, grid=grid(768), stream=stream0)
        buf444 = reinterpret_tensor(buf438, (768, 49), (49, 1), 0); del buf438  # reuse
        # Source Nodes: [x_372], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf441, convolution_42, unsqueeze_167, buf444, 37632, 128, grid=grid(37632), stream=stream0)
        buf445 = empty((768, ), device='cuda', dtype=torch.float32)
        buf446 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_372], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf444, squeeze_67, buf445, buf446, 768, 49, grid=grid(768), stream=stream0)
        buf447 = buf441; del buf441  # reuse
        # Source Nodes: [x_372], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf447, convolution_42, unsqueeze_167, buf445, squeeze_67, buf443, primals_503, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_42
        del primals_503
        del squeeze_67
        del unsqueeze_167
        buf448 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf447, buf448, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf449 = aten.convolution_backward(buf447, view_358, primals_501, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_501
        del view_358
        buf450 = buf449[0]
        buf451 = buf449[1]
        del buf449
        buf452 = reinterpret_tensor(buf444, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf450, primals_499, buf452, 37632, 128, grid=grid(37632), stream=stream0)
        buf453 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf452, buf453, 6272, 6, grid=grid(6272), stream=stream0)
        buf454 = reinterpret_tensor(buf452, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf450, primals_499, mul_474, buf454, 37632, 128, grid=grid(37632), stream=stream0)
        buf455 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf454, buf455, 6272, 6, grid=grid(6272), stream=stream0)
        buf456 = reinterpret_tensor(buf454, (768, 49), (49, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf450, mul_474, buf456, 37632, 128, grid=grid(37632), stream=stream0)
        buf457 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf456, buf457, 768, 49, grid=grid(768), stream=stream0)
        buf458 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf450, buf458, 768, 6272, grid=grid(768), stream=stream0)
        buf459 = buf434; del buf434  # reuse
        buf464 = reinterpret_tensor(buf447, (8, 784, 768), (602112, 768, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf459, div_128, buf450, primals_499, buf453, mul_474, buf455, primals_77, buf464, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_128
        del mul_474
        del primals_499
        buf460 = reinterpret_tensor(buf456, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf456  # reuse
        buf462 = reinterpret_tensor(buf430, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf430  # reuse
        # Source Nodes: [x_367], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf459, mm_19, primals_498, primals_77, buf460, buf462, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_19
        del primals_498
        del primals_77
        buf461 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_367], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf460, buf461, 768, 49, grid=grid(768), stream=stream0)
        buf463 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf462, buf463, 768, 49, grid=grid(768), stream=stream0)
        buf465 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (768, 6272), (1, 768), 0), view_356, out=buf465)
        del view_356
        buf466 = reinterpret_tensor(buf450, (6272, 768), (768, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (6272, 768), (768, 1), 0), permute_412, out=buf466)
        del permute_412
        buf467 = reinterpret_tensor(buf464, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf466, buf467, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf468 = reinterpret_tensor(buf466, (128, 48, 784), (37632, 784, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_415, reinterpret_tensor(buf467, (128, 48, 784), (37632, 784, 1), 0), out=buf468)
        del permute_415
        buf469 = reinterpret_tensor(buf396, (128, 48, 48), (2304, 48, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf467, (128, 48, 784), (37632, 784, 1), 0), permute_416, out=buf469)
        del permute_416
        buf470 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf469, alias_88, buf470, 6144, 48, grid=grid(6144), stream=stream0)
        buf471 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf469, alias_88, buf470, bmm_38, buf471, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_38
        buf472 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf471, buf472, 16, 3, grid=grid(16), stream=stream0)
        buf473 = reinterpret_tensor(buf469, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf473, alias_88, buf470, primals_78, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_88
        del primals_78
        buf474 = reinterpret_tensor(buf467, (128, 784, 48), (37632, 48, 1), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_417, reinterpret_tensor(buf473, (128, 48, 48), (2304, 48, 1), 0), out=buf474)
        del permute_417
        buf475 = reinterpret_tensor(buf437, (128, 48, 784), (37632, 784, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (128, 48, 48), (2304, 48, 1), 0), permute_418, out=buf475)
        del permute_418
        buf476 = reinterpret_tensor(buf401, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf401  # reuse
        # Source Nodes: [k_39], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf474, getitem_218, pow_81, buf476, 43008, 112, grid=grid(43008), stream=stream0)
        buf477 = buf470; del buf470  # reuse
        # Source Nodes: [k_39], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf476, buf477, 6144, 7, grid=grid(6144), stream=stream0)
        buf478 = reinterpret_tensor(buf476, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf476  # reuse
        # Source Nodes: [q_39], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf475, getitem_217, pow_79, buf478, 43008, 112, grid=grid(43008), stream=stream0)
        buf479 = buf400; del buf400  # reuse
        # Source Nodes: [q_39], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf478, buf479, 6144, 7, grid=grid(6144), stream=stream0)
        buf480 = reinterpret_tensor(buf404, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf475, pow_79, buf479, getitem_217, buf474, pow_81, buf477, getitem_218, buf468, buf480, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf468
        del getitem_217
        del getitem_218
        del pow_79
        del pow_81
        buf481 = reinterpret_tensor(buf403, (6272, 2304), (2304, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf480, buf481, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf482 = reinterpret_tensor(buf475, (6272, 768), (768, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf481, permute_421, out=buf482)
        del permute_421
        buf483 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (2304, 6272), (1, 2304), 0), view_346, out=buf483)
        del view_346
        buf484 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf481, buf484, 112896, 128, grid=grid(112896), stream=stream0)
        buf485 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf484, buf485, 2304, 49, grid=grid(2304), stream=stream0)
        buf492 = buf459; del buf459  # reuse
        buf495 = reinterpret_tensor(buf474, (8, 784, 768), (602112, 768, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf492, buf482, primals_493, mul_470, div_137, primals_76, buf495, 6272, 768, grid=grid(6272), stream=stream0)
        del div_137
        del primals_493
        del primals_76
        buf488 = reinterpret_tensor(buf462, (768, 49), (1, 768), 0); del buf462  # reuse
        buf490 = reinterpret_tensor(buf460, (768, 49), (1, 768), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf482, mul_470, buf488, buf490, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_470
        buf489 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf488, buf489, 768, 49, grid=grid(768), stream=stream0)
        buf491 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf490, buf491, 768, 49, grid=grid(768), stream=stream0)
        buf493 = reinterpret_tensor(buf490, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf492, addmm_56, buf493, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_56
        buf494 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf493, buf494, 768, 49, grid=grid(768), stream=stream0)
        buf496 = reinterpret_tensor(buf423, (6272, 3072), (3072, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (6272, 768), (768, 1), 0), permute_425, out=buf496)
        del permute_425
        buf497 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (768, 6272), (1, 768), 0), view_344, out=buf497)
        del view_344
        buf498 = reinterpret_tensor(buf493, (1, 768, 49), (37632, 1, 768), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf495, buf498, 37632, 128, grid=grid(37632), stream=stream0)
        buf499 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf498, buf499, 768, 49, grid=grid(768), stream=stream0)
        buf500 = reinterpret_tensor(buf496, (8, 784, 3072), (2408448, 3072, 1), 0); del buf496  # reuse
        # Source Nodes: [x_359], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf500, addmm_55, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_55
        buf501 = reinterpret_tensor(buf495, (6272, 768), (768, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (6272, 3072), (3072, 1), 0), permute_429, out=buf501)
        del permute_429
        buf502 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (3072, 6272), (1, 3072), 0), view_342, out=buf502)
        del view_342
        buf503 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf500, buf503, 150528, 128, grid=grid(150528), stream=stream0)
        buf504 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf503, buf504, 3072, 49, grid=grid(3072), stream=stream0)
        buf511 = buf492; del buf492  # reuse
        buf514 = reinterpret_tensor(buf482, (8, 784, 768), (602112, 768, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf511, buf501, primals_487, mul_464, div_138, primals_75, buf514, 6272, 768, grid=grid(6272), stream=stream0)
        del div_138
        del primals_487
        del primals_75
        buf507 = reinterpret_tensor(buf498, (768, 49), (1, 768), 0); del buf498  # reuse
        buf509 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf501, mul_464, buf507, buf509, 37632, 128, grid=grid(37632), stream=stream0)
        del buf501
        del mul_464
        buf508 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf507, buf508, 768, 49, grid=grid(768), stream=stream0)
        buf510 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf509, buf510, 768, 49, grid=grid(768), stream=stream0)
        buf512 = reinterpret_tensor(buf509, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf511, convolution_41, buf512, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_41
        buf513 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf512, buf513, 768, 49, grid=grid(768), stream=stream0)
        buf515 = reinterpret_tensor(buf512, (768, 49), (1, 768), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf514, buf515, 37632, 128, grid=grid(37632), stream=stream0)
        buf516 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf515, buf516, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf517 = aten.convolution_backward(reinterpret_tensor(buf514, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_340, primals_485, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_340
        del primals_485
        buf518 = buf517[0]
        buf519 = buf517[1]
        del buf517
        buf520 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf518, buf520, 768, 6272, grid=grid(768), stream=stream0)
        buf521 = reinterpret_tensor(buf515, (768, 49), (49, 1), 0); del buf515  # reuse
        # Source Nodes: [x_353], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf518, convolution_40, unsqueeze_179, buf521, 37632, 128, grid=grid(37632), stream=stream0)
        buf522 = empty((768, ), device='cuda', dtype=torch.float32)
        buf523 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_353], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf521, squeeze_64, buf522, buf523, 768, 49, grid=grid(768), stream=stream0)
        buf524 = buf518; del buf518  # reuse
        # Source Nodes: [x_353], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf524, convolution_40, unsqueeze_179, buf522, squeeze_64, buf520, primals_483, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_40
        del primals_483
        del squeeze_64
        del unsqueeze_179
        buf525 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf524, buf525, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf526 = aten.convolution_backward(buf524, view_340, primals_481, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_481
        del view_340
        buf527 = buf526[0]
        buf528 = buf526[1]
        del buf526
        buf529 = reinterpret_tensor(buf521, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf527, primals_479, buf529, 37632, 128, grid=grid(37632), stream=stream0)
        buf530 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf529, buf530, 6272, 6, grid=grid(6272), stream=stream0)
        buf531 = reinterpret_tensor(buf529, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf527, primals_479, mul_451, buf531, 37632, 128, grid=grid(37632), stream=stream0)
        buf532 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf531, buf532, 6272, 6, grid=grid(6272), stream=stream0)
        buf533 = reinterpret_tensor(buf531, (768, 49), (49, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf527, mul_451, buf533, 37632, 128, grid=grid(37632), stream=stream0)
        buf534 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf533, buf534, 768, 49, grid=grid(768), stream=stream0)
        buf535 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf527, buf535, 768, 6272, grid=grid(768), stream=stream0)
        buf536 = buf511; del buf511  # reuse
        buf541 = reinterpret_tensor(buf524, (8, 784, 768), (602112, 768, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf536, div_139, buf527, primals_479, buf530, mul_451, buf532, primals_73, buf541, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_139
        del mul_451
        del primals_479
        buf537 = reinterpret_tensor(buf533, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf533  # reuse
        buf539 = reinterpret_tensor(buf507, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf507  # reuse
        # Source Nodes: [x_348], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf536, mm_18, primals_478, primals_73, buf537, buf539, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_18
        del primals_478
        del primals_73
        buf538 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_348], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf537, buf538, 768, 49, grid=grid(768), stream=stream0)
        buf540 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf539, buf540, 768, 49, grid=grid(768), stream=stream0)
        buf542 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (768, 6272), (1, 768), 0), view_338, out=buf542)
        del view_338
        buf543 = reinterpret_tensor(buf527, (6272, 768), (768, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (6272, 768), (768, 1), 0), permute_437, out=buf543)
        del permute_437
        buf544 = reinterpret_tensor(buf541, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf543, buf544, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf545 = reinterpret_tensor(buf543, (128, 48, 784), (37632, 784, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_440, reinterpret_tensor(buf544, (128, 48, 784), (37632, 784, 1), 0), out=buf545)
        del permute_440
        buf546 = reinterpret_tensor(buf473, (128, 48, 48), (2304, 48, 1), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf544, (128, 48, 784), (37632, 784, 1), 0), permute_441, out=buf546)
        del permute_441
        buf547 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf546, alias_91, buf547, 6144, 48, grid=grid(6144), stream=stream0)
        buf548 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf546, alias_91, buf547, bmm_36, buf548, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_36
        buf549 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf548, buf549, 16, 3, grid=grid(16), stream=stream0)
        buf550 = reinterpret_tensor(buf546, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf550, alias_91, buf547, primals_74, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_91
        del primals_74
        buf551 = reinterpret_tensor(buf544, (128, 784, 48), (37632, 48, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_442, reinterpret_tensor(buf550, (128, 48, 48), (2304, 48, 1), 0), out=buf551)
        del permute_442
        buf552 = reinterpret_tensor(buf514, (128, 48, 784), (37632, 784, 1), 0); del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf550, (128, 48, 48), (2304, 48, 1), 0), permute_443, out=buf552)
        del permute_443
        buf553 = reinterpret_tensor(buf478, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf478  # reuse
        # Source Nodes: [k_37], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf551, getitem_207, pow_77, buf553, 43008, 112, grid=grid(43008), stream=stream0)
        buf554 = buf547; del buf547  # reuse
        # Source Nodes: [k_37], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf553, buf554, 6144, 7, grid=grid(6144), stream=stream0)
        buf555 = reinterpret_tensor(buf553, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf553  # reuse
        # Source Nodes: [q_37], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf552, getitem_206, pow_75, buf555, 43008, 112, grid=grid(43008), stream=stream0)
        buf556 = buf477; del buf477  # reuse
        # Source Nodes: [q_37], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf555, buf556, 6144, 7, grid=grid(6144), stream=stream0)
        buf557 = reinterpret_tensor(buf481, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf552, pow_75, buf556, getitem_206, buf551, pow_77, buf554, getitem_207, buf545, buf557, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf545
        del getitem_206
        del getitem_207
        del pow_75
        del pow_77
        buf558 = reinterpret_tensor(buf480, (6272, 2304), (2304, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf557, buf558, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf559 = reinterpret_tensor(buf552, (6272, 768), (768, 1), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf558, permute_446, out=buf559)
        del permute_446
        buf560 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (2304, 6272), (1, 2304), 0), view_328, out=buf560)
        del view_328
        buf561 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf558, buf561, 112896, 128, grid=grid(112896), stream=stream0)
        buf562 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf561, buf562, 2304, 49, grid=grid(2304), stream=stream0)
        buf569 = buf536; del buf536  # reuse
        buf572 = reinterpret_tensor(buf551, (8, 784, 768), (602112, 768, 1), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf569, buf559, primals_473, mul_447, div_148, primals_72, buf572, 6272, 768, grid=grid(6272), stream=stream0)
        del div_148
        del primals_473
        del primals_72
        buf565 = reinterpret_tensor(buf539, (768, 49), (1, 768), 0); del buf539  # reuse
        buf567 = reinterpret_tensor(buf537, (768, 49), (1, 768), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf559, mul_447, buf565, buf567, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_447
        buf566 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf565, buf566, 768, 49, grid=grid(768), stream=stream0)
        buf568 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf567, buf568, 768, 49, grid=grid(768), stream=stream0)
        buf570 = reinterpret_tensor(buf567, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf569, addmm_53, buf570, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_53
        buf571 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf570, buf571, 768, 49, grid=grid(768), stream=stream0)
        buf573 = reinterpret_tensor(buf500, (6272, 3072), (3072, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (6272, 768), (768, 1), 0), permute_450, out=buf573)
        del permute_450
        buf574 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (768, 6272), (1, 768), 0), view_326, out=buf574)
        del view_326
        buf575 = reinterpret_tensor(buf570, (1, 768, 49), (37632, 1, 768), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf572, buf575, 37632, 128, grid=grid(37632), stream=stream0)
        buf576 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf575, buf576, 768, 49, grid=grid(768), stream=stream0)
        buf577 = reinterpret_tensor(buf573, (8, 784, 3072), (2408448, 3072, 1), 0); del buf573  # reuse
        # Source Nodes: [x_340], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf577, addmm_52, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_52
        buf578 = reinterpret_tensor(buf572, (6272, 768), (768, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (6272, 3072), (3072, 1), 0), permute_454, out=buf578)
        del permute_454
        buf579 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (3072, 6272), (1, 3072), 0), view_324, out=buf579)
        del view_324
        buf580 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf577, buf580, 150528, 128, grid=grid(150528), stream=stream0)
        buf581 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf580, buf581, 3072, 49, grid=grid(3072), stream=stream0)
        buf588 = buf569; del buf569  # reuse
        buf591 = reinterpret_tensor(buf559, (8, 784, 768), (602112, 768, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf588, buf578, primals_467, mul_441, div_149, primals_71, buf591, 6272, 768, grid=grid(6272), stream=stream0)
        del div_149
        del primals_467
        del primals_71
        buf584 = reinterpret_tensor(buf575, (768, 49), (1, 768), 0); del buf575  # reuse
        buf586 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf578, mul_441, buf584, buf586, 37632, 128, grid=grid(37632), stream=stream0)
        del buf578
        del mul_441
        buf585 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf584, buf585, 768, 49, grid=grid(768), stream=stream0)
        buf587 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf586, buf587, 768, 49, grid=grid(768), stream=stream0)
        buf589 = reinterpret_tensor(buf586, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf588, convolution_39, buf589, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_39
        buf590 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf589, buf590, 768, 49, grid=grid(768), stream=stream0)
        buf592 = reinterpret_tensor(buf589, (768, 49), (1, 768), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf591, buf592, 37632, 128, grid=grid(37632), stream=stream0)
        buf593 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf592, buf593, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf594 = aten.convolution_backward(reinterpret_tensor(buf591, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_323, primals_465, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_323
        del primals_465
        buf595 = buf594[0]
        buf596 = buf594[1]
        del buf594
        buf597 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf595, buf597, 768, 6272, grid=grid(768), stream=stream0)
        buf598 = reinterpret_tensor(buf592, (768, 49), (49, 1), 0); del buf592  # reuse
        # Source Nodes: [x_334], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf595, convolution_38, unsqueeze_191, buf598, 37632, 128, grid=grid(37632), stream=stream0)
        buf599 = empty((768, ), device='cuda', dtype=torch.float32)
        buf600 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_334], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf598, squeeze_61, buf599, buf600, 768, 49, grid=grid(768), stream=stream0)
        buf601 = buf595; del buf595  # reuse
        # Source Nodes: [x_334], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf601, convolution_38, unsqueeze_191, buf599, squeeze_61, buf597, primals_463, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_38
        del primals_463
        del squeeze_61
        del unsqueeze_191
        buf602 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf601, buf602, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf603 = aten.convolution_backward(buf601, view_322, primals_461, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_461
        del view_322
        buf604 = buf603[0]
        buf605 = buf603[1]
        del buf603
        buf606 = reinterpret_tensor(buf598, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf604, primals_459, buf606, 37632, 128, grid=grid(37632), stream=stream0)
        buf607 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf606, buf607, 6272, 6, grid=grid(6272), stream=stream0)
        buf608 = reinterpret_tensor(buf606, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf604, primals_459, mul_428, buf608, 37632, 128, grid=grid(37632), stream=stream0)
        buf609 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf608, buf609, 6272, 6, grid=grid(6272), stream=stream0)
        buf610 = reinterpret_tensor(buf608, (768, 49), (49, 1), 0); del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf604, mul_428, buf610, 37632, 128, grid=grid(37632), stream=stream0)
        buf611 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf610, buf611, 768, 49, grid=grid(768), stream=stream0)
        buf612 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf604, buf612, 768, 6272, grid=grid(768), stream=stream0)
        buf613 = buf588; del buf588  # reuse
        buf618 = reinterpret_tensor(buf601, (8, 784, 768), (602112, 768, 1), 0); del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf613, div_150, buf604, primals_459, buf607, mul_428, buf609, primals_69, buf618, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_150
        del mul_428
        del primals_459
        buf614 = reinterpret_tensor(buf610, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf610  # reuse
        buf616 = reinterpret_tensor(buf584, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf584  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf613, mm_17, primals_458, primals_69, buf614, buf616, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_17
        del primals_458
        del primals_69
        buf615 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf614, buf615, 768, 49, grid=grid(768), stream=stream0)
        buf617 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf616, buf617, 768, 49, grid=grid(768), stream=stream0)
        buf619 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (768, 6272), (1, 768), 0), view_320, out=buf619)
        del view_320
        buf620 = reinterpret_tensor(buf604, (6272, 768), (768, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (6272, 768), (768, 1), 0), permute_462, out=buf620)
        del permute_462
        buf621 = reinterpret_tensor(buf618, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf620, buf621, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf622 = reinterpret_tensor(buf620, (128, 48, 784), (37632, 784, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_465, reinterpret_tensor(buf621, (128, 48, 784), (37632, 784, 1), 0), out=buf622)
        del permute_465
        buf623 = reinterpret_tensor(buf550, (128, 48, 48), (2304, 48, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf621, (128, 48, 784), (37632, 784, 1), 0), permute_466, out=buf623)
        del permute_466
        buf624 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf623, alias_94, buf624, 6144, 48, grid=grid(6144), stream=stream0)
        buf625 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf623, alias_94, buf624, bmm_34, buf625, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_34
        buf626 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf625, buf626, 16, 3, grid=grid(16), stream=stream0)
        buf627 = reinterpret_tensor(buf623, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf627, alias_94, buf624, primals_70, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_94
        del primals_70
        buf628 = reinterpret_tensor(buf621, (128, 784, 48), (37632, 48, 1), 0); del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_467, reinterpret_tensor(buf627, (128, 48, 48), (2304, 48, 1), 0), out=buf628)
        del permute_467
        buf629 = reinterpret_tensor(buf591, (128, 48, 784), (37632, 784, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf627, (128, 48, 48), (2304, 48, 1), 0), permute_468, out=buf629)
        del permute_468
        buf630 = reinterpret_tensor(buf555, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf555  # reuse
        # Source Nodes: [k_35], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf628, getitem_196, pow_73, buf630, 43008, 112, grid=grid(43008), stream=stream0)
        buf631 = buf624; del buf624  # reuse
        # Source Nodes: [k_35], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf630, buf631, 6144, 7, grid=grid(6144), stream=stream0)
        buf632 = reinterpret_tensor(buf630, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf630  # reuse
        # Source Nodes: [q_35], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf629, getitem_195, pow_71, buf632, 43008, 112, grid=grid(43008), stream=stream0)
        buf633 = buf554; del buf554  # reuse
        # Source Nodes: [q_35], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf632, buf633, 6144, 7, grid=grid(6144), stream=stream0)
        buf634 = reinterpret_tensor(buf558, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf629, pow_71, buf633, getitem_195, buf628, pow_73, buf631, getitem_196, buf622, buf634, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf622
        del getitem_195
        del getitem_196
        del pow_71
        del pow_73
        buf635 = reinterpret_tensor(buf557, (6272, 2304), (2304, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf634, buf635, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf636 = reinterpret_tensor(buf629, (6272, 768), (768, 1), 0); del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf635, permute_471, out=buf636)
        del permute_471
        buf637 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (2304, 6272), (1, 2304), 0), view_310, out=buf637)
        del view_310
        buf638 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf635, buf638, 112896, 128, grid=grid(112896), stream=stream0)
        buf639 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf638, buf639, 2304, 49, grid=grid(2304), stream=stream0)
        buf646 = buf613; del buf613  # reuse
        buf649 = reinterpret_tensor(buf628, (8, 784, 768), (602112, 768, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf646, buf636, primals_453, mul_424, div_159, primals_68, buf649, 6272, 768, grid=grid(6272), stream=stream0)
        del div_159
        del primals_453
        del primals_68
        buf642 = reinterpret_tensor(buf616, (768, 49), (1, 768), 0); del buf616  # reuse
        buf644 = reinterpret_tensor(buf614, (768, 49), (1, 768), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf636, mul_424, buf642, buf644, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_424
        buf643 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf642, buf643, 768, 49, grid=grid(768), stream=stream0)
        buf645 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf644, buf645, 768, 49, grid=grid(768), stream=stream0)
        buf647 = reinterpret_tensor(buf644, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf646, addmm_50, buf647, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_50
        buf648 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf647, buf648, 768, 49, grid=grid(768), stream=stream0)
        buf650 = reinterpret_tensor(buf577, (6272, 3072), (3072, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf649, (6272, 768), (768, 1), 0), permute_475, out=buf650)
        del permute_475
        buf651 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf649, (768, 6272), (1, 768), 0), view_308, out=buf651)
        del view_308
        buf652 = reinterpret_tensor(buf647, (1, 768, 49), (37632, 1, 768), 0); del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf649, buf652, 37632, 128, grid=grid(37632), stream=stream0)
        buf653 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf652, buf653, 768, 49, grid=grid(768), stream=stream0)
        buf654 = reinterpret_tensor(buf650, (8, 784, 3072), (2408448, 3072, 1), 0); del buf650  # reuse
        # Source Nodes: [x_321], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf654, addmm_49, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_49
        buf655 = reinterpret_tensor(buf649, (6272, 768), (768, 1), 0); del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (6272, 3072), (3072, 1), 0), permute_479, out=buf655)
        del permute_479
        buf656 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (3072, 6272), (1, 3072), 0), view_306, out=buf656)
        del view_306
        buf657 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf654, buf657, 150528, 128, grid=grid(150528), stream=stream0)
        buf658 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf657, buf658, 3072, 49, grid=grid(3072), stream=stream0)
        buf665 = buf646; del buf646  # reuse
        buf668 = reinterpret_tensor(buf636, (8, 784, 768), (602112, 768, 1), 0); del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf665, buf655, primals_447, mul_418, div_160, primals_67, buf668, 6272, 768, grid=grid(6272), stream=stream0)
        del div_160
        del primals_447
        del primals_67
        buf661 = reinterpret_tensor(buf652, (768, 49), (1, 768), 0); del buf652  # reuse
        buf663 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf655, mul_418, buf661, buf663, 37632, 128, grid=grid(37632), stream=stream0)
        del buf655
        del mul_418
        buf662 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf661, buf662, 768, 49, grid=grid(768), stream=stream0)
        buf664 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf663, buf664, 768, 49, grid=grid(768), stream=stream0)
        buf666 = reinterpret_tensor(buf663, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf665, convolution_37, buf666, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_37
        buf667 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf666, buf667, 768, 49, grid=grid(768), stream=stream0)
        buf669 = reinterpret_tensor(buf666, (768, 49), (1, 768), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf668, buf669, 37632, 128, grid=grid(37632), stream=stream0)
        buf670 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf669, buf670, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf671 = aten.convolution_backward(reinterpret_tensor(buf668, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_306, primals_445, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_306
        del primals_445
        buf672 = buf671[0]
        buf673 = buf671[1]
        del buf671
        buf674 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf672, buf674, 768, 6272, grid=grid(768), stream=stream0)
        buf675 = reinterpret_tensor(buf669, (768, 49), (49, 1), 0); del buf669  # reuse
        # Source Nodes: [x_315], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf672, convolution_36, unsqueeze_203, buf675, 37632, 128, grid=grid(37632), stream=stream0)
        buf676 = empty((768, ), device='cuda', dtype=torch.float32)
        buf677 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf675, squeeze_58, buf676, buf677, 768, 49, grid=grid(768), stream=stream0)
        buf678 = buf672; del buf672  # reuse
        # Source Nodes: [x_315], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf678, convolution_36, unsqueeze_203, buf676, squeeze_58, buf674, primals_443, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_36
        del primals_443
        del squeeze_58
        del unsqueeze_203
        buf679 = buf676; del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf678, buf679, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf680 = aten.convolution_backward(buf678, view_304, primals_441, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_441
        del view_304
        buf681 = buf680[0]
        buf682 = buf680[1]
        del buf680
        buf683 = reinterpret_tensor(buf675, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf681, primals_439, buf683, 37632, 128, grid=grid(37632), stream=stream0)
        buf684 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf683, buf684, 6272, 6, grid=grid(6272), stream=stream0)
        buf685 = reinterpret_tensor(buf683, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf681, primals_439, mul_405, buf685, 37632, 128, grid=grid(37632), stream=stream0)
        buf686 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf685, buf686, 6272, 6, grid=grid(6272), stream=stream0)
        buf687 = reinterpret_tensor(buf685, (768, 49), (49, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf681, mul_405, buf687, 37632, 128, grid=grid(37632), stream=stream0)
        buf688 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf687, buf688, 768, 49, grid=grid(768), stream=stream0)
        buf689 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf681, buf689, 768, 6272, grid=grid(768), stream=stream0)
        buf690 = buf665; del buf665  # reuse
        buf695 = reinterpret_tensor(buf678, (8, 784, 768), (602112, 768, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf690, div_161, buf681, primals_439, buf684, mul_405, buf686, primals_65, buf695, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_161
        del mul_405
        del primals_439
        buf691 = reinterpret_tensor(buf687, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf687  # reuse
        buf693 = reinterpret_tensor(buf661, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf661  # reuse
        # Source Nodes: [x_310], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf690, mm_16, primals_438, primals_65, buf691, buf693, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_16
        del primals_438
        del primals_65
        buf692 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_310], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf691, buf692, 768, 49, grid=grid(768), stream=stream0)
        buf694 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf693, buf694, 768, 49, grid=grid(768), stream=stream0)
        buf696 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (768, 6272), (1, 768), 0), view_302, out=buf696)
        del view_302
        buf697 = reinterpret_tensor(buf681, (6272, 768), (768, 1), 0); del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (6272, 768), (768, 1), 0), permute_487, out=buf697)
        del permute_487
        buf698 = reinterpret_tensor(buf695, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf697, buf698, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf699 = reinterpret_tensor(buf697, (128, 48, 784), (37632, 784, 1), 0); del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_490, reinterpret_tensor(buf698, (128, 48, 784), (37632, 784, 1), 0), out=buf699)
        del permute_490
        buf700 = reinterpret_tensor(buf627, (128, 48, 48), (2304, 48, 1), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf698, (128, 48, 784), (37632, 784, 1), 0), permute_491, out=buf700)
        del permute_491
        buf701 = buf633; del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf700, alias_97, buf701, 6144, 48, grid=grid(6144), stream=stream0)
        buf702 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf700, alias_97, buf701, bmm_32, buf702, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_32
        buf703 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf702, buf703, 16, 3, grid=grid(16), stream=stream0)
        buf704 = reinterpret_tensor(buf700, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf704, alias_97, buf701, primals_66, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_97
        del primals_66
        buf705 = reinterpret_tensor(buf698, (128, 784, 48), (37632, 48, 1), 0); del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_492, reinterpret_tensor(buf704, (128, 48, 48), (2304, 48, 1), 0), out=buf705)
        del permute_492
        buf706 = reinterpret_tensor(buf668, (128, 48, 784), (37632, 784, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf704, (128, 48, 48), (2304, 48, 1), 0), permute_493, out=buf706)
        del permute_493
        buf707 = reinterpret_tensor(buf632, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf632  # reuse
        # Source Nodes: [k_33], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf705, getitem_185, pow_69, buf707, 43008, 112, grid=grid(43008), stream=stream0)
        buf708 = buf701; del buf701  # reuse
        # Source Nodes: [k_33], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf707, buf708, 6144, 7, grid=grid(6144), stream=stream0)
        buf709 = reinterpret_tensor(buf707, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf707  # reuse
        # Source Nodes: [q_33], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf706, getitem_184, pow_67, buf709, 43008, 112, grid=grid(43008), stream=stream0)
        buf710 = buf631; del buf631  # reuse
        # Source Nodes: [q_33], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf709, buf710, 6144, 7, grid=grid(6144), stream=stream0)
        buf711 = reinterpret_tensor(buf635, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf706, pow_67, buf710, getitem_184, buf705, pow_69, buf708, getitem_185, buf699, buf711, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf699
        del getitem_184
        del getitem_185
        del pow_67
        del pow_69
        buf712 = reinterpret_tensor(buf634, (6272, 2304), (2304, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf711, buf712, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf713 = reinterpret_tensor(buf706, (6272, 768), (768, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf712, permute_496, out=buf713)
        del permute_496
        buf714 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf712, (2304, 6272), (1, 2304), 0), view_292, out=buf714)
        del view_292
        buf715 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf712, buf715, 112896, 128, grid=grid(112896), stream=stream0)
        buf716 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf715, buf716, 2304, 49, grid=grid(2304), stream=stream0)
        buf723 = buf690; del buf690  # reuse
        buf726 = reinterpret_tensor(buf705, (8, 784, 768), (602112, 768, 1), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf723, buf713, primals_433, mul_401, div_170, primals_64, buf726, 6272, 768, grid=grid(6272), stream=stream0)
        del div_170
        del primals_433
        del primals_64
        buf719 = reinterpret_tensor(buf693, (768, 49), (1, 768), 0); del buf693  # reuse
        buf721 = reinterpret_tensor(buf691, (768, 49), (1, 768), 0); del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf713, mul_401, buf719, buf721, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_401
        buf720 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf719, buf720, 768, 49, grid=grid(768), stream=stream0)
        buf722 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf721, buf722, 768, 49, grid=grid(768), stream=stream0)
        buf724 = reinterpret_tensor(buf721, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf723, addmm_47, buf724, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_47
        buf725 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf724, buf725, 768, 49, grid=grid(768), stream=stream0)
        buf727 = reinterpret_tensor(buf654, (6272, 3072), (3072, 1), 0); del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf726, (6272, 768), (768, 1), 0), permute_500, out=buf727)
        del permute_500
        buf728 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf726, (768, 6272), (1, 768), 0), view_290, out=buf728)
        del view_290
        buf729 = reinterpret_tensor(buf724, (1, 768, 49), (37632, 1, 768), 0); del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf726, buf729, 37632, 128, grid=grid(37632), stream=stream0)
        buf730 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf729, buf730, 768, 49, grid=grid(768), stream=stream0)
        buf731 = reinterpret_tensor(buf727, (8, 784, 3072), (2408448, 3072, 1), 0); del buf727  # reuse
        # Source Nodes: [x_302], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf731, addmm_46, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_46
        buf732 = reinterpret_tensor(buf726, (6272, 768), (768, 1), 0); del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (6272, 3072), (3072, 1), 0), permute_504, out=buf732)
        del permute_504
        buf733 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (3072, 6272), (1, 3072), 0), view_288, out=buf733)
        del view_288
        buf734 = buf657; del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf731, buf734, 150528, 128, grid=grid(150528), stream=stream0)
        buf735 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf734, buf735, 3072, 49, grid=grid(3072), stream=stream0)
        buf742 = buf723; del buf723  # reuse
        buf745 = reinterpret_tensor(buf713, (8, 784, 768), (602112, 768, 1), 0); del buf713  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf742, buf732, primals_427, mul_395, div_171, primals_63, buf745, 6272, 768, grid=grid(6272), stream=stream0)
        del div_171
        del primals_427
        del primals_63
        buf738 = reinterpret_tensor(buf729, (768, 49), (1, 768), 0); del buf729  # reuse
        buf740 = buf719; del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf732, mul_395, buf738, buf740, 37632, 128, grid=grid(37632), stream=stream0)
        del buf732
        del mul_395
        buf739 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf738, buf739, 768, 49, grid=grid(768), stream=stream0)
        buf741 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf740, buf741, 768, 49, grid=grid(768), stream=stream0)
        buf743 = reinterpret_tensor(buf740, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf742, convolution_35, buf743, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_35
        buf744 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf743, buf744, 768, 49, grid=grid(768), stream=stream0)
        buf746 = reinterpret_tensor(buf743, (768, 49), (1, 768), 0); del buf743  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf745, buf746, 37632, 128, grid=grid(37632), stream=stream0)
        buf747 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf746, buf747, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf748 = aten.convolution_backward(reinterpret_tensor(buf745, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_289, primals_425, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_289
        del primals_425
        buf749 = buf748[0]
        buf750 = buf748[1]
        del buf748
        buf751 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf749, buf751, 768, 6272, grid=grid(768), stream=stream0)
        buf752 = reinterpret_tensor(buf746, (768, 49), (49, 1), 0); del buf746  # reuse
        # Source Nodes: [x_296], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf749, convolution_34, unsqueeze_215, buf752, 37632, 128, grid=grid(37632), stream=stream0)
        buf753 = empty((768, ), device='cuda', dtype=torch.float32)
        buf754 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf752, squeeze_55, buf753, buf754, 768, 49, grid=grid(768), stream=stream0)
        buf755 = buf749; del buf749  # reuse
        # Source Nodes: [x_296], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf755, convolution_34, unsqueeze_215, buf753, squeeze_55, buf751, primals_423, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_34
        del primals_423
        del squeeze_55
        del unsqueeze_215
        buf756 = buf753; del buf753  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf755, buf756, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf757 = aten.convolution_backward(buf755, view_286, primals_421, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_421
        del view_286
        buf758 = buf757[0]
        buf759 = buf757[1]
        del buf757
        buf760 = reinterpret_tensor(buf752, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf758, primals_419, buf760, 37632, 128, grid=grid(37632), stream=stream0)
        buf761 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf760, buf761, 6272, 6, grid=grid(6272), stream=stream0)
        buf762 = reinterpret_tensor(buf760, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf758, primals_419, mul_382, buf762, 37632, 128, grid=grid(37632), stream=stream0)
        buf763 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf762, buf763, 6272, 6, grid=grid(6272), stream=stream0)
        buf764 = reinterpret_tensor(buf762, (768, 49), (49, 1), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf758, mul_382, buf764, 37632, 128, grid=grid(37632), stream=stream0)
        buf765 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf764, buf765, 768, 49, grid=grid(768), stream=stream0)
        buf766 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf758, buf766, 768, 6272, grid=grid(768), stream=stream0)
        buf767 = buf742; del buf742  # reuse
        buf772 = reinterpret_tensor(buf755, (8, 784, 768), (602112, 768, 1), 0); del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf767, div_172, buf758, primals_419, buf761, mul_382, buf763, primals_61, buf772, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_172
        del mul_382
        del primals_419
        buf768 = reinterpret_tensor(buf764, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf764  # reuse
        buf770 = reinterpret_tensor(buf738, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf738  # reuse
        # Source Nodes: [x_291], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf767, mm_15, primals_418, primals_61, buf768, buf770, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_15
        del primals_418
        del primals_61
        buf769 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf768, buf769, 768, 49, grid=grid(768), stream=stream0)
        buf771 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf770, buf771, 768, 49, grid=grid(768), stream=stream0)
        buf773 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (768, 6272), (1, 768), 0), view_284, out=buf773)
        del view_284
        buf774 = reinterpret_tensor(buf758, (6272, 768), (768, 1), 0); del buf758  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (6272, 768), (768, 1), 0), permute_512, out=buf774)
        del permute_512
        buf775 = reinterpret_tensor(buf772, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf774, buf775, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf776 = reinterpret_tensor(buf774, (128, 48, 784), (37632, 784, 1), 0); del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_515, reinterpret_tensor(buf775, (128, 48, 784), (37632, 784, 1), 0), out=buf776)
        del permute_515
        buf777 = reinterpret_tensor(buf704, (128, 48, 48), (2304, 48, 1), 0); del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf775, (128, 48, 784), (37632, 784, 1), 0), permute_516, out=buf777)
        del permute_516
        buf778 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf777, alias_100, buf778, 6144, 48, grid=grid(6144), stream=stream0)
        buf779 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf777, alias_100, buf778, bmm_30, buf779, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_30
        buf780 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf779, buf780, 16, 3, grid=grid(16), stream=stream0)
        buf781 = reinterpret_tensor(buf777, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf781, alias_100, buf778, primals_62, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_100
        del primals_62
        buf782 = reinterpret_tensor(buf775, (128, 784, 48), (37632, 48, 1), 0); del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_517, reinterpret_tensor(buf781, (128, 48, 48), (2304, 48, 1), 0), out=buf782)
        del permute_517
        buf783 = reinterpret_tensor(buf745, (128, 48, 784), (37632, 784, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf781, (128, 48, 48), (2304, 48, 1), 0), permute_518, out=buf783)
        del permute_518
        buf784 = reinterpret_tensor(buf709, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf709  # reuse
        # Source Nodes: [k_31], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf782, getitem_174, pow_65, buf784, 43008, 112, grid=grid(43008), stream=stream0)
        buf785 = buf778; del buf778  # reuse
        # Source Nodes: [k_31], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf784, buf785, 6144, 7, grid=grid(6144), stream=stream0)
        buf786 = reinterpret_tensor(buf784, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf784  # reuse
        # Source Nodes: [q_31], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf783, getitem_173, pow_63, buf786, 43008, 112, grid=grid(43008), stream=stream0)
        buf787 = buf708; del buf708  # reuse
        # Source Nodes: [q_31], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf786, buf787, 6144, 7, grid=grid(6144), stream=stream0)
        buf788 = reinterpret_tensor(buf712, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf783, pow_63, buf787, getitem_173, buf782, pow_65, buf785, getitem_174, buf776, buf788, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf776
        del getitem_173
        del getitem_174
        del pow_63
        del pow_65
        buf789 = reinterpret_tensor(buf711, (6272, 2304), (2304, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf788, buf789, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf790 = reinterpret_tensor(buf783, (6272, 768), (768, 1), 0); del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf789, permute_521, out=buf790)
        del permute_521
        buf791 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf789, (2304, 6272), (1, 2304), 0), view_274, out=buf791)
        del view_274
        buf792 = buf715; del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf789, buf792, 112896, 128, grid=grid(112896), stream=stream0)
        buf793 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf792, buf793, 2304, 49, grid=grid(2304), stream=stream0)
        buf800 = buf767; del buf767  # reuse
        buf803 = reinterpret_tensor(buf782, (8, 784, 768), (602112, 768, 1), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf800, buf790, primals_413, mul_378, div_181, primals_60, buf803, 6272, 768, grid=grid(6272), stream=stream0)
        del div_181
        del primals_413
        del primals_60
        buf796 = reinterpret_tensor(buf770, (768, 49), (1, 768), 0); del buf770  # reuse
        buf798 = reinterpret_tensor(buf768, (768, 49), (1, 768), 0); del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf790, mul_378, buf796, buf798, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_378
        buf797 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf796, buf797, 768, 49, grid=grid(768), stream=stream0)
        buf799 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf798, buf799, 768, 49, grid=grid(768), stream=stream0)
        buf801 = reinterpret_tensor(buf798, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf800, addmm_44, buf801, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_44
        buf802 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf801, buf802, 768, 49, grid=grid(768), stream=stream0)
        buf804 = reinterpret_tensor(buf731, (6272, 3072), (3072, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf803, (6272, 768), (768, 1), 0), permute_525, out=buf804)
        del permute_525
        buf805 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf803, (768, 6272), (1, 768), 0), view_272, out=buf805)
        del view_272
        buf806 = reinterpret_tensor(buf801, (1, 768, 49), (37632, 1, 768), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf803, buf806, 37632, 128, grid=grid(37632), stream=stream0)
        buf807 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf806, buf807, 768, 49, grid=grid(768), stream=stream0)
        buf808 = reinterpret_tensor(buf804, (8, 784, 3072), (2408448, 3072, 1), 0); del buf804  # reuse
        # Source Nodes: [x_283], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf808, addmm_43, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_43
        buf809 = reinterpret_tensor(buf803, (6272, 768), (768, 1), 0); del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf808, (6272, 3072), (3072, 1), 0), permute_529, out=buf809)
        del permute_529
        buf810 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf808, (3072, 6272), (1, 3072), 0), view_270, out=buf810)
        del view_270
        buf811 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf808, buf811, 150528, 128, grid=grid(150528), stream=stream0)
        buf812 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf811, buf812, 3072, 49, grid=grid(3072), stream=stream0)
        buf819 = buf800; del buf800  # reuse
        buf822 = reinterpret_tensor(buf790, (8, 784, 768), (602112, 768, 1), 0); del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf819, buf809, primals_407, mul_372, div_182, primals_59, buf822, 6272, 768, grid=grid(6272), stream=stream0)
        del div_182
        del primals_407
        del primals_59
        buf815 = reinterpret_tensor(buf806, (768, 49), (1, 768), 0); del buf806  # reuse
        buf817 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf809, mul_372, buf815, buf817, 37632, 128, grid=grid(37632), stream=stream0)
        del buf809
        del mul_372
        buf816 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf815, buf816, 768, 49, grid=grid(768), stream=stream0)
        buf818 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf817, buf818, 768, 49, grid=grid(768), stream=stream0)
        buf820 = reinterpret_tensor(buf817, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf819, convolution_33, buf820, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_33
        buf821 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf820, buf821, 768, 49, grid=grid(768), stream=stream0)
        buf823 = reinterpret_tensor(buf820, (768, 49), (1, 768), 0); del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf822, buf823, 37632, 128, grid=grid(37632), stream=stream0)
        buf824 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf823, buf824, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf825 = aten.convolution_backward(reinterpret_tensor(buf822, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_272, primals_405, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_272
        del primals_405
        buf826 = buf825[0]
        buf827 = buf825[1]
        del buf825
        buf828 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf826, buf828, 768, 6272, grid=grid(768), stream=stream0)
        buf829 = reinterpret_tensor(buf823, (768, 49), (49, 1), 0); del buf823  # reuse
        # Source Nodes: [x_277], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf826, convolution_32, unsqueeze_227, buf829, 37632, 128, grid=grid(37632), stream=stream0)
        buf830 = empty((768, ), device='cuda', dtype=torch.float32)
        buf831 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf829, squeeze_52, buf830, buf831, 768, 49, grid=grid(768), stream=stream0)
        buf832 = buf826; del buf826  # reuse
        # Source Nodes: [x_277], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf832, convolution_32, unsqueeze_227, buf830, squeeze_52, buf828, primals_403, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_32
        del primals_403
        del squeeze_52
        del unsqueeze_227
        buf833 = buf830; del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf832, buf833, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf834 = aten.convolution_backward(buf832, view_268, primals_401, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_401
        del view_268
        buf835 = buf834[0]
        buf836 = buf834[1]
        del buf834
        buf837 = reinterpret_tensor(buf829, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf829  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf835, primals_399, buf837, 37632, 128, grid=grid(37632), stream=stream0)
        buf838 = buf763; del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf837, buf838, 6272, 6, grid=grid(6272), stream=stream0)
        buf839 = reinterpret_tensor(buf837, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf837  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf835, primals_399, mul_359, buf839, 37632, 128, grid=grid(37632), stream=stream0)
        buf840 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf839, buf840, 6272, 6, grid=grid(6272), stream=stream0)
        buf841 = reinterpret_tensor(buf839, (768, 49), (49, 1), 0); del buf839  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf835, mul_359, buf841, 37632, 128, grid=grid(37632), stream=stream0)
        buf842 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf841, buf842, 768, 49, grid=grid(768), stream=stream0)
        buf843 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf835, buf843, 768, 6272, grid=grid(768), stream=stream0)
        buf844 = buf819; del buf819  # reuse
        buf849 = reinterpret_tensor(buf832, (8, 784, 768), (602112, 768, 1), 0); del buf832  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf844, div_183, buf835, primals_399, buf838, mul_359, buf840, primals_57, buf849, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_183
        del mul_359
        del primals_399
        buf845 = reinterpret_tensor(buf841, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf841  # reuse
        buf847 = reinterpret_tensor(buf815, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf815  # reuse
        # Source Nodes: [x_272], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf844, mm_14, primals_398, primals_57, buf845, buf847, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_14
        del primals_398
        del primals_57
        buf846 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf845, buf846, 768, 49, grid=grid(768), stream=stream0)
        buf848 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf847, buf848, 768, 49, grid=grid(768), stream=stream0)
        buf850 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (768, 6272), (1, 768), 0), view_266, out=buf850)
        del view_266
        buf851 = reinterpret_tensor(buf835, (6272, 768), (768, 1), 0); del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf849, (6272, 768), (768, 1), 0), permute_537, out=buf851)
        del permute_537
        buf852 = reinterpret_tensor(buf849, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf849  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf851, buf852, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf853 = reinterpret_tensor(buf851, (128, 48, 784), (37632, 784, 1), 0); del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_540, reinterpret_tensor(buf852, (128, 48, 784), (37632, 784, 1), 0), out=buf853)
        del permute_540
        buf854 = reinterpret_tensor(buf781, (128, 48, 48), (2304, 48, 1), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf852, (128, 48, 784), (37632, 784, 1), 0), permute_541, out=buf854)
        del permute_541
        buf855 = buf787; del buf787  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf854, alias_103, buf855, 6144, 48, grid=grid(6144), stream=stream0)
        buf856 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf854, alias_103, buf855, bmm_28, buf856, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_28
        buf857 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf856, buf857, 16, 3, grid=grid(16), stream=stream0)
        buf858 = reinterpret_tensor(buf854, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf854  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf858, alias_103, buf855, primals_58, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_103
        del primals_58
        buf859 = reinterpret_tensor(buf852, (128, 784, 48), (37632, 48, 1), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_542, reinterpret_tensor(buf858, (128, 48, 48), (2304, 48, 1), 0), out=buf859)
        del permute_542
        buf860 = reinterpret_tensor(buf822, (128, 48, 784), (37632, 784, 1), 0); del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf858, (128, 48, 48), (2304, 48, 1), 0), permute_543, out=buf860)
        del permute_543
        buf861 = reinterpret_tensor(buf786, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf786  # reuse
        # Source Nodes: [k_29], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf859, getitem_163, pow_61, buf861, 43008, 112, grid=grid(43008), stream=stream0)
        buf862 = buf855; del buf855  # reuse
        # Source Nodes: [k_29], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf861, buf862, 6144, 7, grid=grid(6144), stream=stream0)
        buf863 = reinterpret_tensor(buf861, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf861  # reuse
        # Source Nodes: [q_29], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf860, getitem_162, pow_59, buf863, 43008, 112, grid=grid(43008), stream=stream0)
        buf864 = buf785; del buf785  # reuse
        # Source Nodes: [q_29], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf863, buf864, 6144, 7, grid=grid(6144), stream=stream0)
        buf865 = reinterpret_tensor(buf789, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf860, pow_59, buf864, getitem_162, buf859, pow_61, buf862, getitem_163, buf853, buf865, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf853
        del getitem_162
        del getitem_163
        del pow_59
        del pow_61
        buf866 = reinterpret_tensor(buf788, (6272, 2304), (2304, 1), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf865, buf866, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf867 = reinterpret_tensor(buf860, (6272, 768), (768, 1), 0); del buf860  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf866, permute_546, out=buf867)
        del permute_546
        buf868 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf866, (2304, 6272), (1, 2304), 0), view_256, out=buf868)
        del view_256
        buf869 = buf792; del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf866, buf869, 112896, 128, grid=grid(112896), stream=stream0)
        buf870 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf869, buf870, 2304, 49, grid=grid(2304), stream=stream0)
        buf877 = buf844; del buf844  # reuse
        buf880 = reinterpret_tensor(buf859, (8, 784, 768), (602112, 768, 1), 0); del buf859  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf877, buf867, primals_393, mul_355, div_192, primals_56, buf880, 6272, 768, grid=grid(6272), stream=stream0)
        del div_192
        del primals_393
        del primals_56
        buf873 = reinterpret_tensor(buf847, (768, 49), (1, 768), 0); del buf847  # reuse
        buf875 = reinterpret_tensor(buf845, (768, 49), (1, 768), 0); del buf845  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf867, mul_355, buf873, buf875, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_355
        buf874 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf873, buf874, 768, 49, grid=grid(768), stream=stream0)
        buf876 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf875, buf876, 768, 49, grid=grid(768), stream=stream0)
        buf878 = reinterpret_tensor(buf875, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf875  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf877, addmm_41, buf878, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_41
        buf879 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf878, buf879, 768, 49, grid=grid(768), stream=stream0)
        buf881 = reinterpret_tensor(buf808, (6272, 3072), (3072, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf880, (6272, 768), (768, 1), 0), permute_550, out=buf881)
        del permute_550
        buf882 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf880, (768, 6272), (1, 768), 0), view_254, out=buf882)
        del view_254
        buf883 = reinterpret_tensor(buf878, (1, 768, 49), (37632, 1, 768), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf880, buf883, 37632, 128, grid=grid(37632), stream=stream0)
        buf884 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf883, buf884, 768, 49, grid=grid(768), stream=stream0)
        buf885 = reinterpret_tensor(buf881, (8, 784, 3072), (2408448, 3072, 1), 0); del buf881  # reuse
        # Source Nodes: [x_264], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf885, addmm_40, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_40
        buf886 = reinterpret_tensor(buf880, (6272, 768), (768, 1), 0); del buf880  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (6272, 3072), (3072, 1), 0), permute_554, out=buf886)
        del permute_554
        buf887 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf885, (3072, 6272), (1, 3072), 0), view_252, out=buf887)
        del view_252
        buf888 = buf811; del buf811  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf885, buf888, 150528, 128, grid=grid(150528), stream=stream0)
        buf889 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf888, buf889, 3072, 49, grid=grid(3072), stream=stream0)
        buf896 = buf877; del buf877  # reuse
        buf899 = reinterpret_tensor(buf867, (8, 784, 768), (602112, 768, 1), 0); del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf896, buf886, primals_387, mul_349, div_193, primals_55, buf899, 6272, 768, grid=grid(6272), stream=stream0)
        del div_193
        del primals_387
        del primals_55
        buf892 = reinterpret_tensor(buf883, (768, 49), (1, 768), 0); del buf883  # reuse
        buf894 = buf873; del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf886, mul_349, buf892, buf894, 37632, 128, grid=grid(37632), stream=stream0)
        del buf886
        del mul_349
        buf893 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf892, buf893, 768, 49, grid=grid(768), stream=stream0)
        buf895 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf894, buf895, 768, 49, grid=grid(768), stream=stream0)
        buf897 = reinterpret_tensor(buf894, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf894  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf896, convolution_31, buf897, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_31
        buf898 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf897, buf898, 768, 49, grid=grid(768), stream=stream0)
        buf900 = reinterpret_tensor(buf897, (768, 49), (1, 768), 0); del buf897  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf899, buf900, 37632, 128, grid=grid(37632), stream=stream0)
        buf901 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf900, buf901, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf902 = aten.convolution_backward(reinterpret_tensor(buf899, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_255, primals_385, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_255
        del primals_385
        buf903 = buf902[0]
        buf904 = buf902[1]
        del buf902
        buf905 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf903, buf905, 768, 6272, grid=grid(768), stream=stream0)
        buf906 = reinterpret_tensor(buf900, (768, 49), (49, 1), 0); del buf900  # reuse
        # Source Nodes: [x_258], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf903, convolution_30, unsqueeze_239, buf906, 37632, 128, grid=grid(37632), stream=stream0)
        buf907 = empty((768, ), device='cuda', dtype=torch.float32)
        buf908 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf906, squeeze_49, buf907, buf908, 768, 49, grid=grid(768), stream=stream0)
        buf909 = buf903; del buf903  # reuse
        # Source Nodes: [x_258], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf909, convolution_30, unsqueeze_239, buf907, squeeze_49, buf905, primals_383, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_30
        del primals_383
        del squeeze_49
        del unsqueeze_239
        buf910 = buf907; del buf907  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf909, buf910, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf911 = aten.convolution_backward(buf909, view_250, primals_381, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_381
        del view_250
        buf912 = buf911[0]
        buf913 = buf911[1]
        del buf911
        buf914 = reinterpret_tensor(buf906, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf906  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf912, primals_379, buf914, 37632, 128, grid=grid(37632), stream=stream0)
        buf915 = buf840; del buf840  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf914, buf915, 6272, 6, grid=grid(6272), stream=stream0)
        buf916 = reinterpret_tensor(buf914, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf914  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf912, primals_379, mul_336, buf916, 37632, 128, grid=grid(37632), stream=stream0)
        buf917 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf916, buf917, 6272, 6, grid=grid(6272), stream=stream0)
        buf918 = reinterpret_tensor(buf916, (768, 49), (49, 1), 0); del buf916  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf912, mul_336, buf918, 37632, 128, grid=grid(37632), stream=stream0)
        buf919 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf918, buf919, 768, 49, grid=grid(768), stream=stream0)
        buf920 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf912, buf920, 768, 6272, grid=grid(768), stream=stream0)
        buf921 = buf896; del buf896  # reuse
        buf926 = reinterpret_tensor(buf909, (8, 784, 768), (602112, 768, 1), 0); del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf921, div_194, buf912, primals_379, buf915, mul_336, buf917, primals_53, buf926, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_194
        del mul_336
        del primals_379
        buf922 = reinterpret_tensor(buf918, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf918  # reuse
        buf924 = reinterpret_tensor(buf892, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf892  # reuse
        # Source Nodes: [x_253], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf921, mm_13, primals_378, primals_53, buf922, buf924, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_13
        del primals_378
        del primals_53
        buf923 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf922, buf923, 768, 49, grid=grid(768), stream=stream0)
        buf925 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf924, buf925, 768, 49, grid=grid(768), stream=stream0)
        buf927 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (768, 6272), (1, 768), 0), view_248, out=buf927)
        del view_248
        buf928 = reinterpret_tensor(buf912, (6272, 768), (768, 1), 0); del buf912  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (6272, 768), (768, 1), 0), permute_562, out=buf928)
        del permute_562
        buf929 = reinterpret_tensor(buf926, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf928, buf929, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf930 = reinterpret_tensor(buf928, (128, 48, 784), (37632, 784, 1), 0); del buf928  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_565, reinterpret_tensor(buf929, (128, 48, 784), (37632, 784, 1), 0), out=buf930)
        del permute_565
        buf931 = reinterpret_tensor(buf858, (128, 48, 48), (2304, 48, 1), 0); del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf929, (128, 48, 784), (37632, 784, 1), 0), permute_566, out=buf931)
        del permute_566
        buf932 = buf864; del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf931, alias_106, buf932, 6144, 48, grid=grid(6144), stream=stream0)
        buf933 = buf856; del buf856  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf931, alias_106, buf932, bmm_26, buf933, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_26
        buf934 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf933, buf934, 16, 3, grid=grid(16), stream=stream0)
        buf935 = reinterpret_tensor(buf931, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf935, alias_106, buf932, primals_54, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_106
        del primals_54
        buf936 = reinterpret_tensor(buf929, (128, 784, 48), (37632, 48, 1), 0); del buf929  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_567, reinterpret_tensor(buf935, (128, 48, 48), (2304, 48, 1), 0), out=buf936)
        del permute_567
        buf937 = reinterpret_tensor(buf899, (128, 48, 784), (37632, 784, 1), 0); del buf899  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf935, (128, 48, 48), (2304, 48, 1), 0), permute_568, out=buf937)
        del permute_568
        buf938 = reinterpret_tensor(buf863, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf863  # reuse
        # Source Nodes: [k_27], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf936, getitem_152, pow_57, buf938, 43008, 112, grid=grid(43008), stream=stream0)
        buf939 = buf932; del buf932  # reuse
        # Source Nodes: [k_27], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf938, buf939, 6144, 7, grid=grid(6144), stream=stream0)
        buf940 = reinterpret_tensor(buf938, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf938  # reuse
        # Source Nodes: [q_27], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf937, getitem_151, pow_55, buf940, 43008, 112, grid=grid(43008), stream=stream0)
        buf941 = buf862; del buf862  # reuse
        # Source Nodes: [q_27], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf940, buf941, 6144, 7, grid=grid(6144), stream=stream0)
        buf942 = reinterpret_tensor(buf866, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf937, pow_55, buf941, getitem_151, buf936, pow_57, buf939, getitem_152, buf930, buf942, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf930
        del getitem_151
        del getitem_152
        del pow_55
        del pow_57
        buf943 = reinterpret_tensor(buf865, (6272, 2304), (2304, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf942, buf943, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf944 = reinterpret_tensor(buf937, (6272, 768), (768, 1), 0); del buf937  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf943, permute_571, out=buf944)
        del permute_571
        buf945 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf943, (2304, 6272), (1, 2304), 0), view_238, out=buf945)
        del view_238
        buf946 = buf869; del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf943, buf946, 112896, 128, grid=grid(112896), stream=stream0)
        buf947 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf946, buf947, 2304, 49, grid=grid(2304), stream=stream0)
        buf954 = buf921; del buf921  # reuse
        buf957 = reinterpret_tensor(buf936, (8, 784, 768), (602112, 768, 1), 0); del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf954, buf944, primals_373, mul_332, div_203, primals_52, buf957, 6272, 768, grid=grid(6272), stream=stream0)
        del div_203
        del primals_373
        del primals_52
        buf950 = reinterpret_tensor(buf924, (768, 49), (1, 768), 0); del buf924  # reuse
        buf952 = reinterpret_tensor(buf922, (768, 49), (1, 768), 0); del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf944, mul_332, buf950, buf952, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_332
        buf951 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf950, buf951, 768, 49, grid=grid(768), stream=stream0)
        buf953 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf952, buf953, 768, 49, grid=grid(768), stream=stream0)
        buf955 = reinterpret_tensor(buf952, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf952  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf954, addmm_38, buf955, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_38
        buf956 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf955, buf956, 768, 49, grid=grid(768), stream=stream0)
        buf958 = reinterpret_tensor(buf885, (6272, 3072), (3072, 1), 0); del buf885  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf957, (6272, 768), (768, 1), 0), permute_575, out=buf958)
        del permute_575
        buf959 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf957, (768, 6272), (1, 768), 0), view_236, out=buf959)
        del view_236
        buf960 = reinterpret_tensor(buf955, (1, 768, 49), (37632, 1, 768), 0); del buf955  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf957, buf960, 37632, 128, grid=grid(37632), stream=stream0)
        buf961 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf960, buf961, 768, 49, grid=grid(768), stream=stream0)
        buf962 = reinterpret_tensor(buf958, (8, 784, 3072), (2408448, 3072, 1), 0); del buf958  # reuse
        # Source Nodes: [x_245], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf962, addmm_37, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_37
        buf963 = reinterpret_tensor(buf957, (6272, 768), (768, 1), 0); del buf957  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf962, (6272, 3072), (3072, 1), 0), permute_579, out=buf963)
        del permute_579
        buf964 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf962, (3072, 6272), (1, 3072), 0), view_234, out=buf964)
        del view_234
        buf965 = buf888; del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf962, buf965, 150528, 128, grid=grid(150528), stream=stream0)
        buf966 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf965, buf966, 3072, 49, grid=grid(3072), stream=stream0)
        buf973 = buf954; del buf954  # reuse
        buf976 = reinterpret_tensor(buf944, (8, 784, 768), (602112, 768, 1), 0); del buf944  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf973, buf963, primals_367, mul_326, div_204, primals_51, buf976, 6272, 768, grid=grid(6272), stream=stream0)
        del div_204
        del primals_367
        del primals_51
        buf969 = reinterpret_tensor(buf960, (768, 49), (1, 768), 0); del buf960  # reuse
        buf971 = buf950; del buf950  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf963, mul_326, buf969, buf971, 37632, 128, grid=grid(37632), stream=stream0)
        del buf963
        del mul_326
        buf970 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf969, buf970, 768, 49, grid=grid(768), stream=stream0)
        buf972 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf971, buf972, 768, 49, grid=grid(768), stream=stream0)
        buf974 = reinterpret_tensor(buf971, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf971  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf973, convolution_29, buf974, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_29
        buf975 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf974, buf975, 768, 49, grid=grid(768), stream=stream0)
        buf977 = reinterpret_tensor(buf974, (768, 49), (1, 768), 0); del buf974  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf976, buf977, 37632, 128, grid=grid(37632), stream=stream0)
        buf978 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf977, buf978, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf979 = aten.convolution_backward(reinterpret_tensor(buf976, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_238, primals_365, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_238
        del primals_365
        buf980 = buf979[0]
        buf981 = buf979[1]
        del buf979
        buf982 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf980, buf982, 768, 6272, grid=grid(768), stream=stream0)
        buf983 = reinterpret_tensor(buf977, (768, 49), (49, 1), 0); del buf977  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf980, convolution_28, unsqueeze_251, buf983, 37632, 128, grid=grid(37632), stream=stream0)
        buf984 = empty((768, ), device='cuda', dtype=torch.float32)
        buf985 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf983, squeeze_46, buf984, buf985, 768, 49, grid=grid(768), stream=stream0)
        buf986 = buf980; del buf980  # reuse
        # Source Nodes: [x_239], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf986, convolution_28, unsqueeze_251, buf984, squeeze_46, buf982, primals_363, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_28
        del primals_363
        del squeeze_46
        del unsqueeze_251
        buf987 = buf984; del buf984  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf986, buf987, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf988 = aten.convolution_backward(buf986, view_232, primals_361, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_361
        del view_232
        buf989 = buf988[0]
        buf990 = buf988[1]
        del buf988
        buf991 = reinterpret_tensor(buf983, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf989, primals_359, buf991, 37632, 128, grid=grid(37632), stream=stream0)
        buf992 = buf917; del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf991, buf992, 6272, 6, grid=grid(6272), stream=stream0)
        buf993 = reinterpret_tensor(buf991, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf991  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf989, primals_359, mul_313, buf993, 37632, 128, grid=grid(37632), stream=stream0)
        buf994 = buf915; del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf993, buf994, 6272, 6, grid=grid(6272), stream=stream0)
        buf995 = reinterpret_tensor(buf993, (768, 49), (49, 1), 0); del buf993  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf989, mul_313, buf995, 37632, 128, grid=grid(37632), stream=stream0)
        buf996 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf995, buf996, 768, 49, grid=grid(768), stream=stream0)
        buf997 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf989, buf997, 768, 6272, grid=grid(768), stream=stream0)
        buf998 = buf973; del buf973  # reuse
        buf1003 = reinterpret_tensor(buf986, (8, 784, 768), (602112, 768, 1), 0); del buf986  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf998, div_205, buf989, primals_359, buf992, mul_313, buf994, primals_49, buf1003, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_205
        del mul_313
        del primals_359
        buf999 = reinterpret_tensor(buf995, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf995  # reuse
        buf1001 = reinterpret_tensor(buf969, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf969  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf998, mm_12, primals_358, primals_49, buf999, buf1001, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_12
        del primals_358
        del primals_49
        buf1000 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf999, buf1000, 768, 49, grid=grid(768), stream=stream0)
        buf1002 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1001, buf1002, 768, 49, grid=grid(768), stream=stream0)
        buf1004 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1003, (768, 6272), (1, 768), 0), view_230, out=buf1004)
        del view_230
        buf1005 = reinterpret_tensor(buf989, (6272, 768), (768, 1), 0); del buf989  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1003, (6272, 768), (768, 1), 0), permute_587, out=buf1005)
        del permute_587
        buf1006 = reinterpret_tensor(buf1003, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1003  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1005, buf1006, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1007 = reinterpret_tensor(buf1005, (128, 48, 784), (37632, 784, 1), 0); del buf1005  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_590, reinterpret_tensor(buf1006, (128, 48, 784), (37632, 784, 1), 0), out=buf1007)
        del permute_590
        buf1008 = reinterpret_tensor(buf935, (128, 48, 48), (2304, 48, 1), 0); del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1006, (128, 48, 784), (37632, 784, 1), 0), permute_591, out=buf1008)
        del permute_591
        buf1009 = buf941; del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1008, alias_109, buf1009, 6144, 48, grid=grid(6144), stream=stream0)
        buf1010 = buf933; del buf933  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1008, alias_109, buf1009, bmm_24, buf1010, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_24
        buf1011 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1010, buf1011, 16, 3, grid=grid(16), stream=stream0)
        buf1012 = reinterpret_tensor(buf1008, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1008  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1012, alias_109, buf1009, primals_50, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_109
        del primals_50
        buf1013 = reinterpret_tensor(buf1006, (128, 784, 48), (37632, 48, 1), 0); del buf1006  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_592, reinterpret_tensor(buf1012, (128, 48, 48), (2304, 48, 1), 0), out=buf1013)
        del permute_592
        buf1014 = reinterpret_tensor(buf976, (128, 48, 784), (37632, 784, 1), 0); del buf976  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1012, (128, 48, 48), (2304, 48, 1), 0), permute_593, out=buf1014)
        del permute_593
        buf1015 = reinterpret_tensor(buf940, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf940  # reuse
        # Source Nodes: [k_25], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1013, getitem_141, pow_53, buf1015, 43008, 112, grid=grid(43008), stream=stream0)
        buf1016 = buf1009; del buf1009  # reuse
        # Source Nodes: [k_25], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1015, buf1016, 6144, 7, grid=grid(6144), stream=stream0)
        buf1017 = reinterpret_tensor(buf1015, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1015  # reuse
        # Source Nodes: [q_25], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1014, getitem_140, pow_51, buf1017, 43008, 112, grid=grid(43008), stream=stream0)
        buf1018 = buf939; del buf939  # reuse
        # Source Nodes: [q_25], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1017, buf1018, 6144, 7, grid=grid(6144), stream=stream0)
        buf1019 = reinterpret_tensor(buf943, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf943  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1014, pow_51, buf1018, getitem_140, buf1013, pow_53, buf1016, getitem_141, buf1007, buf1019, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1007
        del getitem_140
        del getitem_141
        del pow_51
        del pow_53
        buf1020 = reinterpret_tensor(buf942, (6272, 2304), (2304, 1), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1019, buf1020, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1021 = reinterpret_tensor(buf1014, (6272, 768), (768, 1), 0); del buf1014  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1020, permute_596, out=buf1021)
        del permute_596
        buf1022 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1020, (2304, 6272), (1, 2304), 0), view_220, out=buf1022)
        del view_220
        buf1023 = buf946; del buf946  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1020, buf1023, 112896, 128, grid=grid(112896), stream=stream0)
        buf1024 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1023, buf1024, 2304, 49, grid=grid(2304), stream=stream0)
        buf1031 = buf998; del buf998  # reuse
        buf1034 = reinterpret_tensor(buf1013, (8, 784, 768), (602112, 768, 1), 0); del buf1013  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1031, buf1021, primals_353, mul_309, div_214, primals_48, buf1034, 6272, 768, grid=grid(6272), stream=stream0)
        del div_214
        del primals_353
        del primals_48
        buf1027 = reinterpret_tensor(buf1001, (768, 49), (1, 768), 0); del buf1001  # reuse
        buf1029 = reinterpret_tensor(buf999, (768, 49), (1, 768), 0); del buf999  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1021, mul_309, buf1027, buf1029, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_309
        buf1028 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1027, buf1028, 768, 49, grid=grid(768), stream=stream0)
        buf1030 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1029, buf1030, 768, 49, grid=grid(768), stream=stream0)
        buf1032 = reinterpret_tensor(buf1029, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1029  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1031, addmm_35, buf1032, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_35
        buf1033 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1032, buf1033, 768, 49, grid=grid(768), stream=stream0)
        buf1035 = reinterpret_tensor(buf962, (6272, 3072), (3072, 1), 0); del buf962  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1034, (6272, 768), (768, 1), 0), permute_600, out=buf1035)
        del permute_600
        buf1036 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1034, (768, 6272), (1, 768), 0), view_218, out=buf1036)
        del view_218
        buf1037 = reinterpret_tensor(buf1032, (1, 768, 49), (37632, 1, 768), 0); del buf1032  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1034, buf1037, 37632, 128, grid=grid(37632), stream=stream0)
        buf1038 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1037, buf1038, 768, 49, grid=grid(768), stream=stream0)
        buf1039 = reinterpret_tensor(buf1035, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1035  # reuse
        # Source Nodes: [x_226], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1039, addmm_34, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_34
        buf1040 = reinterpret_tensor(buf1034, (6272, 768), (768, 1), 0); del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1039, (6272, 3072), (3072, 1), 0), permute_604, out=buf1040)
        del permute_604
        buf1041 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1039, (3072, 6272), (1, 3072), 0), view_216, out=buf1041)
        del view_216
        buf1042 = buf965; del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1039, buf1042, 150528, 128, grid=grid(150528), stream=stream0)
        buf1043 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1042, buf1043, 3072, 49, grid=grid(3072), stream=stream0)
        buf1050 = buf1031; del buf1031  # reuse
        buf1053 = reinterpret_tensor(buf1021, (8, 784, 768), (602112, 768, 1), 0); del buf1021  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1050, buf1040, primals_347, mul_303, div_215, primals_47, buf1053, 6272, 768, grid=grid(6272), stream=stream0)
        del div_215
        del primals_347
        del primals_47
        buf1046 = reinterpret_tensor(buf1037, (768, 49), (1, 768), 0); del buf1037  # reuse
        buf1048 = buf1027; del buf1027  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1040, mul_303, buf1046, buf1048, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1040
        del mul_303
        buf1047 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1046, buf1047, 768, 49, grid=grid(768), stream=stream0)
        buf1049 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1048, buf1049, 768, 49, grid=grid(768), stream=stream0)
        buf1051 = reinterpret_tensor(buf1048, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1048  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1050, convolution_27, buf1051, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_27
        buf1052 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1051, buf1052, 768, 49, grid=grid(768), stream=stream0)
        buf1054 = reinterpret_tensor(buf1051, (768, 49), (1, 768), 0); del buf1051  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1053, buf1054, 37632, 128, grid=grid(37632), stream=stream0)
        buf1055 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1054, buf1055, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1056 = aten.convolution_backward(reinterpret_tensor(buf1053, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_221, primals_345, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_221
        del primals_345
        buf1057 = buf1056[0]
        buf1058 = buf1056[1]
        del buf1056
        buf1059 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1057, buf1059, 768, 6272, grid=grid(768), stream=stream0)
        buf1060 = reinterpret_tensor(buf1054, (768, 49), (49, 1), 0); del buf1054  # reuse
        # Source Nodes: [x_220], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1057, convolution_26, unsqueeze_263, buf1060, 37632, 128, grid=grid(37632), stream=stream0)
        buf1061 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1062 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1060, squeeze_43, buf1061, buf1062, 768, 49, grid=grid(768), stream=stream0)
        buf1063 = buf1057; del buf1057  # reuse
        # Source Nodes: [x_220], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1063, convolution_26, unsqueeze_263, buf1061, squeeze_43, buf1059, primals_343, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_26
        del primals_343
        del squeeze_43
        del unsqueeze_263
        buf1064 = buf1061; del buf1061  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1063, buf1064, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1065 = aten.convolution_backward(buf1063, view_214, primals_341, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_341
        del view_214
        buf1066 = buf1065[0]
        buf1067 = buf1065[1]
        del buf1065
        buf1068 = reinterpret_tensor(buf1060, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1060  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1066, primals_339, buf1068, 37632, 128, grid=grid(37632), stream=stream0)
        buf1069 = buf994; del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1068, buf1069, 6272, 6, grid=grid(6272), stream=stream0)
        buf1070 = reinterpret_tensor(buf1068, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1068  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1066, primals_339, mul_290, buf1070, 37632, 128, grid=grid(37632), stream=stream0)
        buf1071 = buf992; del buf992  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1070, buf1071, 6272, 6, grid=grid(6272), stream=stream0)
        buf1072 = reinterpret_tensor(buf1070, (768, 49), (49, 1), 0); del buf1070  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1066, mul_290, buf1072, 37632, 128, grid=grid(37632), stream=stream0)
        buf1073 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1072, buf1073, 768, 49, grid=grid(768), stream=stream0)
        buf1074 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1066, buf1074, 768, 6272, grid=grid(768), stream=stream0)
        buf1075 = buf1050; del buf1050  # reuse
        buf1080 = reinterpret_tensor(buf1063, (8, 784, 768), (602112, 768, 1), 0); del buf1063  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1075, div_216, buf1066, primals_339, buf1069, mul_290, buf1071, primals_45, buf1080, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_216
        del mul_290
        del primals_339
        buf1076 = reinterpret_tensor(buf1072, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1072  # reuse
        buf1078 = reinterpret_tensor(buf1046, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1046  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1075, mm_11, primals_338, primals_45, buf1076, buf1078, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_11
        del primals_338
        del primals_45
        buf1077 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1076, buf1077, 768, 49, grid=grid(768), stream=stream0)
        buf1079 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1078, buf1079, 768, 49, grid=grid(768), stream=stream0)
        buf1081 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (768, 6272), (1, 768), 0), view_212, out=buf1081)
        del view_212
        buf1082 = reinterpret_tensor(buf1066, (6272, 768), (768, 1), 0); del buf1066  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1080, (6272, 768), (768, 1), 0), permute_612, out=buf1082)
        del permute_612
        buf1083 = reinterpret_tensor(buf1080, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1080  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1082, buf1083, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1084 = reinterpret_tensor(buf1082, (128, 48, 784), (37632, 784, 1), 0); del buf1082  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_615, reinterpret_tensor(buf1083, (128, 48, 784), (37632, 784, 1), 0), out=buf1084)
        del permute_615
        buf1085 = reinterpret_tensor(buf1012, (128, 48, 48), (2304, 48, 1), 0); del buf1012  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1083, (128, 48, 784), (37632, 784, 1), 0), permute_616, out=buf1085)
        del permute_616
        buf1086 = buf1018; del buf1018  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1085, alias_112, buf1086, 6144, 48, grid=grid(6144), stream=stream0)
        buf1087 = buf1010; del buf1010  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1085, alias_112, buf1086, bmm_22, buf1087, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_22
        buf1088 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1087, buf1088, 16, 3, grid=grid(16), stream=stream0)
        buf1089 = reinterpret_tensor(buf1085, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1085  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1089, alias_112, buf1086, primals_46, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_112
        del primals_46
        buf1090 = reinterpret_tensor(buf1083, (128, 784, 48), (37632, 48, 1), 0); del buf1083  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_617, reinterpret_tensor(buf1089, (128, 48, 48), (2304, 48, 1), 0), out=buf1090)
        del permute_617
        buf1091 = reinterpret_tensor(buf1053, (128, 48, 784), (37632, 784, 1), 0); del buf1053  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1089, (128, 48, 48), (2304, 48, 1), 0), permute_618, out=buf1091)
        del permute_618
        buf1092 = reinterpret_tensor(buf1017, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1017  # reuse
        # Source Nodes: [k_23], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1090, getitem_130, pow_49, buf1092, 43008, 112, grid=grid(43008), stream=stream0)
        buf1093 = buf1086; del buf1086  # reuse
        # Source Nodes: [k_23], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1092, buf1093, 6144, 7, grid=grid(6144), stream=stream0)
        buf1094 = reinterpret_tensor(buf1092, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1092  # reuse
        # Source Nodes: [q_23], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1091, getitem_129, pow_47, buf1094, 43008, 112, grid=grid(43008), stream=stream0)
        buf1095 = buf1016; del buf1016  # reuse
        # Source Nodes: [q_23], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1094, buf1095, 6144, 7, grid=grid(6144), stream=stream0)
        buf1096 = reinterpret_tensor(buf1020, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1020  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1091, pow_47, buf1095, getitem_129, buf1090, pow_49, buf1093, getitem_130, buf1084, buf1096, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1084
        del getitem_129
        del getitem_130
        del pow_47
        del pow_49
        buf1097 = reinterpret_tensor(buf1019, (6272, 2304), (2304, 1), 0); del buf1019  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1096, buf1097, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1098 = reinterpret_tensor(buf1091, (6272, 768), (768, 1), 0); del buf1091  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1097, permute_621, out=buf1098)
        del permute_621
        buf1099 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1097, (2304, 6272), (1, 2304), 0), view_202, out=buf1099)
        del view_202
        buf1100 = buf1023; del buf1023  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1097, buf1100, 112896, 128, grid=grid(112896), stream=stream0)
        buf1101 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1100, buf1101, 2304, 49, grid=grid(2304), stream=stream0)
        buf1108 = buf1075; del buf1075  # reuse
        buf1111 = reinterpret_tensor(buf1090, (8, 784, 768), (602112, 768, 1), 0); del buf1090  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1108, buf1098, primals_333, mul_286, div_225, primals_44, buf1111, 6272, 768, grid=grid(6272), stream=stream0)
        del div_225
        del primals_333
        del primals_44
        buf1104 = reinterpret_tensor(buf1078, (768, 49), (1, 768), 0); del buf1078  # reuse
        buf1106 = reinterpret_tensor(buf1076, (768, 49), (1, 768), 0); del buf1076  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1098, mul_286, buf1104, buf1106, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_286
        buf1105 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1104, buf1105, 768, 49, grid=grid(768), stream=stream0)
        buf1107 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1106, buf1107, 768, 49, grid=grid(768), stream=stream0)
        buf1109 = reinterpret_tensor(buf1106, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1106  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1108, addmm_32, buf1109, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_32
        buf1110 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1109, buf1110, 768, 49, grid=grid(768), stream=stream0)
        buf1112 = reinterpret_tensor(buf1039, (6272, 3072), (3072, 1), 0); del buf1039  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1111, (6272, 768), (768, 1), 0), permute_625, out=buf1112)
        del permute_625
        buf1113 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1111, (768, 6272), (1, 768), 0), view_200, out=buf1113)
        del view_200
        buf1114 = reinterpret_tensor(buf1109, (1, 768, 49), (37632, 1, 768), 0); del buf1109  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1111, buf1114, 37632, 128, grid=grid(37632), stream=stream0)
        buf1115 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1114, buf1115, 768, 49, grid=grid(768), stream=stream0)
        buf1116 = reinterpret_tensor(buf1112, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1112  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1116, addmm_31, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_31
        buf1117 = reinterpret_tensor(buf1111, (6272, 768), (768, 1), 0); del buf1111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1116, (6272, 3072), (3072, 1), 0), permute_629, out=buf1117)
        del permute_629
        buf1118 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1116, (3072, 6272), (1, 3072), 0), view_198, out=buf1118)
        del view_198
        buf1119 = buf1042; del buf1042  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1116, buf1119, 150528, 128, grid=grid(150528), stream=stream0)
        buf1120 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1119, buf1120, 3072, 49, grid=grid(3072), stream=stream0)
        buf1127 = buf1108; del buf1108  # reuse
        buf1130 = reinterpret_tensor(buf1098, (8, 784, 768), (602112, 768, 1), 0); del buf1098  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1127, buf1117, primals_327, mul_280, div_226, primals_43, buf1130, 6272, 768, grid=grid(6272), stream=stream0)
        del div_226
        del primals_327
        del primals_43
        buf1123 = reinterpret_tensor(buf1114, (768, 49), (1, 768), 0); del buf1114  # reuse
        buf1125 = buf1104; del buf1104  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1117, mul_280, buf1123, buf1125, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1117
        del mul_280
        buf1124 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1123, buf1124, 768, 49, grid=grid(768), stream=stream0)
        buf1126 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1125, buf1126, 768, 49, grid=grid(768), stream=stream0)
        buf1128 = reinterpret_tensor(buf1125, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1125  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1127, convolution_25, buf1128, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_25
        buf1129 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1128, buf1129, 768, 49, grid=grid(768), stream=stream0)
        buf1131 = reinterpret_tensor(buf1128, (768, 49), (1, 768), 0); del buf1128  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1130, buf1131, 37632, 128, grid=grid(37632), stream=stream0)
        buf1132 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1131, buf1132, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1133 = aten.convolution_backward(reinterpret_tensor(buf1130, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_204, primals_325, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_204
        del primals_325
        buf1134 = buf1133[0]
        buf1135 = buf1133[1]
        del buf1133
        buf1136 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1134, buf1136, 768, 6272, grid=grid(768), stream=stream0)
        buf1137 = reinterpret_tensor(buf1131, (768, 49), (49, 1), 0); del buf1131  # reuse
        # Source Nodes: [x_201], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1134, convolution_24, unsqueeze_275, buf1137, 37632, 128, grid=grid(37632), stream=stream0)
        buf1138 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1139 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1137, squeeze_40, buf1138, buf1139, 768, 49, grid=grid(768), stream=stream0)
        buf1140 = buf1134; del buf1134  # reuse
        # Source Nodes: [x_201], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1140, convolution_24, unsqueeze_275, buf1138, squeeze_40, buf1136, primals_323, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_24
        del primals_323
        del squeeze_40
        del unsqueeze_275
        buf1141 = buf1138; del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1140, buf1141, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1142 = aten.convolution_backward(buf1140, view_196, primals_321, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_321
        del view_196
        buf1143 = buf1142[0]
        buf1144 = buf1142[1]
        del buf1142
        buf1145 = reinterpret_tensor(buf1137, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1143, primals_319, buf1145, 37632, 128, grid=grid(37632), stream=stream0)
        buf1146 = buf1071; del buf1071  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1145, buf1146, 6272, 6, grid=grid(6272), stream=stream0)
        buf1147 = reinterpret_tensor(buf1145, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1145  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1143, primals_319, mul_267, buf1147, 37632, 128, grid=grid(37632), stream=stream0)
        buf1148 = buf1069; del buf1069  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1147, buf1148, 6272, 6, grid=grid(6272), stream=stream0)
        buf1149 = reinterpret_tensor(buf1147, (768, 49), (49, 1), 0); del buf1147  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1143, mul_267, buf1149, 37632, 128, grid=grid(37632), stream=stream0)
        buf1150 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1149, buf1150, 768, 49, grid=grid(768), stream=stream0)
        buf1151 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1143, buf1151, 768, 6272, grid=grid(768), stream=stream0)
        buf1152 = buf1127; del buf1127  # reuse
        buf1157 = reinterpret_tensor(buf1140, (8, 784, 768), (602112, 768, 1), 0); del buf1140  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1152, div_227, buf1143, primals_319, buf1146, mul_267, buf1148, primals_41, buf1157, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_227
        del mul_267
        del primals_319
        buf1153 = reinterpret_tensor(buf1149, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1149  # reuse
        buf1155 = reinterpret_tensor(buf1123, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1123  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1152, mm_10, primals_318, primals_41, buf1153, buf1155, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_10
        del primals_318
        del primals_41
        buf1154 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1153, buf1154, 768, 49, grid=grid(768), stream=stream0)
        buf1156 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1155, buf1156, 768, 49, grid=grid(768), stream=stream0)
        buf1158 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1157, (768, 6272), (1, 768), 0), view_194, out=buf1158)
        del view_194
        buf1159 = reinterpret_tensor(buf1143, (6272, 768), (768, 1), 0); del buf1143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1157, (6272, 768), (768, 1), 0), permute_637, out=buf1159)
        del permute_637
        buf1160 = reinterpret_tensor(buf1157, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1157  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1159, buf1160, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1161 = reinterpret_tensor(buf1159, (128, 48, 784), (37632, 784, 1), 0); del buf1159  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_640, reinterpret_tensor(buf1160, (128, 48, 784), (37632, 784, 1), 0), out=buf1161)
        del permute_640
        buf1162 = reinterpret_tensor(buf1089, (128, 48, 48), (2304, 48, 1), 0); del buf1089  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1160, (128, 48, 784), (37632, 784, 1), 0), permute_641, out=buf1162)
        del permute_641
        buf1163 = buf1095; del buf1095  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1162, alias_115, buf1163, 6144, 48, grid=grid(6144), stream=stream0)
        buf1164 = buf1087; del buf1087  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1162, alias_115, buf1163, bmm_20, buf1164, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_20
        buf1165 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1164, buf1165, 16, 3, grid=grid(16), stream=stream0)
        buf1166 = reinterpret_tensor(buf1162, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1162  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1166, alias_115, buf1163, primals_42, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_115
        del primals_42
        buf1167 = reinterpret_tensor(buf1160, (128, 784, 48), (37632, 48, 1), 0); del buf1160  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_642, reinterpret_tensor(buf1166, (128, 48, 48), (2304, 48, 1), 0), out=buf1167)
        del permute_642
        buf1168 = reinterpret_tensor(buf1130, (128, 48, 784), (37632, 784, 1), 0); del buf1130  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1166, (128, 48, 48), (2304, 48, 1), 0), permute_643, out=buf1168)
        del permute_643
        buf1169 = reinterpret_tensor(buf1094, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1094  # reuse
        # Source Nodes: [k_21], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1167, getitem_119, pow_45, buf1169, 43008, 112, grid=grid(43008), stream=stream0)
        buf1170 = buf1163; del buf1163  # reuse
        # Source Nodes: [k_21], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1169, buf1170, 6144, 7, grid=grid(6144), stream=stream0)
        buf1171 = reinterpret_tensor(buf1169, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1169  # reuse
        # Source Nodes: [q_21], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1168, getitem_118, pow_43, buf1171, 43008, 112, grid=grid(43008), stream=stream0)
        buf1172 = buf1093; del buf1093  # reuse
        # Source Nodes: [q_21], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1171, buf1172, 6144, 7, grid=grid(6144), stream=stream0)
        buf1173 = reinterpret_tensor(buf1097, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1097  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1168, pow_43, buf1172, getitem_118, buf1167, pow_45, buf1170, getitem_119, buf1161, buf1173, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1161
        del getitem_118
        del getitem_119
        del pow_43
        del pow_45
        buf1174 = reinterpret_tensor(buf1096, (6272, 2304), (2304, 1), 0); del buf1096  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1173, buf1174, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1175 = reinterpret_tensor(buf1168, (6272, 768), (768, 1), 0); del buf1168  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1174, permute_646, out=buf1175)
        del permute_646
        buf1176 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1174, (2304, 6272), (1, 2304), 0), view_184, out=buf1176)
        del view_184
        buf1177 = buf1100; del buf1100  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1174, buf1177, 112896, 128, grid=grid(112896), stream=stream0)
        buf1178 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1177, buf1178, 2304, 49, grid=grid(2304), stream=stream0)
        buf1185 = buf1152; del buf1152  # reuse
        buf1188 = reinterpret_tensor(buf1167, (8, 784, 768), (602112, 768, 1), 0); del buf1167  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1185, buf1175, primals_313, mul_263, div_236, primals_40, buf1188, 6272, 768, grid=grid(6272), stream=stream0)
        del div_236
        del primals_313
        del primals_40
        buf1181 = reinterpret_tensor(buf1155, (768, 49), (1, 768), 0); del buf1155  # reuse
        buf1183 = reinterpret_tensor(buf1153, (768, 49), (1, 768), 0); del buf1153  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1175, mul_263, buf1181, buf1183, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_263
        buf1182 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1181, buf1182, 768, 49, grid=grid(768), stream=stream0)
        buf1184 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1183, buf1184, 768, 49, grid=grid(768), stream=stream0)
        buf1186 = reinterpret_tensor(buf1183, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1183  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1185, addmm_29, buf1186, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_29
        buf1187 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1186, buf1187, 768, 49, grid=grid(768), stream=stream0)
        buf1189 = reinterpret_tensor(buf1116, (6272, 3072), (3072, 1), 0); del buf1116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1188, (6272, 768), (768, 1), 0), permute_650, out=buf1189)
        del permute_650
        buf1190 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1188, (768, 6272), (1, 768), 0), view_182, out=buf1190)
        del view_182
        buf1191 = reinterpret_tensor(buf1186, (1, 768, 49), (37632, 1, 768), 0); del buf1186  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1188, buf1191, 37632, 128, grid=grid(37632), stream=stream0)
        buf1192 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1191, buf1192, 768, 49, grid=grid(768), stream=stream0)
        buf1193 = reinterpret_tensor(buf1189, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1189  # reuse
        # Source Nodes: [x_188], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1193, addmm_28, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_28
        buf1194 = reinterpret_tensor(buf1188, (6272, 768), (768, 1), 0); del buf1188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1193, (6272, 3072), (3072, 1), 0), permute_654, out=buf1194)
        del permute_654
        buf1195 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1193, (3072, 6272), (1, 3072), 0), view_180, out=buf1195)
        del view_180
        buf1196 = buf1119; del buf1119  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1193, buf1196, 150528, 128, grid=grid(150528), stream=stream0)
        buf1197 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1196, buf1197, 3072, 49, grid=grid(3072), stream=stream0)
        buf1204 = buf1185; del buf1185  # reuse
        buf1207 = reinterpret_tensor(buf1175, (8, 784, 768), (602112, 768, 1), 0); del buf1175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1204, buf1194, primals_307, mul_257, div_237, primals_39, buf1207, 6272, 768, grid=grid(6272), stream=stream0)
        del div_237
        del primals_307
        del primals_39
        buf1200 = reinterpret_tensor(buf1191, (768, 49), (1, 768), 0); del buf1191  # reuse
        buf1202 = buf1181; del buf1181  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1194, mul_257, buf1200, buf1202, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1194
        del mul_257
        buf1201 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1200, buf1201, 768, 49, grid=grid(768), stream=stream0)
        buf1203 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1202, buf1203, 768, 49, grid=grid(768), stream=stream0)
        buf1205 = reinterpret_tensor(buf1202, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1202  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1204, convolution_23, buf1205, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_23
        buf1206 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1205, buf1206, 768, 49, grid=grid(768), stream=stream0)
        buf1208 = reinterpret_tensor(buf1205, (768, 49), (1, 768), 0); del buf1205  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1207, buf1208, 37632, 128, grid=grid(37632), stream=stream0)
        buf1209 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1208, buf1209, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1210 = aten.convolution_backward(reinterpret_tensor(buf1207, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_187, primals_305, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_187
        del primals_305
        buf1211 = buf1210[0]
        buf1212 = buf1210[1]
        del buf1210
        buf1213 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1211, buf1213, 768, 6272, grid=grid(768), stream=stream0)
        buf1214 = reinterpret_tensor(buf1208, (768, 49), (49, 1), 0); del buf1208  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1211, convolution_22, unsqueeze_287, buf1214, 37632, 128, grid=grid(37632), stream=stream0)
        buf1215 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1216 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1214, squeeze_37, buf1215, buf1216, 768, 49, grid=grid(768), stream=stream0)
        buf1217 = buf1211; del buf1211  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1217, convolution_22, unsqueeze_287, buf1215, squeeze_37, buf1213, primals_303, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_22
        del primals_303
        del squeeze_37
        del unsqueeze_287
        buf1218 = buf1215; del buf1215  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1217, buf1218, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1219 = aten.convolution_backward(buf1217, view_178, primals_301, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_301
        del view_178
        buf1220 = buf1219[0]
        buf1221 = buf1219[1]
        del buf1219
        buf1222 = reinterpret_tensor(buf1214, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1214  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1220, primals_299, buf1222, 37632, 128, grid=grid(37632), stream=stream0)
        buf1223 = buf1148; del buf1148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1222, buf1223, 6272, 6, grid=grid(6272), stream=stream0)
        buf1224 = reinterpret_tensor(buf1222, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1220, primals_299, mul_244, buf1224, 37632, 128, grid=grid(37632), stream=stream0)
        buf1225 = buf1146; del buf1146  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1224, buf1225, 6272, 6, grid=grid(6272), stream=stream0)
        buf1226 = reinterpret_tensor(buf1224, (768, 49), (49, 1), 0); del buf1224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1220, mul_244, buf1226, 37632, 128, grid=grid(37632), stream=stream0)
        buf1227 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1226, buf1227, 768, 49, grid=grid(768), stream=stream0)
        buf1228 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1220, buf1228, 768, 6272, grid=grid(768), stream=stream0)
        buf1229 = buf1204; del buf1204  # reuse
        buf1234 = reinterpret_tensor(buf1217, (8, 784, 768), (602112, 768, 1), 0); del buf1217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1229, div_238, buf1220, primals_299, buf1223, mul_244, buf1225, primals_37, buf1234, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_238
        del mul_244
        del primals_299
        buf1230 = reinterpret_tensor(buf1226, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1226  # reuse
        buf1232 = reinterpret_tensor(buf1200, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1200  # reuse
        # Source Nodes: [x_177], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1229, mm_9, primals_298, primals_37, buf1230, buf1232, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_9
        del primals_298
        del primals_37
        buf1231 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1230, buf1231, 768, 49, grid=grid(768), stream=stream0)
        buf1233 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1232, buf1233, 768, 49, grid=grid(768), stream=stream0)
        buf1235 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1234, (768, 6272), (1, 768), 0), view_176, out=buf1235)
        del view_176
        buf1236 = reinterpret_tensor(buf1220, (6272, 768), (768, 1), 0); del buf1220  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1234, (6272, 768), (768, 1), 0), permute_662, out=buf1236)
        del permute_662
        buf1237 = reinterpret_tensor(buf1234, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1234  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1236, buf1237, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1238 = reinterpret_tensor(buf1236, (128, 48, 784), (37632, 784, 1), 0); del buf1236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_665, reinterpret_tensor(buf1237, (128, 48, 784), (37632, 784, 1), 0), out=buf1238)
        del permute_665
        buf1239 = reinterpret_tensor(buf1166, (128, 48, 48), (2304, 48, 1), 0); del buf1166  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1237, (128, 48, 784), (37632, 784, 1), 0), permute_666, out=buf1239)
        del permute_666
        buf1240 = buf1172; del buf1172  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1239, alias_118, buf1240, 6144, 48, grid=grid(6144), stream=stream0)
        buf1241 = buf1164; del buf1164  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1239, alias_118, buf1240, bmm_18, buf1241, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_18
        buf1242 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1241, buf1242, 16, 3, grid=grid(16), stream=stream0)
        buf1243 = reinterpret_tensor(buf1239, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1239  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1243, alias_118, buf1240, primals_38, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_118
        del primals_38
        buf1244 = reinterpret_tensor(buf1237, (128, 784, 48), (37632, 48, 1), 0); del buf1237  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_667, reinterpret_tensor(buf1243, (128, 48, 48), (2304, 48, 1), 0), out=buf1244)
        del permute_667
        buf1245 = reinterpret_tensor(buf1207, (128, 48, 784), (37632, 784, 1), 0); del buf1207  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1243, (128, 48, 48), (2304, 48, 1), 0), permute_668, out=buf1245)
        del permute_668
        buf1246 = reinterpret_tensor(buf1171, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1171  # reuse
        # Source Nodes: [k_19], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1244, getitem_108, pow_41, buf1246, 43008, 112, grid=grid(43008), stream=stream0)
        buf1247 = buf1240; del buf1240  # reuse
        # Source Nodes: [k_19], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1246, buf1247, 6144, 7, grid=grid(6144), stream=stream0)
        buf1248 = reinterpret_tensor(buf1246, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1246  # reuse
        # Source Nodes: [q_19], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1245, getitem_107, pow_39, buf1248, 43008, 112, grid=grid(43008), stream=stream0)
        buf1249 = buf1170; del buf1170  # reuse
        # Source Nodes: [q_19], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1248, buf1249, 6144, 7, grid=grid(6144), stream=stream0)
        buf1250 = reinterpret_tensor(buf1174, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1174  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1245, pow_39, buf1249, getitem_107, buf1244, pow_41, buf1247, getitem_108, buf1238, buf1250, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1238
        del getitem_107
        del getitem_108
        del pow_39
        del pow_41
        buf1251 = reinterpret_tensor(buf1173, (6272, 2304), (2304, 1), 0); del buf1173  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1250, buf1251, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1252 = reinterpret_tensor(buf1245, (6272, 768), (768, 1), 0); del buf1245  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1251, permute_671, out=buf1252)
        del permute_671
        buf1253 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1251, (2304, 6272), (1, 2304), 0), view_166, out=buf1253)
        del view_166
        buf1254 = buf1177; del buf1177  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1251, buf1254, 112896, 128, grid=grid(112896), stream=stream0)
        buf1255 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1254, buf1255, 2304, 49, grid=grid(2304), stream=stream0)
        buf1262 = buf1229; del buf1229  # reuse
        buf1265 = reinterpret_tensor(buf1244, (8, 784, 768), (602112, 768, 1), 0); del buf1244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1262, buf1252, primals_293, mul_240, div_247, primals_36, buf1265, 6272, 768, grid=grid(6272), stream=stream0)
        del div_247
        del primals_293
        del primals_36
        buf1258 = reinterpret_tensor(buf1232, (768, 49), (1, 768), 0); del buf1232  # reuse
        buf1260 = reinterpret_tensor(buf1230, (768, 49), (1, 768), 0); del buf1230  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1252, mul_240, buf1258, buf1260, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_240
        buf1259 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1258, buf1259, 768, 49, grid=grid(768), stream=stream0)
        buf1261 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1260, buf1261, 768, 49, grid=grid(768), stream=stream0)
        buf1263 = reinterpret_tensor(buf1260, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1260  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1262, addmm_26, buf1263, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_26
        buf1264 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1263, buf1264, 768, 49, grid=grid(768), stream=stream0)
        buf1266 = reinterpret_tensor(buf1193, (6272, 3072), (3072, 1), 0); del buf1193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1265, (6272, 768), (768, 1), 0), permute_675, out=buf1266)
        del permute_675
        buf1267 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1265, (768, 6272), (1, 768), 0), view_164, out=buf1267)
        del view_164
        buf1268 = reinterpret_tensor(buf1263, (1, 768, 49), (37632, 1, 768), 0); del buf1263  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1265, buf1268, 37632, 128, grid=grid(37632), stream=stream0)
        buf1269 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1268, buf1269, 768, 49, grid=grid(768), stream=stream0)
        buf1270 = reinterpret_tensor(buf1266, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1266  # reuse
        # Source Nodes: [x_169], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1270, addmm_25, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_25
        buf1271 = reinterpret_tensor(buf1265, (6272, 768), (768, 1), 0); del buf1265  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1270, (6272, 3072), (3072, 1), 0), permute_679, out=buf1271)
        del permute_679
        buf1272 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1270, (3072, 6272), (1, 3072), 0), view_162, out=buf1272)
        del view_162
        buf1273 = buf1196; del buf1196  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1270, buf1273, 150528, 128, grid=grid(150528), stream=stream0)
        buf1274 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1273, buf1274, 3072, 49, grid=grid(3072), stream=stream0)
        buf1281 = buf1262; del buf1262  # reuse
        buf1284 = reinterpret_tensor(buf1252, (8, 784, 768), (602112, 768, 1), 0); del buf1252  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1281, buf1271, primals_287, mul_234, div_248, primals_35, buf1284, 6272, 768, grid=grid(6272), stream=stream0)
        del div_248
        del primals_287
        del primals_35
        buf1277 = reinterpret_tensor(buf1268, (768, 49), (1, 768), 0); del buf1268  # reuse
        buf1279 = buf1258; del buf1258  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1271, mul_234, buf1277, buf1279, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1271
        del mul_234
        buf1278 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1277, buf1278, 768, 49, grid=grid(768), stream=stream0)
        buf1280 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1279, buf1280, 768, 49, grid=grid(768), stream=stream0)
        buf1282 = reinterpret_tensor(buf1279, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1279  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1281, convolution_21, buf1282, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_21
        buf1283 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1282, buf1283, 768, 49, grid=grid(768), stream=stream0)
        buf1285 = reinterpret_tensor(buf1282, (768, 49), (1, 768), 0); del buf1282  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1284, buf1285, 37632, 128, grid=grid(37632), stream=stream0)
        buf1286 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1285, buf1286, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1287 = aten.convolution_backward(reinterpret_tensor(buf1284, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_170, primals_285, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_170
        del primals_285
        buf1288 = buf1287[0]
        buf1289 = buf1287[1]
        del buf1287
        buf1290 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1288, buf1290, 768, 6272, grid=grid(768), stream=stream0)
        buf1291 = reinterpret_tensor(buf1285, (768, 49), (49, 1), 0); del buf1285  # reuse
        # Source Nodes: [x_163], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1288, convolution_20, unsqueeze_299, buf1291, 37632, 128, grid=grid(37632), stream=stream0)
        buf1292 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1293 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1291, squeeze_34, buf1292, buf1293, 768, 49, grid=grid(768), stream=stream0)
        buf1294 = buf1288; del buf1288  # reuse
        # Source Nodes: [x_163], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1294, convolution_20, unsqueeze_299, buf1292, squeeze_34, buf1290, primals_283, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_20
        del primals_283
        del squeeze_34
        del unsqueeze_299
        buf1295 = buf1292; del buf1292  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1294, buf1295, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1296 = aten.convolution_backward(buf1294, view_160, primals_281, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_281
        del view_160
        buf1297 = buf1296[0]
        buf1298 = buf1296[1]
        del buf1296
        buf1299 = reinterpret_tensor(buf1291, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1291  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1297, primals_279, buf1299, 37632, 128, grid=grid(37632), stream=stream0)
        buf1300 = buf1225; del buf1225  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1299, buf1300, 6272, 6, grid=grid(6272), stream=stream0)
        buf1301 = reinterpret_tensor(buf1299, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1297, primals_279, mul_221, buf1301, 37632, 128, grid=grid(37632), stream=stream0)
        buf1302 = buf1223; del buf1223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1301, buf1302, 6272, 6, grid=grid(6272), stream=stream0)
        buf1303 = reinterpret_tensor(buf1301, (768, 49), (49, 1), 0); del buf1301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1297, mul_221, buf1303, 37632, 128, grid=grid(37632), stream=stream0)
        buf1304 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1303, buf1304, 768, 49, grid=grid(768), stream=stream0)
        buf1305 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1297, buf1305, 768, 6272, grid=grid(768), stream=stream0)
        buf1306 = buf1281; del buf1281  # reuse
        buf1311 = reinterpret_tensor(buf1294, (8, 784, 768), (602112, 768, 1), 0); del buf1294  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1306, div_249, buf1297, primals_279, buf1300, mul_221, buf1302, primals_33, buf1311, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_249
        del mul_221
        del primals_279
        buf1307 = reinterpret_tensor(buf1303, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1303  # reuse
        buf1309 = reinterpret_tensor(buf1277, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1277  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1306, mm_8, primals_278, primals_33, buf1307, buf1309, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_8
        del primals_278
        del primals_33
        buf1308 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1307, buf1308, 768, 49, grid=grid(768), stream=stream0)
        buf1310 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1309, buf1310, 768, 49, grid=grid(768), stream=stream0)
        buf1312 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1311, (768, 6272), (1, 768), 0), view_158, out=buf1312)
        del view_158
        buf1313 = reinterpret_tensor(buf1297, (6272, 768), (768, 1), 0); del buf1297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1311, (6272, 768), (768, 1), 0), permute_687, out=buf1313)
        del permute_687
        buf1314 = reinterpret_tensor(buf1311, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1311  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1313, buf1314, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1315 = reinterpret_tensor(buf1313, (128, 48, 784), (37632, 784, 1), 0); del buf1313  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_690, reinterpret_tensor(buf1314, (128, 48, 784), (37632, 784, 1), 0), out=buf1315)
        del permute_690
        buf1316 = reinterpret_tensor(buf1243, (128, 48, 48), (2304, 48, 1), 0); del buf1243  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1314, (128, 48, 784), (37632, 784, 1), 0), permute_691, out=buf1316)
        del permute_691
        buf1317 = buf1249; del buf1249  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1316, alias_121, buf1317, 6144, 48, grid=grid(6144), stream=stream0)
        buf1318 = buf1241; del buf1241  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1316, alias_121, buf1317, bmm_16, buf1318, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_16
        buf1319 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1318, buf1319, 16, 3, grid=grid(16), stream=stream0)
        buf1320 = reinterpret_tensor(buf1316, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1316  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1320, alias_121, buf1317, primals_34, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_121
        del primals_34
        buf1321 = reinterpret_tensor(buf1314, (128, 784, 48), (37632, 48, 1), 0); del buf1314  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_692, reinterpret_tensor(buf1320, (128, 48, 48), (2304, 48, 1), 0), out=buf1321)
        del permute_692
        buf1322 = reinterpret_tensor(buf1284, (128, 48, 784), (37632, 784, 1), 0); del buf1284  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1320, (128, 48, 48), (2304, 48, 1), 0), permute_693, out=buf1322)
        del permute_693
        buf1323 = reinterpret_tensor(buf1248, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1248  # reuse
        # Source Nodes: [k_17], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1321, getitem_97, pow_37, buf1323, 43008, 112, grid=grid(43008), stream=stream0)
        buf1324 = buf1317; del buf1317  # reuse
        # Source Nodes: [k_17], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1323, buf1324, 6144, 7, grid=grid(6144), stream=stream0)
        buf1325 = reinterpret_tensor(buf1323, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1323  # reuse
        # Source Nodes: [q_17], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1322, getitem_96, pow_35, buf1325, 43008, 112, grid=grid(43008), stream=stream0)
        buf1326 = buf1247; del buf1247  # reuse
        # Source Nodes: [q_17], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1325, buf1326, 6144, 7, grid=grid(6144), stream=stream0)
        buf1327 = reinterpret_tensor(buf1251, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1251  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1322, pow_35, buf1326, getitem_96, buf1321, pow_37, buf1324, getitem_97, buf1315, buf1327, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1315
        del getitem_96
        del getitem_97
        del pow_35
        del pow_37
        buf1328 = reinterpret_tensor(buf1250, (6272, 2304), (2304, 1), 0); del buf1250  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1327, buf1328, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1329 = reinterpret_tensor(buf1322, (6272, 768), (768, 1), 0); del buf1322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1328, permute_696, out=buf1329)
        del permute_696
        buf1330 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1328, (2304, 6272), (1, 2304), 0), view_148, out=buf1330)
        del view_148
        buf1331 = buf1254; del buf1254  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1328, buf1331, 112896, 128, grid=grid(112896), stream=stream0)
        buf1332 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1331, buf1332, 2304, 49, grid=grid(2304), stream=stream0)
        buf1339 = buf1306; del buf1306  # reuse
        buf1342 = reinterpret_tensor(buf1321, (8, 784, 768), (602112, 768, 1), 0); del buf1321  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1339, buf1329, primals_273, mul_217, div_258, primals_32, buf1342, 6272, 768, grid=grid(6272), stream=stream0)
        del div_258
        del primals_273
        del primals_32
        buf1335 = reinterpret_tensor(buf1309, (768, 49), (1, 768), 0); del buf1309  # reuse
        buf1337 = reinterpret_tensor(buf1307, (768, 49), (1, 768), 0); del buf1307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1329, mul_217, buf1335, buf1337, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_217
        buf1336 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1335, buf1336, 768, 49, grid=grid(768), stream=stream0)
        buf1338 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1337, buf1338, 768, 49, grid=grid(768), stream=stream0)
        buf1340 = reinterpret_tensor(buf1337, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1337  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1339, addmm_23, buf1340, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_23
        buf1341 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1340, buf1341, 768, 49, grid=grid(768), stream=stream0)
        buf1343 = reinterpret_tensor(buf1270, (6272, 3072), (3072, 1), 0); del buf1270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1342, (6272, 768), (768, 1), 0), permute_700, out=buf1343)
        del permute_700
        buf1344 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1342, (768, 6272), (1, 768), 0), view_146, out=buf1344)
        del view_146
        buf1345 = reinterpret_tensor(buf1340, (1, 768, 49), (37632, 1, 768), 0); del buf1340  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1342, buf1345, 37632, 128, grid=grid(37632), stream=stream0)
        buf1346 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1345, buf1346, 768, 49, grid=grid(768), stream=stream0)
        buf1347 = reinterpret_tensor(buf1343, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1343  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1347, addmm_22, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_22
        buf1348 = reinterpret_tensor(buf1342, (6272, 768), (768, 1), 0); del buf1342  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1347, (6272, 3072), (3072, 1), 0), permute_704, out=buf1348)
        del permute_704
        buf1349 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1347, (3072, 6272), (1, 3072), 0), view_144, out=buf1349)
        del view_144
        buf1350 = buf1273; del buf1273  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1347, buf1350, 150528, 128, grid=grid(150528), stream=stream0)
        buf1351 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1350, buf1351, 3072, 49, grid=grid(3072), stream=stream0)
        buf1358 = buf1339; del buf1339  # reuse
        buf1361 = reinterpret_tensor(buf1329, (8, 784, 768), (602112, 768, 1), 0); del buf1329  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1358, buf1348, primals_267, mul_211, div_259, primals_31, buf1361, 6272, 768, grid=grid(6272), stream=stream0)
        del div_259
        del primals_267
        del primals_31
        buf1354 = reinterpret_tensor(buf1345, (768, 49), (1, 768), 0); del buf1345  # reuse
        buf1356 = buf1335; del buf1335  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1348, mul_211, buf1354, buf1356, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1348
        del mul_211
        buf1355 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1354, buf1355, 768, 49, grid=grid(768), stream=stream0)
        buf1357 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1356, buf1357, 768, 49, grid=grid(768), stream=stream0)
        buf1359 = reinterpret_tensor(buf1356, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1356  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1358, convolution_19, buf1359, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_19
        buf1360 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1359, buf1360, 768, 49, grid=grid(768), stream=stream0)
        buf1362 = reinterpret_tensor(buf1359, (768, 49), (1, 768), 0); del buf1359  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1361, buf1362, 37632, 128, grid=grid(37632), stream=stream0)
        buf1363 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1362, buf1363, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1364 = aten.convolution_backward(reinterpret_tensor(buf1361, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_153, primals_265, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_153
        del primals_265
        buf1365 = buf1364[0]
        buf1366 = buf1364[1]
        del buf1364
        buf1367 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1365, buf1367, 768, 6272, grid=grid(768), stream=stream0)
        buf1368 = reinterpret_tensor(buf1362, (768, 49), (49, 1), 0); del buf1362  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1365, convolution_18, unsqueeze_311, buf1368, 37632, 128, grid=grid(37632), stream=stream0)
        buf1369 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1370 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1368, squeeze_31, buf1369, buf1370, 768, 49, grid=grid(768), stream=stream0)
        buf1371 = buf1365; del buf1365  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1371, convolution_18, unsqueeze_311, buf1369, squeeze_31, buf1367, primals_263, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_18
        del primals_263
        del squeeze_31
        del unsqueeze_311
        buf1372 = buf1369; del buf1369  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1371, buf1372, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1373 = aten.convolution_backward(buf1371, view_142, primals_261, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_261
        del view_142
        buf1374 = buf1373[0]
        buf1375 = buf1373[1]
        del buf1373
        buf1376 = reinterpret_tensor(buf1368, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1368  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1374, primals_259, buf1376, 37632, 128, grid=grid(37632), stream=stream0)
        buf1377 = buf1302; del buf1302  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1376, buf1377, 6272, 6, grid=grid(6272), stream=stream0)
        buf1378 = reinterpret_tensor(buf1376, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1376  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1374, primals_259, mul_198, buf1378, 37632, 128, grid=grid(37632), stream=stream0)
        buf1379 = buf1300; del buf1300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1378, buf1379, 6272, 6, grid=grid(6272), stream=stream0)
        buf1380 = reinterpret_tensor(buf1378, (768, 49), (49, 1), 0); del buf1378  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1374, mul_198, buf1380, 37632, 128, grid=grid(37632), stream=stream0)
        buf1381 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1380, buf1381, 768, 49, grid=grid(768), stream=stream0)
        buf1382 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1374, buf1382, 768, 6272, grid=grid(768), stream=stream0)
        buf1383 = buf1358; del buf1358  # reuse
        buf1388 = reinterpret_tensor(buf1371, (8, 784, 768), (602112, 768, 1), 0); del buf1371  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1383, div_260, buf1374, primals_259, buf1377, mul_198, buf1379, primals_29, buf1388, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_260
        del mul_198
        del primals_259
        buf1384 = reinterpret_tensor(buf1380, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1380  # reuse
        buf1386 = reinterpret_tensor(buf1354, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1354  # reuse
        # Source Nodes: [x_139], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1383, mm_7, primals_258, primals_29, buf1384, buf1386, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_7
        del primals_258
        del primals_29
        buf1385 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1384, buf1385, 768, 49, grid=grid(768), stream=stream0)
        buf1387 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1386, buf1387, 768, 49, grid=grid(768), stream=stream0)
        buf1389 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1388, (768, 6272), (1, 768), 0), view_140, out=buf1389)
        del view_140
        buf1390 = reinterpret_tensor(buf1374, (6272, 768), (768, 1), 0); del buf1374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1388, (6272, 768), (768, 1), 0), permute_712, out=buf1390)
        del permute_712
        buf1391 = reinterpret_tensor(buf1388, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1388  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1390, buf1391, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1392 = reinterpret_tensor(buf1390, (128, 48, 784), (37632, 784, 1), 0); del buf1390  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_715, reinterpret_tensor(buf1391, (128, 48, 784), (37632, 784, 1), 0), out=buf1392)
        del permute_715
        buf1393 = reinterpret_tensor(buf1320, (128, 48, 48), (2304, 48, 1), 0); del buf1320  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1391, (128, 48, 784), (37632, 784, 1), 0), permute_716, out=buf1393)
        del permute_716
        buf1394 = buf1326; del buf1326  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1393, alias_124, buf1394, 6144, 48, grid=grid(6144), stream=stream0)
        buf1395 = buf1318; del buf1318  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1393, alias_124, buf1394, bmm_14, buf1395, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_14
        buf1396 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1395, buf1396, 16, 3, grid=grid(16), stream=stream0)
        buf1397 = reinterpret_tensor(buf1393, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1393  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1397, alias_124, buf1394, primals_30, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_124
        del primals_30
        buf1398 = reinterpret_tensor(buf1391, (128, 784, 48), (37632, 48, 1), 0); del buf1391  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_717, reinterpret_tensor(buf1397, (128, 48, 48), (2304, 48, 1), 0), out=buf1398)
        del permute_717
        buf1399 = reinterpret_tensor(buf1361, (128, 48, 784), (37632, 784, 1), 0); del buf1361  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1397, (128, 48, 48), (2304, 48, 1), 0), permute_718, out=buf1399)
        del permute_718
        buf1400 = reinterpret_tensor(buf1325, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1325  # reuse
        # Source Nodes: [k_15], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1398, getitem_86, pow_33, buf1400, 43008, 112, grid=grid(43008), stream=stream0)
        buf1401 = buf1394; del buf1394  # reuse
        # Source Nodes: [k_15], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1400, buf1401, 6144, 7, grid=grid(6144), stream=stream0)
        buf1402 = reinterpret_tensor(buf1400, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1400  # reuse
        # Source Nodes: [q_15], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1399, getitem_85, pow_31, buf1402, 43008, 112, grid=grid(43008), stream=stream0)
        buf1403 = buf1324; del buf1324  # reuse
        # Source Nodes: [q_15], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1402, buf1403, 6144, 7, grid=grid(6144), stream=stream0)
        buf1404 = reinterpret_tensor(buf1328, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1328  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1399, pow_31, buf1403, getitem_85, buf1398, pow_33, buf1401, getitem_86, buf1392, buf1404, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1392
        del getitem_85
        del getitem_86
        del pow_31
        del pow_33
        buf1405 = reinterpret_tensor(buf1327, (6272, 2304), (2304, 1), 0); del buf1327  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1404, buf1405, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1406 = reinterpret_tensor(buf1399, (6272, 768), (768, 1), 0); del buf1399  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1405, permute_721, out=buf1406)
        del permute_721
        buf1407 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1405, (2304, 6272), (1, 2304), 0), view_130, out=buf1407)
        del view_130
        buf1408 = buf1331; del buf1331  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1405, buf1408, 112896, 128, grid=grid(112896), stream=stream0)
        buf1409 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1408, buf1409, 2304, 49, grid=grid(2304), stream=stream0)
        buf1416 = buf1383; del buf1383  # reuse
        buf1419 = reinterpret_tensor(buf1398, (8, 784, 768), (602112, 768, 1), 0); del buf1398  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1416, buf1406, primals_253, mul_194, div_269, primals_28, buf1419, 6272, 768, grid=grid(6272), stream=stream0)
        del div_269
        del primals_253
        del primals_28
        buf1412 = reinterpret_tensor(buf1386, (768, 49), (1, 768), 0); del buf1386  # reuse
        buf1414 = reinterpret_tensor(buf1384, (768, 49), (1, 768), 0); del buf1384  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1406, mul_194, buf1412, buf1414, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_194
        buf1413 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1412, buf1413, 768, 49, grid=grid(768), stream=stream0)
        buf1415 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1414, buf1415, 768, 49, grid=grid(768), stream=stream0)
        buf1417 = reinterpret_tensor(buf1414, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1414  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1416, addmm_20, buf1417, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_20
        buf1418 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1417, buf1418, 768, 49, grid=grid(768), stream=stream0)
        buf1420 = reinterpret_tensor(buf1347, (6272, 3072), (3072, 1), 0); del buf1347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1419, (6272, 768), (768, 1), 0), permute_725, out=buf1420)
        del permute_725
        buf1421 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1419, (768, 6272), (1, 768), 0), view_128, out=buf1421)
        del view_128
        buf1422 = reinterpret_tensor(buf1417, (1, 768, 49), (37632, 1, 768), 0); del buf1417  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1419, buf1422, 37632, 128, grid=grid(37632), stream=stream0)
        buf1423 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1422, buf1423, 768, 49, grid=grid(768), stream=stream0)
        buf1424 = reinterpret_tensor(buf1420, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1420  # reuse
        # Source Nodes: [x_131], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1424, addmm_19, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_19
        buf1425 = reinterpret_tensor(buf1419, (6272, 768), (768, 1), 0); del buf1419  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1424, (6272, 3072), (3072, 1), 0), permute_729, out=buf1425)
        del permute_729
        buf1426 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1424, (3072, 6272), (1, 3072), 0), view_126, out=buf1426)
        del view_126
        buf1427 = buf1350; del buf1350  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1424, buf1427, 150528, 128, grid=grid(150528), stream=stream0)
        buf1428 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1427, buf1428, 3072, 49, grid=grid(3072), stream=stream0)
        buf1435 = buf1416; del buf1416  # reuse
        buf1438 = reinterpret_tensor(buf1406, (8, 784, 768), (602112, 768, 1), 0); del buf1406  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1435, buf1425, primals_247, mul_188, div_270, primals_27, buf1438, 6272, 768, grid=grid(6272), stream=stream0)
        del div_270
        del primals_247
        del primals_27
        buf1431 = reinterpret_tensor(buf1422, (768, 49), (1, 768), 0); del buf1422  # reuse
        buf1433 = buf1412; del buf1412  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1425, mul_188, buf1431, buf1433, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1425
        del mul_188
        buf1432 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1431, buf1432, 768, 49, grid=grid(768), stream=stream0)
        buf1434 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1433, buf1434, 768, 49, grid=grid(768), stream=stream0)
        buf1436 = reinterpret_tensor(buf1433, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1433  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1435, convolution_17, buf1436, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_17
        buf1437 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1436, buf1437, 768, 49, grid=grid(768), stream=stream0)
        buf1439 = reinterpret_tensor(buf1436, (768, 49), (1, 768), 0); del buf1436  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1438, buf1439, 37632, 128, grid=grid(37632), stream=stream0)
        buf1440 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1439, buf1440, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1441 = aten.convolution_backward(reinterpret_tensor(buf1438, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_136, primals_245, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_136
        del primals_245
        buf1442 = buf1441[0]
        buf1443 = buf1441[1]
        del buf1441
        buf1444 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1442, buf1444, 768, 6272, grid=grid(768), stream=stream0)
        buf1445 = reinterpret_tensor(buf1439, (768, 49), (49, 1), 0); del buf1439  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1442, convolution_16, unsqueeze_323, buf1445, 37632, 128, grid=grid(37632), stream=stream0)
        buf1446 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1447 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1445, squeeze_28, buf1446, buf1447, 768, 49, grid=grid(768), stream=stream0)
        buf1448 = buf1442; del buf1442  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1448, convolution_16, unsqueeze_323, buf1446, squeeze_28, buf1444, primals_243, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_16
        del primals_243
        del squeeze_28
        del unsqueeze_323
        buf1449 = buf1446; del buf1446  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1448, buf1449, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1450 = aten.convolution_backward(buf1448, view_124, primals_241, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_241
        del view_124
        buf1451 = buf1450[0]
        buf1452 = buf1450[1]
        del buf1450
        buf1453 = reinterpret_tensor(buf1445, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1445  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1451, primals_239, buf1453, 37632, 128, grid=grid(37632), stream=stream0)
        buf1454 = buf1379; del buf1379  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1453, buf1454, 6272, 6, grid=grid(6272), stream=stream0)
        buf1455 = reinterpret_tensor(buf1453, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1453  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1451, primals_239, mul_175, buf1455, 37632, 128, grid=grid(37632), stream=stream0)
        buf1456 = buf1377; del buf1377  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1455, buf1456, 6272, 6, grid=grid(6272), stream=stream0)
        buf1457 = reinterpret_tensor(buf1455, (768, 49), (49, 1), 0); del buf1455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1451, mul_175, buf1457, 37632, 128, grid=grid(37632), stream=stream0)
        buf1458 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1457, buf1458, 768, 49, grid=grid(768), stream=stream0)
        buf1459 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1451, buf1459, 768, 6272, grid=grid(768), stream=stream0)
        buf1460 = buf1435; del buf1435  # reuse
        buf1465 = reinterpret_tensor(buf1448, (8, 784, 768), (602112, 768, 1), 0); del buf1448  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1460, div_271, buf1451, primals_239, buf1454, mul_175, buf1456, primals_25, buf1465, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_271
        del mul_175
        del primals_239
        buf1461 = reinterpret_tensor(buf1457, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1457  # reuse
        buf1463 = reinterpret_tensor(buf1431, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1431  # reuse
        # Source Nodes: [x_120], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1460, mm_6, primals_238, primals_25, buf1461, buf1463, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_6
        del primals_238
        del primals_25
        buf1462 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1461, buf1462, 768, 49, grid=grid(768), stream=stream0)
        buf1464 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1463, buf1464, 768, 49, grid=grid(768), stream=stream0)
        buf1466 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1465, (768, 6272), (1, 768), 0), view_122, out=buf1466)
        del view_122
        buf1467 = reinterpret_tensor(buf1451, (6272, 768), (768, 1), 0); del buf1451  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1465, (6272, 768), (768, 1), 0), permute_737, out=buf1467)
        del permute_737
        buf1468 = reinterpret_tensor(buf1465, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1465  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1467, buf1468, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1469 = reinterpret_tensor(buf1467, (128, 48, 784), (37632, 784, 1), 0); del buf1467  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_740, reinterpret_tensor(buf1468, (128, 48, 784), (37632, 784, 1), 0), out=buf1469)
        del permute_740
        buf1470 = reinterpret_tensor(buf1397, (128, 48, 48), (2304, 48, 1), 0); del buf1397  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1468, (128, 48, 784), (37632, 784, 1), 0), permute_741, out=buf1470)
        del permute_741
        buf1471 = buf1403; del buf1403  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1470, alias_127, buf1471, 6144, 48, grid=grid(6144), stream=stream0)
        buf1472 = buf1395; del buf1395  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1470, alias_127, buf1471, bmm_12, buf1472, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_12
        buf1473 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1472, buf1473, 16, 3, grid=grid(16), stream=stream0)
        buf1474 = reinterpret_tensor(buf1470, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1470  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1474, alias_127, buf1471, primals_26, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_127
        del primals_26
        buf1475 = reinterpret_tensor(buf1468, (128, 784, 48), (37632, 48, 1), 0); del buf1468  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_742, reinterpret_tensor(buf1474, (128, 48, 48), (2304, 48, 1), 0), out=buf1475)
        del permute_742
        buf1476 = reinterpret_tensor(buf1438, (128, 48, 784), (37632, 784, 1), 0); del buf1438  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1474, (128, 48, 48), (2304, 48, 1), 0), permute_743, out=buf1476)
        del permute_743
        buf1477 = reinterpret_tensor(buf1402, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1402  # reuse
        # Source Nodes: [k_13], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1475, getitem_75, pow_29, buf1477, 43008, 112, grid=grid(43008), stream=stream0)
        buf1478 = buf1471; del buf1471  # reuse
        # Source Nodes: [k_13], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1477, buf1478, 6144, 7, grid=grid(6144), stream=stream0)
        buf1479 = reinterpret_tensor(buf1477, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1477  # reuse
        # Source Nodes: [q_13], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1476, getitem_74, pow_27, buf1479, 43008, 112, grid=grid(43008), stream=stream0)
        buf1480 = buf1401; del buf1401  # reuse
        # Source Nodes: [q_13], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1479, buf1480, 6144, 7, grid=grid(6144), stream=stream0)
        buf1481 = reinterpret_tensor(buf1405, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1405  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1476, pow_27, buf1480, getitem_74, buf1475, pow_29, buf1478, getitem_75, buf1469, buf1481, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1469
        del getitem_74
        del getitem_75
        del pow_27
        del pow_29
        buf1482 = reinterpret_tensor(buf1404, (6272, 2304), (2304, 1), 0); del buf1404  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1481, buf1482, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1483 = reinterpret_tensor(buf1476, (6272, 768), (768, 1), 0); del buf1476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1482, permute_746, out=buf1483)
        del permute_746
        buf1484 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1482, (2304, 6272), (1, 2304), 0), view_112, out=buf1484)
        del view_112
        buf1485 = buf1408; del buf1408  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1482, buf1485, 112896, 128, grid=grid(112896), stream=stream0)
        buf1486 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1485, buf1486, 2304, 49, grid=grid(2304), stream=stream0)
        buf1493 = buf1460; del buf1460  # reuse
        buf1496 = reinterpret_tensor(buf1475, (8, 784, 768), (602112, 768, 1), 0); del buf1475  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1493, buf1483, primals_233, mul_171, div_280, primals_24, buf1496, 6272, 768, grid=grid(6272), stream=stream0)
        del div_280
        del primals_233
        del primals_24
        buf1489 = reinterpret_tensor(buf1463, (768, 49), (1, 768), 0); del buf1463  # reuse
        buf1491 = reinterpret_tensor(buf1461, (768, 49), (1, 768), 0); del buf1461  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1483, mul_171, buf1489, buf1491, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_171
        buf1490 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1489, buf1490, 768, 49, grid=grid(768), stream=stream0)
        buf1492 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1491, buf1492, 768, 49, grid=grid(768), stream=stream0)
        buf1494 = reinterpret_tensor(buf1491, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1491  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1493, addmm_17, buf1494, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_17
        buf1495 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1494, buf1495, 768, 49, grid=grid(768), stream=stream0)
        buf1497 = reinterpret_tensor(buf1424, (6272, 3072), (3072, 1), 0); del buf1424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1496, (6272, 768), (768, 1), 0), permute_750, out=buf1497)
        del permute_750
        buf1498 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1496, (768, 6272), (1, 768), 0), view_110, out=buf1498)
        del view_110
        buf1499 = reinterpret_tensor(buf1494, (1, 768, 49), (37632, 1, 768), 0); del buf1494  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1496, buf1499, 37632, 128, grid=grid(37632), stream=stream0)
        buf1500 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1499, buf1500, 768, 49, grid=grid(768), stream=stream0)
        buf1501 = reinterpret_tensor(buf1497, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1497  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1501, addmm_16, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_16
        buf1502 = reinterpret_tensor(buf1496, (6272, 768), (768, 1), 0); del buf1496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1501, (6272, 3072), (3072, 1), 0), permute_754, out=buf1502)
        del permute_754
        buf1503 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1501, (3072, 6272), (1, 3072), 0), view_108, out=buf1503)
        del view_108
        buf1504 = buf1427; del buf1427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1501, buf1504, 150528, 128, grid=grid(150528), stream=stream0)
        buf1505 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1504, buf1505, 3072, 49, grid=grid(3072), stream=stream0)
        buf1512 = buf1493; del buf1493  # reuse
        buf1515 = reinterpret_tensor(buf1483, (8, 784, 768), (602112, 768, 1), 0); del buf1483  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1512, buf1502, primals_227, mul_165, div_281, primals_23, buf1515, 6272, 768, grid=grid(6272), stream=stream0)
        del div_281
        del primals_227
        del primals_23
        buf1508 = reinterpret_tensor(buf1499, (768, 49), (1, 768), 0); del buf1499  # reuse
        buf1510 = buf1489; del buf1489  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1502, mul_165, buf1508, buf1510, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1502
        del mul_165
        buf1509 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1508, buf1509, 768, 49, grid=grid(768), stream=stream0)
        buf1511 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1510, buf1511, 768, 49, grid=grid(768), stream=stream0)
        buf1513 = reinterpret_tensor(buf1510, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1510  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1512, convolution_15, buf1513, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_15
        buf1514 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1513, buf1514, 768, 49, grid=grid(768), stream=stream0)
        buf1516 = reinterpret_tensor(buf1513, (768, 49), (1, 768), 0); del buf1513  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1515, buf1516, 37632, 128, grid=grid(37632), stream=stream0)
        buf1517 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1516, buf1517, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1518 = aten.convolution_backward(reinterpret_tensor(buf1515, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_119, primals_225, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_119
        del primals_225
        buf1519 = buf1518[0]
        buf1520 = buf1518[1]
        del buf1518
        buf1521 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1519, buf1521, 768, 6272, grid=grid(768), stream=stream0)
        buf1522 = reinterpret_tensor(buf1516, (768, 49), (49, 1), 0); del buf1516  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1519, convolution_14, unsqueeze_335, buf1522, 37632, 128, grid=grid(37632), stream=stream0)
        buf1523 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1524 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1522, squeeze_25, buf1523, buf1524, 768, 49, grid=grid(768), stream=stream0)
        buf1525 = buf1519; del buf1519  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1525, convolution_14, unsqueeze_335, buf1523, squeeze_25, buf1521, primals_223, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_14
        del primals_223
        del squeeze_25
        del unsqueeze_335
        buf1526 = buf1523; del buf1523  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1525, buf1526, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1527 = aten.convolution_backward(buf1525, view_106, primals_221, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_221
        del view_106
        buf1528 = buf1527[0]
        buf1529 = buf1527[1]
        del buf1527
        buf1530 = reinterpret_tensor(buf1522, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1522  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1528, primals_219, buf1530, 37632, 128, grid=grid(37632), stream=stream0)
        buf1531 = buf1456; del buf1456  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1530, buf1531, 6272, 6, grid=grid(6272), stream=stream0)
        buf1532 = reinterpret_tensor(buf1530, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1530  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1528, primals_219, mul_152, buf1532, 37632, 128, grid=grid(37632), stream=stream0)
        buf1533 = buf1454; del buf1454  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1532, buf1533, 6272, 6, grid=grid(6272), stream=stream0)
        buf1534 = reinterpret_tensor(buf1532, (768, 49), (49, 1), 0); del buf1532  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1528, mul_152, buf1534, 37632, 128, grid=grid(37632), stream=stream0)
        buf1535 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1534, buf1535, 768, 49, grid=grid(768), stream=stream0)
        buf1536 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1528, buf1536, 768, 6272, grid=grid(768), stream=stream0)
        buf1537 = buf1512; del buf1512  # reuse
        buf1542 = reinterpret_tensor(buf1525, (8, 784, 768), (602112, 768, 1), 0); del buf1525  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1537, div_282, buf1528, primals_219, buf1531, mul_152, buf1533, primals_21, buf1542, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_282
        del mul_152
        del primals_219
        buf1538 = reinterpret_tensor(buf1534, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1534  # reuse
        buf1540 = reinterpret_tensor(buf1508, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1508  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1537, mm_5, primals_218, primals_21, buf1538, buf1540, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_5
        del primals_21
        del primals_218
        buf1539 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1538, buf1539, 768, 49, grid=grid(768), stream=stream0)
        buf1541 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1540, buf1541, 768, 49, grid=grid(768), stream=stream0)
        buf1543 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1542, (768, 6272), (1, 768), 0), view_104, out=buf1543)
        del view_104
        buf1544 = reinterpret_tensor(buf1528, (6272, 768), (768, 1), 0); del buf1528  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1542, (6272, 768), (768, 1), 0), permute_762, out=buf1544)
        del permute_762
        buf1545 = reinterpret_tensor(buf1542, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1542  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1544, buf1545, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1546 = reinterpret_tensor(buf1544, (128, 48, 784), (37632, 784, 1), 0); del buf1544  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_765, reinterpret_tensor(buf1545, (128, 48, 784), (37632, 784, 1), 0), out=buf1546)
        del permute_765
        buf1547 = reinterpret_tensor(buf1474, (128, 48, 48), (2304, 48, 1), 0); del buf1474  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1545, (128, 48, 784), (37632, 784, 1), 0), permute_766, out=buf1547)
        del permute_766
        buf1548 = buf1480; del buf1480  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1547, alias_130, buf1548, 6144, 48, grid=grid(6144), stream=stream0)
        buf1549 = buf1472; del buf1472  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1547, alias_130, buf1548, bmm_10, buf1549, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_10
        buf1550 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1549, buf1550, 16, 3, grid=grid(16), stream=stream0)
        buf1551 = reinterpret_tensor(buf1547, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1547  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1551, alias_130, buf1548, primals_22, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_130
        del primals_22
        buf1552 = reinterpret_tensor(buf1545, (128, 784, 48), (37632, 48, 1), 0); del buf1545  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_767, reinterpret_tensor(buf1551, (128, 48, 48), (2304, 48, 1), 0), out=buf1552)
        del permute_767
        buf1553 = reinterpret_tensor(buf1515, (128, 48, 784), (37632, 784, 1), 0); del buf1515  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1551, (128, 48, 48), (2304, 48, 1), 0), permute_768, out=buf1553)
        del permute_768
        buf1554 = reinterpret_tensor(buf1479, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1479  # reuse
        # Source Nodes: [k_11], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1552, getitem_64, pow_25, buf1554, 43008, 112, grid=grid(43008), stream=stream0)
        buf1555 = buf1548; del buf1548  # reuse
        # Source Nodes: [k_11], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1554, buf1555, 6144, 7, grid=grid(6144), stream=stream0)
        buf1556 = reinterpret_tensor(buf1554, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1554  # reuse
        # Source Nodes: [q_11], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1553, getitem_63, pow_23, buf1556, 43008, 112, grid=grid(43008), stream=stream0)
        buf1557 = buf1478; del buf1478  # reuse
        # Source Nodes: [q_11], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1556, buf1557, 6144, 7, grid=grid(6144), stream=stream0)
        buf1558 = reinterpret_tensor(buf1482, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1482  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1553, pow_23, buf1557, getitem_63, buf1552, pow_25, buf1555, getitem_64, buf1546, buf1558, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1546
        del getitem_63
        del getitem_64
        del pow_23
        del pow_25
        buf1559 = reinterpret_tensor(buf1481, (6272, 2304), (2304, 1), 0); del buf1481  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1558, buf1559, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1560 = reinterpret_tensor(buf1553, (6272, 768), (768, 1), 0); del buf1553  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1559, permute_771, out=buf1560)
        del permute_771
        buf1561 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1559, (2304, 6272), (1, 2304), 0), view_94, out=buf1561)
        del view_94
        buf1562 = buf1485; del buf1485  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1559, buf1562, 112896, 128, grid=grid(112896), stream=stream0)
        buf1563 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1562, buf1563, 2304, 49, grid=grid(2304), stream=stream0)
        buf1570 = buf1537; del buf1537  # reuse
        buf1573 = reinterpret_tensor(buf1552, (8, 784, 768), (602112, 768, 1), 0); del buf1552  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1570, buf1560, primals_213, mul_148, div_291, primals_20, buf1573, 6272, 768, grid=grid(6272), stream=stream0)
        del div_291
        del primals_20
        del primals_213
        buf1566 = reinterpret_tensor(buf1540, (768, 49), (1, 768), 0); del buf1540  # reuse
        buf1568 = reinterpret_tensor(buf1538, (768, 49), (1, 768), 0); del buf1538  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1560, mul_148, buf1566, buf1568, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_148
        buf1567 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1566, buf1567, 768, 49, grid=grid(768), stream=stream0)
        buf1569 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1568, buf1569, 768, 49, grid=grid(768), stream=stream0)
        buf1571 = reinterpret_tensor(buf1568, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1568  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1570, addmm_14, buf1571, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_14
        buf1572 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1571, buf1572, 768, 49, grid=grid(768), stream=stream0)
        buf1574 = reinterpret_tensor(buf1501, (6272, 3072), (3072, 1), 0); del buf1501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1573, (6272, 768), (768, 1), 0), permute_775, out=buf1574)
        del permute_775
        buf1575 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1573, (768, 6272), (1, 768), 0), view_92, out=buf1575)
        del view_92
        buf1576 = reinterpret_tensor(buf1571, (1, 768, 49), (37632, 1, 768), 0); del buf1571  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1573, buf1576, 37632, 128, grid=grid(37632), stream=stream0)
        buf1577 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1576, buf1577, 768, 49, grid=grid(768), stream=stream0)
        buf1578 = reinterpret_tensor(buf1574, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1574  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1578, addmm_13, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_13
        buf1579 = reinterpret_tensor(buf1573, (6272, 768), (768, 1), 0); del buf1573  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1578, (6272, 3072), (3072, 1), 0), permute_779, out=buf1579)
        del permute_779
        buf1580 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1578, (3072, 6272), (1, 3072), 0), view_90, out=buf1580)
        del view_90
        buf1581 = buf1504; del buf1504  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1578, buf1581, 150528, 128, grid=grid(150528), stream=stream0)
        buf1582 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1581, buf1582, 3072, 49, grid=grid(3072), stream=stream0)
        buf1589 = buf1570; del buf1570  # reuse
        buf1592 = reinterpret_tensor(buf1560, (8, 784, 768), (602112, 768, 1), 0); del buf1560  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1589, buf1579, primals_207, mul_142, div_292, primals_19, buf1592, 6272, 768, grid=grid(6272), stream=stream0)
        del div_292
        del primals_19
        del primals_207
        buf1585 = reinterpret_tensor(buf1576, (768, 49), (1, 768), 0); del buf1576  # reuse
        buf1587 = buf1566; del buf1566  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1579, mul_142, buf1585, buf1587, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1579
        del mul_142
        buf1586 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1585, buf1586, 768, 49, grid=grid(768), stream=stream0)
        buf1588 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1587, buf1588, 768, 49, grid=grid(768), stream=stream0)
        buf1590 = reinterpret_tensor(buf1587, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1587  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1589, convolution_13, buf1590, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_13
        buf1591 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1590, buf1591, 768, 49, grid=grid(768), stream=stream0)
        buf1593 = reinterpret_tensor(buf1590, (768, 49), (1, 768), 0); del buf1590  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1592, buf1593, 37632, 128, grid=grid(37632), stream=stream0)
        buf1594 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1593, buf1594, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1595 = aten.convolution_backward(reinterpret_tensor(buf1592, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_102, primals_205, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_102
        del primals_205
        buf1596 = buf1595[0]
        buf1597 = buf1595[1]
        del buf1595
        buf1598 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1596, buf1598, 768, 6272, grid=grid(768), stream=stream0)
        buf1599 = reinterpret_tensor(buf1593, (768, 49), (49, 1), 0); del buf1593  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1596, convolution_12, unsqueeze_347, buf1599, 37632, 128, grid=grid(37632), stream=stream0)
        buf1600 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1601 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1599, squeeze_22, buf1600, buf1601, 768, 49, grid=grid(768), stream=stream0)
        buf1602 = buf1596; del buf1596  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1602, convolution_12, unsqueeze_347, buf1600, squeeze_22, buf1598, primals_203, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_12
        del primals_203
        del squeeze_22
        del unsqueeze_347
        buf1603 = buf1600; del buf1600  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1602, buf1603, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1604 = aten.convolution_backward(buf1602, view_88, primals_201, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_201
        del view_88
        buf1605 = buf1604[0]
        buf1606 = buf1604[1]
        del buf1604
        buf1607 = reinterpret_tensor(buf1599, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1599  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1605, primals_199, buf1607, 37632, 128, grid=grid(37632), stream=stream0)
        buf1608 = buf1533; del buf1533  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1607, buf1608, 6272, 6, grid=grid(6272), stream=stream0)
        buf1609 = reinterpret_tensor(buf1607, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1607  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1605, primals_199, mul_129, buf1609, 37632, 128, grid=grid(37632), stream=stream0)
        buf1610 = buf1531; del buf1531  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1609, buf1610, 6272, 6, grid=grid(6272), stream=stream0)
        buf1611 = reinterpret_tensor(buf1609, (768, 49), (49, 1), 0); del buf1609  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1605, mul_129, buf1611, 37632, 128, grid=grid(37632), stream=stream0)
        buf1612 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1611, buf1612, 768, 49, grid=grid(768), stream=stream0)
        buf1613 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1605, buf1613, 768, 6272, grid=grid(768), stream=stream0)
        buf1614 = buf1589; del buf1589  # reuse
        buf1619 = reinterpret_tensor(buf1602, (8, 784, 768), (602112, 768, 1), 0); del buf1602  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1614, div_293, buf1605, primals_199, buf1608, mul_129, buf1610, primals_17, buf1619, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_293
        del mul_129
        del primals_199
        buf1615 = reinterpret_tensor(buf1611, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1611  # reuse
        buf1617 = reinterpret_tensor(buf1585, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1585  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1614, mm_4, primals_198, primals_17, buf1615, buf1617, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_4
        del primals_17
        del primals_198
        buf1616 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1615, buf1616, 768, 49, grid=grid(768), stream=stream0)
        buf1618 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1617, buf1618, 768, 49, grid=grid(768), stream=stream0)
        buf1620 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1619, (768, 6272), (1, 768), 0), view_86, out=buf1620)
        del view_86
        buf1621 = reinterpret_tensor(buf1605, (6272, 768), (768, 1), 0); del buf1605  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1619, (6272, 768), (768, 1), 0), permute_787, out=buf1621)
        del permute_787
        buf1622 = reinterpret_tensor(buf1619, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1619  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1621, buf1622, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1623 = reinterpret_tensor(buf1621, (128, 48, 784), (37632, 784, 1), 0); del buf1621  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_790, reinterpret_tensor(buf1622, (128, 48, 784), (37632, 784, 1), 0), out=buf1623)
        del permute_790
        buf1624 = reinterpret_tensor(buf1551, (128, 48, 48), (2304, 48, 1), 0); del buf1551  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1622, (128, 48, 784), (37632, 784, 1), 0), permute_791, out=buf1624)
        del permute_791
        buf1625 = buf1557; del buf1557  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1624, alias_133, buf1625, 6144, 48, grid=grid(6144), stream=stream0)
        buf1626 = buf1549; del buf1549  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1624, alias_133, buf1625, bmm_8, buf1626, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_8
        buf1627 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1626, buf1627, 16, 3, grid=grid(16), stream=stream0)
        buf1628 = reinterpret_tensor(buf1624, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1624  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1628, alias_133, buf1625, primals_18, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_133
        del primals_18
        buf1629 = reinterpret_tensor(buf1622, (128, 784, 48), (37632, 48, 1), 0); del buf1622  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_792, reinterpret_tensor(buf1628, (128, 48, 48), (2304, 48, 1), 0), out=buf1629)
        del permute_792
        buf1630 = reinterpret_tensor(buf1592, (128, 48, 784), (37632, 784, 1), 0); del buf1592  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1628, (128, 48, 48), (2304, 48, 1), 0), permute_793, out=buf1630)
        del permute_793
        buf1631 = reinterpret_tensor(buf1556, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1556  # reuse
        # Source Nodes: [k_9], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1629, getitem_53, pow_21, buf1631, 43008, 112, grid=grid(43008), stream=stream0)
        buf1632 = buf1625; del buf1625  # reuse
        # Source Nodes: [k_9], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1631, buf1632, 6144, 7, grid=grid(6144), stream=stream0)
        buf1633 = reinterpret_tensor(buf1631, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1631  # reuse
        # Source Nodes: [q_9], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1630, getitem_52, pow_19, buf1633, 43008, 112, grid=grid(43008), stream=stream0)
        buf1634 = buf1555; del buf1555  # reuse
        # Source Nodes: [q_9], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1633, buf1634, 6144, 7, grid=grid(6144), stream=stream0)
        buf1635 = reinterpret_tensor(buf1559, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1559  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1630, pow_19, buf1634, getitem_52, buf1629, pow_21, buf1632, getitem_53, buf1623, buf1635, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1623
        del getitem_52
        del getitem_53
        del pow_19
        del pow_21
        buf1636 = reinterpret_tensor(buf1558, (6272, 2304), (2304, 1), 0); del buf1558  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1635, buf1636, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1637 = reinterpret_tensor(buf1630, (6272, 768), (768, 1), 0); del buf1630  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1636, permute_796, out=buf1637)
        del permute_796
        buf1638 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1636, (2304, 6272), (1, 2304), 0), view_76, out=buf1638)
        del view_76
        buf1639 = buf1562; del buf1562  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1636, buf1639, 112896, 128, grid=grid(112896), stream=stream0)
        buf1640 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1639, buf1640, 2304, 49, grid=grid(2304), stream=stream0)
        buf1647 = buf1614; del buf1614  # reuse
        buf1650 = reinterpret_tensor(buf1629, (8, 784, 768), (602112, 768, 1), 0); del buf1629  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1647, buf1637, primals_193, mul_125, div_302, primals_16, buf1650, 6272, 768, grid=grid(6272), stream=stream0)
        del div_302
        del primals_16
        del primals_193
        buf1643 = reinterpret_tensor(buf1617, (768, 49), (1, 768), 0); del buf1617  # reuse
        buf1645 = reinterpret_tensor(buf1615, (768, 49), (1, 768), 0); del buf1615  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1637, mul_125, buf1643, buf1645, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_125
        buf1644 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1643, buf1644, 768, 49, grid=grid(768), stream=stream0)
        buf1646 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1645, buf1646, 768, 49, grid=grid(768), stream=stream0)
        buf1648 = reinterpret_tensor(buf1645, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1645  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1647, addmm_11, buf1648, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_11
        buf1649 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1648, buf1649, 768, 49, grid=grid(768), stream=stream0)
        buf1651 = reinterpret_tensor(buf1578, (6272, 3072), (3072, 1), 0); del buf1578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1650, (6272, 768), (768, 1), 0), permute_800, out=buf1651)
        del permute_800
        buf1652 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1650, (768, 6272), (1, 768), 0), view_74, out=buf1652)
        del view_74
        buf1653 = reinterpret_tensor(buf1648, (1, 768, 49), (37632, 1, 768), 0); del buf1648  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1650, buf1653, 37632, 128, grid=grid(37632), stream=stream0)
        buf1654 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1653, buf1654, 768, 49, grid=grid(768), stream=stream0)
        buf1655 = reinterpret_tensor(buf1651, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1651  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1655, addmm_10, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_10
        buf1656 = reinterpret_tensor(buf1650, (6272, 768), (768, 1), 0); del buf1650  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1655, (6272, 3072), (3072, 1), 0), permute_804, out=buf1656)
        del permute_804
        buf1657 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1655, (3072, 6272), (1, 3072), 0), view_72, out=buf1657)
        del view_72
        buf1658 = buf1581; del buf1581  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1655, buf1658, 150528, 128, grid=grid(150528), stream=stream0)
        buf1659 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1658, buf1659, 3072, 49, grid=grid(3072), stream=stream0)
        buf1666 = buf1647; del buf1647  # reuse
        buf1669 = reinterpret_tensor(buf1637, (8, 784, 768), (602112, 768, 1), 0); del buf1637  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1666, buf1656, primals_187, mul_119, div_303, primals_15, buf1669, 6272, 768, grid=grid(6272), stream=stream0)
        del div_303
        del primals_15
        del primals_187
        buf1662 = reinterpret_tensor(buf1653, (768, 49), (1, 768), 0); del buf1653  # reuse
        buf1664 = buf1643; del buf1643  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1656, mul_119, buf1662, buf1664, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1656
        del mul_119
        buf1663 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1662, buf1663, 768, 49, grid=grid(768), stream=stream0)
        buf1665 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1664, buf1665, 768, 49, grid=grid(768), stream=stream0)
        buf1667 = reinterpret_tensor(buf1664, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1664  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1666, convolution_11, buf1667, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_11
        buf1668 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1667, buf1668, 768, 49, grid=grid(768), stream=stream0)
        buf1670 = reinterpret_tensor(buf1667, (768, 49), (1, 768), 0); del buf1667  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1669, buf1670, 37632, 128, grid=grid(37632), stream=stream0)
        buf1671 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1670, buf1671, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1672 = aten.convolution_backward(reinterpret_tensor(buf1669, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_85, primals_185, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_85
        del primals_185
        buf1673 = buf1672[0]
        buf1674 = buf1672[1]
        del buf1672
        buf1675 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1673, buf1675, 768, 6272, grid=grid(768), stream=stream0)
        buf1676 = reinterpret_tensor(buf1670, (768, 49), (49, 1), 0); del buf1670  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1673, convolution_10, unsqueeze_359, buf1676, 37632, 128, grid=grid(37632), stream=stream0)
        buf1677 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1678 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1676, squeeze_19, buf1677, buf1678, 768, 49, grid=grid(768), stream=stream0)
        buf1679 = buf1673; del buf1673  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1679, convolution_10, unsqueeze_359, buf1677, squeeze_19, buf1675, primals_183, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_10
        del primals_183
        del squeeze_19
        del unsqueeze_359
        buf1680 = buf1677; del buf1677  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1679, buf1680, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1681 = aten.convolution_backward(buf1679, view_70, primals_181, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_181
        del view_70
        buf1682 = buf1681[0]
        buf1683 = buf1681[1]
        del buf1681
        buf1684 = reinterpret_tensor(buf1676, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1676  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1682, primals_179, buf1684, 37632, 128, grid=grid(37632), stream=stream0)
        buf1685 = buf1610; del buf1610  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1684, buf1685, 6272, 6, grid=grid(6272), stream=stream0)
        buf1686 = reinterpret_tensor(buf1684, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1684  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1682, primals_179, mul_106, buf1686, 37632, 128, grid=grid(37632), stream=stream0)
        buf1687 = buf1608; del buf1608  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1686, buf1687, 6272, 6, grid=grid(6272), stream=stream0)
        buf1688 = reinterpret_tensor(buf1686, (768, 49), (49, 1), 0); del buf1686  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1682, mul_106, buf1688, 37632, 128, grid=grid(37632), stream=stream0)
        buf1689 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1688, buf1689, 768, 49, grid=grid(768), stream=stream0)
        buf1690 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1682, buf1690, 768, 6272, grid=grid(768), stream=stream0)
        buf1691 = buf1666; del buf1666  # reuse
        buf1696 = reinterpret_tensor(buf1679, (8, 784, 768), (602112, 768, 1), 0); del buf1679  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1691, div_304, buf1682, primals_179, buf1685, mul_106, buf1687, primals_13, buf1696, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_304
        del mul_106
        del primals_179
        buf1692 = reinterpret_tensor(buf1688, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1688  # reuse
        buf1694 = reinterpret_tensor(buf1662, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1662  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1691, mm_3, primals_178, primals_13, buf1692, buf1694, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_3
        del primals_13
        del primals_178
        buf1693 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1692, buf1693, 768, 49, grid=grid(768), stream=stream0)
        buf1695 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1694, buf1695, 768, 49, grid=grid(768), stream=stream0)
        buf1697 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1696, (768, 6272), (1, 768), 0), view_68, out=buf1697)
        del view_68
        buf1698 = reinterpret_tensor(buf1682, (6272, 768), (768, 1), 0); del buf1682  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1696, (6272, 768), (768, 1), 0), permute_812, out=buf1698)
        del permute_812
        buf1699 = reinterpret_tensor(buf1696, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1696  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1698, buf1699, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1700 = reinterpret_tensor(buf1698, (128, 48, 784), (37632, 784, 1), 0); del buf1698  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_815, reinterpret_tensor(buf1699, (128, 48, 784), (37632, 784, 1), 0), out=buf1700)
        del permute_815
        buf1701 = reinterpret_tensor(buf1628, (128, 48, 48), (2304, 48, 1), 0); del buf1628  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1699, (128, 48, 784), (37632, 784, 1), 0), permute_816, out=buf1701)
        del permute_816
        buf1702 = buf1634; del buf1634  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1701, alias_136, buf1702, 6144, 48, grid=grid(6144), stream=stream0)
        buf1703 = buf1626; del buf1626  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1701, alias_136, buf1702, bmm_6, buf1703, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_6
        buf1704 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1703, buf1704, 16, 3, grid=grid(16), stream=stream0)
        buf1705 = reinterpret_tensor(buf1701, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1701  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1705, alias_136, buf1702, primals_14, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_136
        del primals_14
        buf1706 = reinterpret_tensor(buf1699, (128, 784, 48), (37632, 48, 1), 0); del buf1699  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_817, reinterpret_tensor(buf1705, (128, 48, 48), (2304, 48, 1), 0), out=buf1706)
        del permute_817
        buf1707 = reinterpret_tensor(buf1669, (128, 48, 784), (37632, 784, 1), 0); del buf1669  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1705, (128, 48, 48), (2304, 48, 1), 0), permute_818, out=buf1707)
        del permute_818
        buf1708 = reinterpret_tensor(buf1633, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1633  # reuse
        # Source Nodes: [k_7], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1706, getitem_42, pow_17, buf1708, 43008, 112, grid=grid(43008), stream=stream0)
        buf1709 = buf1702; del buf1702  # reuse
        # Source Nodes: [k_7], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1708, buf1709, 6144, 7, grid=grid(6144), stream=stream0)
        buf1710 = reinterpret_tensor(buf1708, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1708  # reuse
        # Source Nodes: [q_7], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1707, getitem_41, pow_15, buf1710, 43008, 112, grid=grid(43008), stream=stream0)
        buf1711 = buf1632; del buf1632  # reuse
        # Source Nodes: [q_7], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1710, buf1711, 6144, 7, grid=grid(6144), stream=stream0)
        buf1712 = reinterpret_tensor(buf1636, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1636  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1707, pow_15, buf1711, getitem_41, buf1706, pow_17, buf1709, getitem_42, buf1700, buf1712, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1700
        del getitem_41
        del getitem_42
        del pow_15
        del pow_17
        buf1713 = reinterpret_tensor(buf1635, (6272, 2304), (2304, 1), 0); del buf1635  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1712, buf1713, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1714 = reinterpret_tensor(buf1707, (6272, 768), (768, 1), 0); del buf1707  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1713, permute_821, out=buf1714)
        del permute_821
        buf1715 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1713, (2304, 6272), (1, 2304), 0), view_58, out=buf1715)
        del view_58
        buf1716 = buf1639; del buf1639  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1713, buf1716, 112896, 128, grid=grid(112896), stream=stream0)
        buf1717 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1716, buf1717, 2304, 49, grid=grid(2304), stream=stream0)
        buf1724 = buf1691; del buf1691  # reuse
        buf1727 = reinterpret_tensor(buf1706, (8, 784, 768), (602112, 768, 1), 0); del buf1706  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1724, buf1714, primals_173, mul_102, div_313, primals_12, buf1727, 6272, 768, grid=grid(6272), stream=stream0)
        del div_313
        del primals_12
        del primals_173
        buf1720 = reinterpret_tensor(buf1694, (768, 49), (1, 768), 0); del buf1694  # reuse
        buf1722 = reinterpret_tensor(buf1692, (768, 49), (1, 768), 0); del buf1692  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1714, mul_102, buf1720, buf1722, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_102
        buf1721 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1720, buf1721, 768, 49, grid=grid(768), stream=stream0)
        buf1723 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1722, buf1723, 768, 49, grid=grid(768), stream=stream0)
        buf1725 = reinterpret_tensor(buf1722, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1722  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1724, addmm_8, buf1725, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_8
        buf1726 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1725, buf1726, 768, 49, grid=grid(768), stream=stream0)
        buf1728 = reinterpret_tensor(buf1655, (6272, 3072), (3072, 1), 0); del buf1655  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1727, (6272, 768), (768, 1), 0), permute_825, out=buf1728)
        del permute_825
        buf1729 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1727, (768, 6272), (1, 768), 0), view_56, out=buf1729)
        del view_56
        buf1730 = reinterpret_tensor(buf1725, (1, 768, 49), (37632, 1, 768), 0); del buf1725  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1727, buf1730, 37632, 128, grid=grid(37632), stream=stream0)
        buf1731 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1730, buf1731, 768, 49, grid=grid(768), stream=stream0)
        buf1732 = reinterpret_tensor(buf1728, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1728  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1732, addmm_7, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_7
        buf1733 = reinterpret_tensor(buf1727, (6272, 768), (768, 1), 0); del buf1727  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1732, (6272, 3072), (3072, 1), 0), permute_829, out=buf1733)
        del permute_829
        buf1734 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1732, (3072, 6272), (1, 3072), 0), view_54, out=buf1734)
        del view_54
        buf1735 = buf1658; del buf1658  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1732, buf1735, 150528, 128, grid=grid(150528), stream=stream0)
        buf1736 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1735, buf1736, 3072, 49, grid=grid(3072), stream=stream0)
        buf1743 = buf1724; del buf1724  # reuse
        buf1746 = reinterpret_tensor(buf1714, (8, 784, 768), (602112, 768, 1), 0); del buf1714  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1743, buf1733, primals_167, mul_96, div_314, primals_11, buf1746, 6272, 768, grid=grid(6272), stream=stream0)
        del div_314
        del primals_11
        del primals_167
        buf1739 = reinterpret_tensor(buf1730, (768, 49), (1, 768), 0); del buf1730  # reuse
        buf1741 = buf1720; del buf1720  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1733, mul_96, buf1739, buf1741, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1733
        del mul_96
        buf1740 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1739, buf1740, 768, 49, grid=grid(768), stream=stream0)
        buf1742 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1741, buf1742, 768, 49, grid=grid(768), stream=stream0)
        buf1744 = reinterpret_tensor(buf1741, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1741  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1743, convolution_9, buf1744, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_9
        buf1745 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1744, buf1745, 768, 49, grid=grid(768), stream=stream0)
        buf1747 = reinterpret_tensor(buf1744, (768, 49), (1, 768), 0); del buf1744  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1746, buf1747, 37632, 128, grid=grid(37632), stream=stream0)
        buf1748 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1747, buf1748, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1749 = aten.convolution_backward(reinterpret_tensor(buf1746, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_68, primals_165, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_68
        del primals_165
        buf1750 = buf1749[0]
        buf1751 = buf1749[1]
        del buf1749
        buf1752 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1750, buf1752, 768, 6272, grid=grid(768), stream=stream0)
        buf1753 = reinterpret_tensor(buf1747, (768, 49), (49, 1), 0); del buf1747  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1750, convolution_8, unsqueeze_371, buf1753, 37632, 128, grid=grid(37632), stream=stream0)
        buf1754 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1755 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1753, squeeze_16, buf1754, buf1755, 768, 49, grid=grid(768), stream=stream0)
        buf1756 = buf1750; del buf1750  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1756, convolution_8, unsqueeze_371, buf1754, squeeze_16, buf1752, primals_163, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_8
        del primals_163
        del squeeze_16
        del unsqueeze_371
        buf1757 = buf1754; del buf1754  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1756, buf1757, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1758 = aten.convolution_backward(buf1756, view_52, primals_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_161
        del view_52
        buf1759 = buf1758[0]
        buf1760 = buf1758[1]
        del buf1758
        buf1761 = reinterpret_tensor(buf1753, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1753  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1759, primals_159, buf1761, 37632, 128, grid=grid(37632), stream=stream0)
        buf1762 = buf1687; del buf1687  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1761, buf1762, 6272, 6, grid=grid(6272), stream=stream0)
        buf1763 = reinterpret_tensor(buf1761, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1761  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1759, primals_159, mul_83, buf1763, 37632, 128, grid=grid(37632), stream=stream0)
        buf1764 = buf1685; del buf1685  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1763, buf1764, 6272, 6, grid=grid(6272), stream=stream0)
        buf1765 = reinterpret_tensor(buf1763, (768, 49), (49, 1), 0); del buf1763  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1759, mul_83, buf1765, 37632, 128, grid=grid(37632), stream=stream0)
        buf1766 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1765, buf1766, 768, 49, grid=grid(768), stream=stream0)
        buf1767 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1759, buf1767, 768, 6272, grid=grid(768), stream=stream0)
        buf1768 = buf1743; del buf1743  # reuse
        buf1773 = reinterpret_tensor(buf1756, (8, 784, 768), (602112, 768, 1), 0); del buf1756  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1768, div_315, buf1759, primals_159, buf1762, mul_83, buf1764, primals_9, buf1773, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_315
        del mul_83
        del primals_159
        buf1769 = reinterpret_tensor(buf1765, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1765  # reuse
        buf1771 = reinterpret_tensor(buf1739, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1739  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1768, mm_2, primals_158, primals_9, buf1769, buf1771, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_2
        del primals_158
        del primals_9
        buf1770 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1769, buf1770, 768, 49, grid=grid(768), stream=stream0)
        buf1772 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1771, buf1772, 768, 49, grid=grid(768), stream=stream0)
        buf1774 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1773, (768, 6272), (1, 768), 0), view_50, out=buf1774)
        del view_50
        buf1775 = reinterpret_tensor(buf1759, (6272, 768), (768, 1), 0); del buf1759  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1773, (6272, 768), (768, 1), 0), permute_837, out=buf1775)
        del permute_837
        buf1776 = reinterpret_tensor(buf1773, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1773  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1775, buf1776, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1777 = reinterpret_tensor(buf1775, (128, 48, 784), (37632, 784, 1), 0); del buf1775  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_840, reinterpret_tensor(buf1776, (128, 48, 784), (37632, 784, 1), 0), out=buf1777)
        del permute_840
        buf1778 = reinterpret_tensor(buf1705, (128, 48, 48), (2304, 48, 1), 0); del buf1705  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1776, (128, 48, 784), (37632, 784, 1), 0), permute_841, out=buf1778)
        del permute_841
        buf1779 = buf1711; del buf1711  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1778, alias_139, buf1779, 6144, 48, grid=grid(6144), stream=stream0)
        buf1780 = buf1703; del buf1703  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1778, alias_139, buf1779, bmm_4, buf1780, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_4
        buf1781 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1780, buf1781, 16, 3, grid=grid(16), stream=stream0)
        buf1782 = reinterpret_tensor(buf1778, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1778  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1782, alias_139, buf1779, primals_10, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_139
        del primals_10
        buf1783 = reinterpret_tensor(buf1776, (128, 784, 48), (37632, 48, 1), 0); del buf1776  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_842, reinterpret_tensor(buf1782, (128, 48, 48), (2304, 48, 1), 0), out=buf1783)
        del permute_842
        buf1784 = reinterpret_tensor(buf1746, (128, 48, 784), (37632, 784, 1), 0); del buf1746  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1782, (128, 48, 48), (2304, 48, 1), 0), permute_843, out=buf1784)
        del permute_843
        buf1785 = reinterpret_tensor(buf1710, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1710  # reuse
        # Source Nodes: [k_5], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1783, getitem_31, pow_13, buf1785, 43008, 112, grid=grid(43008), stream=stream0)
        buf1786 = buf1779; del buf1779  # reuse
        # Source Nodes: [k_5], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1785, buf1786, 6144, 7, grid=grid(6144), stream=stream0)
        buf1787 = reinterpret_tensor(buf1785, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1785  # reuse
        # Source Nodes: [q_5], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1784, getitem_30, pow_11, buf1787, 43008, 112, grid=grid(43008), stream=stream0)
        buf1788 = buf1709; del buf1709  # reuse
        # Source Nodes: [q_5], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1787, buf1788, 6144, 7, grid=grid(6144), stream=stream0)
        buf1789 = reinterpret_tensor(buf1713, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1713  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1784, pow_11, buf1788, getitem_30, buf1783, pow_13, buf1786, getitem_31, buf1777, buf1789, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1777
        del getitem_30
        del getitem_31
        del pow_11
        del pow_13
        buf1790 = reinterpret_tensor(buf1712, (6272, 2304), (2304, 1), 0); del buf1712  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1789, buf1790, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1791 = reinterpret_tensor(buf1784, (6272, 768), (768, 1), 0); del buf1784  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1790, permute_846, out=buf1791)
        del permute_846
        buf1792 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1790, (2304, 6272), (1, 2304), 0), view_40, out=buf1792)
        del view_40
        buf1793 = buf1716; del buf1716  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1790, buf1793, 112896, 128, grid=grid(112896), stream=stream0)
        buf1794 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1793, buf1794, 2304, 49, grid=grid(2304), stream=stream0)
        buf1801 = buf1768; del buf1768  # reuse
        buf1804 = reinterpret_tensor(buf1783, (8, 784, 768), (602112, 768, 1), 0); del buf1783  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1801, buf1791, primals_153, mul_79, div_324, primals_8, buf1804, 6272, 768, grid=grid(6272), stream=stream0)
        del div_324
        del primals_153
        del primals_8
        buf1797 = reinterpret_tensor(buf1771, (768, 49), (1, 768), 0); del buf1771  # reuse
        buf1799 = reinterpret_tensor(buf1769, (768, 49), (1, 768), 0); del buf1769  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1791, mul_79, buf1797, buf1799, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_79
        buf1798 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1797, buf1798, 768, 49, grid=grid(768), stream=stream0)
        buf1800 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1799, buf1800, 768, 49, grid=grid(768), stream=stream0)
        buf1802 = reinterpret_tensor(buf1799, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1799  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1801, addmm_5, buf1802, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_5
        buf1803 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1802, buf1803, 768, 49, grid=grid(768), stream=stream0)
        buf1805 = reinterpret_tensor(buf1732, (6272, 3072), (3072, 1), 0); del buf1732  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1804, (6272, 768), (768, 1), 0), permute_850, out=buf1805)
        del permute_850
        buf1806 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1804, (768, 6272), (1, 768), 0), view_38, out=buf1806)
        del view_38
        buf1807 = reinterpret_tensor(buf1802, (1, 768, 49), (37632, 1, 768), 0); del buf1802  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1804, buf1807, 37632, 128, grid=grid(37632), stream=stream0)
        buf1808 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1807, buf1808, 768, 49, grid=grid(768), stream=stream0)
        buf1809 = reinterpret_tensor(buf1805, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1805  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1809, addmm_4, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_4
        buf1810 = reinterpret_tensor(buf1804, (6272, 768), (768, 1), 0); del buf1804  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1809, (6272, 3072), (3072, 1), 0), permute_854, out=buf1810)
        del permute_854
        buf1811 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1809, (3072, 6272), (1, 3072), 0), view_36, out=buf1811)
        del view_36
        buf1812 = buf1735; del buf1735  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1809, buf1812, 150528, 128, grid=grid(150528), stream=stream0)
        buf1813 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1812, buf1813, 3072, 49, grid=grid(3072), stream=stream0)
        buf1820 = buf1801; del buf1801  # reuse
        buf1823 = reinterpret_tensor(buf1791, (8, 784, 768), (602112, 768, 1), 0); del buf1791  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1820, buf1810, primals_147, mul_73, div_325, primals_7, buf1823, 6272, 768, grid=grid(6272), stream=stream0)
        del div_325
        del primals_147
        del primals_7
        buf1816 = reinterpret_tensor(buf1807, (768, 49), (1, 768), 0); del buf1807  # reuse
        buf1818 = buf1797; del buf1797  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1810, mul_73, buf1816, buf1818, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1810
        del mul_73
        buf1817 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1816, buf1817, 768, 49, grid=grid(768), stream=stream0)
        buf1819 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1818, buf1819, 768, 49, grid=grid(768), stream=stream0)
        buf1821 = reinterpret_tensor(buf1818, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1818  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1820, convolution_7, buf1821, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_7
        buf1822 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1821, buf1822, 768, 49, grid=grid(768), stream=stream0)
        buf1824 = reinterpret_tensor(buf1821, (768, 49), (1, 768), 0); del buf1821  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1823, buf1824, 37632, 128, grid=grid(37632), stream=stream0)
        buf1825 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1824, buf1825, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1826 = aten.convolution_backward(reinterpret_tensor(buf1823, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_51, primals_145, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_51
        del primals_145
        buf1827 = buf1826[0]
        buf1828 = buf1826[1]
        del buf1826
        buf1829 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1827, buf1829, 768, 6272, grid=grid(768), stream=stream0)
        buf1830 = reinterpret_tensor(buf1824, (768, 49), (49, 1), 0); del buf1824  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1827, convolution_6, unsqueeze_383, buf1830, 37632, 128, grid=grid(37632), stream=stream0)
        buf1831 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1832 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1830, squeeze_13, buf1831, buf1832, 768, 49, grid=grid(768), stream=stream0)
        buf1833 = buf1827; del buf1827  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1833, convolution_6, unsqueeze_383, buf1831, squeeze_13, buf1829, primals_143, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_6
        del primals_143
        del squeeze_13
        del unsqueeze_383
        buf1834 = buf1831; del buf1831  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1833, buf1834, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1835 = aten.convolution_backward(buf1833, view_34, primals_141, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_141
        del view_34
        buf1836 = buf1835[0]
        buf1837 = buf1835[1]
        del buf1835
        buf1838 = reinterpret_tensor(buf1830, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1830  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1836, primals_139, buf1838, 37632, 128, grid=grid(37632), stream=stream0)
        buf1839 = buf1764; del buf1764  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1838, buf1839, 6272, 6, grid=grid(6272), stream=stream0)
        buf1840 = reinterpret_tensor(buf1838, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1838  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1836, primals_139, mul_60, buf1840, 37632, 128, grid=grid(37632), stream=stream0)
        buf1841 = buf1762; del buf1762  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1840, buf1841, 6272, 6, grid=grid(6272), stream=stream0)
        buf1842 = reinterpret_tensor(buf1840, (768, 49), (49, 1), 0); del buf1840  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1836, mul_60, buf1842, 37632, 128, grid=grid(37632), stream=stream0)
        buf1843 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1842, buf1843, 768, 49, grid=grid(768), stream=stream0)
        buf1844 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1836, buf1844, 768, 6272, grid=grid(768), stream=stream0)
        buf1845 = buf1820; del buf1820  # reuse
        buf1850 = reinterpret_tensor(buf1833, (8, 784, 768), (602112, 768, 1), 0); del buf1833  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1845, div_326, buf1836, primals_139, buf1839, mul_60, buf1841, primals_5, buf1850, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del div_326
        del mul_60
        del primals_139
        buf1846 = reinterpret_tensor(buf1842, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1842  # reuse
        buf1848 = reinterpret_tensor(buf1816, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1816  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1845, mm_1, primals_138, primals_5, buf1846, buf1848, 37632, 128, grid=grid(37632), stream=stream0)
        del mm_1
        del primals_138
        del primals_5
        buf1847 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1846, buf1847, 768, 49, grid=grid(768), stream=stream0)
        buf1849 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1848, buf1849, 768, 49, grid=grid(768), stream=stream0)
        buf1851 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1850, (768, 6272), (1, 768), 0), view_32, out=buf1851)
        del view_32
        buf1852 = reinterpret_tensor(buf1836, (6272, 768), (768, 1), 0); del buf1836  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1850, (6272, 768), (768, 1), 0), permute_862, out=buf1852)
        del permute_862
        buf1853 = reinterpret_tensor(buf1850, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1850  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1852, buf1853, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1854 = reinterpret_tensor(buf1852, (128, 48, 784), (37632, 784, 1), 0); del buf1852  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_865, reinterpret_tensor(buf1853, (128, 48, 784), (37632, 784, 1), 0), out=buf1854)
        del permute_865
        buf1855 = reinterpret_tensor(buf1782, (128, 48, 48), (2304, 48, 1), 0); del buf1782  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1853, (128, 48, 784), (37632, 784, 1), 0), permute_866, out=buf1855)
        del permute_866
        buf1856 = buf1788; del buf1788  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1855, alias_142, buf1856, 6144, 48, grid=grid(6144), stream=stream0)
        buf1857 = buf1780; del buf1780  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1855, alias_142, buf1856, bmm_2, buf1857, 48, 6144, grid=grid(48), stream=stream0)
        del bmm_2
        buf1858 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1857, buf1858, 16, 3, grid=grid(16), stream=stream0)
        buf1859 = reinterpret_tensor(buf1855, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1855  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1859, alias_142, buf1856, primals_6, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_142
        del primals_6
        buf1860 = reinterpret_tensor(buf1853, (128, 784, 48), (37632, 48, 1), 0); del buf1853  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_867, reinterpret_tensor(buf1859, (128, 48, 48), (2304, 48, 1), 0), out=buf1860)
        del permute_867
        buf1861 = reinterpret_tensor(buf1823, (128, 48, 784), (37632, 784, 1), 0); del buf1823  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1859, (128, 48, 48), (2304, 48, 1), 0), permute_868, out=buf1861)
        del permute_868
        buf1862 = reinterpret_tensor(buf1787, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1787  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1860, getitem_20, pow_9, buf1862, 43008, 112, grid=grid(43008), stream=stream0)
        buf1863 = buf1856; del buf1856  # reuse
        # Source Nodes: [k_3], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1862, buf1863, 6144, 7, grid=grid(6144), stream=stream0)
        buf1864 = reinterpret_tensor(buf1862, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1862  # reuse
        # Source Nodes: [q_3], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1861, getitem_19, pow_7, buf1864, 43008, 112, grid=grid(43008), stream=stream0)
        buf1865 = buf1786; del buf1786  # reuse
        # Source Nodes: [q_3], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1864, buf1865, 6144, 7, grid=grid(6144), stream=stream0)
        buf1866 = reinterpret_tensor(buf1790, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1790  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1861, pow_7, buf1865, getitem_19, buf1860, pow_9, buf1863, getitem_20, buf1854, buf1866, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1854
        del getitem_19
        del getitem_20
        del pow_7
        del pow_9
        buf1867 = reinterpret_tensor(buf1789, (6272, 2304), (2304, 1), 0); del buf1789  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1866, buf1867, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        buf1868 = reinterpret_tensor(buf1861, (6272, 768), (768, 1), 0); del buf1861  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1867, permute_871, out=buf1868)
        del permute_871
        buf1869 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1867, (2304, 6272), (1, 2304), 0), view_22, out=buf1869)
        del view_22
        buf1870 = buf1793; del buf1793  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1867, buf1870, 112896, 128, grid=grid(112896), stream=stream0)
        buf1871 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1870, buf1871, 2304, 49, grid=grid(2304), stream=stream0)
        buf1878 = buf1845; del buf1845  # reuse
        buf1881 = reinterpret_tensor(buf1860, (8, 784, 768), (602112, 768, 1), 0); del buf1860  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1878, buf1868, primals_133, mul_56, div_335, primals_4, buf1881, 6272, 768, grid=grid(6272), stream=stream0)
        del div_335
        del primals_133
        del primals_4
        buf1874 = reinterpret_tensor(buf1848, (768, 49), (1, 768), 0); del buf1848  # reuse
        buf1876 = reinterpret_tensor(buf1846, (768, 49), (1, 768), 0); del buf1846  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1868, mul_56, buf1874, buf1876, 37632, 128, grid=grid(37632), stream=stream0)
        del mul_56
        buf1875 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1874, buf1875, 768, 49, grid=grid(768), stream=stream0)
        buf1877 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1876, buf1877, 768, 49, grid=grid(768), stream=stream0)
        buf1879 = reinterpret_tensor(buf1876, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1876  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1878, addmm_2, buf1879, 37632, 128, grid=grid(37632), stream=stream0)
        del addmm_2
        buf1880 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1879, buf1880, 768, 49, grid=grid(768), stream=stream0)
        buf1882 = reinterpret_tensor(buf1809, (6272, 3072), (3072, 1), 0); del buf1809  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1881, (6272, 768), (768, 1), 0), permute_875, out=buf1882)
        del permute_875
        buf1883 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1881, (768, 6272), (1, 768), 0), view_20, out=buf1883)
        del view_20
        buf1884 = reinterpret_tensor(buf1879, (1, 768, 49), (37632, 1, 768), 0); del buf1879  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf1881, buf1884, 37632, 128, grid=grid(37632), stream=stream0)
        buf1885 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_mul_sum_22.run(buf1884, buf1885, 768, 49, grid=grid(768), stream=stream0)
        buf1886 = reinterpret_tensor(buf1882, (8, 784, 3072), (2408448, 3072, 1), 0); del buf1882  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1886, addmm_1, 19267584, grid=grid(19267584), stream=stream0)
        del addmm_1
        buf1887 = reinterpret_tensor(buf1881, (6272, 768), (768, 1), 0); del buf1881  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1886, (6272, 3072), (3072, 1), 0), permute_879, out=buf1887)
        del permute_879
        buf1888 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1886, (3072, 6272), (1, 3072), 0), view_18, out=buf1888)
        del view_18
        buf1889 = buf1812; del buf1812  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1886, buf1889, 150528, 128, grid=grid(150528), stream=stream0)
        del buf1886
        buf1890 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1889, buf1890, 3072, 49, grid=grid(3072), stream=stream0)
        del buf1889
        buf1897 = buf1878; del buf1878  # reuse
        buf1900 = reinterpret_tensor(buf1868, (8, 784, 768), (602112, 768, 1), 0); del buf1868  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_57.run(buf1897, buf1887, primals_127, mul_50, div_336, primals_3, buf1900, 6272, 768, grid=grid(6272), stream=stream0)
        del div_336
        del primals_127
        del primals_3
        buf1893 = reinterpret_tensor(buf1884, (768, 49), (1, 768), 0); del buf1884  # reuse
        buf1895 = buf1874; del buf1874  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1887, mul_50, buf1893, buf1895, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1887
        del mul_50
        buf1894 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1893, buf1894, 768, 49, grid=grid(768), stream=stream0)
        buf1896 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1895, buf1896, 768, 49, grid=grid(768), stream=stream0)
        buf1898 = reinterpret_tensor(buf1895, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1895  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_30.run(buf1897, convolution_5, buf1898, 37632, 128, grid=grid(37632), stream=stream0)
        del convolution_5
        buf1899 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1898, buf1899, 768, 49, grid=grid(768), stream=stream0)
        buf1901 = reinterpret_tensor(buf1898, (768, 49), (1, 768), 0); del buf1898  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf1900, buf1901, 37632, 128, grid=grid(37632), stream=stream0)
        buf1902 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_mul_sum_22.run(buf1901, buf1902, 768, 49, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1903 = aten.convolution_backward(reinterpret_tensor(buf1900, (8, 768, 28, 28), (602112, 1, 21504, 768), 0), add_34, primals_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del add_34
        del primals_125
        buf1904 = buf1903[0]
        buf1905 = buf1903[1]
        del buf1903
        buf1906 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1904, buf1906, 768, 6272, grid=grid(768), stream=stream0)
        buf1907 = reinterpret_tensor(buf1901, (768, 49), (49, 1), 0); del buf1901  # reuse
        # Source Nodes: [x_11], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_red_fused_gelu_native_batch_norm_backward_33.run(buf1904, convolution_4, unsqueeze_395, buf1907, 37632, 128, grid=grid(37632), stream=stream0)
        buf1908 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1909 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.gelu, aten.native_batch_norm_backward]
        triton_per_fused_gelu_native_batch_norm_backward_34.run(buf1907, squeeze_10, buf1908, buf1909, 768, 49, grid=grid(768), stream=stream0)
        buf1910 = buf1904; del buf1904  # reuse
        # Source Nodes: [x_11], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_gelu_gelu_backward_native_batch_norm_backward_35.run(buf1910, convolution_4, unsqueeze_395, buf1908, squeeze_10, buf1906, primals_123, 6144, 784, grid=grid(6144, 784), stream=stream0)
        del convolution_4
        del primals_123
        del squeeze_10
        del unsqueeze_395
        buf1911 = buf1908; del buf1908  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1910, buf1911, 768, 6272, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1912 = aten.convolution_backward(buf1910, view_16, primals_121, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del primals_121
        del view_16
        buf1913 = buf1912[0]
        buf1914 = buf1912[1]
        del buf1912
        buf1915 = reinterpret_tensor(buf1907, (8, 784, 1, 6), (4704, 1, 37632, 784), 0); del buf1907  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_36.run(buf1913, primals_119, buf1915, 37632, 128, grid=grid(37632), stream=stream0)
        buf1916 = buf1841; del buf1841  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_37.run(buf1915, buf1916, 6272, 6, grid=grid(6272), stream=stream0)
        buf1917 = reinterpret_tensor(buf1915, (8, 784, 1, 6), (4704, 6, 37632, 1), 0); del buf1915  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf1913, primals_119, mul_37, buf1917, 37632, 128, grid=grid(37632), stream=stream0)
        buf1918 = buf1839; del buf1839  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf1917, buf1918, 6272, 6, grid=grid(6272), stream=stream0)
        buf1919 = reinterpret_tensor(buf1917, (768, 49), (49, 1), 0); del buf1917  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf1913, mul_37, buf1919, 37632, 128, grid=grid(37632), stream=stream0)
        buf1920 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf1919, buf1920, 768, 49, grid=grid(768), stream=stream0)
        buf1921 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf1913, buf1921, 768, 6272, grid=grid(768), stream=stream0)
        buf1922 = buf1897; del buf1897  # reuse
        buf1927 = reinterpret_tensor(buf1910, (8, 784, 768), (602112, 768, 1), 0); del buf1910  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_poi_fused_add_mul_native_layer_norm_backward_42.run(buf1922, div_337, buf1913, primals_119, buf1916, mul_37, buf1918, primals_1, buf1927, 6272, 768, grid=grid(6272, 768), stream=stream0)
        del buf1916
        del buf1918
        del div_337
        del mul_37
        del primals_119
        buf1923 = reinterpret_tensor(buf1919, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1919  # reuse
        buf1925 = reinterpret_tensor(buf1893, (1, 1, 768, 49), (37632, 37632, 1, 768), 0); del buf1893  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_red_fused_add_mul_sum_43.run(buf1922, mm, primals_118, primals_1, buf1923, buf1925, 37632, 128, grid=grid(37632), stream=stream0)
        del mm
        del primals_1
        del primals_118
        buf1924 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.add, aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1923, buf1924, 768, 49, grid=grid(768), stream=stream0)
        buf1926 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_22.run(buf1925, buf1926, 768, 49, grid=grid(768), stream=stream0)
        buf1928 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1927, (768, 6272), (1, 768), 0), view_14, out=buf1928)
        del view_14
        buf1929 = reinterpret_tensor(buf1913, (6272, 768), (768, 1), 0); del buf1913  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1927, (6272, 768), (768, 1), 0), permute_887, out=buf1929)
        del permute_887
        buf1930 = reinterpret_tensor(buf1927, (8, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1927  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf1929, buf1930, 6144, 784, grid=grid(6144, 784), stream=stream0)
        buf1931 = reinterpret_tensor(buf1929, (128, 48, 784), (37632, 784, 1), 0); del buf1929  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_890, reinterpret_tensor(buf1930, (128, 48, 784), (37632, 784, 1), 0), out=buf1931)
        del permute_890
        buf1932 = reinterpret_tensor(buf1859, (128, 48, 48), (2304, 48, 1), 0); del buf1859  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1930, (128, 48, 784), (37632, 784, 1), 0), permute_891, out=buf1932)
        del permute_891
        buf1933 = buf1865; del buf1865  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_45.run(buf1932, alias_145, buf1933, 6144, 48, grid=grid(6144), stream=stream0)
        buf1934 = buf1857; del buf1857  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_red_fused__softmax_backward_data_mul_sum_46.run(buf1932, alias_145, buf1933, bmm, buf1934, 48, 6144, grid=grid(48), stream=stream0)
        del bmm
        buf1935 = empty((1, 16, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_47.run(buf1934, buf1935, 16, 3, grid=grid(16), stream=stream0)
        del buf1934
        buf1936 = reinterpret_tensor(buf1932, (8, 16, 48, 48), (36864, 2304, 48, 1), 0); del buf1932  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_poi_fused__softmax_backward_data_mul_48.run(buf1936, alias_145, buf1933, primals_2, 128, 2304, grid=grid(128, 2304), stream=stream0)
        del alias_145
        del primals_2
        buf1937 = reinterpret_tensor(buf1930, (128, 784, 48), (37632, 48, 1), 0); del buf1930  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_892, reinterpret_tensor(buf1936, (128, 48, 48), (2304, 48, 1), 0), out=buf1937)
        del permute_892
        buf1938 = reinterpret_tensor(buf1900, (128, 48, 784), (37632, 784, 1), 0); del buf1900  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1936, (128, 48, 48), (2304, 48, 1), 0), permute_893, out=buf1938)
        del buf1936
        del permute_893
        buf1939 = reinterpret_tensor(buf1864, (8, 16, 48, 1, 7), (5376, 336, 1, 43008, 48), 0); del buf1864  # reuse
        # Source Nodes: [k_1], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_49.run(buf1937, getitem_9, pow_5, buf1939, 43008, 112, grid=grid(43008), stream=stream0)
        buf1940 = buf1933; del buf1933  # reuse
        # Source Nodes: [k_1], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_50.run(buf1939, buf1940, 6144, 7, grid=grid(6144), stream=stream0)
        buf1941 = reinterpret_tensor(buf1939, (8, 16, 48, 1, 7), (5376, 336, 7, 43008, 1), 0); del buf1939  # reuse
        # Source Nodes: [q_1], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_red_fused_div_mul_neg_sum_51.run(buf1938, getitem_8, pow_3, buf1941, 43008, 112, grid=grid(43008), stream=stream0)
        buf1942 = buf1863; del buf1863  # reuse
        # Source Nodes: [q_1], Original ATen: [aten.div, aten.mul, aten.neg, aten.sum]
        triton_per_fused_div_mul_neg_sum_52.run(buf1941, buf1942, 6144, 7, grid=grid(6144), stream=stream0)
        del buf1941
        buf1943 = reinterpret_tensor(buf1867, (24, 16, 48, 784), (602112, 37632, 784, 1), 0); del buf1867  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_53.run(buf1938, pow_3, buf1942, getitem_8, buf1937, pow_5, buf1940, getitem_9, buf1931, buf1943, 18432, 784, grid=grid(18432, 784), stream=stream0)
        del buf1931
        del buf1937
        del buf1940
        del buf1942
        del getitem_8
        del getitem_9
        del pow_3
        del pow_5
        buf1944 = reinterpret_tensor(buf1866, (6272, 2304), (2304, 1), 0); del buf1866  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_54.run(buf1943, buf1944, 6272, 2304, grid=grid(6272, 2304), stream=stream0)
        del buf1943
        buf1945 = reinterpret_tensor(buf1938, (6272, 768), (768, 1), 0); del buf1938  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1944, permute_896, out=buf1945)
        del permute_896
        buf1946 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1944, (2304, 6272), (1, 2304), 0), view_4, out=buf1946)
        del view_4
        buf1947 = buf1870; del buf1870  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_55.run(buf1944, buf1947, 112896, 128, grid=grid(112896), stream=stream0)
        del buf1944
        buf1948 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_56.run(buf1947, buf1948, 2304, 49, grid=grid(2304), stream=stream0)
        del buf1947
        buf1955 = buf1922; del buf1922  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_58.run(buf1955, buf1945, primals_113, mul_33, div_346, 6272, 768, grid=grid(6272), stream=stream0)
        del div_346
        del primals_113
        buf1951 = reinterpret_tensor(buf1925, (768, 49), (1, 768), 0); del buf1925  # reuse
        buf1953 = reinterpret_tensor(buf1923, (768, 49), (1, 768), 0); del buf1923  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1945, mul_33, buf1951, buf1953, 37632, 128, grid=grid(37632), stream=stream0)
        del buf1945
        del mul_33
        buf1952 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1951, buf1952, 768, 49, grid=grid(768), stream=stream0)
        buf1954 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1953, buf1954, 768, 49, grid=grid(768), stream=stream0)
        buf1956 = empty((1, 768, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_59.run(buf1955, buf1956, 602112, 8, grid=grid(602112), stream=stream0)
        buf1957 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_60.run(buf1956, buf1957, 768, 784, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1958 = aten.convolution_backward(buf1956, permute_1, primals_111, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1956
        del permute_1
        del primals_111
        buf1959 = buf1958[1]
        del buf1958
        buf1960 = buf1953; del buf1953  # reuse
        buf1962 = buf1951; del buf1951  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_61.run(buf1955, convolution_2, unsqueeze_407, buf1960, buf1962, 37632, 128, grid=grid(37632), stream=stream0)
        buf1961 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_mul_sum_22.run(buf1960, buf1961, 768, 49, grid=grid(768), stream=stream0)
        del buf1960
        buf1963 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1964 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_62.run(buf1962, squeeze_7, buf1963, buf1964, 768, 49, grid=grid(768), stream=stream0)
        del buf1962
        buf1965 = reinterpret_tensor(buf1955, (8, 768, 28, 28), (602112, 1, 21504, 768), 0); del buf1955  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_63.run(buf1965, convolution_2, unsqueeze_407, buf1963, squeeze_7, buf1961, primals_109, 4816896, grid=grid(4816896), stream=stream0)
        del buf1963
        del convolution_2
        del primals_109
        del squeeze_7
        del unsqueeze_407
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf1966 = aten.convolution_backward(buf1965, mul_19, primals_108, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1965
        del mul_19
        del primals_108
        buf1967 = buf1966[0]
        buf1968 = buf1966[1]
        del buf1966
        buf1969 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_red_fused_gelu_backward_native_batch_norm_backward_64.run(buf1967, add_682, buf1969, 75264, 128, grid=grid(75264), stream=stream0)
        buf1970 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_per_fused_gelu_backward_native_batch_norm_backward_65.run(buf1969, buf1970, 384, 196, grid=grid(384), stream=stream0)
        buf1971 = reinterpret_tensor(buf1969, (384, 196), (1, 384), 0); del buf1969  # reuse
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_red_fused_gelu_backward_native_batch_norm_backward_66.run(buf1967, add_682, convolution_1, unsqueeze_419, buf1971, 75264, 128, grid=grid(75264), stream=stream0)
        buf1972 = empty((384, ), device='cuda', dtype=torch.float32)
        buf1973 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_red_fused_gelu_backward_native_batch_norm_backward_67.run(buf1971, squeeze_4, buf1972, buf1973, 384, 196, grid=grid(384), stream=stream0)
        del buf1971
        buf1974 = buf1967; del buf1967  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_gelu_backward_native_batch_norm_backward_68.run(buf1974, add_682, convolution_1, unsqueeze_419, buf1972, squeeze_4, buf1970, primals_106, 3072, 3136, grid=grid(3072, 3136), stream=stream0)
        del add_682
        del buf1972
        del convolution_1
        del primals_106
        del squeeze_4
        del unsqueeze_419
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.gelu_backward, aten.native_batch_norm_backward]
        buf1975 = aten.convolution_backward(buf1974, mul_9, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1974
        del mul_9
        del primals_105
        buf1976 = buf1975[0]
        buf1977 = buf1975[1]
        del buf1975
        buf1978 = empty((192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_red_fused_gelu_backward_native_batch_norm_backward_69.run(buf1976, add_684, buf1978, 98304, 196, grid=grid(98304), stream=stream0)
        buf1979 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_per_fused_gelu_backward_native_batch_norm_backward_70.run(buf1978, buf1979, 192, 512, grid=grid(192), stream=stream0)
        buf1980 = reinterpret_tensor(buf1978, (192, 512), (1, 192), 0); del buf1978  # reuse
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_red_fused_gelu_backward_native_batch_norm_backward_71.run(buf1976, add_684, convolution, unsqueeze_431, buf1980, 98304, 196, grid=grid(98304), stream=stream0)
        buf1981 = empty((192, ), device='cuda', dtype=torch.float32)
        buf1982 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.gelu_backward, aten.native_batch_norm_backward]
        triton_red_fused_gelu_backward_native_batch_norm_backward_72.run(buf1980, squeeze_1, buf1981, buf1982, 192, 512, grid=grid(192), stream=stream0)
        del buf1980
        buf1983 = buf1976; del buf1976  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.gelu_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_gelu_backward_native_batch_norm_backward_73.run(buf1983, add_684, convolution, unsqueeze_431, buf1981, squeeze_1, buf1979, primals_103, 1536, 12544, grid=grid(1536, 12544), stream=stream0)
        del add_684
        del buf1981
        del convolution
        del primals_103
        del squeeze_1
        del unsqueeze_431
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.gelu_backward, aten.native_batch_norm_backward]
        buf1984 = aten.convolution_backward(buf1983, primals_710, primals_102, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1983
        del primals_102
        del primals_710
        buf1985 = buf1984[1]
        return (reinterpret_tensor(buf1924, (768, ), (1, ), 0), reinterpret_tensor(buf1935, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1899, (768, ), (1, ), 0), reinterpret_tensor(buf1880, (768, ), (1, ), 0), reinterpret_tensor(buf1847, (768, ), (1, ), 0), reinterpret_tensor(buf1858, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1822, (768, ), (1, ), 0), reinterpret_tensor(buf1803, (768, ), (1, ), 0), reinterpret_tensor(buf1770, (768, ), (1, ), 0), reinterpret_tensor(buf1781, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1745, (768, ), (1, ), 0), reinterpret_tensor(buf1726, (768, ), (1, ), 0), reinterpret_tensor(buf1693, (768, ), (1, ), 0), reinterpret_tensor(buf1704, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1668, (768, ), (1, ), 0), reinterpret_tensor(buf1649, (768, ), (1, ), 0), reinterpret_tensor(buf1616, (768, ), (1, ), 0), reinterpret_tensor(buf1627, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1591, (768, ), (1, ), 0), reinterpret_tensor(buf1572, (768, ), (1, ), 0), reinterpret_tensor(buf1539, (768, ), (1, ), 0), reinterpret_tensor(buf1550, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1514, (768, ), (1, ), 0), reinterpret_tensor(buf1495, (768, ), (1, ), 0), reinterpret_tensor(buf1462, (768, ), (1, ), 0), reinterpret_tensor(buf1473, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1437, (768, ), (1, ), 0), reinterpret_tensor(buf1418, (768, ), (1, ), 0), reinterpret_tensor(buf1385, (768, ), (1, ), 0), reinterpret_tensor(buf1396, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1360, (768, ), (1, ), 0), reinterpret_tensor(buf1341, (768, ), (1, ), 0), reinterpret_tensor(buf1308, (768, ), (1, ), 0), reinterpret_tensor(buf1319, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1283, (768, ), (1, ), 0), reinterpret_tensor(buf1264, (768, ), (1, ), 0), reinterpret_tensor(buf1231, (768, ), (1, ), 0), reinterpret_tensor(buf1242, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1206, (768, ), (1, ), 0), reinterpret_tensor(buf1187, (768, ), (1, ), 0), reinterpret_tensor(buf1154, (768, ), (1, ), 0), reinterpret_tensor(buf1165, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1129, (768, ), (1, ), 0), reinterpret_tensor(buf1110, (768, ), (1, ), 0), reinterpret_tensor(buf1077, (768, ), (1, ), 0), reinterpret_tensor(buf1088, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf1052, (768, ), (1, ), 0), reinterpret_tensor(buf1033, (768, ), (1, ), 0), reinterpret_tensor(buf1000, (768, ), (1, ), 0), reinterpret_tensor(buf1011, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf975, (768, ), (1, ), 0), reinterpret_tensor(buf956, (768, ), (1, ), 0), reinterpret_tensor(buf923, (768, ), (1, ), 0), reinterpret_tensor(buf934, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf898, (768, ), (1, ), 0), reinterpret_tensor(buf879, (768, ), (1, ), 0), reinterpret_tensor(buf846, (768, ), (1, ), 0), reinterpret_tensor(buf857, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf821, (768, ), (1, ), 0), reinterpret_tensor(buf802, (768, ), (1, ), 0), reinterpret_tensor(buf769, (768, ), (1, ), 0), reinterpret_tensor(buf780, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf744, (768, ), (1, ), 0), reinterpret_tensor(buf725, (768, ), (1, ), 0), reinterpret_tensor(buf692, (768, ), (1, ), 0), reinterpret_tensor(buf703, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf667, (768, ), (1, ), 0), reinterpret_tensor(buf648, (768, ), (1, ), 0), reinterpret_tensor(buf615, (768, ), (1, ), 0), reinterpret_tensor(buf626, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf590, (768, ), (1, ), 0), reinterpret_tensor(buf571, (768, ), (1, ), 0), reinterpret_tensor(buf538, (768, ), (1, ), 0), reinterpret_tensor(buf549, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf513, (768, ), (1, ), 0), reinterpret_tensor(buf494, (768, ), (1, ), 0), reinterpret_tensor(buf461, (768, ), (1, ), 0), reinterpret_tensor(buf472, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf436, (768, ), (1, ), 0), reinterpret_tensor(buf417, (768, ), (1, ), 0), reinterpret_tensor(buf384, (768, ), (1, ), 0), reinterpret_tensor(buf395, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf359, (768, ), (1, ), 0), reinterpret_tensor(buf340, (768, ), (1, ), 0), reinterpret_tensor(buf307, (768, ), (1, ), 0), reinterpret_tensor(buf318, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf282, (768, ), (1, ), 0), reinterpret_tensor(buf263, (768, ), (1, ), 0), reinterpret_tensor(buf230, (768, ), (1, ), 0), reinterpret_tensor(buf241, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf205, (768, ), (1, ), 0), reinterpret_tensor(buf186, (768, ), (1, ), 0), reinterpret_tensor(buf153, (768, ), (1, ), 0), reinterpret_tensor(buf164, (16, 1, 1), (1, 1, 1), 0), reinterpret_tensor(buf128, (768, ), (1, ), 0), reinterpret_tensor(buf109, (768, ), (1, ), 0), buf107, reinterpret_tensor(buf75, (768, ), (1, ), 0), reinterpret_tensor(buf58, (768, ), (1, ), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), reinterpret_tensor(buf9, (768, ), (1, ), 0), buf1985, buf1982, buf1979, buf1977, buf1973, buf1970, buf1968, buf1964, buf1961, buf1959, buf1957, buf1952, buf1954, reinterpret_tensor(buf1946, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1948, (2304, ), (1, ), 0), reinterpret_tensor(buf1928, (768, 768), (768, 1), 0), reinterpret_tensor(buf1926, (768, ), (1, ), 0), buf1920, buf1921, buf1914, buf1911, buf1909, buf1906, buf1905, buf1902, buf1894, buf1896, reinterpret_tensor(buf1888, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1890, (3072, ), (1, ), 0), reinterpret_tensor(buf1883, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1885, (768, ), (1, ), 0), buf1875, buf1877, reinterpret_tensor(buf1869, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1871, (2304, ), (1, ), 0), reinterpret_tensor(buf1851, (768, 768), (768, 1), 0), reinterpret_tensor(buf1849, (768, ), (1, ), 0), buf1843, buf1844, buf1837, buf1834, buf1832, buf1829, buf1828, buf1825, buf1817, buf1819, reinterpret_tensor(buf1811, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1813, (3072, ), (1, ), 0), reinterpret_tensor(buf1806, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1808, (768, ), (1, ), 0), buf1798, buf1800, reinterpret_tensor(buf1792, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1794, (2304, ), (1, ), 0), reinterpret_tensor(buf1774, (768, 768), (768, 1), 0), reinterpret_tensor(buf1772, (768, ), (1, ), 0), buf1766, buf1767, buf1760, buf1757, buf1755, buf1752, buf1751, buf1748, buf1740, buf1742, reinterpret_tensor(buf1734, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1736, (3072, ), (1, ), 0), reinterpret_tensor(buf1729, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1731, (768, ), (1, ), 0), buf1721, buf1723, reinterpret_tensor(buf1715, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1717, (2304, ), (1, ), 0), reinterpret_tensor(buf1697, (768, 768), (768, 1), 0), reinterpret_tensor(buf1695, (768, ), (1, ), 0), buf1689, buf1690, buf1683, buf1680, buf1678, buf1675, buf1674, buf1671, buf1663, buf1665, reinterpret_tensor(buf1657, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1659, (3072, ), (1, ), 0), reinterpret_tensor(buf1652, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1654, (768, ), (1, ), 0), buf1644, buf1646, reinterpret_tensor(buf1638, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1640, (2304, ), (1, ), 0), reinterpret_tensor(buf1620, (768, 768), (768, 1), 0), reinterpret_tensor(buf1618, (768, ), (1, ), 0), buf1612, buf1613, buf1606, buf1603, buf1601, buf1598, buf1597, buf1594, buf1586, buf1588, reinterpret_tensor(buf1580, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1582, (3072, ), (1, ), 0), reinterpret_tensor(buf1575, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1577, (768, ), (1, ), 0), buf1567, buf1569, reinterpret_tensor(buf1561, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1563, (2304, ), (1, ), 0), reinterpret_tensor(buf1543, (768, 768), (768, 1), 0), reinterpret_tensor(buf1541, (768, ), (1, ), 0), buf1535, buf1536, buf1529, buf1526, buf1524, buf1521, buf1520, buf1517, buf1509, buf1511, reinterpret_tensor(buf1503, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1505, (3072, ), (1, ), 0), reinterpret_tensor(buf1498, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1500, (768, ), (1, ), 0), buf1490, buf1492, reinterpret_tensor(buf1484, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1486, (2304, ), (1, ), 0), reinterpret_tensor(buf1466, (768, 768), (768, 1), 0), reinterpret_tensor(buf1464, (768, ), (1, ), 0), buf1458, buf1459, buf1452, buf1449, buf1447, buf1444, buf1443, buf1440, buf1432, buf1434, reinterpret_tensor(buf1426, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1428, (3072, ), (1, ), 0), reinterpret_tensor(buf1421, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1423, (768, ), (1, ), 0), buf1413, buf1415, reinterpret_tensor(buf1407, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1409, (2304, ), (1, ), 0), reinterpret_tensor(buf1389, (768, 768), (768, 1), 0), reinterpret_tensor(buf1387, (768, ), (1, ), 0), buf1381, buf1382, buf1375, buf1372, buf1370, buf1367, buf1366, buf1363, buf1355, buf1357, reinterpret_tensor(buf1349, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1351, (3072, ), (1, ), 0), reinterpret_tensor(buf1344, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1346, (768, ), (1, ), 0), buf1336, buf1338, reinterpret_tensor(buf1330, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1332, (2304, ), (1, ), 0), reinterpret_tensor(buf1312, (768, 768), (768, 1), 0), reinterpret_tensor(buf1310, (768, ), (1, ), 0), buf1304, buf1305, buf1298, buf1295, buf1293, buf1290, buf1289, buf1286, buf1278, buf1280, reinterpret_tensor(buf1272, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1274, (3072, ), (1, ), 0), reinterpret_tensor(buf1267, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1269, (768, ), (1, ), 0), buf1259, buf1261, reinterpret_tensor(buf1253, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1255, (2304, ), (1, ), 0), reinterpret_tensor(buf1235, (768, 768), (768, 1), 0), reinterpret_tensor(buf1233, (768, ), (1, ), 0), buf1227, buf1228, buf1221, buf1218, buf1216, buf1213, buf1212, buf1209, buf1201, buf1203, reinterpret_tensor(buf1195, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1197, (3072, ), (1, ), 0), reinterpret_tensor(buf1190, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1192, (768, ), (1, ), 0), buf1182, buf1184, reinterpret_tensor(buf1176, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1178, (2304, ), (1, ), 0), reinterpret_tensor(buf1158, (768, 768), (768, 1), 0), reinterpret_tensor(buf1156, (768, ), (1, ), 0), buf1150, buf1151, buf1144, buf1141, buf1139, buf1136, buf1135, buf1132, buf1124, buf1126, reinterpret_tensor(buf1118, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1120, (3072, ), (1, ), 0), reinterpret_tensor(buf1113, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1115, (768, ), (1, ), 0), buf1105, buf1107, reinterpret_tensor(buf1099, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1101, (2304, ), (1, ), 0), reinterpret_tensor(buf1081, (768, 768), (768, 1), 0), reinterpret_tensor(buf1079, (768, ), (1, ), 0), buf1073, buf1074, buf1067, buf1064, buf1062, buf1059, buf1058, buf1055, buf1047, buf1049, reinterpret_tensor(buf1041, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1043, (3072, ), (1, ), 0), reinterpret_tensor(buf1036, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1038, (768, ), (1, ), 0), buf1028, buf1030, reinterpret_tensor(buf1022, (2304, 768), (768, 1), 0), reinterpret_tensor(buf1024, (2304, ), (1, ), 0), reinterpret_tensor(buf1004, (768, 768), (768, 1), 0), reinterpret_tensor(buf1002, (768, ), (1, ), 0), buf996, buf997, buf990, buf987, buf985, buf982, buf981, buf978, buf970, buf972, reinterpret_tensor(buf964, (3072, 768), (768, 1), 0), reinterpret_tensor(buf966, (3072, ), (1, ), 0), reinterpret_tensor(buf959, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf961, (768, ), (1, ), 0), buf951, buf953, reinterpret_tensor(buf945, (2304, 768), (768, 1), 0), reinterpret_tensor(buf947, (2304, ), (1, ), 0), reinterpret_tensor(buf927, (768, 768), (768, 1), 0), reinterpret_tensor(buf925, (768, ), (1, ), 0), buf919, buf920, buf913, buf910, buf908, buf905, buf904, buf901, buf893, buf895, reinterpret_tensor(buf887, (3072, 768), (768, 1), 0), reinterpret_tensor(buf889, (3072, ), (1, ), 0), reinterpret_tensor(buf882, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf884, (768, ), (1, ), 0), buf874, buf876, reinterpret_tensor(buf868, (2304, 768), (768, 1), 0), reinterpret_tensor(buf870, (2304, ), (1, ), 0), reinterpret_tensor(buf850, (768, 768), (768, 1), 0), reinterpret_tensor(buf848, (768, ), (1, ), 0), buf842, buf843, buf836, buf833, buf831, buf828, buf827, buf824, buf816, buf818, reinterpret_tensor(buf810, (3072, 768), (768, 1), 0), reinterpret_tensor(buf812, (3072, ), (1, ), 0), reinterpret_tensor(buf805, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf807, (768, ), (1, ), 0), buf797, buf799, reinterpret_tensor(buf791, (2304, 768), (768, 1), 0), reinterpret_tensor(buf793, (2304, ), (1, ), 0), reinterpret_tensor(buf773, (768, 768), (768, 1), 0), reinterpret_tensor(buf771, (768, ), (1, ), 0), buf765, buf766, buf759, buf756, buf754, buf751, buf750, buf747, buf739, buf741, reinterpret_tensor(buf733, (3072, 768), (768, 1), 0), reinterpret_tensor(buf735, (3072, ), (1, ), 0), reinterpret_tensor(buf728, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf730, (768, ), (1, ), 0), buf720, buf722, reinterpret_tensor(buf714, (2304, 768), (768, 1), 0), reinterpret_tensor(buf716, (2304, ), (1, ), 0), reinterpret_tensor(buf696, (768, 768), (768, 1), 0), reinterpret_tensor(buf694, (768, ), (1, ), 0), buf688, buf689, buf682, buf679, buf677, buf674, buf673, buf670, buf662, buf664, reinterpret_tensor(buf656, (3072, 768), (768, 1), 0), reinterpret_tensor(buf658, (3072, ), (1, ), 0), reinterpret_tensor(buf651, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf653, (768, ), (1, ), 0), buf643, buf645, reinterpret_tensor(buf637, (2304, 768), (768, 1), 0), reinterpret_tensor(buf639, (2304, ), (1, ), 0), reinterpret_tensor(buf619, (768, 768), (768, 1), 0), reinterpret_tensor(buf617, (768, ), (1, ), 0), buf611, buf612, buf605, buf602, buf600, buf597, buf596, buf593, buf585, buf587, reinterpret_tensor(buf579, (3072, 768), (768, 1), 0), reinterpret_tensor(buf581, (3072, ), (1, ), 0), reinterpret_tensor(buf574, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf576, (768, ), (1, ), 0), buf566, buf568, reinterpret_tensor(buf560, (2304, 768), (768, 1), 0), reinterpret_tensor(buf562, (2304, ), (1, ), 0), reinterpret_tensor(buf542, (768, 768), (768, 1), 0), reinterpret_tensor(buf540, (768, ), (1, ), 0), buf534, buf535, buf528, buf525, buf523, buf520, buf519, buf516, buf508, buf510, reinterpret_tensor(buf502, (3072, 768), (768, 1), 0), reinterpret_tensor(buf504, (3072, ), (1, ), 0), reinterpret_tensor(buf497, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf499, (768, ), (1, ), 0), buf489, buf491, reinterpret_tensor(buf483, (2304, 768), (768, 1), 0), reinterpret_tensor(buf485, (2304, ), (1, ), 0), reinterpret_tensor(buf465, (768, 768), (768, 1), 0), reinterpret_tensor(buf463, (768, ), (1, ), 0), buf457, buf458, buf451, buf448, buf446, buf443, buf442, buf439, buf431, buf433, reinterpret_tensor(buf425, (3072, 768), (768, 1), 0), reinterpret_tensor(buf427, (3072, ), (1, ), 0), reinterpret_tensor(buf420, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf422, (768, ), (1, ), 0), buf412, buf414, reinterpret_tensor(buf406, (2304, 768), (768, 1), 0), reinterpret_tensor(buf408, (2304, ), (1, ), 0), reinterpret_tensor(buf388, (768, 768), (768, 1), 0), reinterpret_tensor(buf386, (768, ), (1, ), 0), buf380, buf381, buf374, buf371, buf369, buf366, buf365, buf362, buf354, buf356, reinterpret_tensor(buf348, (3072, 768), (768, 1), 0), reinterpret_tensor(buf350, (3072, ), (1, ), 0), reinterpret_tensor(buf343, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf345, (768, ), (1, ), 0), buf335, buf337, reinterpret_tensor(buf329, (2304, 768), (768, 1), 0), reinterpret_tensor(buf331, (2304, ), (1, ), 0), reinterpret_tensor(buf311, (768, 768), (768, 1), 0), reinterpret_tensor(buf309, (768, ), (1, ), 0), buf303, buf304, buf297, buf294, buf292, buf289, buf288, buf285, buf277, buf279, reinterpret_tensor(buf271, (3072, 768), (768, 1), 0), reinterpret_tensor(buf273, (3072, ), (1, ), 0), reinterpret_tensor(buf266, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf268, (768, ), (1, ), 0), buf258, buf260, reinterpret_tensor(buf252, (2304, 768), (768, 1), 0), reinterpret_tensor(buf254, (2304, ), (1, ), 0), reinterpret_tensor(buf234, (768, 768), (768, 1), 0), reinterpret_tensor(buf232, (768, ), (1, ), 0), buf226, buf227, buf220, buf217, buf215, buf212, buf211, buf208, buf200, buf202, reinterpret_tensor(buf194, (3072, 768), (768, 1), 0), reinterpret_tensor(buf196, (3072, ), (1, ), 0), reinterpret_tensor(buf189, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf191, (768, ), (1, ), 0), buf181, buf183, reinterpret_tensor(buf175, (2304, 768), (768, 1), 0), reinterpret_tensor(buf177, (2304, ), (1, ), 0), reinterpret_tensor(buf157, (768, 768), (768, 1), 0), reinterpret_tensor(buf155, (768, ), (1, ), 0), buf149, buf150, buf143, buf140, buf138, buf135, buf134, buf131, buf123, buf125, reinterpret_tensor(buf117, (3072, 768), (768, 1), 0), reinterpret_tensor(buf119, (3072, ), (1, ), 0), reinterpret_tensor(buf112, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf114, (768, ), (1, ), 0), buf103, buf105, reinterpret_tensor(buf97, (768, 768), (768, 1), 0), reinterpret_tensor(buf98, (768, ), (1, ), 0), reinterpret_tensor(buf93, (768, 768), (768, 1), 0), reinterpret_tensor(buf95, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, 768), (768, 1), 0), reinterpret_tensor(buf91, (768, ), (1, ), 0), reinterpret_tensor(buf78, (768, 768), (768, 1), 0), reinterpret_tensor(buf79, (768, ), (1, ), 0), buf71, buf73, reinterpret_tensor(buf65, (3072, 768), (768, 1), 0), reinterpret_tensor(buf63, (3072, ), (1, ), 0), reinterpret_tensor(buf61, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf62, (768, ), (1, ), 0), buf54, buf56, reinterpret_tensor(buf48, (768, 768), (768, 1), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf44, (768, 768), (768, 1), 0), reinterpret_tensor(buf46, (768, ), (1, ), 0), reinterpret_tensor(buf40, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf29, (768, 768), (768, 1), 0), reinterpret_tensor(buf30, (768, ), (1, ), 0), buf22, buf24, reinterpret_tensor(buf16, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf12, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf13, (768, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_19 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1, 64, 28, 28), (50176, 1, 1792, 64), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_8 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_3 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_5 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_34 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_20 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_7 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_9 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_2 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_51 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_11 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_13 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_4 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_83 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_68 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_54 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_58 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_42 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_15 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_17 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_6 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_106 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_85 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_119 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_76 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_52 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_19 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_21 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_8 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_129 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_102 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_142 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_148 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_94 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_64 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_23 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_25 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_10 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_119 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_165 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_171 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_27 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_29 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_12 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_175 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_124 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_136 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_188 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_194 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_31 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_33 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_14 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_198 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_142 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_153 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_211 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_217 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_96 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_35 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_37 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_16 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_158 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_221 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_160 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_170 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_234 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_240 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_108 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_39 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_41 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_18 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_9 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_244 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_187 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_257 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_182 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_263 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_118 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_43 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_45 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_20 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_10 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_267 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_204 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_280 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_32 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_286 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_202 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_129 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_130 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_47 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_49 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_22 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_290 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_221 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_303 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_309 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_140 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_51 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_53 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_24 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_12 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_313 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_232 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_238 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_326 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_234 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_37 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_332 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_152 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_55 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_57 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_26 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_248 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_13 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_336 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_255 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_349 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_252 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_254 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_355 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_256 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_59 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_61 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_28 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_14 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_359 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_268 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_272 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_372 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_270 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_43 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_272 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_378 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_173 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_174 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_63 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_65 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_30 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_284 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_15 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_382 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_286 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_289 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_395 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_288 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_290 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_401 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_292 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_184 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_185 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_67 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_69 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_32 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_302 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_16 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_405 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_304 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_306 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_418 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_306 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_49 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_308 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_50 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_424 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_310 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_195 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_196 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_71 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_73 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_34 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_320 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_17 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_428 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_323 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_441 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_324 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_326 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_447 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_328 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_206 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_207 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_75 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_77 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_36 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_338 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_18 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_451 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_340 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_340 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_464 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_55 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_344 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_56 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_470 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_346 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_217 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_218 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_79 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_81 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_38 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_356 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_19 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_474 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_358 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_357 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_487 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_360 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_362 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_59 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_493 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_364 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_228 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_229 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_83 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_85 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_40 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_374 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_20 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_497 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_376 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_374 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_510 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_378 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_61 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_380 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_516 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_382 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_239 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_240 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_87 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_89 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_42 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_21 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_520 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_394 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_391 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_533 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_396 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_398 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_65 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_539 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_400 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_250 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_251 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_91 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_93 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_44 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_410 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_22 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_543 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_412 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_408 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_556 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_414 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_67 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_416 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_68 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_562 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_418 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_261 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    getitem_262 = rand_strided((8, 16, 48, 784), (1806336, 48, 1, 2304), device='cuda:0', dtype=torch.float32)
    pow_95 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    pow_97 = rand_strided((8, 16, 48, 1), (768, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    bmm_46 = rand_strided((128, 48, 48), (2304, 48, 1), device='cuda:0', dtype=torch.float32)
    view_428 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_23 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_566 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_430 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_425 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda:0', dtype=torch.float32)
    mul_579 = rand_strided((8, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    view_432 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_434 = rand_strided((6272, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_71 = rand_strided((6272, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_271 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_99 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    select = rand_strided((8, 768), (602880, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    view_437 = rand_strided((6280, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    getitem_273 = rand_strided((8, 16, 32), (512, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_444 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_588 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    view_446 = rand_strided((8, 768), (602880, 1), device='cuda:0', dtype=torch.float32)
    mm_24 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_448 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_76 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_594 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    select_1 = rand_strided((8, 768), (602880, 1), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    view_451 = rand_strided((6280, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((8, 16, 785, 48), (602880, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    getitem_281 = rand_strided((8, 16, 32), (512, 32, 1), device='cuda:0', dtype=torch.float32)
    getitem_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_458 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_597 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    view_460 = rand_strided((8, 768), (602880, 1), device='cuda:0', dtype=torch.float32)
    mm_25 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_462 = rand_strided((8, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_81 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_603 = rand_strided((8, 785, 768), (602880, 768, 1), device='cuda:0', dtype=torch.float32)
    clone_271 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_74 = rand_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_75 = rand_strided((8, 16, 1, 48), (768, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_119 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_316 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_76 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_93 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_94 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_131 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_95 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_341 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_79 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_342 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_104 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_105 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_143 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_106 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_362 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_366 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_82 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_367 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_115 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_116 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_155 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_117 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_391 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_85 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_392 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_396 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_126 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_400 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_127 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_167 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_128 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_412 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_88 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_137 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_138 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_179 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_139 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_441 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_91 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_148 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_149 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_191 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_150 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_462 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_466 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_94 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_159 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_479 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_160 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_203 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_161 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_491 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_97 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_493 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_170 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_171 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_215 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_172 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_512 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_515 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_516 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_100 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_181 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_529 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_182 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_227 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_183 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_537 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_541 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_103 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_542 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_543 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_546 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_192 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_554 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_193 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_239 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_194 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_562 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_565 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_566 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_106 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_568 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_203 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_575 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_204 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_251 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_205 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_587 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_591 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_109 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_592 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_593 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_214 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_604 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_215 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_263 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_216 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_612 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_615 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_616 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_112 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_617 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_618 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_225 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_226 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_275 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_227 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_637 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_640 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_641 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_115 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_642 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_643 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_646 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_236 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_650 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_237 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_287 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_238 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_662 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_665 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_666 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_118 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_668 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_671 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_247 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_675 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_679 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_248 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_299 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_249 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_690 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_691 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_121 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_692 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_693 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_696 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_258 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_704 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_259 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_311 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_260 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_712 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_715 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_716 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_124 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_718 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_269 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_270 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_323 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_271 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_737 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_740 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_741 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_127 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_742 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_743 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_746 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_280 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_754 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_281 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_335 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_282 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_762 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_765 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_766 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_130 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_767 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_768 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_771 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_291 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_775 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_779 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_292 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_347 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_293 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_787 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_790 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_791 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_133 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_792 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_793 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_796 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_302 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_800 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_804 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_303 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_359 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_304 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_812 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_815 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_816 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_136 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_817 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_818 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_821 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_313 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_825 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_829 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_314 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_371 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_315 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_837 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_840 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_841 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_139 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_842 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_843 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_846 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_324 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_850 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_854 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_325 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_383 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_326 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_862 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_865 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_866 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_142 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_867 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_868 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_871 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_335 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_875 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_879 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_336 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_395 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_337 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_887 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_890 = rand_strided((128, 48, 48), (2304, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_891 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    alias_145 = rand_strided((8, 16, 48, 48), (36864, 1, 768, 16), device='cuda:0', dtype=torch.float32)
    permute_892 = rand_strided((128, 784, 48), (37632, 1, 784), device='cuda:0', dtype=torch.float32)
    permute_893 = rand_strided((128, 48, 784), (37632, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_896 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_346 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_407 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_682 = rand_strided((8, 384, 56, 56), (1204224, 1, 21504, 384), device='cuda:0', dtype=torch.float32)
    unsqueeze_419 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_684 = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    unsqueeze_431 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_113, primals_118, primals_119, primals_121, primals_123, primals_125, primals_127, primals_133, primals_138, primals_139, primals_141, primals_143, primals_145, primals_147, primals_153, primals_158, primals_159, primals_161, primals_163, primals_165, primals_167, primals_173, primals_178, primals_179, primals_181, primals_183, primals_185, primals_187, primals_193, primals_198, primals_199, primals_201, primals_203, primals_205, primals_207, primals_213, primals_218, primals_219, primals_221, primals_223, primals_225, primals_227, primals_233, primals_238, primals_239, primals_241, primals_243, primals_245, primals_247, primals_253, primals_258, primals_259, primals_261, primals_263, primals_265, primals_267, primals_273, primals_278, primals_279, primals_281, primals_283, primals_285, primals_287, primals_293, primals_298, primals_299, primals_301, primals_303, primals_305, primals_307, primals_313, primals_318, primals_319, primals_321, primals_323, primals_325, primals_327, primals_333, primals_338, primals_339, primals_341, primals_343, primals_345, primals_347, primals_353, primals_358, primals_359, primals_361, primals_363, primals_365, primals_367, primals_373, primals_378, primals_379, primals_381, primals_383, primals_385, primals_387, primals_393, primals_398, primals_399, primals_401, primals_403, primals_405, primals_407, primals_413, primals_418, primals_419, primals_421, primals_423, primals_425, primals_427, primals_433, primals_438, primals_439, primals_441, primals_443, primals_445, primals_447, primals_453, primals_458, primals_459, primals_461, primals_463, primals_465, primals_467, primals_473, primals_478, primals_479, primals_481, primals_483, primals_485, primals_487, primals_493, primals_498, primals_499, primals_501, primals_503, primals_505, primals_507, primals_513, primals_518, primals_519, primals_521, primals_523, primals_525, primals_527, primals_533, primals_538, primals_539, primals_541, primals_543, primals_545, primals_547, primals_553, primals_558, primals_559, primals_561, primals_563, primals_565, primals_567, primals_573, primals_578, primals_579, primals_581, primals_583, primals_585, primals_587, primals_593, primals_603, primals_606, primals_609, primals_619, primals_622, primals_625, primals_710, convolution, squeeze_1, mul_9, convolution_1, squeeze_4, mul_19, convolution_2, squeeze_7, permute_1, mul_33, view_4, getitem_8, getitem_9, pow_3, pow_5, bmm, view_14, mm, mul_37, view_16, convolution_4, squeeze_10, add_34, convolution_5, mul_50, view_18, addmm_1, view_20, addmm_2, mul_56, view_22, getitem_19, getitem_20, pow_7, pow_9, bmm_2, view_32, mm_1, mul_60, view_34, convolution_6, squeeze_13, add_51, convolution_7, mul_73, view_36, addmm_4, view_38, addmm_5, mul_79, view_40, getitem_30, getitem_31, pow_11, pow_13, bmm_4, view_50, mm_2, mul_83, view_52, convolution_8, squeeze_16, add_68, convolution_9, mul_96, view_54, addmm_7, view_56, addmm_8, mul_102, view_58, getitem_41, getitem_42, pow_15, pow_17, bmm_6, view_68, mm_3, mul_106, view_70, convolution_10, squeeze_19, add_85, convolution_11, mul_119, view_72, addmm_10, view_74, addmm_11, mul_125, view_76, getitem_52, getitem_53, pow_19, pow_21, bmm_8, view_86, mm_4, mul_129, view_88, convolution_12, squeeze_22, add_102, convolution_13, mul_142, view_90, addmm_13, view_92, addmm_14, mul_148, view_94, getitem_63, getitem_64, pow_23, pow_25, bmm_10, view_104, mm_5, mul_152, view_106, convolution_14, squeeze_25, add_119, convolution_15, mul_165, view_108, addmm_16, view_110, addmm_17, mul_171, view_112, getitem_74, getitem_75, pow_27, pow_29, bmm_12, view_122, mm_6, mul_175, view_124, convolution_16, squeeze_28, add_136, convolution_17, mul_188, view_126, addmm_19, view_128, addmm_20, mul_194, view_130, getitem_85, getitem_86, pow_31, pow_33, bmm_14, view_140, mm_7, mul_198, view_142, convolution_18, squeeze_31, add_153, convolution_19, mul_211, view_144, addmm_22, view_146, addmm_23, mul_217, view_148, getitem_96, getitem_97, pow_35, pow_37, bmm_16, view_158, mm_8, mul_221, view_160, convolution_20, squeeze_34, add_170, convolution_21, mul_234, view_162, addmm_25, view_164, addmm_26, mul_240, view_166, getitem_107, getitem_108, pow_39, pow_41, bmm_18, view_176, mm_9, mul_244, view_178, convolution_22, squeeze_37, add_187, convolution_23, mul_257, view_180, addmm_28, view_182, addmm_29, mul_263, view_184, getitem_118, getitem_119, pow_43, pow_45, bmm_20, view_194, mm_10, mul_267, view_196, convolution_24, squeeze_40, add_204, convolution_25, mul_280, view_198, addmm_31, view_200, addmm_32, mul_286, view_202, getitem_129, getitem_130, pow_47, pow_49, bmm_22, view_212, mm_11, mul_290, view_214, convolution_26, squeeze_43, add_221, convolution_27, mul_303, view_216, addmm_34, view_218, addmm_35, mul_309, view_220, getitem_140, getitem_141, pow_51, pow_53, bmm_24, view_230, mm_12, mul_313, view_232, convolution_28, squeeze_46, add_238, convolution_29, mul_326, view_234, addmm_37, view_236, addmm_38, mul_332, view_238, getitem_151, getitem_152, pow_55, pow_57, bmm_26, view_248, mm_13, mul_336, view_250, convolution_30, squeeze_49, add_255, convolution_31, mul_349, view_252, addmm_40, view_254, addmm_41, mul_355, view_256, getitem_162, getitem_163, pow_59, pow_61, bmm_28, view_266, mm_14, mul_359, view_268, convolution_32, squeeze_52, add_272, convolution_33, mul_372, view_270, addmm_43, view_272, addmm_44, mul_378, view_274, getitem_173, getitem_174, pow_63, pow_65, bmm_30, view_284, mm_15, mul_382, view_286, convolution_34, squeeze_55, add_289, convolution_35, mul_395, view_288, addmm_46, view_290, addmm_47, mul_401, view_292, getitem_184, getitem_185, pow_67, pow_69, bmm_32, view_302, mm_16, mul_405, view_304, convolution_36, squeeze_58, add_306, convolution_37, mul_418, view_306, addmm_49, view_308, addmm_50, mul_424, view_310, getitem_195, getitem_196, pow_71, pow_73, bmm_34, view_320, mm_17, mul_428, view_322, convolution_38, squeeze_61, add_323, convolution_39, mul_441, view_324, addmm_52, view_326, addmm_53, mul_447, view_328, getitem_206, getitem_207, pow_75, pow_77, bmm_36, view_338, mm_18, mul_451, view_340, convolution_40, squeeze_64, add_340, convolution_41, mul_464, view_342, addmm_55, view_344, addmm_56, mul_470, view_346, getitem_217, getitem_218, pow_79, pow_81, bmm_38, view_356, mm_19, mul_474, view_358, convolution_42, squeeze_67, add_357, convolution_43, mul_487, view_360, addmm_58, view_362, addmm_59, mul_493, view_364, getitem_228, getitem_229, pow_83, pow_85, bmm_40, view_374, mm_20, mul_497, view_376, convolution_44, squeeze_70, add_374, convolution_45, mul_510, view_378, addmm_61, view_380, addmm_62, mul_516, view_382, getitem_239, getitem_240, pow_87, pow_89, bmm_42, view_392, mm_21, mul_520, view_394, convolution_46, squeeze_73, add_391, convolution_47, mul_533, view_396, addmm_64, view_398, addmm_65, mul_539, view_400, getitem_250, getitem_251, pow_91, pow_93, bmm_44, view_410, mm_22, mul_543, view_412, convolution_48, squeeze_76, add_408, convolution_49, mul_556, view_414, addmm_67, view_416, addmm_68, mul_562, view_418, getitem_261, getitem_262, pow_95, pow_97, bmm_46, view_428, mm_23, mul_566, view_430, convolution_50, squeeze_79, add_425, convolution_51, mul_579, view_432, addmm_70, view_434, addmm_71, cat_3, getitem_271, rsqrt_99, select, permute_220, view_437, permute_222, permute_224, getitem_273, getitem_274, getitem_275, view_444, cat_4, mul_588, view_446, mm_24, view_448, addmm_76, mul_594, select_1, permute_230, view_451, permute_232, permute_234, getitem_281, getitem_282, getitem_283, view_458, cat_6, mul_597, view_460, mm_25, view_462, addmm_81, mul_603, clone_271, permute_240, div_78, permute_244, permute_250, div_79, permute_252, alias_74, permute_258, permute_263, permute_268, div_80, permute_272, permute_278, div_81, permute_280, alias_75, permute_286, permute_291, permute_296, permute_300, permute_304, div_83, unsqueeze_119, div_84, permute_312, permute_315, permute_316, alias_76, permute_317, permute_318, permute_321, div_93, permute_325, permute_329, div_94, unsqueeze_131, div_95, permute_337, permute_340, permute_341, alias_79, permute_342, permute_343, permute_346, div_104, permute_350, permute_354, div_105, unsqueeze_143, div_106, permute_362, permute_365, permute_366, alias_82, permute_367, permute_368, permute_371, div_115, permute_375, permute_379, div_116, unsqueeze_155, div_117, permute_387, permute_390, permute_391, alias_85, permute_392, permute_393, permute_396, div_126, permute_400, permute_404, div_127, unsqueeze_167, div_128, permute_412, permute_415, permute_416, alias_88, permute_417, permute_418, permute_421, div_137, permute_425, permute_429, div_138, unsqueeze_179, div_139, permute_437, permute_440, permute_441, alias_91, permute_442, permute_443, permute_446, div_148, permute_450, permute_454, div_149, unsqueeze_191, div_150, permute_462, permute_465, permute_466, alias_94, permute_467, permute_468, permute_471, div_159, permute_475, permute_479, div_160, unsqueeze_203, div_161, permute_487, permute_490, permute_491, alias_97, permute_492, permute_493, permute_496, div_170, permute_500, permute_504, div_171, unsqueeze_215, div_172, permute_512, permute_515, permute_516, alias_100, permute_517, permute_518, permute_521, div_181, permute_525, permute_529, div_182, unsqueeze_227, div_183, permute_537, permute_540, permute_541, alias_103, permute_542, permute_543, permute_546, div_192, permute_550, permute_554, div_193, unsqueeze_239, div_194, permute_562, permute_565, permute_566, alias_106, permute_567, permute_568, permute_571, div_203, permute_575, permute_579, div_204, unsqueeze_251, div_205, permute_587, permute_590, permute_591, alias_109, permute_592, permute_593, permute_596, div_214, permute_600, permute_604, div_215, unsqueeze_263, div_216, permute_612, permute_615, permute_616, alias_112, permute_617, permute_618, permute_621, div_225, permute_625, permute_629, div_226, unsqueeze_275, div_227, permute_637, permute_640, permute_641, alias_115, permute_642, permute_643, permute_646, div_236, permute_650, permute_654, div_237, unsqueeze_287, div_238, permute_662, permute_665, permute_666, alias_118, permute_667, permute_668, permute_671, div_247, permute_675, permute_679, div_248, unsqueeze_299, div_249, permute_687, permute_690, permute_691, alias_121, permute_692, permute_693, permute_696, div_258, permute_700, permute_704, div_259, unsqueeze_311, div_260, permute_712, permute_715, permute_716, alias_124, permute_717, permute_718, permute_721, div_269, permute_725, permute_729, div_270, unsqueeze_323, div_271, permute_737, permute_740, permute_741, alias_127, permute_742, permute_743, permute_746, div_280, permute_750, permute_754, div_281, unsqueeze_335, div_282, permute_762, permute_765, permute_766, alias_130, permute_767, permute_768, permute_771, div_291, permute_775, permute_779, div_292, unsqueeze_347, div_293, permute_787, permute_790, permute_791, alias_133, permute_792, permute_793, permute_796, div_302, permute_800, permute_804, div_303, unsqueeze_359, div_304, permute_812, permute_815, permute_816, alias_136, permute_817, permute_818, permute_821, div_313, permute_825, permute_829, div_314, unsqueeze_371, div_315, permute_837, permute_840, permute_841, alias_139, permute_842, permute_843, permute_846, div_324, permute_850, permute_854, div_325, unsqueeze_383, div_326, permute_862, permute_865, permute_866, alias_142, permute_867, permute_868, permute_871, div_335, permute_875, permute_879, div_336, unsqueeze_395, div_337, permute_887, permute_890, permute_891, alias_145, permute_892, permute_893, permute_896, div_346, unsqueeze_407, add_682, unsqueeze_419, add_684, unsqueeze_431, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('xcit_large_24_p8_224', benchmark_compiled_module)
