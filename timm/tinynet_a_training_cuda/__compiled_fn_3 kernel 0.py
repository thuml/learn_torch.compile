
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7df2wzqzj65kpbrw3kvwd46dfszadkdpcxhoozudelgek5bfog5.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5gih6ncjc7ypb5fcsg42idzxuteaiaapioy2gucuejra4xd5pl.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (36864*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (110592*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxknkjemlumh4epohm5ki2fs6u47vjwjq3pveg72ag64a23ye5qu.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (294912*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbkbvhdkx4ruv3g6jwoif37uo2woy23tg2zjs3jicccdrmom5mt.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl77qxo52a75er2hei25kp6fvqajpkaaxrnocuvitalxjwfzkd25.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (116*x0)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*r2) + (3712*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (32*r2) + (3712*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (32*r2) + (3712*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvkigyvbppm4e655ynpbjziyvrzwcszauryvujyt5ncfftic6sk.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 73728.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000013563552023
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cnic2cxwi4gokentvl2ofkkr2dcvw2lsgnjp6sj55v4glxdqfams.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut => mul_7, sigmoid
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 73728.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfjwudojhfifehns3m3mfqltvykoghzoja7owvchs5ysejxkwh5.py
# Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
# x_6 => add_6, add_9, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 73728.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2kk5nzzkswctxu3hyogjpbzz3gnmwwvbptkoy64emaws5yxuyp.py
# Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
# x_9 => mul_15, sigmoid_1
# x_se => mean
triton_red_fused_mean_silu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbydillorgpgfiihtpbjwkhjmga5lr2wnyw7es2jeolrcxhbsk3.py
# Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
# x_9 => mul_15, sigmoid_1
# x_se => mean
triton_red_fused_mean_silu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (2304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 9216.0
    tmp5 = tmp2 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkkm2k6m5t2qtdnhwxrvn35cqb3cydr7vdayjad57cyr45bjcjz.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
# x_se_1 => convolution_2
# x_se_2 => mul_16, sigmoid_2
triton_poi_fused_convolution_silu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coifngayzour3tqvv5dp3wxrpe354dtbeh26hrryxnpk4yl764sk.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_3
triton_poi_fused_convolution_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybjpby2fjin7g5lj3yoivvcb64t55yufjyuodw5rjkrddipjrd5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_17
# x_9 => mul_15, sigmoid_1
triton_poi_fused_mul_sigmoid_silu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    x2 = (xindex // 294912)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlk362xye4fui7ewsi4dtfrec2o43rbpebkg2apqhn7hazrwoz7.py
# Source Nodes: [x_11], Original ATen: [aten.convolution]
# x_11 => convolution_4
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (147456*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqirwopfnumih5xtfyrsiouem6svfivhh7v7rgkuylgnuiut5roh.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pmzosra375mao56v62pe2nh5fxbtenht3v5ivxcehv65oelhrw.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (116*x0)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (16*r2) + (1856*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (16*r2) + (1856*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (16*r2) + (1856*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (16*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (16*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (16*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4kjsctoieh5pooyiyvwp7wvnyt7374jha2j34hciserbyn4iky.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => add_11, add_12, add_13, mul_19, mul_20, mul_21, mul_22, mul_23, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 73728.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000013563552023
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskxwy3tjhp3f5rdibdloi4hcyccphodblzoop2sqrvhxjqmv3la.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => add_11, add_14, mul_18, mul_24, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 73728.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwgmtg4dafjik56wbgqpyy6q6eqbzllhczvmlzcrdlvonwb23o7.py
# Source Nodes: [x_16], Original ATen: [aten.convolution]
# x_16 => convolution_5
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (884736*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz27dvvsozbgsvizag2cmddjuct53xb54empyahskmad5brzyks7.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 55296
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74gliiftu2hhr52piygsryccl3qk3sth3ckxz2h6ghdcmkjqg7g.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (116*x0)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (96*r2) + (11136*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.load(in_ptr1 + (x1 + (96*r2) + (11136*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.load(in_ptr2 + (x1 + (96*r2) + (11136*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (96*x0)), tmp15, xmask)
    tl.store(out_ptr1 + (x1 + (96*x0)), tmp16, xmask)
    tl.store(out_ptr2 + (x1 + (96*x0)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckixj3lchc5q5pw64fi7srilh2pw5bwmucqywvlzcggiufvp3uh7.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => add_16, add_17, add_18, mul_26, mul_27, mul_28, mul_29, mul_30, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 73728.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000013563552023
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknmyvuox7p56fonrmsf3o4bsxg42dmufxikjgzuielufphk6vej.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_17 => add_16, add_19, mul_25, mul_31, rsqrt_3, sub_3, var_mean_3
# x_20 => mul_32, sigmoid_4
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7077888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 73728.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbv7ppga2qndfcnchgczwo4k7eq3iyk2jzmn6z73dbksxwzi3vic.py
# Source Nodes: [x_21], Original ATen: [aten.convolution]
# x_21 => convolution_6
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (221184*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyvlhmecsfbfee3lokayb5neg6i32u2plfcgyomfa2mphuuv366.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caola7i7nmbdi2efqtok5ap67y3ig35dh6npwrvxj4swhwvzzqmu.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (6912*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (6912*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (96*r2) + (6912*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (96*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (96*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (96*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftakfzzknizndxfiymkzh7obwhxn5czb3i2czou7ambucxb3fzu.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => add_21, add_22, add_23, mul_34, mul_35, mul_36, mul_37, mul_38, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (96*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 18432.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000542564158212
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdokm2lrq3tyggeecse7hww7ck3vpzfkefons7us72m73cexmbe5.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => add_21, add_24, mul_33, mul_39, rsqrt_4, sub_4, var_mean_4
triton_poi_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 18432.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cev2f2tgwhe5bytchnwsx73drybfgefybydolheischkl36vancr.py
# Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_25 => mul_40, sigmoid_5
# x_se_4 => mean_1
triton_red_fused_mean_silu_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/clubaczzcybuvdi64inapy3agdabncapdwvtmswxpk3ra66oqn3g.py
# Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_25 => mul_40, sigmoid_5
# x_se_4 => mean_1
triton_per_fused_mean_silu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 96
    x1 = (xindex // 96)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (1728*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 2304.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbesdq3qbzogn5mbkr63md7baoz2bhpatwrcrr2jmt6zepqztp7.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
# x_se_5 => convolution_7
# x_se_6 => mul_41, sigmoid_6
triton_poi_fused_convolution_silu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhdzawxog7ldjyta7ro4zznicl3yhpxhc5iamdhcsl3std7w6o6.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_8
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujkeeyto4bxklrliybzvorat2pl6txfttuxfjhr4aofmn5jyter.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_25 => mul_40, sigmoid_5
# x_26 => mul_42
triton_poi_fused_mul_sigmoid_silu_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 221184)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (96*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrp2lmmcj657ztzrle66fs3arlqop3nf7hfl5ldwto5ln3bowie.py
# Source Nodes: [x_27], Original ATen: [aten.convolution]
# x_27 => convolution_9
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (55296*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2w7nq5zqb6zj2iysxsuvtxab7uw7o45xvxbpiq4mkkxn6kdan3s.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhnkhzyynlwpp2tkycf2mfqmzqk6hkf6iv3l5qfakapnpp5k736.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (24*r2) + (1728*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (1728*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (24*r2) + (1728*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (24*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (24*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (24*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/cljgjggpczjgsy2jz6ub3vgkkqlnfs3jofkfu6dx6oexjdpazjid.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => add_26, add_27, add_28, mul_44, mul_45, mul_46, mul_47, mul_48, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 18432.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000542564158212
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nd/cndmgs3em7fiqxenrxwzviigkmy5f4yk7fvlp4oqfkvrqt2myqy2.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => add_26, add_29, mul_43, mul_49, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 18432.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuumtmfqmdsuaanm652kxbrs3ervs3qpvpus6kvaepo7haxbga5l.py
# Source Nodes: [x_32], Original ATen: [aten.convolution]
# x_32 => convolution_10
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (331776*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqq6y4ze5ltsal4i44y4y77ymm52f3ggejqqzsxeixyrhcxzcidl.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6i6sl64jeapetplsqcnwcnx436st3j7enlgh27yjek6cf64rord.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (144*r2) + (10368*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r2) + (10368*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (144*r2) + (10368*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (144*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (144*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (144*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5xiavhvrsuxmqsykykqycsxmfybvfe3nectmzgil6d3nml6j72.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => add_31, add_32, add_33, mul_51, mul_52, mul_53, mul_54, mul_55, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 18432.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000542564158212
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjkx2sqfng4bepgulfdoqfmrgjppo26kpywis7s2eqvkk5vxfw7.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_33 => add_31, add_34, mul_50, mul_56, rsqrt_6, sub_6, var_mean_6
# x_36 => mul_57, sigmoid_8
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 18432.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cch2fuhgra74f3h7nt2grhgmfq2j3vi7c65ttbfakoidw6s3kum5.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => add_36, add_39, mul_58, mul_64, rsqrt_7, sub_7, var_mean_7
triton_poi_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 18432.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czowedkkkkot6ihinkr4vp4haajktmovon5l66ilowy36we4fh4o.py
# Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_41 => mul_65, sigmoid_9
# x_se_8 => mean_2
triton_red_fused_mean_silu_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20736
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqzkimi6o2bk5l3ymydhmu6udueh7ilgbtsx7lp3sknw2ekadgw.py
# Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_41 => mul_65, sigmoid_9
# x_se_8 => mean_2
triton_per_fused_mean_silu_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (2592*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 2304.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyja5talgtunhsyalmyttyknkm6xjffhpuamv33r2h6v6rkpvuab.py
# Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
# x_se_10 => mul_66, sigmoid_10
# x_se_9 => convolution_12
triton_poi_fused_convolution_silu_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cipnqpq4hcy33n5rcdhzmdcdxrnyfs3w3tdjdwye6kofvepoojar.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution]
# x_se_11 => convolution_13
triton_poi_fused_convolution_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6aeybasmxxkt6sgyb6rsbs3jlaua3nllskvqbx5gphimszazp2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_41 => mul_65, sigmoid_9
# x_42 => mul_67
triton_poi_fused_mul_sigmoid_silu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 331776)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbncgbxvv7sxbqhrdjq3ks6qvfglkk6ncjjkjebblpv4vqzkv6b.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_45
# x_44 => add_41, add_44, mul_68, mul_74, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_add_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 18432.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdl2wo7jslwgkxfqphvsgi4slcaj4uvppi5vysyuz2urx2fegico.py
# Source Nodes: [x_54], Original ATen: [aten.convolution]
# x_54 => convolution_16
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (82944*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgcre6jin2cvaumgf32pzsr44ds3g6m2jzxeul4wwre4aiakw4ht.py
# Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
# x_55 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5184
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgp4q4smani6akwr2myj4wvbabwo7gpehrdjx4llnqju3ztxngzj.py
# Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
# x_55 => add_52, add_53, add_54, mul_84, mul_85, mul_86, mul_87, mul_88, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (144*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 4608.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0002170609941394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/csw5r74spj2n5lprp5r7yufckcngb6p4wmixnbsjmmkefxqsvbqh.py
# Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
# x_55 => add_52, add_55, mul_83, mul_89, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4608.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cukphounye3ycadfxe7a2sflnrqvqaaspbv67zycl7sky5645ifa.py
# Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
# x_58 => mul_90, sigmoid_13
# x_se_12 => mean_3
triton_red_fused_mean_silu_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144) % 5
    x0 = xindex % 144
    x2 = (xindex // 720)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (116*x1)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (144*((r3 + (116*x1)) % 576)) + (82944*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnv7zn4qb5suosyhlqfko4yn7lntuj5yxr5hoxcdhlxtmiegmjdv.py
# Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
# x_58 => mul_90, sigmoid_13
# x_se_12 => mean_3
triton_per_fused_mean_silu_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 576.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxdr5i6agjiazficfuep4ebu6lmzsfo46bca2t6pkd2xk6hnrhp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_58 => mul_90, sigmoid_13
# x_59 => mul_92
triton_poi_fused_mul_sigmoid_silu_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 82944)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4c4bhpditflucqgo5l3qvfd33lyd5tbgtfgfnyc2p5bf5dobl6.py
# Source Nodes: [x_60], Original ATen: [aten.convolution]
# x_60 => convolution_19
triton_poi_fused_convolution_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (40*x2) + (23040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5rjoff4sdnqgxhtgetfvssl5vjaho55si4zggf2keruuocdddn.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
# x_61 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1440
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 40
    x1 = (xindex // 40)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (40*r2) + (5120*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbzlrlie2sfjo4pmmrxpe5zmham4pcyu5xohxxi34jslw3wj4fa.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
# x_61 => add_57, add_58, add_59, mul_94, mul_95, mul_96, mul_97, mul_98, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (40*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (40*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 4608.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0002170609941394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdnuhvbu6j4pdkok4s22ki2thkmxbxgcpy4acrwzemc3vv5urxs.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
# x_61 => add_57, add_60, mul_93, mul_99, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 184320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4608.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cid3l7xp2fjsstpwkxtsytm3fixcj4wdlxqpngnopji2ytikzkya.py
# Source Nodes: [x_65], Original ATen: [aten.convolution]
# x_65 => convolution_20
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (138240*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblpi5mzmlfaroak777zg35y7elhw4zsp5kgl3qqd2ni7fgnm5b5.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8640
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvz3akab4tk2ssnx2pzgvdxslepbpcgcgbiarropekgowpnjenmp.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => add_62, add_63, add_64, mul_101, mul_102, mul_103, mul_104, mul_105, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 4608.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0002170609941394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazi6jljdhyo2jirajtkbcyqytuvvtcqqn3ll7udi3tqo56rguc6.py
# Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_66 => add_62, add_65, mul_100, mul_106, rsqrt_12, sub_12, var_mean_12
# x_69 => mul_107, sigmoid_16
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4608.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mg/cmgfxszb2bpykmaibtupj72ouz34wyx2xyikklsopy6mb4klv52i.py
# Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
# x_71 => add_67, add_70, mul_108, mul_114, rsqrt_13, sub_13, var_mean_13
triton_poi_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4608.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2ixzcvqbpxzseixjxogcuzlomgnvue5er44szmlyqibueen7xe.py
# Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_74 => mul_115, sigmoid_17
# x_se_16 => mean_4
triton_red_fused_mean_silu_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9600
    rnumel = 116
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240) % 5
    x0 = xindex % 240
    x2 = (xindex // 1200)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (116*x1)
        tmp1 = tl.full([1, 1], 576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (240*((r3 + (116*x1)) % 576)) + (138240*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfgyppdquwr7i5tllx75mtmqrs2dkrjl6zol64rti7m6pgpzu5p.py
# Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_74 => mul_115, sigmoid_17
# x_se_16 => mean_4
triton_per_fused_mean_silu_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_67', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 240
    x1 = (xindex // 240)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (1200*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 576.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54nbk2qb3ywpgwrt5oj2wlwemh7tvavflqohhijt5qzv25x3iuv.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
# x_se_17 => convolution_22
# x_se_18 => mul_116, sigmoid_18
triton_poi_fused_convolution_silu_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 10
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7c37zrb4cv7qalouq5zmgemn2fklyhrxacc4evu6avpuvzh3gn.py
# Source Nodes: [x_se_19], Original ATen: [aten.convolution]
# x_se_19 => convolution_23
triton_poi_fused_convolution_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caopy4hoaslg2suwzfz4kf3i5ioh367r557bpt2ozxpjkqwh2gzr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_74 => mul_115, sigmoid_17
# x_75 => mul_117
triton_poi_fused_mul_sigmoid_silu_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 138240)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (240*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyq25udfvf52ypgf3udk5zy3slledyomsrkfuwscy6cmita47yum.py
# Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_5 => add_76
# x_77 => add_72, add_75, mul_118, mul_124, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_add_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 184320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 4608.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c377jsjbvyvumbcvq3l7rs5twjz4r75y4qkdwm7oc24nwi55ngpd.py
# Source Nodes: [x_87], Original ATen: [aten.convolution]
# x_87 => convolution_26
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (34560*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cibogui6edi45ppl4ryrpg3i22cicp3owewtbpwmmdgxaktlpizo.py
# Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
# x_88 => var_mean_16
triton_red_fused__native_batch_norm_legit_functional_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2160
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3e6ft2bvfpu2ednhqwanln3bnrcmlg5geirnmpx3rngb7emuyon.py
# Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
# x_88 => add_83, add_84, add_85, mul_134, mul_135, mul_136, mul_137, mul_138, rsqrt_16, squeeze_49, var_mean_16
triton_per_fused__native_batch_norm_legit_functional_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_74', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1152.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000868809730669
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cntwznbsqfcn2fhqavh25h2oytkgxqutp3dmou7q7pedssyhhj7q.py
# Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
# x_88 => add_83, add_86, mul_133, mul_139, rsqrt_16, sub_16, var_mean_16
triton_poi_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 276480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cididdqpi4eze6g7v26eukutmhkdjpg5omudvhhotxnh4nt7tkyj.py
# Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_91 => mul_140, sigmoid_21
# x_se_20 => mean_5
triton_red_fused_mean_silu_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (17280*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmcp6h6z5v3jk5k6ejvcbwcnhbrrtfq3qvgzle2m4qec2427mme.py
# Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_91 => mul_140, sigmoid_21
# x_se_20 => mean_5
triton_per_fused_mean_silu_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 240
    x1 = (xindex // 240)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (480*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 144.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73znu7zenewfgjbxu2lhkmp6ypyboda6ixh5b7d7dbhx2jmwu4u.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_91 => mul_140, sigmoid_21
# x_92 => mul_142
triton_poi_fused_mul_sigmoid_silu_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 276480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 34560)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (240*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajcisnwdpxd732cusbriz7ashmllixtcu2k7rllbdlryq5l5bqt.py
# Source Nodes: [x_93], Original ATen: [aten.convolution]
# x_93 => convolution_29
triton_poi_fused_convolution_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (11520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsgxzete5k6ywlsmwr4yxjoaeqb5xu3nzy6pfoo4wayxzbgjifp.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
# x_94 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 80
    x1 = (xindex // 80)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (80*r2) + (10240*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54jhcghorbi6pvfvyqb4teqratbbsytkflo6edez4eae4geyss3.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
# x_94 => add_88, add_89, add_90, mul_144, mul_145, mul_146, mul_147, mul_148, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1152.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000868809730669
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/cljcwi2pxpqnffwkhoeumbbuu4yvpj6qzxihp6suj4rxbbmwdtxt.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
# x_94 => add_88, add_91, mul_143, mul_149, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tfylgyn2welzq3vp4anc2q3xlrb42uptfv66tdaobb5hgtixxy.py
# Source Nodes: [x_98], Original ATen: [aten.convolution]
# x_98 => convolution_30
triton_poi_fused_convolution_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (480*x2) + (69120*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7k/c7k6bygrcwnso2gpbrwsol6vgpvttxgfsvye73o2zwu42dsape7j.py
# Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
# x_99 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4320
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (61440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5x4qyhvptajvddfuescrhluq7gr26a6b4fam6xdmkfanexz6dj.py
# Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
# x_99 => add_93, add_94, add_95, mul_151, mul_152, mul_153, mul_154, mul_155, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_85', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1152.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000868809730669
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn672fythrplybjtepf54yxhibktuuwetpyocxztciygifcgckmf.py
# Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_102 => mul_157, sigmoid_24
# x_99 => add_93, add_96, mul_150, mul_156, rsqrt_18, sub_18, var_mean_18
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 552960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbj27nvvomnm32qlym44kyvqqe4d6kv6y64y4sgwaoh76farlg3.py
# Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
# x_104 => add_101, add_98, mul_158, mul_164, rsqrt_19, sub_19, var_mean_19
triton_poi_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 552960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q6cuaesluqqecqlm34d3n25wcschg2nwhxb5eqsh3ldy6ni4rv.py
# Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
# x_107 => mul_165, sigmoid_25
# x_se_24 => mean_6
triton_red_fused_mean_silu_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (34560*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegmi67r24whvdkkzkbyujhyrykxoq6dzvzlvj7cp6uxjnje3hym.py
# Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
# x_107 => mul_165, sigmoid_25
# x_se_24 => mean_6
triton_per_fused_mean_silu_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_89', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 480
    x1 = (xindex // 480)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (960*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 144.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkz3vm4cf52f7uwku25k2jsptfrkm22ruiw7vi2xhghwssbdcpx.py
# Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
# x_se_25 => convolution_32
# x_se_26 => mul_166, sigmoid_26
triton_poi_fused_convolution_silu_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_90', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyj563jycrifnzhpk3cf5rvivhf7v3ggurgdozegpz5t36fuo2r.py
# Source Nodes: [x_se_27], Original ATen: [aten.convolution]
# x_se_27 => convolution_33
triton_poi_fused_convolution_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_91', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qwvpdmftzoapcgtzsrbyhsmyp4kqwksklmhaqkljyk7xy2k3nf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_27
# x_107 => mul_165, sigmoid_25
# x_108 => mul_167
triton_poi_fused_mul_sigmoid_silu_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 552960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 480
    x2 = (xindex // 69120)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (480*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zb/czbdnxdmisnmv7pn7nvarpm2nxv6klthjoxzpifsinv5c4sanca6.py
# Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_7 => add_107
# x_110 => add_103, add_106, mul_168, mul_174, rsqrt_20, sub_20, var_mean_20
triton_poi_fused__native_batch_norm_legit_functional_add_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4x36ynjlf74smsrolrtudchgocslvjf6f7azglev5ybvfsgl24.py
# Source Nodes: [x_160], Original ATen: [aten.convolution]
# x_160 => convolution_49
triton_poi_fused_convolution_94 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (16128*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mw/cmw52sugrwqi3zdykkd5pmonw6mnbervzct2j64lh62enyt4ngzo.py
# Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
# x_161 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1008
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 112
    x1 = (xindex // 112)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (112*r2) + (14336*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3uhme2s2kmdrwdbim5m3gsust7opukr3ouci7n5owrm2j4mszw.py
# Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
# x_161 => add_151, add_152, add_153, mul_244, mul_245, mul_246, mul_247, mul_248, rsqrt_29, squeeze_88, var_mean_29
triton_per_fused__native_batch_norm_legit_functional_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_96', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (112*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1152.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000868809730669
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gi4ykusxzhrpdlxsvi2e7owl5e5knwzdgetincutiqa2sodrjf.py
# Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
# x_161 => add_151, add_154, mul_243, mul_249, rsqrt_29, sub_29, var_mean_29
triton_poi_fused__native_batch_norm_legit_functional_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpx3bkqzbrw2ge3skrigqencj4o46c3d5r2vzgh4u4uxfkuwvzhv.py
# Source Nodes: [x_165], Original ATen: [aten.convolution]
# x_165 => convolution_50
triton_poi_fused_convolution_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (672*x2) + (96768*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/cikzutvb7x75vmsytz44praeh3y5vtn2j37ppyupxkjxg57v2qoi.py
# Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
# x_166 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (86016*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crfjecicsy6argufwbykpwiirb7otdlebwdpxckf54qa4txfiah3.py
# Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
# x_166 => add_156, add_157, add_158, mul_251, mul_252, mul_253, mul_254, mul_255, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_100', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1152.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000868809730669
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cichycuaiubzgjsoowmbpdy2j3ssvs4tprsd5il7k44e6knstyrr.py
# Source Nodes: [x_166, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_166 => add_156, add_159, mul_250, mul_256, rsqrt_30, sub_30, var_mean_30
# x_169 => mul_257, sigmoid_40
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrqlbdquj3w3moxldbaswcagfthwgdsj4egw6no5mdtxmdtkgzm.py
# Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
# x_171 => add_161, add_164, mul_258, mul_264, rsqrt_31, sub_31, var_mean_31
triton_poi_fused__native_batch_norm_legit_functional_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2prem4rzvzmpbk3lkuc47rnaj6y47o4764vaesokqpyzkpglrh.py
# Source Nodes: [x_174, x_se_40], Original ATen: [aten.mean, aten.silu]
# x_174 => mul_265, sigmoid_41
# x_se_40 => mean_10
triton_red_fused_mean_silu_103 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (48384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/costfqpkenvas3vu55ns2vtn7kunwg2zxwjpmogvyzhxvrtqp4sv.py
# Source Nodes: [x_174, x_se_40], Original ATen: [aten.mean, aten.silu]
# x_174 => mul_265, sigmoid_41
# x_se_40 => mean_10
triton_per_fused_mean_silu_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_104', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (1344*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 144.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafdbnulw6wuya4ln3l4uclwrbfovqa5y47mr5wxsz5iywc6v5wv.py
# Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.silu]
# x_se_41 => convolution_52
# x_se_42 => mul_266, sigmoid_42
triton_poi_fused_convolution_silu_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_105', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cetk6sn5mhfwwwb7b272aaoyvmf3pnfjl5ctfq26o7qfpzc25mfz.py
# Source Nodes: [x_se_43], Original ATen: [aten.convolution]
# x_se_43 => convolution_53
triton_poi_fused_convolution_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_106', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxsaetfoups3t7quwdwoxqhig57nloyo56xp6gkgkv46kjrpfur.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174, x_175], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_43
# x_174 => mul_265, sigmoid_41
# x_175 => mul_267
triton_poi_fused_mul_sigmoid_silu_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 96768)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (672*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zpwr3plehaki574lv734qaxvqsipvd3uh2v65iay5unswfzyuk.py
# Source Nodes: [shortcut_11, x_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_11 => add_170
# x_177 => add_166, add_169, mul_268, mul_274, rsqrt_32, sub_32, var_mean_32
triton_poi_fused__native_batch_norm_legit_functional_add_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1152.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckm5nzhwp7tvdcbv5rq42lpdwsddgwipw75itrotkcica7ud3zqq.py
# Source Nodes: [x_221], Original ATen: [aten.convolution]
# x_221 => convolution_66
triton_poi_fused_convolution_109 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (672*x2) + (24192*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccntzgkgiqjtgte7ys6f5vciyx3zom2tzc7z27ifwhs3nirypce.py
# Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
# x_222 => var_mean_40
triton_red_fused__native_batch_norm_legit_functional_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2016
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (64512*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhebkbtlx5xjoglarur67jpyl7ps5q6bntmp7ib6bpefpxvtv7u.py
# Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
# x_222 => add_209, add_210, add_211, mul_334, mul_335, mul_336, mul_337, mul_338, rsqrt_40, squeeze_121, var_mean_40
triton_per_fused__native_batch_norm_legit_functional_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_111', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 288.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0034843205574913
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36xn6ytgi3h4hgvmegt3up7iazurchcv5b4u43mcsypmxd7squh.py
# Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
# x_222 => add_209, add_212, mul_333, mul_339, rsqrt_40, sub_40, var_mean_40
triton_poi_fused__native_batch_norm_legit_functional_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 193536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjw5cf7rddeyqhk56mhvpqz4awlejmx24g7lgkpcnddfc44yfrmn.py
# Source Nodes: [x_225, x_se_52], Original ATen: [aten.mean, aten.silu]
# x_225 => mul_340, sigmoid_53
# x_se_52 => mean_13
triton_per_fused_mean_silu_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_113', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (24192*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cibl4kyr5nahry77yupegbnbedjhzfne5x2lj3j6wkpqt4wjzdbz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225, x_226], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
# x_225 => mul_340, sigmoid_53
# x_226 => mul_342
triton_poi_fused_mul_sigmoid_silu_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_114', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 193536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 24192)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5j7flgwtynvwgxqqp2s7qqfje2zm5rpeyp6m7nk6kqedvhq46iq.py
# Source Nodes: [x_227], Original ATen: [aten.convolution]
# x_227 => convolution_69
triton_poi_fused_convolution_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (6912*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccoclvivz7bttpbuziuob3tuqbg33lxcyunznwlqujt54vo5wclm.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => var_mean_41
triton_red_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (18432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hh/chhp3lrzqosvjicgxnz4mzzhzfvbfzfw6cf26h223ifycjs776g3.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => add_214, add_215, add_216, mul_344, mul_345, mul_346, mul_347, mul_348, rsqrt_41, squeeze_124, var_mean_41
triton_per_fused__native_batch_norm_legit_functional_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_117', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 288.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0034843205574913
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwlfxfcikkzv3lwp2mkqx2cnfmvrfacblgyrvyw5jso5ysmcyka.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => add_214, add_217, mul_343, mul_349, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfsvvat2s2r3mcdmg7wdne3qegcr76amvwmuibizu2yyzwc6iub.py
# Source Nodes: [x_232], Original ATen: [aten.convolution]
# x_232 => convolution_70
triton_poi_fused_convolution_119 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_119', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1152*x2) + (41472*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7m63avuwl5ngwmonzbebrevrlyfprz4uvvrgplpmh4yofnk4pd6.py
# Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
# x_233 => var_mean_42
triton_red_fused__native_batch_norm_legit_functional_120 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3456
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (110592*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbymt7m5sgr4f4vvobwjjhsawxvlagxneosv4grlv3ulyqso32s.py
# Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
# x_233 => add_219, add_220, add_221, mul_351, mul_352, mul_353, mul_354, mul_355, rsqrt_42, squeeze_127, var_mean_42
triton_per_fused__native_batch_norm_legit_functional_121 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_121', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 288.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0034843205574913
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54tj2mafkbq6rwdhrxp4dfrfijtomxlfh5276vbg63dkzlw2emy.py
# Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_233 => add_219, add_222, mul_350, mul_356, rsqrt_42, sub_42, var_mean_42
# x_236 => mul_357, sigmoid_56
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x2), tmp15, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehgvufqkd4lcl3j75j262t3yaq33tkqkiisi4im3hi75tsuzadt.py
# Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
# x_238 => add_224, add_227, mul_358, mul_364, rsqrt_43, sub_43, var_mean_43
triton_poi_fused__native_batch_norm_legit_functional_123 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgs46jduwfngzkhp4htrcgrogzz4ik2bwspk2er35bwoir3bubyg.py
# Source Nodes: [x_241, x_se_56], Original ATen: [aten.mean, aten.silu]
# x_241 => mul_365, sigmoid_57
# x_se_56 => mean_14
triton_per_fused_mean_silu_124 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_124', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (41472*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4wyzcs7xxtrzuarssio6dj2ro5b6wo4iwaklccivxvsqvaupgt.py
# Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.silu]
# x_se_57 => convolution_72
# x_se_58 => mul_366, sigmoid_58
triton_poi_fused_convolution_silu_125 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_125', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7ilhkm53u4sauz75u56e6udxlfqvt3g4cnn2orcix3zpqe36el.py
# Source Nodes: [x_se_59], Original ATen: [aten.convolution]
# x_se_59 => convolution_73
triton_poi_fused_convolution_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_126', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczk3agwpxamsr2uidtojjr54tdsq6qj3xdcevhvioospeuyo3iy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241, x_242], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_59
# x_241 => mul_365, sigmoid_57
# x_242 => mul_367
triton_poi_fused_mul_sigmoid_silu_127 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_127', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1152
    x2 = (xindex // 41472)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (1152*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbb5wjirqm46rxnj7t7x3tdbwkxosfm6gh4baujcguezugycx4nz.py
# Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_15 => add_233
# x_244 => add_229, add_232, mul_368, mul_374, rsqrt_44, sub_44, var_mean_44
triton_poi_fused__native_batch_norm_legit_functional_add_128 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_128', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/coti3jbusqdeuyzhq26xuhpbpkaqjfs5ydn7rmjye5jnrfvi3232.py
# Source Nodes: [x_311], Original ATen: [aten.convolution]
# x_311 => convolution_94
triton_poi_fused_convolution_129 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (11520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gnae64vyx3h64pbbdjpv3tg4apsghchekhtvafeux3mmpaqzwi.py
# Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
# x_312 => var_mean_56
triton_red_fused__native_batch_norm_legit_functional_130 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwg4yq34lht7fztwht7tikbhlpfrrsua6qecw64alna4hzmfh3p.py
# Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
# x_312 => add_293, add_294, add_295, mul_469, mul_470, mul_471, mul_472, mul_473, rsqrt_56, squeeze_169, var_mean_56
triton_per_fused__native_batch_norm_legit_functional_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_131', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 288.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0034843205574913
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfl72gpunagal2pno4c3depc7aqpsomzxco4md5pcdgawiqyp6t.py
# Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
# x_312 => add_293, add_296, mul_468, mul_474, rsqrt_56, sub_56, var_mean_56
triton_poi_fused__native_batch_norm_legit_functional_132 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33gvei2zjzwzmutuqx6apmw4yq43ru7oyxewzdrnasf4xetxyd2.py
# Source Nodes: [x_317], Original ATen: [aten.convolution]
# x_317 => convolution_95
triton_poi_fused_convolution_133 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_133', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (46080*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctaeppaff4j7xfoy4r26yu7o33khqe2srvueqxrqwf327yy4m556.py
# Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional]
# x_318 => var_mean_57
triton_red_fused__native_batch_norm_legit_functional_134 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (122880*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnsva2jvrbe5f6kl4onze4hyday2m6vesi67bn4vm6cosaah2aa.py
# Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional]
# x_318 => add_298, add_299, add_300, mul_476, mul_477, mul_478, mul_479, mul_480, rsqrt_57, squeeze_172, var_mean_57
triton_per_fused__native_batch_norm_legit_functional_135 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_135', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 288.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0034843205574913
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfohw7lumtvrrl6cstxz4em4yiksa3alistbus3zj5pdm7i4rvkl.py
# Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_318 => add_298, add_301, mul_475, mul_481, rsqrt_57, sub_57, var_mean_57
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 288.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cam3ivtknx4ciytggymtkso2maw4pqrbui4pwsm6objjwcrn6ca3.py
# Source Nodes: [x_322, x_323, x_325], Original ATen: [aten.mean, aten.silu, aten.view]
# x_322 => mul_482, sigmoid_76
# x_323 => mean_19
# x_325 => view
triton_per_fused_mean_silu_view_137 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_view_137', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (46080*x1)), rmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/cooclris4j3hfihizar4r5aqnf2beh4tmap2fpuspahkznxot3pc.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_138 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_138', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (144, ), (1, ))
    assert_size_stride(primals_14, (144, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_16, (144, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (144, ), (1, ))
    assert_size_stride(primals_20, (144, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_22, (144, ), (1, ))
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_24, (40, ), (1, ))
    assert_size_stride(primals_25, (240, ), (1, ))
    assert_size_stride(primals_26, (240, ), (1, ))
    assert_size_stride(primals_27, (240, ), (1, ))
    assert_size_stride(primals_28, (240, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_30, (40, ), (1, ))
    assert_size_stride(primals_31, (240, ), (1, ))
    assert_size_stride(primals_32, (240, ), (1, ))
    assert_size_stride(primals_33, (240, ), (1, ))
    assert_size_stride(primals_34, (240, ), (1, ))
    assert_size_stride(primals_35, (80, ), (1, ))
    assert_size_stride(primals_36, (80, ), (1, ))
    assert_size_stride(primals_37, (480, ), (1, ))
    assert_size_stride(primals_38, (480, ), (1, ))
    assert_size_stride(primals_39, (480, ), (1, ))
    assert_size_stride(primals_40, (480, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_42, (80, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_44, (480, ), (1, ))
    assert_size_stride(primals_45, (480, ), (1, ))
    assert_size_stride(primals_46, (480, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_48, (80, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_50, (480, ), (1, ))
    assert_size_stride(primals_51, (480, ), (1, ))
    assert_size_stride(primals_52, (480, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_54, (80, ), (1, ))
    assert_size_stride(primals_55, (480, ), (1, ))
    assert_size_stride(primals_56, (480, ), (1, ))
    assert_size_stride(primals_57, (480, ), (1, ))
    assert_size_stride(primals_58, (480, ), (1, ))
    assert_size_stride(primals_59, (112, ), (1, ))
    assert_size_stride(primals_60, (112, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_62, (672, ), (1, ))
    assert_size_stride(primals_63, (672, ), (1, ))
    assert_size_stride(primals_64, (672, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_66, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_68, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_70, (672, ), (1, ))
    assert_size_stride(primals_71, (112, ), (1, ))
    assert_size_stride(primals_72, (112, ), (1, ))
    assert_size_stride(primals_73, (672, ), (1, ))
    assert_size_stride(primals_74, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_76, (672, ), (1, ))
    assert_size_stride(primals_77, (112, ), (1, ))
    assert_size_stride(primals_78, (112, ), (1, ))
    assert_size_stride(primals_79, (672, ), (1, ))
    assert_size_stride(primals_80, (672, ), (1, ))
    assert_size_stride(primals_81, (672, ), (1, ))
    assert_size_stride(primals_82, (672, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_84, (192, ), (1, ))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_86, (1152, ), (1, ))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_88, (1152, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_90, (192, ), (1, ))
    assert_size_stride(primals_91, (1152, ), (1, ))
    assert_size_stride(primals_92, (1152, ), (1, ))
    assert_size_stride(primals_93, (1152, ), (1, ))
    assert_size_stride(primals_94, (1152, ), (1, ))
    assert_size_stride(primals_95, (192, ), (1, ))
    assert_size_stride(primals_96, (192, ), (1, ))
    assert_size_stride(primals_97, (1152, ), (1, ))
    assert_size_stride(primals_98, (1152, ), (1, ))
    assert_size_stride(primals_99, (1152, ), (1, ))
    assert_size_stride(primals_100, (1152, ), (1, ))
    assert_size_stride(primals_101, (192, ), (1, ))
    assert_size_stride(primals_102, (192, ), (1, ))
    assert_size_stride(primals_103, (1152, ), (1, ))
    assert_size_stride(primals_104, (1152, ), (1, ))
    assert_size_stride(primals_105, (1152, ), (1, ))
    assert_size_stride(primals_106, (1152, ), (1, ))
    assert_size_stride(primals_107, (192, ), (1, ))
    assert_size_stride(primals_108, (192, ), (1, ))
    assert_size_stride(primals_109, (1152, ), (1, ))
    assert_size_stride(primals_110, (1152, ), (1, ))
    assert_size_stride(primals_111, (1152, ), (1, ))
    assert_size_stride(primals_112, (1152, ), (1, ))
    assert_size_stride(primals_113, (320, ), (1, ))
    assert_size_stride(primals_114, (320, ), (1, ))
    assert_size_stride(primals_115, (1280, ), (1, ))
    assert_size_stride(primals_116, (1280, ), (1, ))
    assert_size_stride(primals_117, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_118, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_120, (8, ), (1, ))
    assert_size_stride(primals_121, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_122, (32, ), (1, ))
    assert_size_stride(primals_123, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_125, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_126, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_127, (4, ), (1, ))
    assert_size_stride(primals_128, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_129, (96, ), (1, ))
    assert_size_stride(primals_130, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_131, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_132, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_134, (6, ), (1, ))
    assert_size_stride(primals_135, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_136, (144, ), (1, ))
    assert_size_stride(primals_137, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_138, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_139, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_140, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_141, (6, ), (1, ))
    assert_size_stride(primals_142, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_143, (144, ), (1, ))
    assert_size_stride(primals_144, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_145, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_146, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_147, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_148, (10, ), (1, ))
    assert_size_stride(primals_149, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_150, (240, ), (1, ))
    assert_size_stride(primals_151, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_152, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_153, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_154, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_155, (10, ), (1, ))
    assert_size_stride(primals_156, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_157, (240, ), (1, ))
    assert_size_stride(primals_158, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_159, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_160, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_162, (20, ), (1, ))
    assert_size_stride(primals_163, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_164, (480, ), (1, ))
    assert_size_stride(primals_165, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_166, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_167, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_169, (20, ), (1, ))
    assert_size_stride(primals_170, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_171, (480, ), (1, ))
    assert_size_stride(primals_172, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_173, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_174, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_176, (20, ), (1, ))
    assert_size_stride(primals_177, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_178, (480, ), (1, ))
    assert_size_stride(primals_179, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_180, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_181, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_182, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_183, (20, ), (1, ))
    assert_size_stride(primals_184, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_185, (480, ), (1, ))
    assert_size_stride(primals_186, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_187, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_188, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_189, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_190, (28, ), (1, ))
    assert_size_stride(primals_191, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_192, (672, ), (1, ))
    assert_size_stride(primals_193, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_194, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_195, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_197, (28, ), (1, ))
    assert_size_stride(primals_198, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_199, (672, ), (1, ))
    assert_size_stride(primals_200, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_201, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_202, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_203, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_204, (28, ), (1, ))
    assert_size_stride(primals_205, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_206, (672, ), (1, ))
    assert_size_stride(primals_207, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_208, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_209, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_210, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_211, (28, ), (1, ))
    assert_size_stride(primals_212, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_213, (672, ), (1, ))
    assert_size_stride(primals_214, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_215, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_216, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_217, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_218, (48, ), (1, ))
    assert_size_stride(primals_219, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_220, (1152, ), (1, ))
    assert_size_stride(primals_221, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_222, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_223, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_224, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_225, (48, ), (1, ))
    assert_size_stride(primals_226, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_227, (1152, ), (1, ))
    assert_size_stride(primals_228, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_229, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_230, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_232, (48, ), (1, ))
    assert_size_stride(primals_233, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_234, (1152, ), (1, ))
    assert_size_stride(primals_235, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_236, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_237, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_238, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_239, (48, ), (1, ))
    assert_size_stride(primals_240, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_241, (1152, ), (1, ))
    assert_size_stride(primals_242, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_243, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_244, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_245, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_246, (48, ), (1, ))
    assert_size_stride(primals_247, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_248, (1152, ), (1, ))
    assert_size_stride(primals_249, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_250, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_251, (1000, 1280), (1280, 1))
    assert_size_stride(primals_252, (1000, ), (1, ))
    assert_size_stride(primals_253, (), ())
    assert_size_stride(primals_254, (32, ), (1, ))
    assert_size_stride(primals_255, (32, ), (1, ))
    assert_size_stride(primals_256, (), ())
    assert_size_stride(primals_257, (32, ), (1, ))
    assert_size_stride(primals_258, (32, ), (1, ))
    assert_size_stride(primals_259, (), ())
    assert_size_stride(primals_260, (16, ), (1, ))
    assert_size_stride(primals_261, (16, ), (1, ))
    assert_size_stride(primals_262, (), ())
    assert_size_stride(primals_263, (96, ), (1, ))
    assert_size_stride(primals_264, (96, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (96, ), (1, ))
    assert_size_stride(primals_267, (96, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (24, ), (1, ))
    assert_size_stride(primals_270, (24, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (144, ), (1, ))
    assert_size_stride(primals_273, (144, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (144, ), (1, ))
    assert_size_stride(primals_276, (144, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (24, ), (1, ))
    assert_size_stride(primals_279, (24, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (144, ), (1, ))
    assert_size_stride(primals_282, (144, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (144, ), (1, ))
    assert_size_stride(primals_285, (144, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (40, ), (1, ))
    assert_size_stride(primals_288, (40, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (240, ), (1, ))
    assert_size_stride(primals_291, (240, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (240, ), (1, ))
    assert_size_stride(primals_294, (240, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (40, ), (1, ))
    assert_size_stride(primals_297, (40, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (240, ), (1, ))
    assert_size_stride(primals_300, (240, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (240, ), (1, ))
    assert_size_stride(primals_303, (240, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (80, ), (1, ))
    assert_size_stride(primals_306, (80, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (480, ), (1, ))
    assert_size_stride(primals_309, (480, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (480, ), (1, ))
    assert_size_stride(primals_312, (480, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (80, ), (1, ))
    assert_size_stride(primals_315, (80, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (480, ), (1, ))
    assert_size_stride(primals_318, (480, ), (1, ))
    assert_size_stride(primals_319, (), ())
    assert_size_stride(primals_320, (480, ), (1, ))
    assert_size_stride(primals_321, (480, ), (1, ))
    assert_size_stride(primals_322, (), ())
    assert_size_stride(primals_323, (80, ), (1, ))
    assert_size_stride(primals_324, (80, ), (1, ))
    assert_size_stride(primals_325, (), ())
    assert_size_stride(primals_326, (480, ), (1, ))
    assert_size_stride(primals_327, (480, ), (1, ))
    assert_size_stride(primals_328, (), ())
    assert_size_stride(primals_329, (480, ), (1, ))
    assert_size_stride(primals_330, (480, ), (1, ))
    assert_size_stride(primals_331, (), ())
    assert_size_stride(primals_332, (80, ), (1, ))
    assert_size_stride(primals_333, (80, ), (1, ))
    assert_size_stride(primals_334, (), ())
    assert_size_stride(primals_335, (480, ), (1, ))
    assert_size_stride(primals_336, (480, ), (1, ))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (480, ), (1, ))
    assert_size_stride(primals_339, (480, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (112, ), (1, ))
    assert_size_stride(primals_342, (112, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (672, ), (1, ))
    assert_size_stride(primals_345, (672, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (672, ), (1, ))
    assert_size_stride(primals_348, (672, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (112, ), (1, ))
    assert_size_stride(primals_351, (112, ), (1, ))
    assert_size_stride(primals_352, (), ())
    assert_size_stride(primals_353, (672, ), (1, ))
    assert_size_stride(primals_354, (672, ), (1, ))
    assert_size_stride(primals_355, (), ())
    assert_size_stride(primals_356, (672, ), (1, ))
    assert_size_stride(primals_357, (672, ), (1, ))
    assert_size_stride(primals_358, (), ())
    assert_size_stride(primals_359, (112, ), (1, ))
    assert_size_stride(primals_360, (112, ), (1, ))
    assert_size_stride(primals_361, (), ())
    assert_size_stride(primals_362, (672, ), (1, ))
    assert_size_stride(primals_363, (672, ), (1, ))
    assert_size_stride(primals_364, (), ())
    assert_size_stride(primals_365, (672, ), (1, ))
    assert_size_stride(primals_366, (672, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (112, ), (1, ))
    assert_size_stride(primals_369, (112, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (672, ), (1, ))
    assert_size_stride(primals_372, (672, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (672, ), (1, ))
    assert_size_stride(primals_375, (672, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (192, ), (1, ))
    assert_size_stride(primals_378, (192, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (1152, ), (1, ))
    assert_size_stride(primals_381, (1152, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (1152, ), (1, ))
    assert_size_stride(primals_384, (1152, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (192, ), (1, ))
    assert_size_stride(primals_387, (192, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (1152, ), (1, ))
    assert_size_stride(primals_390, (1152, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (1152, ), (1, ))
    assert_size_stride(primals_393, (1152, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (192, ), (1, ))
    assert_size_stride(primals_396, (192, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (1152, ), (1, ))
    assert_size_stride(primals_399, (1152, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (1152, ), (1, ))
    assert_size_stride(primals_402, (1152, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (192, ), (1, ))
    assert_size_stride(primals_405, (192, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (1152, ), (1, ))
    assert_size_stride(primals_408, (1152, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (1152, ), (1, ))
    assert_size_stride(primals_411, (1152, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (192, ), (1, ))
    assert_size_stride(primals_414, (192, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (1152, ), (1, ))
    assert_size_stride(primals_417, (1152, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (1152, ), (1, ))
    assert_size_stride(primals_420, (1152, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (320, ), (1, ))
    assert_size_stride(primals_423, (320, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (1280, ), (1, ))
    assert_size_stride(primals_426, (1280, ), (1, ))
    assert_size_stride(primals_427, (8, 3, 192, 192), (110592, 36864, 192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_117, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_117
        buf1 = empty_strided((8, 3, 192, 192), (110592, 1, 576, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_427, buf1, 24, 36864, grid=grid(24, 36864), stream=stream0)
        del primals_427
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 96, 96), (294912, 9216, 96, 1))
        buf3 = empty_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 256, 9216, grid=grid(256, 9216), stream=stream0)
        buf4 = empty_strided((1, 32, 1, 1, 576), (18432, 1, 18432, 18432, 32), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 32, 1, 1, 576), (18432, 1, 18432, 18432, 32), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 32, 1, 1, 576), (18432, 1, 18432, 18432, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf3, buf4, buf5, buf6, 18432, 128, grid=grid(18432), stream=stream0)
        buf7 = empty_strided((1, 32, 1, 1, 5), (160, 1, 160, 160, 32), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 32, 1, 1, 5), (160, 1, 160, 160, 32), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1, 32, 1, 1, 5), (160, 1, 160, 160, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf4, buf5, buf6, buf7, buf8, buf9, 160, 116, grid=grid(160), stream=stream0)
        buf10 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf13 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_254, primals_255, buf10, buf11, buf13, primals_254, primals_255, 32, 5, grid=grid(32), stream=stream0)
        del primals_254
        del primals_255
        buf15 = reinterpret_tensor(buf2, (8, 32, 96, 96), (294912, 1, 3072, 32), 0); del buf2  # reuse
        buf818 = empty_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_6.run(buf3, buf10, buf11, primals_1, primals_2, buf15, buf818, 2359296, grid=grid(2359296), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf16, (8, 32, 96, 96), (294912, 9216, 96, 1))
        buf17 = empty_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf16, buf17, 256, 9216, grid=grid(256, 9216), stream=stream0)
        buf18 = buf6; del buf6  # reuse
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf17, buf18, buf19, buf20, 18432, 128, grid=grid(18432), stream=stream0)
        buf21 = buf9; del buf9  # reuse
        buf22 = buf8; del buf8  # reuse
        buf23 = buf7; del buf7  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf18, buf19, buf20, buf21, buf22, buf23, 160, 116, grid=grid(160), stream=stream0)
        del buf18
        del buf19
        buf24 = buf11; del buf11  # reuse
        buf25 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf27 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf21, buf22, buf23, primals_257, primals_258, buf24, buf25, buf27, primals_257, primals_258, 32, 5, grid=grid(32), stream=stream0)
        del primals_257
        del primals_258
        buf28 = reinterpret_tensor(buf16, (8, 32, 96, 96), (294912, 1, 3072, 32), 0); del buf16  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_7.run(buf17, buf24, buf25, primals_3, primals_4, buf28, 2359296, grid=grid(2359296), stream=stream0)
        del primals_4
        buf29 = reinterpret_tensor(buf20, (8, 32, 1, 1, 72), (2304, 1, 18432, 18432, 32), 0); del buf20  # reuse
        # Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_8.run(buf28, buf29, 18432, 128, grid=grid(18432), stream=stream0)
        buf30 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf31 = reinterpret_tensor(buf30, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf30  # reuse
        # Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_9.run(buf31, buf29, 256, 72, grid=grid(256), stream=stream0)
        del buf29
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 8, 1, 1), (8, 1, 1, 1))
        buf33 = reinterpret_tensor(buf32, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf32  # reuse
        buf34 = empty_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_10.run(buf33, primals_120, buf34, 64, grid=grid(64), stream=stream0)
        del primals_120
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 32, 1, 1), (32, 1, 1, 1))
        buf36 = reinterpret_tensor(buf35, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf35  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf36, primals_122, 256, grid=grid(256), stream=stream0)
        del primals_122
        buf37 = empty_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_12.run(buf28, buf36, buf37, 2359296, grid=grid(2359296), stream=stream0)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 16, 96, 96), (147456, 9216, 96, 1))
        buf39 = empty_strided((8, 16, 96, 96), (147456, 1, 1536, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf38, buf39, 128, 9216, grid=grid(128, 9216), stream=stream0)
        buf40 = empty_strided((1, 16, 1, 1, 576), (9216, 1, 9216, 9216, 16), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 16, 1, 1, 576), (9216, 1, 9216, 9216, 16), device='cuda', dtype=torch.float32)
        buf42 = empty_strided((1, 16, 1, 1, 576), (9216, 1, 9216, 9216, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf39, buf40, buf41, buf42, 9216, 128, grid=grid(9216), stream=stream0)
        buf43 = empty_strided((1, 16, 1, 1, 5), (80, 1, 80, 80, 16), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 16, 1, 1, 5), (80, 1, 80, 80, 16), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((1, 16, 1, 1, 5), (80, 1, 80, 80, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf40, buf41, buf42, buf43, buf44, buf45, 80, 116, grid=grid(80), stream=stream0)
        buf46 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf49 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf43, buf44, buf45, primals_260, primals_261, buf46, buf47, buf49, primals_260, primals_261, 16, 5, grid=grid(16), stream=stream0)
        del primals_260
        del primals_261
        buf50 = reinterpret_tensor(buf38, (8, 16, 96, 96), (147456, 1, 1536, 16), 0); del buf38  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_17.run(buf39, buf46, buf47, primals_5, primals_6, buf50, 1179648, grid=grid(1179648), stream=stream0)
        del buf47
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 96, 96, 96), (884736, 9216, 96, 1))
        buf52 = empty_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf51, buf52, 768, 9216, grid=grid(768, 9216), stream=stream0)
        buf53 = empty_strided((1, 96, 1, 1, 576), (55296, 1, 55296, 55296, 96), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((1, 96, 1, 1, 576), (55296, 1, 55296, 55296, 96), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((1, 96, 1, 1, 576), (55296, 1, 55296, 55296, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf52, buf53, buf54, buf55, 55296, 128, grid=grid(55296), stream=stream0)
        buf56 = empty_strided((1, 96, 1, 1, 5), (480, 1, 480, 480, 96), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((1, 96, 1, 1, 5), (480, 1, 480, 480, 96), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 96, 1, 1, 5), (480, 1, 480, 480, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf53, buf54, buf55, buf56, buf57, buf58, 480, 116, grid=grid(480), stream=stream0)
        buf59 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf62 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf56, buf57, buf58, primals_263, primals_264, buf59, buf60, buf62, primals_263, primals_264, 96, 5, grid=grid(96), stream=stream0)
        del primals_263
        del primals_264
        buf64 = reinterpret_tensor(buf51, (8, 96, 96, 96), (884736, 1, 9216, 96), 0); del buf51  # reuse
        buf817 = empty_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_22.run(buf52, buf59, buf60, primals_7, primals_8, buf64, buf817, 7077888, grid=grid(7077888), stream=stream0)
        del primals_8
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_125, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf65, (8, 96, 48, 48), (221184, 2304, 48, 1))
        buf66 = empty_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf65, buf66, 768, 2304, grid=grid(768, 2304), stream=stream0)
        buf67 = empty_strided((1, 96, 1, 1, 144), (13824, 1, 13824, 13824, 96), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((1, 96, 1, 1, 144), (13824, 1, 13824, 13824, 96), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((1, 96, 1, 1, 144), (13824, 1, 13824, 13824, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf66, buf67, buf68, buf69, 13824, 128, grid=grid(13824), stream=stream0)
        buf70 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf67, buf68, buf69, buf70, buf71, buf72, 192, 72, grid=grid(192), stream=stream0)
        del buf67
        del buf68
        buf73 = buf60; del buf60  # reuse
        buf74 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf76 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf70, buf71, buf72, primals_266, primals_267, buf73, buf74, buf76, primals_266, primals_267, 96, 2, grid=grid(96), stream=stream0)
        del primals_266
        del primals_267
        buf77 = reinterpret_tensor(buf65, (8, 96, 48, 48), (221184, 1, 4608, 96), 0); del buf65  # reuse
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_27.run(buf66, buf73, buf74, primals_9, primals_10, buf77, 1769472, grid=grid(1769472), stream=stream0)
        del buf74
        del primals_10
        buf78 = reinterpret_tensor(buf69, (8, 96, 1, 1, 18), (1728, 1, 13824, 13824, 96), 0); del buf69  # reuse
        # Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_28.run(buf77, buf78, 13824, 128, grid=grid(13824), stream=stream0)
        buf79 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf80 = reinterpret_tensor(buf79, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf79  # reuse
        # Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_29.run(buf80, buf78, 768, 18, grid=grid(768), stream=stream0)
        del buf78
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 4, 1, 1), (4, 1, 1, 1))
        buf82 = reinterpret_tensor(buf81, (8, 4, 1, 1), (4, 1, 4, 4), 0); del buf81  # reuse
        buf83 = reinterpret_tensor(buf25, (8, 4, 1, 1), (4, 1, 4, 4), 0); del buf25  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_30.run(buf82, primals_127, buf83, 32, grid=grid(32), stream=stream0)
        del primals_127
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 96, 1, 1), (96, 1, 1, 1))
        buf85 = reinterpret_tensor(buf84, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf84  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf85, primals_129, 768, grid=grid(768), stream=stream0)
        del primals_129
        buf86 = empty_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_32.run(buf77, buf85, buf86, 1769472, grid=grid(1769472), stream=stream0)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 24, 48, 48), (55296, 2304, 48, 1))
        buf88 = empty_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf87, buf88, 192, 2304, grid=grid(192, 2304), stream=stream0)
        buf89 = empty_strided((1, 24, 1, 1, 144), (3456, 1, 3456, 3456, 24), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 24, 1, 1, 144), (3456, 1, 3456, 3456, 24), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 24, 1, 1, 144), (3456, 1, 3456, 3456, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf88, buf89, buf90, buf91, 3456, 128, grid=grid(3456), stream=stream0)
        buf92 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf89, buf90, buf91, buf92, buf93, buf94, 48, 72, grid=grid(48), stream=stream0)
        buf95 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf96 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf98 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf92, buf93, buf94, primals_269, primals_270, buf95, buf96, buf98, primals_269, primals_270, 24, 2, grid=grid(24), stream=stream0)
        del primals_269
        del primals_270
        buf99 = reinterpret_tensor(buf87, (8, 24, 48, 48), (55296, 1, 1152, 24), 0); del buf87  # reuse
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_37.run(buf88, buf95, buf96, primals_11, primals_12, buf99, 442368, grid=grid(442368), stream=stream0)
        del primals_12
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 144, 48, 48), (331776, 2304, 48, 1))
        buf101 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf100, buf101, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        buf102 = empty_strided((1, 144, 1, 1, 144), (20736, 1, 20736, 20736, 144), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((1, 144, 1, 1, 144), (20736, 1, 20736, 20736, 144), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((1, 144, 1, 1, 144), (20736, 1, 20736, 20736, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf101, buf102, buf103, buf104, 20736, 128, grid=grid(20736), stream=stream0)
        buf105 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf106 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf107 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf102, buf103, buf104, buf105, buf106, buf107, 288, 72, grid=grid(288), stream=stream0)
        buf108 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf111 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf105, buf106, buf107, primals_272, primals_273, buf108, buf109, buf111, primals_272, primals_273, 144, 2, grid=grid(144), stream=stream0)
        del primals_272
        del primals_273
        buf113 = reinterpret_tensor(buf100, (8, 144, 48, 48), (331776, 1, 6912, 144), 0); del buf100  # reuse
        buf816 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_42.run(buf101, buf108, buf109, primals_13, primals_14, buf113, buf816, 2654208, grid=grid(2654208), stream=stream0)
        del primals_14
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf114, (8, 144, 48, 48), (331776, 2304, 48, 1))
        buf115 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf114, buf115, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        buf116 = buf104; del buf104  # reuse
        buf117 = buf103; del buf103  # reuse
        buf118 = buf102; del buf102  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf115, buf116, buf117, buf118, 20736, 128, grid=grid(20736), stream=stream0)
        buf119 = buf107; del buf107  # reuse
        buf120 = buf106; del buf106  # reuse
        buf121 = buf105; del buf105  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf116, buf117, buf118, buf119, buf120, buf121, 288, 72, grid=grid(288), stream=stream0)
        buf122 = buf109; del buf109  # reuse
        buf123 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf125 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf119, buf120, buf121, primals_275, primals_276, buf122, buf123, buf125, primals_275, primals_276, 144, 2, grid=grid(144), stream=stream0)
        del primals_275
        del primals_276
        buf126 = reinterpret_tensor(buf114, (8, 144, 48, 48), (331776, 1, 6912, 144), 0); del buf114  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_43.run(buf115, buf122, buf123, primals_15, primals_16, buf126, 2654208, grid=grid(2654208), stream=stream0)
        del primals_16
        buf127 = reinterpret_tensor(buf118, (8, 144, 1, 1, 18), (2592, 1, 20736, 20736, 144), 0); del buf118  # reuse
        # Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_44.run(buf126, buf127, 20736, 128, grid=grid(20736), stream=stream0)
        buf128 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf129 = reinterpret_tensor(buf128, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf128  # reuse
        # Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_45.run(buf129, buf127, 1152, 18, grid=grid(1152), stream=stream0)
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 6, 1, 1), (6, 1, 1, 1))
        buf131 = reinterpret_tensor(buf130, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf130  # reuse
        buf132 = reinterpret_tensor(buf94, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf94  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_46.run(buf131, primals_134, buf132, 48, grid=grid(48), stream=stream0)
        del primals_134
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 144, 1, 1), (144, 1, 1, 1))
        buf134 = reinterpret_tensor(buf133, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf133  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf134, primals_136, 1152, grid=grid(1152), stream=stream0)
        del primals_136
        buf135 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_48.run(buf126, buf134, buf135, 2654208, grid=grid(2654208), stream=stream0)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 24, 48, 48), (55296, 2304, 48, 1))
        buf137 = empty_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf136, buf137, 192, 2304, grid=grid(192, 2304), stream=stream0)
        buf138 = buf91; del buf91  # reuse
        buf139 = buf90; del buf90  # reuse
        buf140 = buf89; del buf89  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf137, buf138, buf139, buf140, 3456, 128, grid=grid(3456), stream=stream0)
        buf141 = buf93; del buf93  # reuse
        buf142 = buf92; del buf92  # reuse
        buf143 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf138, buf139, buf140, buf141, buf142, buf143, 48, 72, grid=grid(48), stream=stream0)
        buf144 = buf96; del buf96  # reuse
        buf145 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf147 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf141, buf142, buf143, primals_278, primals_279, buf144, buf145, buf147, primals_278, primals_279, 24, 2, grid=grid(24), stream=stream0)
        del buf141
        del buf142
        del primals_278
        del primals_279
        buf148 = reinterpret_tensor(buf136, (8, 24, 48, 48), (55296, 1, 1152, 24), 0); del buf136  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_49.run(buf137, buf144, buf145, primals_17, primals_18, buf99, buf148, 442368, grid=grid(442368), stream=stream0)
        del buf145
        del primals_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 144, 48, 48), (331776, 2304, 48, 1))
        buf150 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf149, buf150, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        buf151 = reinterpret_tensor(buf127, (1, 144, 1, 1, 144), (20736, 1, 20736, 20736, 144), 0); del buf127  # reuse
        buf152 = buf117; del buf117  # reuse
        buf153 = buf116; del buf116  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf150, buf151, buf152, buf153, 20736, 128, grid=grid(20736), stream=stream0)
        buf154 = buf121; del buf121  # reuse
        buf155 = buf120; del buf120  # reuse
        buf156 = buf119; del buf119  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf151, buf152, buf153, buf154, buf155, buf156, 288, 72, grid=grid(288), stream=stream0)
        del buf151
        del buf152
        del buf153
        buf157 = buf123; del buf123  # reuse
        buf158 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf160 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf154, buf155, buf156, primals_281, primals_282, buf157, buf158, buf160, primals_281, primals_282, 144, 2, grid=grid(144), stream=stream0)
        del buf154
        del buf155
        del buf156
        del primals_281
        del primals_282
        buf162 = reinterpret_tensor(buf149, (8, 144, 48, 48), (331776, 1, 6912, 144), 0); del buf149  # reuse
        buf815 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_42.run(buf150, buf157, buf158, primals_19, primals_20, buf162, buf815, 2654208, grid=grid(2654208), stream=stream0)
        del primals_20
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_139, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf163, (8, 144, 24, 24), (82944, 576, 24, 1))
        buf164 = empty_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf163, buf164, 1152, 576, grid=grid(1152, 576), stream=stream0)
        buf165 = empty_strided((1, 144, 1, 1, 36), (5184, 1, 5184, 5184, 144), device='cuda', dtype=torch.float32)
        buf166 = empty_strided((1, 144, 1, 1, 36), (5184, 1, 5184, 5184, 144), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 144, 1, 1, 36), (5184, 1, 5184, 5184, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf164, buf165, buf166, buf167, 5184, 128, grid=grid(5184), stream=stream0)
        buf168 = buf158; del buf158  # reuse
        buf169 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf171 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf165, buf166, buf167, primals_284, primals_285, buf168, buf169, buf171, primals_284, primals_285, 144, 36, grid=grid(144), stream=stream0)
        del buf165
        del buf166
        del buf167
        del primals_284
        del primals_285
        buf172 = reinterpret_tensor(buf163, (8, 144, 24, 24), (82944, 1, 3456, 144), 0); del buf163  # reuse
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_53.run(buf164, buf168, buf169, primals_21, primals_22, buf172, 663552, grid=grid(663552), stream=stream0)
        del buf169
        del primals_22
        buf173 = empty_strided((8, 144, 1, 1, 5), (720, 1, 5760, 5760, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_54.run(buf172, buf173, 5760, 116, grid=grid(5760), stream=stream0)
        buf174 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf175 = reinterpret_tensor(buf174, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf174  # reuse
        # Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_55.run(buf175, buf173, 1152, 5, grid=grid(1152), stream=stream0)
        del buf173
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 6, 1, 1), (6, 1, 1, 1))
        buf177 = reinterpret_tensor(buf176, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf176  # reuse
        buf178 = reinterpret_tensor(buf143, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf143  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_46.run(buf177, primals_141, buf178, 48, grid=grid(48), stream=stream0)
        del primals_141
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 144, 1, 1), (144, 1, 1, 1))
        buf180 = reinterpret_tensor(buf179, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf179  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf180, primals_143, 1152, grid=grid(1152), stream=stream0)
        del primals_143
        buf181 = empty_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_56.run(buf172, buf180, buf181, 663552, grid=grid(663552), stream=stream0)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 40, 24, 24), (23040, 576, 24, 1))
        buf183 = empty_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf182, buf183, 320, 576, grid=grid(320, 576), stream=stream0)
        buf184 = empty_strided((1, 40, 1, 1, 36), (1440, 1, 1440, 1440, 40), device='cuda', dtype=torch.float32)
        buf185 = empty_strided((1, 40, 1, 1, 36), (1440, 1, 1440, 1440, 40), device='cuda', dtype=torch.float32)
        buf186 = empty_strided((1, 40, 1, 1, 36), (1440, 1, 1440, 1440, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf183, buf184, buf185, buf186, 1440, 128, grid=grid(1440), stream=stream0)
        buf187 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf190 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf184, buf185, buf186, primals_287, primals_288, buf187, buf188, buf190, primals_287, primals_288, 40, 36, grid=grid(40), stream=stream0)
        del primals_287
        del primals_288
        buf191 = reinterpret_tensor(buf182, (8, 40, 24, 24), (23040, 1, 960, 40), 0); del buf182  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_60.run(buf183, buf187, buf188, primals_23, primals_24, buf191, 184320, grid=grid(184320), stream=stream0)
        del primals_24
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 240, 24, 24), (138240, 576, 24, 1))
        buf193 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf192, buf193, 1920, 576, grid=grid(1920, 576), stream=stream0)
        buf194 = empty_strided((1, 240, 1, 1, 36), (8640, 1, 8640, 8640, 240), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((1, 240, 1, 1, 36), (8640, 1, 8640, 8640, 240), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((1, 240, 1, 1, 36), (8640, 1, 8640, 8640, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf193, buf194, buf195, buf196, 8640, 128, grid=grid(8640), stream=stream0)
        buf197 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf200 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf194, buf195, buf196, primals_290, primals_291, buf197, buf198, buf200, primals_290, primals_291, 240, 36, grid=grid(240), stream=stream0)
        del primals_290
        del primals_291
        buf202 = reinterpret_tensor(buf192, (8, 240, 24, 24), (138240, 1, 5760, 240), 0); del buf192  # reuse
        buf814 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64.run(buf193, buf197, buf198, primals_25, primals_26, buf202, buf814, 1105920, grid=grid(1105920), stream=stream0)
        del primals_26
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_146, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf203, (8, 240, 24, 24), (138240, 576, 24, 1))
        buf204 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf203, buf204, 1920, 576, grid=grid(1920, 576), stream=stream0)
        buf205 = buf196; del buf196  # reuse
        buf206 = buf195; del buf195  # reuse
        buf207 = buf194; del buf194  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf204, buf205, buf206, buf207, 8640, 128, grid=grid(8640), stream=stream0)
        buf208 = buf198; del buf198  # reuse
        buf209 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf211 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf205, buf206, buf207, primals_293, primals_294, buf208, buf209, buf211, primals_293, primals_294, 240, 36, grid=grid(240), stream=stream0)
        del primals_293
        del primals_294
        buf212 = reinterpret_tensor(buf203, (8, 240, 24, 24), (138240, 1, 5760, 240), 0); del buf203  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_65.run(buf204, buf208, buf209, primals_27, primals_28, buf212, 1105920, grid=grid(1105920), stream=stream0)
        del primals_28
        buf213 = empty_strided((8, 240, 1, 1, 5), (1200, 1, 9600, 9600, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_66.run(buf212, buf213, 9600, 116, grid=grid(9600), stream=stream0)
        buf214 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf215 = reinterpret_tensor(buf214, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf214  # reuse
        # Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_67.run(buf215, buf213, 1920, 5, grid=grid(1920), stream=stream0)
        del buf213
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 10, 1, 1), (10, 1, 1, 1))
        buf217 = reinterpret_tensor(buf216, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf216  # reuse
        buf218 = reinterpret_tensor(buf45, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf45  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_68.run(buf217, primals_148, buf218, 80, grid=grid(80), stream=stream0)
        del primals_148
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 240, 1, 1), (240, 1, 1, 1))
        buf220 = reinterpret_tensor(buf219, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf219  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf220, primals_150, 1920, grid=grid(1920), stream=stream0)
        del primals_150
        buf221 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_70.run(buf212, buf220, buf221, 1105920, grid=grid(1105920), stream=stream0)
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 40, 24, 24), (23040, 576, 24, 1))
        buf223 = empty_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf222, buf223, 320, 576, grid=grid(320, 576), stream=stream0)
        buf224 = buf186; del buf186  # reuse
        buf225 = buf185; del buf185  # reuse
        buf226 = buf184; del buf184  # reuse
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf223, buf224, buf225, buf226, 1440, 128, grid=grid(1440), stream=stream0)
        buf227 = buf188; del buf188  # reuse
        buf228 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf230 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf224, buf225, buf226, primals_296, primals_297, buf227, buf228, buf230, primals_296, primals_297, 40, 36, grid=grid(40), stream=stream0)
        del buf224
        del buf225
        del buf226
        del primals_296
        del primals_297
        buf231 = reinterpret_tensor(buf222, (8, 40, 24, 24), (23040, 1, 960, 40), 0); del buf222  # reuse
        # Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_71.run(buf223, buf227, buf228, primals_29, primals_30, buf191, buf231, 184320, grid=grid(184320), stream=stream0)
        del buf228
        del primals_30
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 240, 24, 24), (138240, 576, 24, 1))
        buf233 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf232, buf233, 1920, 576, grid=grid(1920, 576), stream=stream0)
        buf234 = buf207; del buf207  # reuse
        buf235 = buf206; del buf206  # reuse
        buf236 = buf205; del buf205  # reuse
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf233, buf234, buf235, buf236, 8640, 128, grid=grid(8640), stream=stream0)
        buf237 = buf209; del buf209  # reuse
        buf238 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf240 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf234, buf235, buf236, primals_299, primals_300, buf237, buf238, buf240, primals_299, primals_300, 240, 36, grid=grid(240), stream=stream0)
        del buf234
        del buf235
        del buf236
        del primals_299
        del primals_300
        buf242 = reinterpret_tensor(buf232, (8, 240, 24, 24), (138240, 1, 5760, 240), 0); del buf232  # reuse
        buf813 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_64.run(buf233, buf237, buf238, primals_31, primals_32, buf242, buf813, 1105920, grid=grid(1105920), stream=stream0)
        del primals_32
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_153, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf243, (8, 240, 12, 12), (34560, 144, 12, 1))
        buf244 = empty_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf243, buf244, 1920, 144, grid=grid(1920, 144), stream=stream0)
        buf245 = empty_strided((1, 240, 1, 1, 9), (2160, 1, 2160, 2160, 240), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((1, 240, 1, 1, 9), (2160, 1, 2160, 2160, 240), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((1, 240, 1, 1, 9), (2160, 1, 2160, 2160, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf244, buf245, buf246, buf247, 2160, 128, grid=grid(2160), stream=stream0)
        buf248 = buf238; del buf238  # reuse
        buf249 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf251 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf245, buf246, buf247, primals_302, primals_303, buf248, buf249, buf251, primals_302, primals_303, 240, 9, grid=grid(240), stream=stream0)
        del buf245
        del buf246
        del buf247
        del primals_302
        del primals_303
        buf252 = reinterpret_tensor(buf243, (8, 240, 12, 12), (34560, 1, 2880, 240), 0); del buf243  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_75.run(buf244, buf248, buf249, primals_33, primals_34, buf252, 276480, grid=grid(276480), stream=stream0)
        del buf249
        del primals_34
        buf253 = empty_strided((8, 240, 1, 1, 2), (480, 1, 3840, 3840, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_76.run(buf252, buf253, 3840, 72, grid=grid(3840), stream=stream0)
        buf254 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf255 = reinterpret_tensor(buf254, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf254  # reuse
        # Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_77.run(buf255, buf253, 1920, 2, grid=grid(1920), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 10, 1, 1), (10, 1, 1, 1))
        buf257 = reinterpret_tensor(buf256, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf256  # reuse
        buf258 = reinterpret_tensor(buf44, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf44  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_68.run(buf257, primals_155, buf258, 80, grid=grid(80), stream=stream0)
        del primals_155
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 240, 1, 1), (240, 1, 1, 1))
        buf260 = reinterpret_tensor(buf259, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf259  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf260, primals_157, 1920, grid=grid(1920), stream=stream0)
        del primals_157
        buf261 = empty_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_78.run(buf252, buf260, buf261, 276480, grid=grid(276480), stream=stream0)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 80, 12, 12), (11520, 144, 12, 1))
        buf263 = empty_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf262, buf263, 640, 144, grid=grid(640, 144), stream=stream0)
        buf264 = empty_strided((1, 80, 1, 1, 9), (720, 1, 720, 720, 80), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((1, 80, 1, 1, 9), (720, 1, 720, 720, 80), device='cuda', dtype=torch.float32)
        buf266 = empty_strided((1, 80, 1, 1, 9), (720, 1, 720, 720, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf263, buf264, buf265, buf266, 720, 128, grid=grid(720), stream=stream0)
        buf267 = reinterpret_tensor(buf43, (1, 80, 1, 1), (80, 1, 80, 80), 0); del buf43  # reuse
        buf268 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf270 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf264, buf265, buf266, primals_305, primals_306, buf267, buf268, buf270, primals_305, primals_306, 80, 9, grid=grid(80), stream=stream0)
        del primals_305
        del primals_306
        buf271 = reinterpret_tensor(buf262, (8, 80, 12, 12), (11520, 1, 960, 80), 0); del buf262  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_82.run(buf263, buf267, buf268, primals_35, primals_36, buf271, 92160, grid=grid(92160), stream=stream0)
        del primals_36
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf273 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf272, buf273, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf274 = empty_strided((1, 480, 1, 1, 9), (4320, 1, 4320, 4320, 480), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((1, 480, 1, 1, 9), (4320, 1, 4320, 4320, 480), device='cuda', dtype=torch.float32)
        buf276 = empty_strided((1, 480, 1, 1, 9), (4320, 1, 4320, 4320, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf273, buf274, buf275, buf276, 4320, 128, grid=grid(4320), stream=stream0)
        buf277 = reinterpret_tensor(buf58, (1, 480, 1, 1), (480, 1, 480, 480), 0); del buf58  # reuse
        buf278 = reinterpret_tensor(buf57, (1, 480, 1, 1), (480, 1, 480, 480), 0); del buf57  # reuse
        buf280 = reinterpret_tensor(buf56, (480, ), (1, ), 0); del buf56  # reuse
        # Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf274, buf275, buf276, primals_308, primals_309, buf277, buf278, buf280, primals_308, primals_309, 480, 9, grid=grid(480), stream=stream0)
        del primals_308
        del primals_309
        buf282 = reinterpret_tensor(buf272, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf272  # reuse
        buf812 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_86.run(buf273, buf277, buf278, primals_37, primals_38, buf282, buf812, 552960, grid=grid(552960), stream=stream0)
        del primals_38
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf283, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf284 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf283, buf284, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf285 = buf276; del buf276  # reuse
        buf286 = buf275; del buf275  # reuse
        buf287 = buf274; del buf274  # reuse
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf284, buf285, buf286, buf287, 4320, 128, grid=grid(4320), stream=stream0)
        buf288 = buf278; del buf278  # reuse
        buf289 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf291 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf285, buf286, buf287, primals_311, primals_312, buf288, buf289, buf291, primals_311, primals_312, 480, 9, grid=grid(480), stream=stream0)
        del primals_311
        del primals_312
        buf292 = reinterpret_tensor(buf283, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf283  # reuse
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_87.run(buf284, buf288, buf289, primals_39, primals_40, buf292, 552960, grid=grid(552960), stream=stream0)
        del primals_40
        buf293 = empty_strided((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_88.run(buf292, buf293, 7680, 72, grid=grid(7680), stream=stream0)
        buf294 = reinterpret_tensor(buf253, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf253  # reuse
        buf295 = reinterpret_tensor(buf294, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf294  # reuse
        # Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_89.run(buf295, buf293, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 20, 1, 1), (20, 1, 1, 1))
        buf297 = reinterpret_tensor(buf296, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf296  # reuse
        buf298 = reinterpret_tensor(buf23, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf23  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_90.run(buf297, primals_162, buf298, 160, grid=grid(160), stream=stream0)
        del primals_162
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 480, 1, 1), (480, 1, 1, 1))
        buf300 = reinterpret_tensor(buf299, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf299  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf300, primals_164, 3840, grid=grid(3840), stream=stream0)
        del primals_164
        buf301 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_92.run(buf292, buf300, buf301, 552960, grid=grid(552960), stream=stream0)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 80, 12, 12), (11520, 144, 12, 1))
        buf303 = empty_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf302, buf303, 640, 144, grid=grid(640, 144), stream=stream0)
        buf304 = buf266; del buf266  # reuse
        buf305 = buf265; del buf265  # reuse
        buf306 = buf264; del buf264  # reuse
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf303, buf304, buf305, buf306, 720, 128, grid=grid(720), stream=stream0)
        buf307 = buf268; del buf268  # reuse
        buf308 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf310 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf304, buf305, buf306, primals_314, primals_315, buf307, buf308, buf310, primals_314, primals_315, 80, 9, grid=grid(80), stream=stream0)
        del primals_314
        del primals_315
        buf311 = reinterpret_tensor(buf302, (8, 80, 12, 12), (11520, 1, 960, 80), 0); del buf302  # reuse
        # Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_93.run(buf303, buf307, buf308, primals_41, primals_42, buf271, buf311, 92160, grid=grid(92160), stream=stream0)
        del primals_42
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf313 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf312, buf313, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf314 = buf287; del buf287  # reuse
        buf315 = buf286; del buf286  # reuse
        buf316 = buf285; del buf285  # reuse
        # Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf313, buf314, buf315, buf316, 4320, 128, grid=grid(4320), stream=stream0)
        buf317 = buf289; del buf289  # reuse
        buf318 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf320 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf314, buf315, buf316, primals_317, primals_318, buf317, buf318, buf320, primals_317, primals_318, 480, 9, grid=grid(480), stream=stream0)
        del primals_317
        del primals_318
        buf322 = reinterpret_tensor(buf312, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf312  # reuse
        buf811 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_86.run(buf313, buf317, buf318, primals_43, primals_44, buf322, buf811, 552960, grid=grid(552960), stream=stream0)
        del primals_44
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf323, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf324 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf323, buf324, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf325 = buf316; del buf316  # reuse
        buf326 = buf315; del buf315  # reuse
        buf327 = buf314; del buf314  # reuse
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf324, buf325, buf326, buf327, 4320, 128, grid=grid(4320), stream=stream0)
        buf328 = buf318; del buf318  # reuse
        buf329 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf331 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf325, buf326, buf327, primals_320, primals_321, buf328, buf329, buf331, primals_320, primals_321, 480, 9, grid=grid(480), stream=stream0)
        del primals_320
        del primals_321
        buf332 = reinterpret_tensor(buf323, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf323  # reuse
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_87.run(buf324, buf328, buf329, primals_45, primals_46, buf332, 552960, grid=grid(552960), stream=stream0)
        del primals_46
        buf333 = buf293; del buf293  # reuse
        # Source Nodes: [x_124, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_88.run(buf332, buf333, 7680, 72, grid=grid(7680), stream=stream0)
        buf334 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf335 = reinterpret_tensor(buf334, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf334  # reuse
        # Source Nodes: [x_124, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_89.run(buf335, buf333, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 20, 1, 1), (20, 1, 1, 1))
        buf337 = reinterpret_tensor(buf336, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf336  # reuse
        buf338 = reinterpret_tensor(buf22, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf22  # reuse
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_90.run(buf337, primals_169, buf338, 160, grid=grid(160), stream=stream0)
        del primals_169
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (8, 480, 1, 1), (480, 1, 1, 1))
        buf340 = reinterpret_tensor(buf339, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf339  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf340, primals_171, 3840, grid=grid(3840), stream=stream0)
        del primals_171
        buf341 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_92.run(buf332, buf340, buf341, 552960, grid=grid(552960), stream=stream0)
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 80, 12, 12), (11520, 144, 12, 1))
        buf343 = empty_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf342, buf343, 640, 144, grid=grid(640, 144), stream=stream0)
        buf344 = buf306; del buf306  # reuse
        buf345 = buf305; del buf305  # reuse
        buf346 = buf304; del buf304  # reuse
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf343, buf344, buf345, buf346, 720, 128, grid=grid(720), stream=stream0)
        buf347 = buf308; del buf308  # reuse
        buf348 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf350 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf344, buf345, buf346, primals_323, primals_324, buf347, buf348, buf350, primals_323, primals_324, 80, 9, grid=grid(80), stream=stream0)
        del primals_323
        del primals_324
        buf351 = reinterpret_tensor(buf342, (8, 80, 12, 12), (11520, 1, 960, 80), 0); del buf342  # reuse
        # Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_93.run(buf343, buf347, buf348, primals_47, primals_48, buf311, buf351, 92160, grid=grid(92160), stream=stream0)
        del primals_48
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf353 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf352, buf353, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf354 = buf327; del buf327  # reuse
        buf355 = buf326; del buf326  # reuse
        buf356 = buf325; del buf325  # reuse
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf353, buf354, buf355, buf356, 4320, 128, grid=grid(4320), stream=stream0)
        buf357 = buf329; del buf329  # reuse
        buf358 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf360 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf354, buf355, buf356, primals_326, primals_327, buf357, buf358, buf360, primals_326, primals_327, 480, 9, grid=grid(480), stream=stream0)
        del primals_326
        del primals_327
        buf362 = reinterpret_tensor(buf352, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf352  # reuse
        buf810 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_86.run(buf353, buf357, buf358, primals_49, primals_50, buf362, buf810, 552960, grid=grid(552960), stream=stream0)
        del primals_50
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf363, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf364 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf363, buf364, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf365 = buf356; del buf356  # reuse
        buf366 = buf355; del buf355  # reuse
        buf367 = buf354; del buf354  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf364, buf365, buf366, buf367, 4320, 128, grid=grid(4320), stream=stream0)
        buf368 = buf358; del buf358  # reuse
        buf369 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf371 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf365, buf366, buf367, primals_329, primals_330, buf368, buf369, buf371, primals_329, primals_330, 480, 9, grid=grid(480), stream=stream0)
        del primals_329
        del primals_330
        buf372 = reinterpret_tensor(buf363, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf363  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_87.run(buf364, buf368, buf369, primals_51, primals_52, buf372, 552960, grid=grid(552960), stream=stream0)
        del primals_52
        buf373 = buf333; del buf333  # reuse
        # Source Nodes: [x_141, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_88.run(buf372, buf373, 7680, 72, grid=grid(7680), stream=stream0)
        buf374 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf375 = reinterpret_tensor(buf374, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf374  # reuse
        # Source Nodes: [x_141, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_89.run(buf375, buf373, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 20, 1, 1), (20, 1, 1, 1))
        buf377 = reinterpret_tensor(buf376, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf376  # reuse
        buf378 = reinterpret_tensor(buf21, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf21  # reuse
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_90.run(buf377, primals_176, buf378, 160, grid=grid(160), stream=stream0)
        del primals_176
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 480, 1, 1), (480, 1, 1, 1))
        buf380 = reinterpret_tensor(buf379, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf379  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf380, primals_178, 3840, grid=grid(3840), stream=stream0)
        del primals_178
        buf381 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_141, x_142], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_92.run(buf372, buf380, buf381, 552960, grid=grid(552960), stream=stream0)
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 80, 12, 12), (11520, 144, 12, 1))
        buf383 = empty_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf382, buf383, 640, 144, grid=grid(640, 144), stream=stream0)
        buf384 = buf346; del buf346  # reuse
        buf385 = buf345; del buf345  # reuse
        buf386 = buf344; del buf344  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_80.run(buf383, buf384, buf385, buf386, 720, 128, grid=grid(720), stream=stream0)
        buf387 = buf348; del buf348  # reuse
        buf388 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf390 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_81.run(buf384, buf385, buf386, primals_332, primals_333, buf387, buf388, buf390, primals_332, primals_333, 80, 9, grid=grid(80), stream=stream0)
        del buf384
        del buf385
        del buf386
        del primals_332
        del primals_333
        buf391 = reinterpret_tensor(buf382, (8, 80, 12, 12), (11520, 1, 960, 80), 0); del buf382  # reuse
        # Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_93.run(buf383, buf387, buf388, primals_53, primals_54, buf351, buf391, 92160, grid=grid(92160), stream=stream0)
        del buf388
        del primals_54
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf393 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf392, buf393, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf394 = buf367; del buf367  # reuse
        buf395 = buf366; del buf366  # reuse
        buf396 = buf365; del buf365  # reuse
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf393, buf394, buf395, buf396, 4320, 128, grid=grid(4320), stream=stream0)
        buf397 = buf369; del buf369  # reuse
        buf398 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf400 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf394, buf395, buf396, primals_335, primals_336, buf397, buf398, buf400, primals_335, primals_336, 480, 9, grid=grid(480), stream=stream0)
        del primals_335
        del primals_336
        buf402 = reinterpret_tensor(buf392, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf392  # reuse
        buf809 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150, x_153], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_86.run(buf393, buf397, buf398, primals_55, primals_56, buf402, buf809, 552960, grid=grid(552960), stream=stream0)
        del primals_56
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_181, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf403, (8, 480, 12, 12), (69120, 144, 12, 1))
        buf404 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf403, buf404, 3840, 144, grid=grid(3840, 144), stream=stream0)
        buf405 = buf396; del buf396  # reuse
        buf406 = buf395; del buf395  # reuse
        buf407 = buf394; del buf394  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf404, buf405, buf406, buf407, 4320, 128, grid=grid(4320), stream=stream0)
        buf408 = buf398; del buf398  # reuse
        buf409 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf411 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf405, buf406, buf407, primals_338, primals_339, buf408, buf409, buf411, primals_338, primals_339, 480, 9, grid=grid(480), stream=stream0)
        del buf405
        del buf406
        del buf407
        del primals_338
        del primals_339
        buf412 = reinterpret_tensor(buf403, (8, 480, 12, 12), (69120, 1, 5760, 480), 0); del buf403  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_87.run(buf404, buf408, buf409, primals_57, primals_58, buf412, 552960, grid=grid(552960), stream=stream0)
        del buf409
        del primals_58
        buf413 = buf373; del buf373  # reuse
        # Source Nodes: [x_158, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_88.run(buf412, buf413, 7680, 72, grid=grid(7680), stream=stream0)
        buf414 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf415 = reinterpret_tensor(buf414, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf414  # reuse
        # Source Nodes: [x_158, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_89.run(buf415, buf413, 3840, 2, grid=grid(3840), stream=stream0)
        del buf413
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 20, 1, 1), (20, 1, 1, 1))
        buf417 = reinterpret_tensor(buf416, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf416  # reuse
        buf418 = empty_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_90.run(buf417, primals_183, buf418, 160, grid=grid(160), stream=stream0)
        del primals_183
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 480, 1, 1), (480, 1, 1, 1))
        buf420 = reinterpret_tensor(buf419, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf419  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf420, primals_185, 3840, grid=grid(3840), stream=stream0)
        del primals_185
        buf421 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_158, x_159], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_92.run(buf412, buf420, buf421, 552960, grid=grid(552960), stream=stream0)
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 112, 12, 12), (16128, 144, 12, 1))
        buf423 = empty_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf422, buf423, 896, 144, grid=grid(896, 144), stream=stream0)
        buf424 = empty_strided((1, 112, 1, 1, 9), (1008, 1, 1008, 1008, 112), device='cuda', dtype=torch.float32)
        buf425 = empty_strided((1, 112, 1, 1, 9), (1008, 1, 1008, 1008, 112), device='cuda', dtype=torch.float32)
        buf426 = empty_strided((1, 112, 1, 1, 9), (1008, 1, 1008, 1008, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf423, buf424, buf425, buf426, 1008, 128, grid=grid(1008), stream=stream0)
        buf427 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf428 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf430 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf424, buf425, buf426, primals_341, primals_342, buf427, buf428, buf430, primals_341, primals_342, 112, 9, grid=grid(112), stream=stream0)
        del primals_341
        del primals_342
        buf431 = reinterpret_tensor(buf422, (8, 112, 12, 12), (16128, 1, 1344, 112), 0); del buf422  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_97.run(buf423, buf427, buf428, primals_59, primals_60, buf431, 129024, grid=grid(129024), stream=stream0)
        del primals_60
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf433 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf432, buf433, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf434 = empty_strided((1, 672, 1, 1, 9), (6048, 1, 6048, 6048, 672), device='cuda', dtype=torch.float32)
        buf435 = empty_strided((1, 672, 1, 1, 9), (6048, 1, 6048, 6048, 672), device='cuda', dtype=torch.float32)
        buf436 = empty_strided((1, 672, 1, 1, 9), (6048, 1, 6048, 6048, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf433, buf434, buf435, buf436, 6048, 128, grid=grid(6048), stream=stream0)
        buf437 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf438 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf440 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf434, buf435, buf436, primals_344, primals_345, buf437, buf438, buf440, primals_344, primals_345, 672, 9, grid=grid(672), stream=stream0)
        del primals_344
        del primals_345
        buf442 = reinterpret_tensor(buf432, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf432  # reuse
        buf808 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_101.run(buf433, buf437, buf438, primals_61, primals_62, buf442, buf808, 774144, grid=grid(774144), stream=stream0)
        del primals_62
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_188, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf443, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf444 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf443, buf444, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf445 = buf436; del buf436  # reuse
        buf446 = buf435; del buf435  # reuse
        buf447 = buf434; del buf434  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf444, buf445, buf446, buf447, 6048, 128, grid=grid(6048), stream=stream0)
        buf448 = buf438; del buf438  # reuse
        buf449 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf451 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf445, buf446, buf447, primals_347, primals_348, buf448, buf449, buf451, primals_347, primals_348, 672, 9, grid=grid(672), stream=stream0)
        del primals_347
        del primals_348
        buf452 = reinterpret_tensor(buf443, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf443  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_102.run(buf444, buf448, buf449, primals_63, primals_64, buf452, 774144, grid=grid(774144), stream=stream0)
        del primals_64
        buf453 = empty_strided((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_103.run(buf452, buf453, 10752, 72, grid=grid(10752), stream=stream0)
        buf454 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf455 = reinterpret_tensor(buf454, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf454  # reuse
        # Source Nodes: [x_174, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_104.run(buf455, buf453, 5376, 2, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 28, 1, 1), (28, 1, 1, 1))
        buf457 = reinterpret_tensor(buf456, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf456  # reuse
        buf458 = empty_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_105.run(buf457, primals_190, buf458, 224, grid=grid(224), stream=stream0)
        del primals_190
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (8, 672, 1, 1), (672, 1, 1, 1))
        buf460 = reinterpret_tensor(buf459, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf459  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf460, primals_192, 5376, grid=grid(5376), stream=stream0)
        del primals_192
        buf461 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174, x_175], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_107.run(buf452, buf460, buf461, 774144, grid=grid(774144), stream=stream0)
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 112, 12, 12), (16128, 144, 12, 1))
        buf463 = empty_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf462, buf463, 896, 144, grid=grid(896, 144), stream=stream0)
        buf464 = buf426; del buf426  # reuse
        buf465 = buf425; del buf425  # reuse
        buf466 = buf424; del buf424  # reuse
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf463, buf464, buf465, buf466, 1008, 128, grid=grid(1008), stream=stream0)
        buf467 = buf428; del buf428  # reuse
        buf468 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf470 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf464, buf465, buf466, primals_350, primals_351, buf467, buf468, buf470, primals_350, primals_351, 112, 9, grid=grid(112), stream=stream0)
        del primals_350
        del primals_351
        buf471 = reinterpret_tensor(buf462, (8, 112, 12, 12), (16128, 1, 1344, 112), 0); del buf462  # reuse
        # Source Nodes: [shortcut_11, x_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_108.run(buf463, buf467, buf468, primals_65, primals_66, buf431, buf471, 129024, grid=grid(129024), stream=stream0)
        del primals_66
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf473 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf472, buf473, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf474 = buf447; del buf447  # reuse
        buf475 = buf446; del buf446  # reuse
        buf476 = buf445; del buf445  # reuse
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf473, buf474, buf475, buf476, 6048, 128, grid=grid(6048), stream=stream0)
        buf477 = buf449; del buf449  # reuse
        buf478 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf480 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf474, buf475, buf476, primals_353, primals_354, buf477, buf478, buf480, primals_353, primals_354, 672, 9, grid=grid(672), stream=stream0)
        del primals_353
        del primals_354
        buf482 = reinterpret_tensor(buf472, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf472  # reuse
        buf807 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_101.run(buf473, buf477, buf478, primals_67, primals_68, buf482, buf807, 774144, grid=grid(774144), stream=stream0)
        del primals_68
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_195, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf483, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf484 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf483, buf484, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf485 = buf476; del buf476  # reuse
        buf486 = buf475; del buf475  # reuse
        buf487 = buf474; del buf474  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf484, buf485, buf486, buf487, 6048, 128, grid=grid(6048), stream=stream0)
        buf488 = buf478; del buf478  # reuse
        buf489 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf491 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf485, buf486, buf487, primals_356, primals_357, buf488, buf489, buf491, primals_356, primals_357, 672, 9, grid=grid(672), stream=stream0)
        del primals_356
        del primals_357
        buf492 = reinterpret_tensor(buf483, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf483  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_102.run(buf484, buf488, buf489, primals_69, primals_70, buf492, 774144, grid=grid(774144), stream=stream0)
        del primals_70
        buf493 = buf453; del buf453  # reuse
        # Source Nodes: [x_191, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_103.run(buf492, buf493, 10752, 72, grid=grid(10752), stream=stream0)
        buf494 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf495 = reinterpret_tensor(buf494, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf494  # reuse
        # Source Nodes: [x_191, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_104.run(buf495, buf493, 5376, 2, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (8, 28, 1, 1), (28, 1, 1, 1))
        buf497 = reinterpret_tensor(buf496, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf496  # reuse
        buf498 = empty_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_105.run(buf497, primals_197, buf498, 224, grid=grid(224), stream=stream0)
        del primals_197
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (8, 672, 1, 1), (672, 1, 1, 1))
        buf500 = reinterpret_tensor(buf499, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf499  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf500, primals_199, 5376, grid=grid(5376), stream=stream0)
        del primals_199
        buf501 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_191, x_192], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_107.run(buf492, buf500, buf501, 774144, grid=grid(774144), stream=stream0)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (8, 112, 12, 12), (16128, 144, 12, 1))
        buf503 = empty_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf502, buf503, 896, 144, grid=grid(896, 144), stream=stream0)
        buf504 = buf466; del buf466  # reuse
        buf505 = buf465; del buf465  # reuse
        buf506 = buf464; del buf464  # reuse
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf503, buf504, buf505, buf506, 1008, 128, grid=grid(1008), stream=stream0)
        buf507 = buf468; del buf468  # reuse
        buf508 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf510 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf504, buf505, buf506, primals_359, primals_360, buf507, buf508, buf510, primals_359, primals_360, 112, 9, grid=grid(112), stream=stream0)
        del primals_359
        del primals_360
        buf511 = reinterpret_tensor(buf502, (8, 112, 12, 12), (16128, 1, 1344, 112), 0); del buf502  # reuse
        # Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_108.run(buf503, buf507, buf508, primals_71, primals_72, buf471, buf511, 129024, grid=grid(129024), stream=stream0)
        del primals_72
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf513 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf512, buf513, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf514 = buf487; del buf487  # reuse
        buf515 = buf486; del buf486  # reuse
        buf516 = buf485; del buf485  # reuse
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf513, buf514, buf515, buf516, 6048, 128, grid=grid(6048), stream=stream0)
        buf517 = buf489; del buf489  # reuse
        buf518 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf520 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf514, buf515, buf516, primals_362, primals_363, buf517, buf518, buf520, primals_362, primals_363, 672, 9, grid=grid(672), stream=stream0)
        del primals_362
        del primals_363
        buf522 = reinterpret_tensor(buf512, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf512  # reuse
        buf806 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_101.run(buf513, buf517, buf518, primals_73, primals_74, buf522, buf806, 774144, grid=grid(774144), stream=stream0)
        del primals_74
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_202, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf523, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf524 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf523, buf524, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf525 = buf516; del buf516  # reuse
        buf526 = buf515; del buf515  # reuse
        buf527 = buf514; del buf514  # reuse
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf524, buf525, buf526, buf527, 6048, 128, grid=grid(6048), stream=stream0)
        buf528 = buf518; del buf518  # reuse
        buf529 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf531 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf525, buf526, buf527, primals_365, primals_366, buf528, buf529, buf531, primals_365, primals_366, 672, 9, grid=grid(672), stream=stream0)
        del primals_365
        del primals_366
        buf532 = reinterpret_tensor(buf523, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf523  # reuse
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_102.run(buf524, buf528, buf529, primals_75, primals_76, buf532, 774144, grid=grid(774144), stream=stream0)
        del primals_76
        buf533 = buf493; del buf493  # reuse
        # Source Nodes: [x_208, x_se_48], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_103.run(buf532, buf533, 10752, 72, grid=grid(10752), stream=stream0)
        buf534 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf535 = reinterpret_tensor(buf534, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf534  # reuse
        # Source Nodes: [x_208, x_se_48], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_104.run(buf535, buf533, 5376, 2, grid=grid(5376), stream=stream0)
        del buf533
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 28, 1, 1), (28, 1, 1, 1))
        buf537 = reinterpret_tensor(buf536, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf536  # reuse
        buf538 = empty_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_105.run(buf537, primals_204, buf538, 224, grid=grid(224), stream=stream0)
        del primals_204
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (8, 672, 1, 1), (672, 1, 1, 1))
        buf540 = reinterpret_tensor(buf539, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf539  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf540, primals_206, 5376, grid=grid(5376), stream=stream0)
        del primals_206
        buf541 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_208, x_209], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_107.run(buf532, buf540, buf541, 774144, grid=grid(774144), stream=stream0)
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 112, 12, 12), (16128, 144, 12, 1))
        buf543 = empty_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf542, buf543, 896, 144, grid=grid(896, 144), stream=stream0)
        buf544 = buf506; del buf506  # reuse
        buf545 = buf505; del buf505  # reuse
        buf546 = buf504; del buf504  # reuse
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_95.run(buf543, buf544, buf545, buf546, 1008, 128, grid=grid(1008), stream=stream0)
        buf547 = buf508; del buf508  # reuse
        buf548 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf550 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_96.run(buf544, buf545, buf546, primals_368, primals_369, buf547, buf548, buf550, primals_368, primals_369, 112, 9, grid=grid(112), stream=stream0)
        del buf544
        del buf545
        del buf546
        del primals_368
        del primals_369
        buf551 = reinterpret_tensor(buf542, (8, 112, 12, 12), (16128, 1, 1344, 112), 0); del buf542  # reuse
        # Source Nodes: [shortcut_13, x_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_108.run(buf543, buf547, buf548, primals_77, primals_78, buf511, buf551, 129024, grid=grid(129024), stream=stream0)
        del buf548
        del primals_78
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (8, 672, 12, 12), (96768, 144, 12, 1))
        buf553 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_98.run(buf552, buf553, 5376, 144, grid=grid(5376, 144), stream=stream0)
        buf554 = buf527; del buf527  # reuse
        buf555 = buf526; del buf526  # reuse
        buf556 = buf525; del buf525  # reuse
        # Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_99.run(buf553, buf554, buf555, buf556, 6048, 128, grid=grid(6048), stream=stream0)
        buf557 = buf529; del buf529  # reuse
        buf558 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf560 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_100.run(buf554, buf555, buf556, primals_371, primals_372, buf557, buf558, buf560, primals_371, primals_372, 672, 9, grid=grid(672), stream=stream0)
        del buf554
        del buf555
        del buf556
        del primals_371
        del primals_372
        buf562 = reinterpret_tensor(buf552, (8, 672, 12, 12), (96768, 1, 8064, 672), 0); del buf552  # reuse
        buf805 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_101.run(buf553, buf557, buf558, primals_79, primals_80, buf562, buf805, 774144, grid=grid(774144), stream=stream0)
        del primals_80
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_209, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf563, (8, 672, 6, 6), (24192, 36, 6, 1))
        buf564 = empty_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_109.run(buf563, buf564, 5376, 36, grid=grid(5376, 36), stream=stream0)
        buf565 = empty_strided((1, 672, 1, 1, 3), (2016, 1, 2016, 2016, 672), device='cuda', dtype=torch.float32)
        buf566 = empty_strided((1, 672, 1, 1, 3), (2016, 1, 2016, 2016, 672), device='cuda', dtype=torch.float32)
        buf567 = empty_strided((1, 672, 1, 1, 3), (2016, 1, 2016, 2016, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_110.run(buf564, buf565, buf566, buf567, 2016, 96, grid=grid(2016), stream=stream0)
        buf568 = buf558; del buf558  # reuse
        buf569 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf571 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_111.run(buf565, buf566, buf567, primals_374, primals_375, buf568, buf569, buf571, primals_374, primals_375, 672, 3, grid=grid(672), stream=stream0)
        del buf565
        del buf566
        del buf567
        del primals_374
        del primals_375
        buf572 = reinterpret_tensor(buf563, (8, 672, 6, 6), (24192, 1, 4032, 672), 0); del buf563  # reuse
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_112.run(buf564, buf568, buf569, primals_81, primals_82, buf572, 193536, grid=grid(193536), stream=stream0)
        del buf569
        del primals_82
        buf573 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf574 = reinterpret_tensor(buf573, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf573  # reuse
        # Source Nodes: [x_225, x_se_52], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_113.run(buf574, buf572, 5376, 36, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (8, 28, 1, 1), (28, 1, 1, 1))
        buf576 = reinterpret_tensor(buf575, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf575  # reuse
        buf577 = empty_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_105.run(buf576, primals_211, buf577, 224, grid=grid(224), stream=stream0)
        del primals_211
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf578 = extern_kernels.convolution(buf577, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (8, 672, 1, 1), (672, 1, 1, 1))
        buf579 = reinterpret_tensor(buf578, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf578  # reuse
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf579, primals_213, 5376, grid=grid(5376), stream=stream0)
        del primals_213
        buf580 = empty_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225, x_226], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_114.run(buf572, buf579, buf580, 193536, grid=grid(193536), stream=stream0)
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 192, 6, 6), (6912, 36, 6, 1))
        buf582 = reinterpret_tensor(buf55, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf55  # reuse
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf581, buf582, 1536, 36, grid=grid(1536, 36), stream=stream0)
        buf583 = empty_strided((1, 192, 1, 1, 3), (576, 1, 576, 576, 192), device='cuda', dtype=torch.float32)
        buf584 = empty_strided((1, 192, 1, 1, 3), (576, 1, 576, 576, 192), device='cuda', dtype=torch.float32)
        buf585 = empty_strided((1, 192, 1, 1, 3), (576, 1, 576, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf582, buf583, buf584, buf585, 576, 96, grid=grid(576), stream=stream0)
        buf586 = reinterpret_tensor(buf72, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf72  # reuse
        buf587 = reinterpret_tensor(buf71, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf71  # reuse
        buf589 = reinterpret_tensor(buf70, (192, ), (1, ), 0); del buf70  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf583, buf584, buf585, primals_377, primals_378, buf586, buf587, buf589, primals_377, primals_378, 192, 3, grid=grid(192), stream=stream0)
        del primals_377
        del primals_378
        buf590 = reinterpret_tensor(buf581, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf581  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_118.run(buf582, buf586, buf587, primals_83, primals_84, buf590, 55296, grid=grid(55296), stream=stream0)
        del primals_84
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf590, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf592 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf591, buf592, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf593 = reinterpret_tensor(buf140, (1, 1152, 1, 1, 3), (3456, 1, 3456, 3456, 1152), 0); del buf140  # reuse
        buf594 = reinterpret_tensor(buf139, (1, 1152, 1, 1, 3), (3456, 1, 3456, 3456, 1152), 0); del buf139  # reuse
        buf595 = reinterpret_tensor(buf138, (1, 1152, 1, 1, 3), (3456, 1, 3456, 3456, 1152), 0); del buf138  # reuse
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf592, buf593, buf594, buf595, 3456, 96, grid=grid(3456), stream=stream0)
        buf596 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf597 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf599 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf593, buf594, buf595, primals_380, primals_381, buf596, buf597, buf599, primals_380, primals_381, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_380
        del primals_381
        buf601 = reinterpret_tensor(buf591, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf591  # reuse
        buf804 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122.run(buf592, buf596, buf597, primals_85, primals_86, buf601, buf804, 331776, grid=grid(331776), stream=stream0)
        del primals_86
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_216, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf602, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf603 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf602, buf603, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf604 = buf595; del buf595  # reuse
        buf605 = buf594; del buf594  # reuse
        buf606 = buf593; del buf593  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf603, buf604, buf605, buf606, 3456, 96, grid=grid(3456), stream=stream0)
        buf607 = buf597; del buf597  # reuse
        buf608 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf610 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf604, buf605, buf606, primals_383, primals_384, buf607, buf608, buf610, primals_383, primals_384, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_383
        del primals_384
        buf611 = reinterpret_tensor(buf602, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf602  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_123.run(buf603, buf607, buf608, primals_87, primals_88, buf611, 331776, grid=grid(331776), stream=stream0)
        del primals_88
        buf612 = reinterpret_tensor(buf42, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf42  # reuse
        buf613 = reinterpret_tensor(buf612, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf612  # reuse
        # Source Nodes: [x_241, x_se_56], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_124.run(buf613, buf611, 9216, 36, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 48, 1, 1), (48, 1, 1, 1))
        buf615 = reinterpret_tensor(buf614, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf614  # reuse
        buf616 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_125.run(buf615, primals_218, buf616, 384, grid=grid(384), stream=stream0)
        del primals_218
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf618 = reinterpret_tensor(buf617, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf617  # reuse
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf618, primals_220, 9216, grid=grid(9216), stream=stream0)
        del primals_220
        buf619 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241, x_242], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_127.run(buf611, buf618, buf619, 331776, grid=grid(331776), stream=stream0)
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (8, 192, 6, 6), (6912, 36, 6, 1))
        buf621 = reinterpret_tensor(buf54, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf54  # reuse
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf620, buf621, 1536, 36, grid=grid(1536, 36), stream=stream0)
        buf622 = buf585; del buf585  # reuse
        buf623 = buf584; del buf584  # reuse
        buf624 = buf583; del buf583  # reuse
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf621, buf622, buf623, buf624, 576, 96, grid=grid(576), stream=stream0)
        buf625 = buf587; del buf587  # reuse
        buf626 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf628 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf622, buf623, buf624, primals_386, primals_387, buf625, buf626, buf628, primals_386, primals_387, 192, 3, grid=grid(192), stream=stream0)
        del primals_386
        del primals_387
        buf629 = reinterpret_tensor(buf620, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf620  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_128.run(buf621, buf625, buf626, primals_89, primals_90, buf590, buf629, 55296, grid=grid(55296), stream=stream0)
        del primals_90
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf631 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf630, buf631, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf632 = buf606; del buf606  # reuse
        buf633 = buf605; del buf605  # reuse
        buf634 = buf604; del buf604  # reuse
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf631, buf632, buf633, buf634, 3456, 96, grid=grid(3456), stream=stream0)
        buf635 = buf608; del buf608  # reuse
        buf636 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf638 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf632, buf633, buf634, primals_389, primals_390, buf635, buf636, buf638, primals_389, primals_390, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_389
        del primals_390
        buf640 = reinterpret_tensor(buf630, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf630  # reuse
        buf803 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122.run(buf631, buf635, buf636, primals_91, primals_92, buf640, buf803, 331776, grid=grid(331776), stream=stream0)
        del primals_92
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_223, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf641, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf642 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf641, buf642, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf643 = buf634; del buf634  # reuse
        buf644 = buf633; del buf633  # reuse
        buf645 = buf632; del buf632  # reuse
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf642, buf643, buf644, buf645, 3456, 96, grid=grid(3456), stream=stream0)
        buf646 = buf636; del buf636  # reuse
        buf647 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf649 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf643, buf644, buf645, primals_392, primals_393, buf646, buf647, buf649, primals_392, primals_393, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_392
        del primals_393
        buf650 = reinterpret_tensor(buf641, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf641  # reuse
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_123.run(buf642, buf646, buf647, primals_93, primals_94, buf650, 331776, grid=grid(331776), stream=stream0)
        del primals_94
        buf651 = reinterpret_tensor(buf41, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf41  # reuse
        buf652 = reinterpret_tensor(buf651, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf651  # reuse
        # Source Nodes: [x_258, x_se_60], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_124.run(buf652, buf650, 9216, 36, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf653 = extern_kernels.convolution(buf652, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf653, (8, 48, 1, 1), (48, 1, 1, 1))
        buf654 = reinterpret_tensor(buf653, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf653  # reuse
        buf655 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_125.run(buf654, primals_225, buf655, 384, grid=grid(384), stream=stream0)
        del primals_225
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf657 = reinterpret_tensor(buf656, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf656  # reuse
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf657, primals_227, 9216, grid=grid(9216), stream=stream0)
        del primals_227
        buf658 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_258, x_259], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_127.run(buf650, buf657, buf658, 331776, grid=grid(331776), stream=stream0)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(buf658, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (8, 192, 6, 6), (6912, 36, 6, 1))
        buf660 = reinterpret_tensor(buf53, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf53  # reuse
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf659, buf660, 1536, 36, grid=grid(1536, 36), stream=stream0)
        buf661 = buf624; del buf624  # reuse
        buf662 = buf623; del buf623  # reuse
        buf663 = buf622; del buf622  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf660, buf661, buf662, buf663, 576, 96, grid=grid(576), stream=stream0)
        buf664 = buf626; del buf626  # reuse
        buf665 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf667 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf661, buf662, buf663, primals_395, primals_396, buf664, buf665, buf667, primals_395, primals_396, 192, 3, grid=grid(192), stream=stream0)
        del primals_395
        del primals_396
        buf668 = reinterpret_tensor(buf659, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf659  # reuse
        # Source Nodes: [shortcut_16, x_261], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_128.run(buf660, buf664, buf665, primals_95, primals_96, buf629, buf668, 55296, grid=grid(55296), stream=stream0)
        del primals_96
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf669 = extern_kernels.convolution(buf668, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf669, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf670 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf669, buf670, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf671 = buf645; del buf645  # reuse
        buf672 = buf644; del buf644  # reuse
        buf673 = buf643; del buf643  # reuse
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf670, buf671, buf672, buf673, 3456, 96, grid=grid(3456), stream=stream0)
        buf674 = buf647; del buf647  # reuse
        buf675 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf677 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf671, buf672, buf673, primals_398, primals_399, buf674, buf675, buf677, primals_398, primals_399, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_398
        del primals_399
        buf679 = reinterpret_tensor(buf669, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf669  # reuse
        buf802 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267, x_270], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122.run(buf670, buf674, buf675, primals_97, primals_98, buf679, buf802, 331776, grid=grid(331776), stream=stream0)
        del primals_98
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf680 = extern_kernels.convolution(buf679, primals_230, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf680, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf681 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf680, buf681, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf682 = buf673; del buf673  # reuse
        buf683 = buf672; del buf672  # reuse
        buf684 = buf671; del buf671  # reuse
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf681, buf682, buf683, buf684, 3456, 96, grid=grid(3456), stream=stream0)
        buf685 = buf675; del buf675  # reuse
        buf686 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf688 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf682, buf683, buf684, primals_401, primals_402, buf685, buf686, buf688, primals_401, primals_402, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_401
        del primals_402
        buf689 = reinterpret_tensor(buf680, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf680  # reuse
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_123.run(buf681, buf685, buf686, primals_99, primals_100, buf689, 331776, grid=grid(331776), stream=stream0)
        del primals_100
        buf690 = reinterpret_tensor(buf40, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf40  # reuse
        buf691 = reinterpret_tensor(buf690, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf690  # reuse
        # Source Nodes: [x_275, x_se_64], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_124.run(buf691, buf689, 9216, 36, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_65], Original ATen: [aten.convolution]
        buf692 = extern_kernels.convolution(buf691, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf692, (8, 48, 1, 1), (48, 1, 1, 1))
        buf693 = reinterpret_tensor(buf692, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf692  # reuse
        buf694 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_65, x_se_66], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_125.run(buf693, primals_232, buf694, 384, grid=grid(384), stream=stream0)
        del primals_232
        # Source Nodes: [x_se_67], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf694, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf695, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf696 = reinterpret_tensor(buf695, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf695  # reuse
        # Source Nodes: [x_se_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf696, primals_234, 9216, grid=grid(9216), stream=stream0)
        del primals_234
        buf697 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_275, x_276], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_127.run(buf689, buf696, buf697, 331776, grid=grid(331776), stream=stream0)
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf698 = extern_kernels.convolution(buf697, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf698, (8, 192, 6, 6), (6912, 36, 6, 1))
        buf699 = empty_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf698, buf699, 1536, 36, grid=grid(1536, 36), stream=stream0)
        buf700 = buf663; del buf663  # reuse
        buf701 = buf662; del buf662  # reuse
        buf702 = buf661; del buf661  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf699, buf700, buf701, buf702, 576, 96, grid=grid(576), stream=stream0)
        buf703 = buf665; del buf665  # reuse
        buf704 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf706 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf700, buf701, buf702, primals_404, primals_405, buf703, buf704, buf706, primals_404, primals_405, 192, 3, grid=grid(192), stream=stream0)
        del primals_404
        del primals_405
        buf707 = reinterpret_tensor(buf698, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf698  # reuse
        # Source Nodes: [shortcut_17, x_278], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_128.run(buf699, buf703, buf704, primals_101, primals_102, buf668, buf707, 55296, grid=grid(55296), stream=stream0)
        del primals_102
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf708 = extern_kernels.convolution(buf707, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf708, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf709 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf708, buf709, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf710 = buf684; del buf684  # reuse
        buf711 = buf683; del buf683  # reuse
        buf712 = buf682; del buf682  # reuse
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf709, buf710, buf711, buf712, 3456, 96, grid=grid(3456), stream=stream0)
        buf713 = buf686; del buf686  # reuse
        buf714 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf716 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf710, buf711, buf712, primals_407, primals_408, buf713, buf714, buf716, primals_407, primals_408, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_407
        del primals_408
        buf718 = reinterpret_tensor(buf708, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf708  # reuse
        buf801 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284, x_287], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122.run(buf709, buf713, buf714, primals_103, primals_104, buf718, buf801, 331776, grid=grid(331776), stream=stream0)
        del primals_104
        # Source Nodes: [x_288], Original ATen: [aten.convolution]
        buf719 = extern_kernels.convolution(buf718, primals_237, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf719, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf720 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_288], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf719, buf720, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf721 = buf712; del buf712  # reuse
        buf722 = buf711; del buf711  # reuse
        buf723 = buf710; del buf710  # reuse
        # Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf720, buf721, buf722, buf723, 3456, 96, grid=grid(3456), stream=stream0)
        buf724 = buf714; del buf714  # reuse
        buf725 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf727 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf721, buf722, buf723, primals_410, primals_411, buf724, buf725, buf727, primals_410, primals_411, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_410
        del primals_411
        buf728 = reinterpret_tensor(buf719, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf719  # reuse
        # Source Nodes: [x_289], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_123.run(buf720, buf724, buf725, primals_105, primals_106, buf728, 331776, grid=grid(331776), stream=stream0)
        del primals_106
        buf729 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf730 = reinterpret_tensor(buf729, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf729  # reuse
        # Source Nodes: [x_292, x_se_68], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_124.run(buf730, buf728, 9216, 36, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_69], Original ATen: [aten.convolution]
        buf731 = extern_kernels.convolution(buf730, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf731, (8, 48, 1, 1), (48, 1, 1, 1))
        buf732 = reinterpret_tensor(buf731, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf731  # reuse
        buf733 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_69, x_se_70], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_125.run(buf732, primals_239, buf733, 384, grid=grid(384), stream=stream0)
        del primals_239
        # Source Nodes: [x_se_71], Original ATen: [aten.convolution]
        buf734 = extern_kernels.convolution(buf733, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf734, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf735 = reinterpret_tensor(buf734, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf734  # reuse
        # Source Nodes: [x_se_71], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf735, primals_241, 9216, grid=grid(9216), stream=stream0)
        del primals_241
        buf736 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_292, x_293], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_127.run(buf728, buf735, buf736, 331776, grid=grid(331776), stream=stream0)
        # Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf737 = extern_kernels.convolution(buf736, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf737, (8, 192, 6, 6), (6912, 36, 6, 1))
        buf738 = empty_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf737, buf738, 1536, 36, grid=grid(1536, 36), stream=stream0)
        buf739 = buf702; del buf702  # reuse
        buf740 = buf701; del buf701  # reuse
        buf741 = buf700; del buf700  # reuse
        # Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf738, buf739, buf740, buf741, 576, 96, grid=grid(576), stream=stream0)
        buf742 = buf704; del buf704  # reuse
        buf743 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf745 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf739, buf740, buf741, primals_413, primals_414, buf742, buf743, buf745, primals_413, primals_414, 192, 3, grid=grid(192), stream=stream0)
        del buf739
        del buf740
        del buf741
        del primals_413
        del primals_414
        buf746 = reinterpret_tensor(buf737, (8, 192, 6, 6), (6912, 1, 1152, 192), 0); del buf737  # reuse
        # Source Nodes: [shortcut_18, x_295], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_128.run(buf738, buf742, buf743, primals_107, primals_108, buf707, buf746, 55296, grid=grid(55296), stream=stream0)
        del buf743
        del primals_108
        # Source Nodes: [x_300], Original ATen: [aten.convolution]
        buf747 = extern_kernels.convolution(buf746, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf747, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf748 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf747, buf748, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf749 = buf723; del buf723  # reuse
        buf750 = buf722; del buf722  # reuse
        buf751 = buf721; del buf721  # reuse
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf748, buf749, buf750, buf751, 3456, 96, grid=grid(3456), stream=stream0)
        buf752 = buf725; del buf725  # reuse
        buf753 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf755 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf749, buf750, buf751, primals_416, primals_417, buf752, buf753, buf755, primals_416, primals_417, 1152, 3, grid=grid(1152), stream=stream0)
        del primals_416
        del primals_417
        buf757 = reinterpret_tensor(buf747, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf747  # reuse
        buf800 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301, x_304], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_122.run(buf748, buf752, buf753, primals_109, primals_110, buf757, buf800, 331776, grid=grid(331776), stream=stream0)
        del primals_110
        # Source Nodes: [x_305], Original ATen: [aten.convolution]
        buf758 = extern_kernels.convolution(buf757, primals_244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf758, (8, 1152, 6, 6), (41472, 36, 6, 1))
        buf759 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_305], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_119.run(buf758, buf759, 9216, 36, grid=grid(9216, 36), stream=stream0)
        buf760 = buf751; del buf751  # reuse
        buf761 = buf750; del buf750  # reuse
        buf762 = buf749; del buf749  # reuse
        # Source Nodes: [x_306], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf759, buf760, buf761, buf762, 3456, 96, grid=grid(3456), stream=stream0)
        buf763 = buf753; del buf753  # reuse
        buf764 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf766 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_306], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_121.run(buf760, buf761, buf762, primals_419, primals_420, buf763, buf764, buf766, primals_419, primals_420, 1152, 3, grid=grid(1152), stream=stream0)
        del buf760
        del buf761
        del buf762
        del primals_419
        del primals_420
        buf767 = reinterpret_tensor(buf758, (8, 1152, 6, 6), (41472, 1, 6912, 1152), 0); del buf758  # reuse
        # Source Nodes: [x_306], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_123.run(buf759, buf763, buf764, primals_111, primals_112, buf767, 331776, grid=grid(331776), stream=stream0)
        del buf764
        del primals_112
        buf768 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf769 = reinterpret_tensor(buf768, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf768  # reuse
        # Source Nodes: [x_309, x_se_72], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_124.run(buf769, buf767, 9216, 36, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_73], Original ATen: [aten.convolution]
        buf770 = extern_kernels.convolution(buf769, primals_245, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf770, (8, 48, 1, 1), (48, 1, 1, 1))
        buf771 = reinterpret_tensor(buf770, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf770  # reuse
        buf772 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_73, x_se_74], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_125.run(buf771, primals_246, buf772, 384, grid=grid(384), stream=stream0)
        del primals_246
        # Source Nodes: [x_se_75], Original ATen: [aten.convolution]
        buf773 = extern_kernels.convolution(buf772, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf773, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf774 = reinterpret_tensor(buf773, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf773  # reuse
        # Source Nodes: [x_se_75], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf774, primals_248, 9216, grid=grid(9216), stream=stream0)
        del primals_248
        buf775 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_309, x_310], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_127.run(buf767, buf774, buf775, 331776, grid=grid(331776), stream=stream0)
        # Source Nodes: [x_311], Original ATen: [aten.convolution]
        buf776 = extern_kernels.convolution(buf775, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf776, (8, 320, 6, 6), (11520, 36, 6, 1))
        buf777 = empty_strided((8, 320, 6, 6), (11520, 1, 1920, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_311], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_129.run(buf776, buf777, 2560, 36, grid=grid(2560, 36), stream=stream0)
        buf778 = empty_strided((1, 320, 1, 1, 3), (960, 1, 960, 960, 320), device='cuda', dtype=torch.float32)
        buf779 = empty_strided((1, 320, 1, 1, 3), (960, 1, 960, 960, 320), device='cuda', dtype=torch.float32)
        buf780 = empty_strided((1, 320, 1, 1, 3), (960, 1, 960, 960, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_130.run(buf777, buf778, buf779, buf780, 960, 96, grid=grid(960), stream=stream0)
        buf781 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf782 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf784 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_131.run(buf778, buf779, buf780, primals_422, primals_423, buf781, buf782, buf784, primals_422, primals_423, 320, 3, grid=grid(320), stream=stream0)
        del buf778
        del buf779
        del buf780
        del primals_422
        del primals_423
        buf785 = reinterpret_tensor(buf776, (8, 320, 6, 6), (11520, 1, 1920, 320), 0); del buf776  # reuse
        # Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_132.run(buf777, buf781, buf782, primals_113, primals_114, buf785, 92160, grid=grid(92160), stream=stream0)
        del buf782
        del primals_114
        # Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf786 = extern_kernels.convolution(buf785, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf786, (8, 1280, 6, 6), (46080, 36, 6, 1))
        buf787 = empty_strided((8, 1280, 6, 6), (46080, 1, 7680, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_317], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf786, buf787, 10240, 36, grid=grid(10240, 36), stream=stream0)
        buf788 = empty_strided((1, 1280, 1, 1, 3), (3840, 1, 3840, 3840, 1280), device='cuda', dtype=torch.float32)
        buf789 = empty_strided((1, 1280, 1, 1, 3), (3840, 1, 3840, 3840, 1280), device='cuda', dtype=torch.float32)
        buf790 = empty_strided((1, 1280, 1, 1, 3), (3840, 1, 3840, 3840, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_134.run(buf787, buf788, buf789, buf790, 3840, 96, grid=grid(3840), stream=stream0)
        buf791 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf792 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf794 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_135.run(buf788, buf789, buf790, primals_425, primals_426, buf791, buf792, buf794, primals_425, primals_426, 1280, 3, grid=grid(1280), stream=stream0)
        del buf788
        del buf789
        del buf790
        del primals_425
        del primals_426
        buf795 = reinterpret_tensor(buf786, (8, 1280, 6, 6), (46080, 1, 7680, 1280), 0); del buf786  # reuse
        buf799 = empty_strided((8, 1280, 6, 6), (46080, 1, 7680, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_136.run(buf787, buf791, buf792, primals_115, primals_116, buf795, buf799, 368640, grid=grid(368640), stream=stream0)
        del buf792
        del primals_116
        buf796 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf797 = reinterpret_tensor(buf796, (8, 1280), (1280, 1), 0); del buf796  # reuse
        # Source Nodes: [x_322, x_323, x_325], Original ATen: [aten.mean, aten.silu, aten.view]
        triton_per_fused_mean_silu_view_137.run(buf797, buf795, 10240, 36, grid=grid(10240), stream=stream0)
        del buf795
        buf798 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_252, buf797, reinterpret_tensor(primals_251, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf798)
        del primals_252
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_253, primals_253, 1, grid=grid(1), stream=stream0)
        del primals_253
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_256, primals_256, 1, grid=grid(1), stream=stream0)
        del primals_256
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_259, primals_259, 1, grid=grid(1), stream=stream0)
        del primals_259
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_262, primals_262, 1, grid=grid(1), stream=stream0)
        del primals_262
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_265, primals_265, 1, grid=grid(1), stream=stream0)
        del primals_265
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_268, primals_268, 1, grid=grid(1), stream=stream0)
        del primals_268
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_271, primals_271, 1, grid=grid(1), stream=stream0)
        del primals_271
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_274, primals_274, 1, grid=grid(1), stream=stream0)
        del primals_274
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_277, primals_277, 1, grid=grid(1), stream=stream0)
        del primals_277
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_280, primals_280, 1, grid=grid(1), stream=stream0)
        del primals_280
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_283, primals_283, 1, grid=grid(1), stream=stream0)
        del primals_283
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_286, primals_286, 1, grid=grid(1), stream=stream0)
        del primals_286
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_289, primals_289, 1, grid=grid(1), stream=stream0)
        del primals_289
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_292, primals_292, 1, grid=grid(1), stream=stream0)
        del primals_292
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_313, primals_313, 1, grid=grid(1), stream=stream0)
        del primals_313
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_316, primals_316, 1, grid=grid(1), stream=stream0)
        del primals_316
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_319, primals_319, 1, grid=grid(1), stream=stream0)
        del primals_319
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_322, primals_322, 1, grid=grid(1), stream=stream0)
        del primals_322
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_325, primals_325, 1, grid=grid(1), stream=stream0)
        del primals_325
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_328, primals_328, 1, grid=grid(1), stream=stream0)
        del primals_328
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_331, primals_331, 1, grid=grid(1), stream=stream0)
        del primals_331
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_334, primals_334, 1, grid=grid(1), stream=stream0)
        del primals_334
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_337, primals_337, 1, grid=grid(1), stream=stream0)
        del primals_337
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_340, primals_340, 1, grid=grid(1), stream=stream0)
        del primals_340
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_343, primals_343, 1, grid=grid(1), stream=stream0)
        del primals_343
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_346, primals_346, 1, grid=grid(1), stream=stream0)
        del primals_346
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_349, primals_349, 1, grid=grid(1), stream=stream0)
        del primals_349
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_352, primals_352, 1, grid=grid(1), stream=stream0)
        del primals_352
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_355, primals_355, 1, grid=grid(1), stream=stream0)
        del primals_355
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_358, primals_358, 1, grid=grid(1), stream=stream0)
        del primals_358
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_361, primals_361, 1, grid=grid(1), stream=stream0)
        del primals_361
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_364, primals_364, 1, grid=grid(1), stream=stream0)
        del primals_364
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_367, primals_367, 1, grid=grid(1), stream=stream0)
        del primals_367
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_370, primals_370, 1, grid=grid(1), stream=stream0)
        del primals_370
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_373, primals_373, 1, grid=grid(1), stream=stream0)
        del primals_373
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_376, primals_376, 1, grid=grid(1), stream=stream0)
        del primals_376
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_379, primals_379, 1, grid=grid(1), stream=stream0)
        del primals_379
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_382, primals_382, 1, grid=grid(1), stream=stream0)
        del primals_382
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_385, primals_385, 1, grid=grid(1), stream=stream0)
        del primals_385
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_388, primals_388, 1, grid=grid(1), stream=stream0)
        del primals_388
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_391, primals_391, 1, grid=grid(1), stream=stream0)
        del primals_391
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_394, primals_394, 1, grid=grid(1), stream=stream0)
        del primals_394
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_397, primals_397, 1, grid=grid(1), stream=stream0)
        del primals_397
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_400, primals_400, 1, grid=grid(1), stream=stream0)
        del primals_400
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_403, primals_403, 1, grid=grid(1), stream=stream0)
        del primals_403
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_406, primals_406, 1, grid=grid(1), stream=stream0)
        del primals_406
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_409, primals_409, 1, grid=grid(1), stream=stream0)
        del primals_409
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_412, primals_412, 1, grid=grid(1), stream=stream0)
        del primals_412
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_415, primals_415, 1, grid=grid(1), stream=stream0)
        del primals_415
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_418, primals_418, 1, grid=grid(1), stream=stream0)
        del primals_418
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_421, primals_421, 1, grid=grid(1), stream=stream0)
        del primals_421
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_138.run(primals_424, primals_424, 1, grid=grid(1), stream=stream0)
        del primals_424
        return (buf798, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, buf0, primals_118, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_128, primals_130, primals_131, primals_132, primals_133, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_144, primals_145, primals_146, primals_147, primals_149, primals_151, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_187, primals_188, primals_189, primals_191, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_249, primals_250, buf1, buf3, buf13, buf15, buf17, buf27, buf28, buf31, buf33, buf34, buf36, buf37, buf39, buf49, buf50, buf52, buf62, buf64, buf66, buf76, buf77, buf80, buf82, buf83, buf85, buf86, buf88, buf98, buf99, buf101, buf111, buf113, buf115, buf125, buf126, buf129, buf131, buf132, buf134, buf135, buf137, buf147, buf148, buf150, buf160, buf162, buf164, buf171, buf172, buf175, buf177, buf178, buf180, buf181, buf183, buf190, buf191, buf193, buf200, buf202, buf204, buf211, buf212, buf215, buf217, buf218, buf220, buf221, buf223, buf230, buf231, buf233, buf240, buf242, buf244, buf251, buf252, buf255, buf257, buf258, buf260, buf261, buf263, buf270, buf271, buf273, buf280, buf282, buf284, buf291, buf292, buf295, buf297, buf298, buf300, buf301, buf303, buf310, buf311, buf313, buf320, buf322, buf324, buf331, buf332, buf335, buf337, buf338, buf340, buf341, buf343, buf350, buf351, buf353, buf360, buf362, buf364, buf371, buf372, buf375, buf377, buf378, buf380, buf381, buf383, buf390, buf391, buf393, buf400, buf402, buf404, buf411, buf412, buf415, buf417, buf418, buf420, buf421, buf423, buf430, buf431, buf433, buf440, buf442, buf444, buf451, buf452, buf455, buf457, buf458, buf460, buf461, buf463, buf470, buf471, buf473, buf480, buf482, buf484, buf491, buf492, buf495, buf497, buf498, buf500, buf501, buf503, buf510, buf511, buf513, buf520, buf522, buf524, buf531, buf532, buf535, buf537, buf538, buf540, buf541, buf543, buf550, buf551, buf553, buf560, buf562, buf564, buf571, buf572, buf574, buf576, buf577, buf579, buf580, buf582, buf589, buf590, buf592, buf599, buf601, buf603, buf610, buf611, buf613, buf615, buf616, buf618, buf619, buf621, buf628, buf629, buf631, buf638, buf640, buf642, buf649, buf650, buf652, buf654, buf655, buf657, buf658, buf660, buf667, buf668, buf670, buf677, buf679, buf681, buf688, buf689, buf691, buf693, buf694, buf696, buf697, buf699, buf706, buf707, buf709, buf716, buf718, buf720, buf727, buf728, buf730, buf732, buf733, buf735, buf736, buf738, buf745, buf746, buf748, buf755, buf757, buf759, buf766, buf767, buf769, buf771, buf772, buf774, buf775, buf777, buf784, buf785, buf787, buf794, buf797, reinterpret_tensor(primals_251, (1000, 1280), (1280, 1), 0), buf799, reinterpret_tensor(buf791, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf781, (1, 320, 1, 1), (320, 1, 1, 1), 0), reinterpret_tensor(buf763, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf800, reinterpret_tensor(buf752, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf742, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf724, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf801, reinterpret_tensor(buf713, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf703, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf685, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf802, reinterpret_tensor(buf674, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf664, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf646, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf803, reinterpret_tensor(buf635, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf625, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf607, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf804, reinterpret_tensor(buf596, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf586, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf568, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf805, reinterpret_tensor(buf557, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf547, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf528, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf806, reinterpret_tensor(buf517, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf507, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf488, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf807, reinterpret_tensor(buf477, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf467, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf448, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf808, reinterpret_tensor(buf437, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf427, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf408, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf809, reinterpret_tensor(buf397, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf368, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf810, reinterpret_tensor(buf357, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf347, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf328, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf811, reinterpret_tensor(buf317, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf307, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf288, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf812, reinterpret_tensor(buf277, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf248, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf813, reinterpret_tensor(buf237, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf227, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf208, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf814, reinterpret_tensor(buf197, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf168, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf815, reinterpret_tensor(buf157, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf144, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf122, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf816, reinterpret_tensor(buf108, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf95, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf73, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf817, reinterpret_tensor(buf59, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf46, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf818, reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_254 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_257 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_260 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_263 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_269 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_275 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_278 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_284 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_293 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_314 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_320 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_323 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_326 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_329 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_332 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_335 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_338 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_344 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_347 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_350 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_353 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_356 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_359 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_362 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_365 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_368 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_371 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_374 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_377 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_380 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_383 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_386 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_389 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_392 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_395 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_398 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_401 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_404 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_407 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_410 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_413 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_416 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_419 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_422 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_425 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((8, 3, 192, 192), (110592, 36864, 192, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tinynet_a', benchmark_compiled_module)
