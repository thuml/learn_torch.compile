
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


# kernel path: /tmp/torchinductor_youkaichao/ff/cffukygqnecyrgthxf2gzgmdgkouny54larfwliadyy2ckeyc3gz.py
# Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50625
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 225)
    x2 = xindex % 225
    y4 = yindex
    x5 = xindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 224, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (224*x3) + (50176*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (y0 + (3*x5) + (151875*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clk32p3dallahhj3b3au642oeuehx5abnbgpaj4hxrbmnnlj5ws4.py
# Source Nodes: [x_1], Original ATen: [aten.convolution]
# x_1 => convolution
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
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/cory2syeboblbsxmbjwubnvvurupktstlsmdrw2fjbrvzlgmh4mb.py
# Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
# x_2 => var_mean
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
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/26/c26nvt2mfa2vvzc53m4zmz2fesqupinydzilftvskjhzqof3ombw.py
# Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
# x_2 => var_mean
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (3584*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (3584*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (32*r2) + (3584*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (32*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (32*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (32*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfslbwhqbebjoezavueyjiqkzn5ry3qk45q5f2czejqhvamvspy2.py
# Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
# x_2 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
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
    rnumel = 7
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
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


# kernel path: /tmp/torchinductor_youkaichao/6a/c6a5duhlvqxysyzzotugmvqyecwliy4k3elvxhrsuqaoi4i7uvad.py
# Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut => mul_7, sigmoid
# x_2 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
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
    xnumel = 3211264
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
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csyoooh2mny7lot563z5ai7jy5m2tmsp4enqx3orw32qnsfiph7l.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => add_6, add_9, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
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
    xnumel = 3211264
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
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypc4o4euqv6kxwki7f7m72efl6o6orakq65nlzz2ehh3h6i4qn5.py
# Source Nodes: [x_10, x_se], Original ATen: [aten.mean, aten.silu]
# x_10 => mul_15, sigmoid_1
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
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdjcjiknupzvyxxt3ct6r44qgcennac742eszgowph75fm43es7.py
# Source Nodes: [x_10, x_se], Original ATen: [aten.mean, aten.silu]
# x_10 => mul_15, sigmoid_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 12544.0
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4fffskpkmpoxdb47ppf5k7ocar6q2ni7zxk4yixphtcyy2jh3v.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_15, sigmoid_1
# x_11 => mul_17
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
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    x2 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwq5flttlyfbfz5sloyeigcjwdnzkgxksxjzsphu6nid7ml4cn4g.py
# Source Nodes: [x_12], Original ATen: [aten.convolution]
# x_12 => convolution_4
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
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwy7ga4gzwbju6sth3mhyeg53qhwv7kmmdlqerbcqtbgb5ni3oz.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
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
    xnumel = 12544
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


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3msx6jozd6oc7ddzmk5upha7fphffks3fopk4l4vpg3dhsqdft.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (1792*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (16*r2) + (1792*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (16*r2) + (1792*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (16*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (16*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (16*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjg2gontfzdiyxl67eff6ryvwzxavadfscz6yg3wpv6frgs4u2ql.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => add_11, add_12, add_13, mul_19, mul_20, mul_21, mul_22, mul_23, rsqrt_2, squeeze_7, var_mean_2
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
    rnumel = 7
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wbqtvfju2photon6thu2xcvu7hcg7d2rvp3b7fahpkaob6pj3q.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => add_11, add_14, mul_18, mul_24, rsqrt_2, sub_2, var_mean_2
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
    xnumel = 1605632
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
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpe7alk62w7hzhdalr3isr554nkiydh4f7he6bjkntuuvielniot.py
# Source Nodes: [x_17], Original ATen: [aten.convolution]
# x_17 => convolution_5
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
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzhxvwlepjjsdrgm255bfjidobmngude5bjkxi33y7355tn3a7a.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_18 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwfugbkdnerijexlwgi7bnrmfti7cmbivso5a6tecooq7wmgsne.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_18 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (10752*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (10752*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (96*r2) + (10752*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5d4io7lb4lvcjygabit63omxplblz5pu7f4zoj3nvf4khudice.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_18 => add_16, add_17, add_18, mul_26, mul_27, mul_28, mul_29, mul_30, rsqrt_3, squeeze_10, var_mean_3
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
    rnumel = 7
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
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


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmxrpu7w2l5xipwnyjruprt5fjnwlebaszcvoof2ol7gp2juger.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_18 => add_16, add_19, mul_25, mul_31, rsqrt_3, sub_3, var_mean_3
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
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
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcix3idz5qhgafmdh7p4pb3gakadh2yngom2v7agsuqwh664y42.py
# Source Nodes: [x_21, x_23], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_21 => mul_32, sigmoid_4
# x_23 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_silu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9806592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10848) % 113
    x1 = (xindex // 96) % 113
    x3 = (xindex // 1225824)
    x4 = xindex % 10848
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (10752*x2) + (1204224*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxcrhqkbwv7qgxvh5afjgnlpxsn2yusgsvfr3rplf4pmlqct6mld.py
# Source Nodes: [x_24], Original ATen: [aten.convolution]
# x_24 => convolution_6
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmxj4hkzijf55luiortkub3nukazsveqbzpaeu55373f4yslg55.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
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


# kernel path: /tmp/torchinductor_youkaichao/md/cmdi4zw7fx37ejirzr236o5p4rjyijnej2bt4nt5pei6wusezfw5.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (9408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (9408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (96*r2) + (9408*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qa/cqao7vfkjq2tkhf4dxuvwjufmde3l7zsbnleflieo56mml6jksq7.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => add_21, add_22, add_23, mul_34, mul_35, mul_36, mul_37, mul_38, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
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


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnux67ms7vfgm3af45o2k7kbidoulpg2s23wxd2rmcenyvsyvwr.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => add_21, add_24, mul_33, mul_39, rsqrt_4, sub_4, var_mean_4
triton_poi_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/criuheol7mtp46dvhgw7nyxnoglercdjnhatpmixl2xjd2ji4hmc.py
# Source Nodes: [x_28, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_28 => mul_40, sigmoid_5
# x_se_4 => mean_1
triton_red_fused_mean_silu_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96) % 25
    x0 = xindex % 96
    x2 = (xindex // 2400)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r3 + (126*x1)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7mr74sg54asb2gsiundizww54hixtg4bdnsvd2ww3x2z2kfq2y.py
# Source Nodes: [x_28, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_28 => mul_40, sigmoid_5
# x_se_4 => mean_1
triton_per_fused_mean_silu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (2400*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnnkd4zvrxylrcvgbxohoxay737rmzl2iwq5g5kd2a4ydanecvj.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
# x_se_5 => convolution_7
# x_se_6 => mul_41, sigmoid_6
triton_poi_fused_convolution_silu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_31', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtj3kbww7tnhelfzbacsbgxpmovbij34qhklloma2ey3xeshlr6.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_8
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/4h/c4ht6byof4to4sje3t7l353io3syn6fmwvwgqc25ejp6lckdvk5s.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28, x_29], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_28 => mul_40, sigmoid_5
# x_29 => mul_42
triton_poi_fused_mul_sigmoid_silu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 301056)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (96*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/ccimpqd4qutzq2vesk2sc6j7b6izxgk7fbxmzk7jn4fser3gnn3b.py
# Source Nodes: [x_30], Original ATen: [aten.convolution]
# x_30 => convolution_9
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlhrnn5wyk2ncuvb3zsstaibjcah4w3pzicqmcpcnu5okthv3fp.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_youkaichao/nk/cnksh2mdvv2jdenvtujt7f5h3fwbfse5wbvs42pxzhoot7nao4ja.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x1 + (24*r2) + (2352*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (2352*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (24*r2) + (2352*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ax/caxs4bsunpbhrph6yesa5cjbjeivklp3v5pqj7jhuovj7d6ir7jv.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => add_26, add_27, add_28, mul_44, mul_45, mul_46, mul_47, mul_48, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
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


# kernel path: /tmp/torchinductor_youkaichao/ad/cadkmkg2nlppcci6eyx4nrtitcb3yugaussh4t7ja6jenuu476eu.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
# x_31 => add_26, add_29, mul_43, mul_49, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldkmvj7x5arwp5gihjmcoz5k5ssg6cbjekmkrhow2wznhlb2sej.py
# Source Nodes: [x_35], Original ATen: [aten.convolution]
# x_35 => convolution_10
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kf/ckf7vsfusd33jrnzspuclknybxg3wirkay3m2jh6ztlga5cd7rrc.py
# Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
# x_36 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
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


# kernel path: /tmp/torchinductor_youkaichao/2y/c2y5iq5solcm2z5yb5ug2o7i4d3rlt6melefjd5eb25mhvnqjxel.py
# Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
# x_36 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x1 + (144*r2) + (14112*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (144*r2) + (14112*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (144*r2) + (14112*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yx/cyx44r3eu47ardtncqsmohkjq7cow3ahc3vjzhik72756as2w2g6.py
# Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
# x_36 => add_31, add_32, add_33, mul_51, mul_52, mul_53, mul_54, mul_55, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
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


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrlz7tofh6qmmdvxdvoz3qkmvk573emzpa5rmeauwch3rgmjp2h.py
# Source Nodes: [x_36, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_36 => add_31, add_34, mul_50, mul_56, rsqrt_6, sub_6, var_mean_6
# x_39 => mul_57, sigmoid_8
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7s6iuw7dpj45j4xrqyeoiefgg4pwwk4aa44af466trggdmofv3.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
# x_41 => add_36, add_39, mul_58, mul_64, rsqrt_7, sub_7, var_mean_7
triton_poi_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyy6um53vqilaogj45r7hcgof6w6g3ibyl2ky4b4ovpmyidwonx.py
# Source Nodes: [x_44, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_44 => mul_65, sigmoid_9
# x_se_8 => mean_2
triton_red_fused_mean_silu_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144) % 25
    x0 = xindex % 144
    x2 = (xindex // 3600)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (144*((r3 + (126*x1)) % 3136)) + (451584*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rymv4n5cw6b4bvpkmqzbgoydhpmrz5lzp5b3pom6w6rmozfprb.py
# Source Nodes: [x_44, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_44 => mul_65, sigmoid_9
# x_se_8 => mean_2
triton_per_fused_mean_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (3600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/cksmlustbtbrhidd4qqg2idd35p6qs2ewavhonkax7gmlu5azlfi.py
# Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
# x_se_10 => mul_66, sigmoid_10
# x_se_9 => convolution_12
triton_poi_fused_convolution_silu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_47', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdnya6apj4rieknfvbwwarf3nihu6jytszoavomvngfmlc3fshv.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution]
# x_se_11 => convolution_13
triton_poi_fused_convolution_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhrvtowf3iaccbatsxb2yumeiw4zjw5w4tge4llwgq5m3uxgzq2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44, x_45], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_44 => mul_65, sigmoid_9
# x_45 => mul_67
triton_poi_fused_mul_sigmoid_silu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 451584)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpn5tdaqgowlzp35jpeqig5phusonicuf3uuyxyfvcpaqjkpvtmh.py
# Source Nodes: [shortcut_3, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_45
# x_47 => add_41, add_44, mul_68, mul_74, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_add_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckc7x4dta6oiq2waqqxh46hswmnrtv2w3aqlrsnrjjq4grmts3us.py
# Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_53 => add_47, add_50, mul_75, mul_81, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ce/ccepr5c6pv5fm6yl6pkah6rhugrkfa7vzrfjrkoftccorhhngtvz.py
# Source Nodes: [x_56, x_58], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_56 => mul_82, sigmoid_12
# x_58 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_silu_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4010112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8496) % 59
    x1 = (xindex // 144) % 59
    x3 = (xindex // 501264)
    x4 = xindex % 8496
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-8208) + x4 + (8064*x2) + (451584*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7exjhiu2nor2odfdi24aqcbnhbqkahfd4tfpartnr6grrpj5jg.py
# Source Nodes: [x_59], Original ATen: [aten.convolution]
# x_59 => convolution_16
triton_poi_fused_convolution_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x2) + (112896*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwe3k3ztgfwz4dnnyx73xydxnuobm2cjnfe4yuo3jx6i52fdw5j.py
# Source Nodes: [x_60], Original ATen: [aten._native_batch_norm_legit_functional]
# x_60 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
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


# kernel path: /tmp/torchinductor_youkaichao/tp/ctpk7knuvlimkxjahsawfaspshq7646ft5f3gbwbms4uio36ze2d.py
# Source Nodes: [x_60], Original ATen: [aten._native_batch_norm_legit_functional]
# x_60 => add_52, add_53, add_54, mul_84, mul_85, mul_86, mul_87, mul_88, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 49
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
    tmp16 = 6272.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001594642002871
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


# kernel path: /tmp/torchinductor_youkaichao/pt/cptoqzgpgjciqelmmqtion7644qiwvhfhihkvfidur6r35365lsj.py
# Source Nodes: [x_60], Original ATen: [aten._native_batch_norm_legit_functional]
# x_60 => add_52, add_55, mul_83, mul_89, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
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
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clgbu6agtvrugfg3cdxqt6xc6mmwk4nojtckco5fqtugsuqg7u4t.py
# Source Nodes: [x_63, x_se_12], Original ATen: [aten.mean, aten.silu]
# x_63 => mul_90, sigmoid_13
# x_se_12 => mean_3
triton_red_fused_mean_silu_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8064
    rnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (16128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4y5jk2a2dqmaoe5lfwmuu43wwqpbz5okhdxnmmiijjahfzdctrc.py
# Source Nodes: [x_63, x_se_12], Original ATen: [aten.mean, aten.silu]
# x_63 => mul_90, sigmoid_13
# x_se_12 => mean_3
triton_per_fused_mean_silu_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (1008*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mw2x5fzzflcrefuodezdw3pag34rupxdw4za2uzbpkgboi3dag.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63, x_64], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_63 => mul_90, sigmoid_13
# x_64 => mul_92
triton_poi_fused_mul_sigmoid_silu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 112896)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cja2ewne54ere4jj4ayg2ihc7mmc7h2x6lo7aulp7fnkyq37dpfe.py
# Source Nodes: [x_65], Original ATen: [aten.convolution]
# x_65 => convolution_19
triton_poi_fused_convolution_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (40*x2) + (31360*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wfeaxd5ttusum3yzdbsxwwj32zczgskplw4ddcsd2kn3kg67jg.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1960
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


# kernel path: /tmp/torchinductor_youkaichao/hq/chqxtvqorxbddlnfjzt772j6jnxza7pdinxjfagaexcfjdyckhad.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => add_57, add_58, add_59, mul_94, mul_95, mul_96, mul_97, mul_98, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 49
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
    tmp16 = 6272.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001594642002871
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


# kernel path: /tmp/torchinductor_youkaichao/ux/cux5fvoalwo5bpgaobnwbcvbucufcy5hy5q4kkocu6jfsxgbse4c.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => add_57, add_60, mul_93, mul_99, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnopdkxyc7eomansfqfx426k5nusfjfastaqjt246x2buarufhg.py
# Source Nodes: [x_70], Original ATen: [aten.convolution]
# x_70 => convolution_20
triton_poi_fused_convolution_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkfnb3wklx636jiznj3xeklr6hlp3lewc6xonqixwiguaone4so.py
# Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
# x_71 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
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


# kernel path: /tmp/torchinductor_youkaichao/uu/cuudhovlnl4kowll4oqfaqy4ivvicjq5xvbr3eqfbzuxv7xt7nrm.py
# Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
# x_71 => add_62, add_63, add_64, mul_101, mul_102, mul_103, mul_104, mul_105, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 49
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
    tmp16 = 6272.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001594642002871
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


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3jmig3ci5dipnxnyjm27ewdu67dwh25ocy3znsyqf42t7ycuhz.py
# Source Nodes: [x_71, x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_71 => add_62, add_65, mul_100, mul_106, rsqrt_12, sub_12, var_mean_12
# x_74 => mul_107, sigmoid_16
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
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
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnvetgchsmlkf32fslbrrnnqt626ihq3zozoxr56tidxtzjxysj.py
# Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
# x_76 => add_67, add_70, mul_108, mul_114, rsqrt_13, sub_13, var_mean_13
triton_poi_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
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
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqkind3n3car66qjqwkdpv64xxti2mrhi6rqrftcq5gz3ntbtae.py
# Source Nodes: [x_79, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_79 => mul_115, sigmoid_17
# x_se_16 => mean_4
triton_red_fused_mean_silu_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13440
    rnumel = 112
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
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ya/cyai2rokbicsbcermtphsu7ilscaachug2ezh2lplqyrv6eq2nfm.py
# Source Nodes: [x_79, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_79 => mul_115, sigmoid_17
# x_se_16 => mean_4
triton_per_fused_mean_silu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_70', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (1680*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvxrhn6pfwuljpuvnn7ybvw3sdpwdps5vzd7mn4azloj4unkwsh.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
# x_se_17 => convolution_22
# x_se_18 => mul_116, sigmoid_18
triton_poi_fused_convolution_silu_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_71', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3w5eipjjwfjuajztwxlw4ulcntucb2vf377r72v5mgm5lil4tuw.py
# Source Nodes: [x_se_19], Original ATen: [aten.convolution]
# x_se_19 => convolution_23
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6scedltyqawlcolxozneqeuon5wdwd3pu32rpydl2el55gyoyh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79, x_80], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_79 => mul_115, sigmoid_17
# x_80 => mul_117
triton_poi_fused_mul_sigmoid_silu_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 188160)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (240*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgacvdcmrcpycexfx4nraugvnjaxtb2jsatbqil4kdjaxxf3fpj.py
# Source Nodes: [shortcut_5, x_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_5 => add_76
# x_82 => add_72, add_75, mul_118, mul_124, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_add_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2mchyrt6xalpwohugjetrdtoxatfdshz3jyrdq6ytdia7vgamr.py
# Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_88 => add_78, add_81, mul_125, mul_131, rsqrt_15, sub_15, var_mean_15
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
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
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ar/carur5xtfu2npkbhql2w33iqzy2k5ozbp6h5boaklkrny4bhp4x4.py
# Source Nodes: [x_91, x_93], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_91 => mul_132, sigmoid_20
# x_93 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_silu_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1614720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6960) % 29
    x1 = (xindex // 240) % 29
    x3 = (xindex // 201840)
    x4 = xindex % 6960
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (6720*x2) + (188160*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cet5pm7plzbdivimzjrml6h4zgbg2j2etiwqlzyfqggyfpizdyau.py
# Source Nodes: [x_94], Original ATen: [aten.convolution]
# x_94 => convolution_26
triton_poi_fused_convolution_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskmgsyshe27pv2uonsxsbphexbhlchsekeyu33ovahb4af7ig7j.py
# Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
# x_95 => var_mean_16
triton_red_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3120
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240)
    x0 = xindex % 240
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (240*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskvrxrygkurrrlq4t6wc22rjl7yu5smbvn7mquu5gcor3ybqy5c.py
# Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
# x_95 => add_83, add_84, add_85, mul_134, mul_135, mul_136, mul_137, mul_138, rsqrt_16, squeeze_49, var_mean_16
triton_per_fused__native_batch_norm_legit_functional_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_79', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 13
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/ij/cij7xbg4kqty2uroe25pqi47nf4u65epdccuztywnqaekksfbvaq.py
# Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
# x_95 => add_83, add_86, mul_133, mul_139, rsqrt_16, sub_16, var_mean_16
triton_poi_fused__native_batch_norm_legit_functional_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clngeki4yixi5bp5hz6do2anyheoa2wvp6h7rdipngj2rxahvhfh.py
# Source Nodes: [x_98, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_98 => mul_140, sigmoid_21
# x_se_20 => mean_5
triton_red_fused_mean_silu_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (23520*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v4/cv4cfrlujilpeha6nn2bf3fvf3subl3bwdaa32lg3gi3gmk46not.py
# Source Nodes: [x_98, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_98 => mul_140, sigmoid_21
# x_se_20 => mean_5
triton_per_fused_mean_silu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_82', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctblrygvpyb3myhfecnrayhaqhhd4vgdtoals3gyyyymp7fvzcqg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_98, x_99], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_98 => mul_140, sigmoid_21
# x_99 => mul_142
triton_poi_fused_mul_sigmoid_silu_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 47040)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (240*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kjwolzjg4lldbbirlttz2l2w2u6jj7e7klcsmkfuopw7lw6bp6.py
# Source Nodes: [x_100], Original ATen: [aten.convolution]
# x_100 => convolution_29
triton_poi_fused_convolution_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6eddghd53rc56sr4emackixkubuihafm3yyi3lvbbblv3y2dpz.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
# x_101 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1040
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5phr4se2jy5x27m23oxops6e4m46nzp3yakllimbbhsqvckcbu.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
# x_101 => add_88, add_89, add_90, mul_144, mul_145, mul_146, mul_147, mul_148, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 13
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtbblrtjivxl3nmxzqpcdxlqzz6dwbdxaxzgkiy2orhu33hfk2w.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
# x_101 => add_88, add_91, mul_143, mul_149, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzgz7sad73u2t4jc5uvv6moiragmymkxsgbnysfivxt4ts3uoel.py
# Source Nodes: [x_105], Original ATen: [aten.convolution]
# x_105 => convolution_30
triton_poi_fused_convolution_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/cslx5f75qo5gd5u3iexnepn7wwvcsz4yrlbel4hqi4qtchssxmcx.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 480)
    x0 = xindex % 480
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (480*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cnepk4fvkvl4uecde7eecdwti7hy3z52h6ljuep4oohup7xyarbq.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => add_93, add_94, add_95, mul_151, mul_152, mul_153, mul_154, mul_155, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 13
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/xc/cxcbzh55pm3lfqqravqilztuwa3lbsrr7dlpcer75y6p7ggd4ae2.py
# Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_106 => add_93, add_96, mul_150, mul_156, rsqrt_18, sub_18, var_mean_18
# x_109 => mul_157, sigmoid_24
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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
    tl.store(out_ptr1 + (x2), tmp15, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg46vu57a5fuvq73mzxviowh2genb75ph4ozkgcz6hu5gxxibvya.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
# x_111 => add_101, add_98, mul_158, mul_164, rsqrt_19, sub_19, var_mean_19
triton_poi_fused__native_batch_norm_legit_functional_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4syvygizou4osyaqohpzc3737tp5kd72vk7oona7mpk4eeylr4b.py
# Source Nodes: [x_114, x_se_24], Original ATen: [aten.mean, aten.silu]
# x_114 => mul_165, sigmoid_25
# x_se_24 => mean_6
triton_red_fused_mean_silu_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqnmon4zbzywnry5yv5l5sxixp4mcmdgsf4hefoyei2bqpargla.py
# Source Nodes: [x_114, x_se_24], Original ATen: [aten.mean, aten.silu]
# x_114 => mul_165, sigmoid_25
# x_se_24 => mean_6
triton_per_fused_mean_silu_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_94', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bmdlzlya4lzbdvnpfvqiqprk4hv3lveluasfyamlrhxagvnizw.py
# Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
# x_se_25 => convolution_32
# x_se_26 => mul_166, sigmoid_26
triton_poi_fused_convolution_silu_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_95', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6w5kgkvf3fudi5a2pzg7pqmjsxy4s5oin2x3maf6wj47iv4uwc4.py
# Source Nodes: [x_se_27], Original ATen: [aten.convolution]
# x_se_27 => convolution_33
triton_poi_fused_convolution_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_96', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/36/c365xstqxnkk42ivljwvu46vpfrsuviaztlyifpk7w6lz4gbli7u.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_114, x_115], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_27
# x_114 => mul_165, sigmoid_25
# x_115 => mul_167
triton_poi_fused_mul_sigmoid_silu_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 480
    x2 = (xindex // 94080)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (480*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/76/c765ilmv273cypkwgexypjvkggdn6i2ct2gn6rqvpayxxs6zvi3e.py
# Source Nodes: [shortcut_7, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_7 => add_107
# x_117 => add_103, add_106, mul_168, mul_174, rsqrt_20, sub_20, var_mean_20
triton_poi_fused__native_batch_norm_legit_functional_add_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawlfizzpru6wq4zchr3lmsez6xkpnsgehmvptgzfnfg35ft5b3l.py
# Source Nodes: [x_150], Original ATen: [aten.convolution]
# x_150 => convolution_44
triton_poi_fused_convolution_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdbgdkkcai76ygrtecr7i7453cwbauvjpotxmq453r7vogtyf7e.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1456
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 112)
    x0 = xindex % 112
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (112*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/ciohnl3zwshq6rqc2hn7tfkhiwuwsmu6fbg3kuraluwfu42ymvb6.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => add_135, add_136, add_137, mul_219, mul_220, mul_221, mul_222, mul_223, rsqrt_26, squeeze_79, var_mean_26
triton_per_fused__native_batch_norm_legit_functional_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_101', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 13
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhmjrubeipf22ienm2btsli4peqf6edrjujfhv5z55gytnazjwg.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => add_135, add_138, mul_218, mul_224, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5f/c5fzz26nc6icik53f7xq4o54ufe5gm5igxmoi56gqsg3x3gee57z.py
# Source Nodes: [x_155], Original ATen: [aten.convolution]
# x_155 => convolution_45
triton_poi_fused_convolution_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdxaqvm5vlssaiwfuy3uboxamvz26psadzapdwkxfjplxfhx5mx.py
# Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
# x_156 => var_mean_27
triton_red_fused__native_batch_norm_legit_functional_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8736
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 672)
    x0 = xindex % 672
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (672*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
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
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vwb7vkx5pgdo3car6cbflaiufwz7rsos4linecm3tnkh4hkkzf.py
# Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
# x_156 => add_140, add_141, add_142, mul_226, mul_227, mul_228, mul_229, mul_230, rsqrt_27, squeeze_82, var_mean_27
triton_per_fused__native_batch_norm_legit_functional_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_105', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 13
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
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


# kernel path: /tmp/torchinductor_youkaichao/uu/cuuqh4byowmnglal76dysrk7wvohsvo4662h5w7ib3nrfwlm6shf.py
# Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_156 => add_140, add_143, mul_225, mul_231, rsqrt_27, sub_27, var_mean_27
# x_159 => mul_232, sigmoid_36
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
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
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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
    tl.store(out_ptr1 + (x2), tmp15, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzugacekxhdalbmmi7gxxtkkhh6xdtjvvoikqm7gv6aia3eb4ny.py
# Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
# x_161 => add_145, add_148, mul_233, mul_239, rsqrt_28, sub_28, var_mean_28
triton_poi_fused__native_batch_norm_legit_functional_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
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
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chijf7lagulmeivbs3my7vztip7syqybyrn6tibob4ei3ifsgf2f.py
# Source Nodes: [x_164, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_164 => mul_240, sigmoid_37
# x_se_36 => mean_9
triton_red_fused_mean_silu_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawltpy2b3sescto62jrqrwalpinfdyvhmgbmm3fidhkuo56gjfc.py
# Source Nodes: [x_164, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_164 => mul_240, sigmoid_37
# x_se_36 => mean_9
triton_per_fused_mean_silu_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_109', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5txqgzuejt4m66isuvzhnwqd3vycyjmppuyldkfyeu7nvvs27c5.py
# Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
# x_se_37 => convolution_47
# x_se_38 => mul_241, sigmoid_38
triton_poi_fused_convolution_silu_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_110', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fp6begbvlxtuqx3q2vsojuqn6njhcdx4dawge5md3aqjmkgi75.py
# Source Nodes: [x_se_39], Original ATen: [aten.convolution]
# x_se_39 => convolution_48
triton_poi_fused_convolution_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_111', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bc/cbcq3vstchi2d5vaaqh7etxrw4zb44yd3kblrufmdkf7h2dc6dvz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_164, x_165], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_164 => mul_240, sigmoid_37
# x_165 => mul_242
triton_poi_fused_mul_sigmoid_silu_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 131712)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ui/cuixubqxoofqsloxutjy35nqq5ypaergv3h2xzbyd442dlymhab6.py
# Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_10 => add_154
# x_167 => add_150, add_153, mul_243, mul_249, rsqrt_29, sub_29, var_mean_29
triton_poi_fused__native_batch_norm_legit_functional_add_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwareyf77ak7xgfekmlpjbocmstojymbli36wfdm7uips4pv4stt.py
# Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_190 => add_172, add_175, mul_275, mul_281, rsqrt_33, sub_33, var_mean_33
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_114', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
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
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdz6hjludxxeu7ucheg2vegfsxfmzmxbfjdn4slk5y6eyj4bimmy.py
# Source Nodes: [x_193, x_195], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_193 => mul_282, sigmoid_44
# x_195 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_silu_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1553664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 11424) % 17
    x1 = (xindex // 672) % 17
    x3 = (xindex // 194208)
    x4 = xindex % 11424
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-10080) + x4 + (9408*x2) + (131712*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqmiagexndxokc2djpg7d7wk25z7yhjuxko3ii5zwzqguazcynt.py
# Source Nodes: [x_196], Original ATen: [aten.convolution]
# x_196 => convolution_56
triton_poi_fused_convolution_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_116', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdl6f4gnpmca57qpwkyljajvqspaixp66ivsy6zqa3bk2kg4yok.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => var_mean_34
triton_red_fused__native_batch_norm_legit_functional_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5joefqe23sx3dhzb3e2rjycgobsray5bswcrantcbeossoo6vdg.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => add_177, add_178, add_179, mul_284, mul_285, mul_286, mul_287, mul_288, rsqrt_34, squeeze_103, var_mean_34
triton_per_fused__native_batch_norm_legit_functional_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_118', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 4
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/a6/ca6gvw4pz4b3pvk7lt3ga4txzsa6lerxkaxke3pgbwvehnexfq2d.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => add_177, add_180, mul_283, mul_289, rsqrt_34, sub_34, var_mean_34
triton_poi_fused__native_batch_norm_legit_functional_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_119', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
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
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7id7rqhvny5jvoec6embnwjjsalzl6izwiufl3qroauselpjvw.py
# Source Nodes: [x_200, x_se_44], Original ATen: [aten.mean, aten.silu]
# x_200 => mul_290, sigmoid_45
# x_se_44 => mean_11
triton_per_fused_mean_silu_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_120', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casepita4zoq2pdwsxicmrpoxg5fawujlddgfbntpmvkrub475kr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200, x_201], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
# x_200 => mul_290, sigmoid_45
# x_201 => mul_292
triton_poi_fused_mul_sigmoid_silu_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_121', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 263424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxalhdkjdw26g254mgkneacoyc7ik44gfhfwqcgqpexyb7sfyn3z.py
# Source Nodes: [x_202], Original ATen: [aten.convolution]
# x_202 => convolution_59
triton_poi_fused_convolution_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_122', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4z5dhbjbuq2wzbdx3tkdcnswysalh2bno6qy2g34j7lviyygfzu.py
# Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
# x_203 => var_mean_35
triton_red_fused__native_batch_norm_legit_functional_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (18816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxxxytjy2k5jmehfcyh5dxkfpx7xnlgv52tqi3gb7pui5skeipy.py
# Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
# x_203 => add_182, add_183, add_184, mul_294, mul_295, mul_296, mul_297, mul_298, rsqrt_35, squeeze_106, var_mean_35
triton_per_fused__native_batch_norm_legit_functional_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_124', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/my/cmyi3kxqkqioy4lmlcn7msfvvei5xmgrtazy5pqduoaa7zhdrzap.py
# Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
# x_203 => add_182, add_185, mul_293, mul_299, rsqrt_35, sub_35, var_mean_35
triton_poi_fused__native_batch_norm_legit_functional_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_125', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2vc4e7czlcjd4ebqonuvnt4kamjn5jvfpomui22dm63muerzxt.py
# Source Nodes: [x_207], Original ATen: [aten.convolution]
# x_207 => convolution_60
triton_poi_fused_convolution_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_126', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxrp4krpmotdm7ebgetyu7vzgwjor5lvfbsmcyxv2brtebjxajh.py
# Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_functional]
# x_208 => var_mean_36
triton_red_fused__native_batch_norm_legit_functional_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (112896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/um/cumuqofpwaaf3rymtswestvqx7ylit7gilqa6cdwwlkhfixykizk.py
# Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_functional]
# x_208 => add_187, add_188, add_189, mul_301, mul_302, mul_303, mul_304, mul_305, rsqrt_36, squeeze_109, var_mean_36
triton_per_fused__native_batch_norm_legit_functional_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_128', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 4
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/bg/cbgupw5sfgsxj2bnbadcq5x6bxsmi3i65awdno4nfh5s5pybd4cu.py
# Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_208 => add_187, add_190, mul_300, mul_306, rsqrt_36, sub_36, var_mean_36
# x_211 => mul_307, sigmoid_48
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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
    tl.store(out_ptr1 + (x2), tmp15, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrhc4ixjvevs4wmtebyaivakvfcq2qgz6xeolbo4mxfpaeaoqgw.py
# Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
# x_213 => add_192, add_195, mul_308, mul_314, rsqrt_37, sub_37, var_mean_37
triton_poi_fused__native_batch_norm_legit_functional_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_130', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdpqiy5nkceh34jhizlh6iqycke4hmhepqciudrt5rewkzywuth.py
# Source Nodes: [x_216, x_se_48], Original ATen: [aten.mean, aten.silu]
# x_216 => mul_315, sigmoid_49
# x_se_48 => mean_12
triton_per_fused_mean_silu_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_131', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (56448*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v4/cv4cipqdcwyugy3argft4qjy4q4vzkt5hmjvcjqnfmcieiixibbw.py
# Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
# x_se_49 => convolution_62
# x_se_50 => mul_316, sigmoid_50
triton_poi_fused_convolution_silu_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_132', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtxppvxbmuuzxglt4hlzrnfdg2q2bhw2365wz5db5cpwgxjnipf.py
# Source Nodes: [x_se_51], Original ATen: [aten.convolution]
# x_se_51 => convolution_63
triton_poi_fused_convolution_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_133', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/76/c76aaqgc6wlhjqrdltemg4iyeb4tkevdmdmn7rykyjwdxz2obi5k.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_216, x_217], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_51
# x_216 => mul_315, sigmoid_49
# x_217 => mul_317
triton_poi_fused_mul_sigmoid_silu_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_134', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1152
    x2 = (xindex // 56448)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (1152*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzh3ezq7pc4lqkqs7rlnxayssskmm6dotrlsyoykapigem6w4rj.py
# Source Nodes: [shortcut_13, x_219], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_13 => add_201
# x_219 => add_197, add_200, mul_318, mul_324, rsqrt_38, sub_38, var_mean_38
triton_poi_fused__native_batch_norm_legit_functional_add_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_135', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdp7tgoigcqys74rrahfp36iubp2qi5eqfnhfvawt4u4m6wde2wl.py
# Source Nodes: [x_269], Original ATen: [aten.convolution]
# x_269 => convolution_79
triton_poi_fused_convolution_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqejdj2pehlqf4p22jg2l34somlsvmacnodovbehv75lkd42kyt.py
# Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
# x_270 => var_mean_47
triton_red_fused__native_batch_norm_legit_functional_137 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_137', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (31360*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ga/cgadj3fcf6fblvjfxvjxtchn7lhwlyv56fcir5o2sy2sommwuiye.py
# Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
# x_270 => add_245, add_246, add_247, mul_394, mul_395, mul_396, mul_397, mul_398, rsqrt_47, squeeze_142, var_mean_47
triton_per_fused__native_batch_norm_legit_functional_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_138', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 4
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4tdkhqt53xq6fzkjcsdfhl3bgwe6zn2gmzir2mubediffsh3ko.py
# Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
# x_270 => add_245, add_248, mul_393, mul_399, rsqrt_47, sub_47, var_mean_47
triton_poi_fused__native_batch_norm_legit_functional_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_139', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cda6yr73y3tu25cheei5ladzbbbqa7lif3ke6fuvjbx3eedrbhc3.py
# Source Nodes: [x_275], Original ATen: [aten.convolution]
# x_275 => convolution_80
triton_poi_fused_convolution_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_140', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (62720*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6w5ozlp22pazluhcziuhoja4yrpwxpzfjlzsuro7g2ocghf5v6a.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
# x_276 => var_mean_48
triton_red_fused__native_batch_norm_legit_functional_141 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_141', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ogiae5lio5dnxhvtf2gc6575g4hqqbolx25jgwreftkwezrv7m.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
# x_276 => add_250, add_251, add_252, mul_401, mul_402, mul_403, mul_404, mul_405, rsqrt_48, squeeze_145, var_mean_48
triton_per_fused__native_batch_norm_legit_functional_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_142', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
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
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqo24fbb3h3btgzjdbrlwbtnotjwxyquerbtdicwp5sszyke52b.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_276 => add_250, add_253, mul_400, mul_406, rsqrt_48, sub_48, var_mean_48
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_143', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
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
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/lm/clmdvtcedg4ifea5iiygy3emnzfx3e3koaexqn5df5biofsnbjw4.py
# Source Nodes: [x_280, x_281, x_283], Original ATen: [aten.mean, aten.silu, aten.view]
# x_280 => mul_407, sigmoid_64
# x_281 => mean_16
# x_283 => view
triton_per_fused_mean_silu_view_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_view_144', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (62720*x1)), rmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesit52fnvde24easfhdrpwxj4kwgkg6yw5uf37qsevz6s6onc6y.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_145', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (96, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (24, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_16, (144, ), (1, ))
    assert_size_stride(primals_17, (144, ), (1, ))
    assert_size_stride(primals_18, (144, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_20, (24, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_22, (144, ), (1, ))
    assert_size_stride(primals_23, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_24, (144, ), (1, ))
    assert_size_stride(primals_25, (144, ), (1, ))
    assert_size_stride(primals_26, (40, ), (1, ))
    assert_size_stride(primals_27, (40, ), (1, ))
    assert_size_stride(primals_28, (240, ), (1, ))
    assert_size_stride(primals_29, (240, ), (1, ))
    assert_size_stride(primals_30, (240, ), (1, ))
    assert_size_stride(primals_31, (240, ), (1, ))
    assert_size_stride(primals_32, (40, ), (1, ))
    assert_size_stride(primals_33, (40, ), (1, ))
    assert_size_stride(primals_34, (240, ), (1, ))
    assert_size_stride(primals_35, (240, ), (1, ))
    assert_size_stride(primals_36, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_38, (240, ), (1, ))
    assert_size_stride(primals_39, (80, ), (1, ))
    assert_size_stride(primals_40, (80, ), (1, ))
    assert_size_stride(primals_41, (480, ), (1, ))
    assert_size_stride(primals_42, (480, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_44, (480, ), (1, ))
    assert_size_stride(primals_45, (80, ), (1, ))
    assert_size_stride(primals_46, (80, ), (1, ))
    assert_size_stride(primals_47, (480, ), (1, ))
    assert_size_stride(primals_48, (480, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_50, (480, ), (1, ))
    assert_size_stride(primals_51, (80, ), (1, ))
    assert_size_stride(primals_52, (80, ), (1, ))
    assert_size_stride(primals_53, (480, ), (1, ))
    assert_size_stride(primals_54, (480, ), (1, ))
    assert_size_stride(primals_55, (480, ), (1, ))
    assert_size_stride(primals_56, (480, ), (1, ))
    assert_size_stride(primals_57, (112, ), (1, ))
    assert_size_stride(primals_58, (112, ), (1, ))
    assert_size_stride(primals_59, (672, ), (1, ))
    assert_size_stride(primals_60, (672, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_62, (672, ), (1, ))
    assert_size_stride(primals_63, (112, ), (1, ))
    assert_size_stride(primals_64, (112, ), (1, ))
    assert_size_stride(primals_65, (672, ), (1, ))
    assert_size_stride(primals_66, (672, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_68, (672, ), (1, ))
    assert_size_stride(primals_69, (112, ), (1, ))
    assert_size_stride(primals_70, (112, ), (1, ))
    assert_size_stride(primals_71, (672, ), (1, ))
    assert_size_stride(primals_72, (672, ), (1, ))
    assert_size_stride(primals_73, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_74, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_76, (192, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_78, (1152, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_80, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_82, (192, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_84, (1152, ), (1, ))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_86, (1152, ), (1, ))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_90, (1152, ), (1, ))
    assert_size_stride(primals_91, (1152, ), (1, ))
    assert_size_stride(primals_92, (1152, ), (1, ))
    assert_size_stride(primals_93, (1152, ), (1, ))
    assert_size_stride(primals_94, (192, ), (1, ))
    assert_size_stride(primals_95, (192, ), (1, ))
    assert_size_stride(primals_96, (1152, ), (1, ))
    assert_size_stride(primals_97, (1152, ), (1, ))
    assert_size_stride(primals_98, (1152, ), (1, ))
    assert_size_stride(primals_99, (1152, ), (1, ))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_101, (320, ), (1, ))
    assert_size_stride(primals_102, (1280, ), (1, ))
    assert_size_stride(primals_103, (1280, ), (1, ))
    assert_size_stride(primals_104, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_106, (8, ), (1, ))
    assert_size_stride(primals_107, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_110, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_111, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_112, (4, ), (1, ))
    assert_size_stride(primals_113, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_114, (96, ), (1, ))
    assert_size_stride(primals_115, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_116, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_117, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_119, (6, ), (1, ))
    assert_size_stride(primals_120, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_121, (144, ), (1, ))
    assert_size_stride(primals_122, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_123, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_124, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_125, (6, ), (1, ))
    assert_size_stride(primals_126, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_127, (144, ), (1, ))
    assert_size_stride(primals_128, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_129, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_130, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_132, (10, ), (1, ))
    assert_size_stride(primals_133, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_134, (240, ), (1, ))
    assert_size_stride(primals_135, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_136, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_137, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_138, (10, ), (1, ))
    assert_size_stride(primals_139, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_140, (240, ), (1, ))
    assert_size_stride(primals_141, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_142, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_143, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_144, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_145, (20, ), (1, ))
    assert_size_stride(primals_146, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_147, (480, ), (1, ))
    assert_size_stride(primals_148, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_149, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_150, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_151, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_152, (20, ), (1, ))
    assert_size_stride(primals_153, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_154, (480, ), (1, ))
    assert_size_stride(primals_155, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_156, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_157, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_158, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_159, (20, ), (1, ))
    assert_size_stride(primals_160, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_161, (480, ), (1, ))
    assert_size_stride(primals_162, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_164, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_165, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_166, (28, ), (1, ))
    assert_size_stride(primals_167, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_168, (672, ), (1, ))
    assert_size_stride(primals_169, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_170, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_171, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_172, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_173, (28, ), (1, ))
    assert_size_stride(primals_174, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_175, (672, ), (1, ))
    assert_size_stride(primals_176, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_177, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_178, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_179, (28, ), (1, ))
    assert_size_stride(primals_180, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_181, (672, ), (1, ))
    assert_size_stride(primals_182, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_183, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_184, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_186, (48, ), (1, ))
    assert_size_stride(primals_187, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_188, (1152, ), (1, ))
    assert_size_stride(primals_189, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_190, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_191, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_192, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_193, (48, ), (1, ))
    assert_size_stride(primals_194, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_195, (1152, ), (1, ))
    assert_size_stride(primals_196, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_197, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_199, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_200, (48, ), (1, ))
    assert_size_stride(primals_201, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_202, (1152, ), (1, ))
    assert_size_stride(primals_203, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_204, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_205, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_207, (48, ), (1, ))
    assert_size_stride(primals_208, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_209, (1152, ), (1, ))
    assert_size_stride(primals_210, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_211, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_212, (1000, 1280), (1280, 1))
    assert_size_stride(primals_213, (1000, ), (1, ))
    assert_size_stride(primals_214, (), ())
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (), ())
    assert_size_stride(primals_218, (32, ), (1, ))
    assert_size_stride(primals_219, (32, ), (1, ))
    assert_size_stride(primals_220, (), ())
    assert_size_stride(primals_221, (16, ), (1, ))
    assert_size_stride(primals_222, (16, ), (1, ))
    assert_size_stride(primals_223, (), ())
    assert_size_stride(primals_224, (96, ), (1, ))
    assert_size_stride(primals_225, (96, ), (1, ))
    assert_size_stride(primals_226, (), ())
    assert_size_stride(primals_227, (96, ), (1, ))
    assert_size_stride(primals_228, (96, ), (1, ))
    assert_size_stride(primals_229, (), ())
    assert_size_stride(primals_230, (24, ), (1, ))
    assert_size_stride(primals_231, (24, ), (1, ))
    assert_size_stride(primals_232, (), ())
    assert_size_stride(primals_233, (144, ), (1, ))
    assert_size_stride(primals_234, (144, ), (1, ))
    assert_size_stride(primals_235, (), ())
    assert_size_stride(primals_236, (144, ), (1, ))
    assert_size_stride(primals_237, (144, ), (1, ))
    assert_size_stride(primals_238, (), ())
    assert_size_stride(primals_239, (24, ), (1, ))
    assert_size_stride(primals_240, (24, ), (1, ))
    assert_size_stride(primals_241, (), ())
    assert_size_stride(primals_242, (144, ), (1, ))
    assert_size_stride(primals_243, (144, ), (1, ))
    assert_size_stride(primals_244, (), ())
    assert_size_stride(primals_245, (144, ), (1, ))
    assert_size_stride(primals_246, (144, ), (1, ))
    assert_size_stride(primals_247, (), ())
    assert_size_stride(primals_248, (40, ), (1, ))
    assert_size_stride(primals_249, (40, ), (1, ))
    assert_size_stride(primals_250, (), ())
    assert_size_stride(primals_251, (240, ), (1, ))
    assert_size_stride(primals_252, (240, ), (1, ))
    assert_size_stride(primals_253, (), ())
    assert_size_stride(primals_254, (240, ), (1, ))
    assert_size_stride(primals_255, (240, ), (1, ))
    assert_size_stride(primals_256, (), ())
    assert_size_stride(primals_257, (40, ), (1, ))
    assert_size_stride(primals_258, (40, ), (1, ))
    assert_size_stride(primals_259, (), ())
    assert_size_stride(primals_260, (240, ), (1, ))
    assert_size_stride(primals_261, (240, ), (1, ))
    assert_size_stride(primals_262, (), ())
    assert_size_stride(primals_263, (240, ), (1, ))
    assert_size_stride(primals_264, (240, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (80, ), (1, ))
    assert_size_stride(primals_267, (80, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_270, (480, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (480, ), (1, ))
    assert_size_stride(primals_273, (480, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (80, ), (1, ))
    assert_size_stride(primals_276, (80, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (480, ), (1, ))
    assert_size_stride(primals_279, (480, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (480, ), (1, ))
    assert_size_stride(primals_282, (480, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (80, ), (1, ))
    assert_size_stride(primals_285, (80, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (480, ), (1, ))
    assert_size_stride(primals_288, (480, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (480, ), (1, ))
    assert_size_stride(primals_291, (480, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (112, ), (1, ))
    assert_size_stride(primals_294, (112, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (672, ), (1, ))
    assert_size_stride(primals_297, (672, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (672, ), (1, ))
    assert_size_stride(primals_300, (672, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (112, ), (1, ))
    assert_size_stride(primals_303, (112, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (672, ), (1, ))
    assert_size_stride(primals_306, (672, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (672, ), (1, ))
    assert_size_stride(primals_309, (672, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (112, ), (1, ))
    assert_size_stride(primals_312, (112, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (672, ), (1, ))
    assert_size_stride(primals_315, (672, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (672, ), (1, ))
    assert_size_stride(primals_318, (672, ), (1, ))
    assert_size_stride(primals_319, (), ())
    assert_size_stride(primals_320, (192, ), (1, ))
    assert_size_stride(primals_321, (192, ), (1, ))
    assert_size_stride(primals_322, (), ())
    assert_size_stride(primals_323, (1152, ), (1, ))
    assert_size_stride(primals_324, (1152, ), (1, ))
    assert_size_stride(primals_325, (), ())
    assert_size_stride(primals_326, (1152, ), (1, ))
    assert_size_stride(primals_327, (1152, ), (1, ))
    assert_size_stride(primals_328, (), ())
    assert_size_stride(primals_329, (192, ), (1, ))
    assert_size_stride(primals_330, (192, ), (1, ))
    assert_size_stride(primals_331, (), ())
    assert_size_stride(primals_332, (1152, ), (1, ))
    assert_size_stride(primals_333, (1152, ), (1, ))
    assert_size_stride(primals_334, (), ())
    assert_size_stride(primals_335, (1152, ), (1, ))
    assert_size_stride(primals_336, (1152, ), (1, ))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (192, ), (1, ))
    assert_size_stride(primals_339, (192, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (1152, ), (1, ))
    assert_size_stride(primals_342, (1152, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (1152, ), (1, ))
    assert_size_stride(primals_345, (1152, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (192, ), (1, ))
    assert_size_stride(primals_348, (192, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (1152, ), (1, ))
    assert_size_stride(primals_351, (1152, ), (1, ))
    assert_size_stride(primals_352, (), ())
    assert_size_stride(primals_353, (1152, ), (1, ))
    assert_size_stride(primals_354, (1152, ), (1, ))
    assert_size_stride(primals_355, (), ())
    assert_size_stride(primals_356, (320, ), (1, ))
    assert_size_stride(primals_357, (320, ), (1, ))
    assert_size_stride(primals_358, (), ())
    assert_size_stride(primals_359, (1280, ), (1, ))
    assert_size_stride(primals_360, (1280, ), (1, ))
    assert_size_stride(primals_361, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_1.run(primals_361, buf1, 24, 50625, grid=grid(24, 50625), stream=stream0)
        del primals_361
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf3 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf4 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf3, buf4, buf5, buf6, 25088, 128, grid=grid(25088), stream=stream0)
        buf7 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf4, buf5, buf6, buf7, buf8, buf9, 224, 112, grid=grid(224), stream=stream0)
        buf10 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf13 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_215, primals_216, buf10, buf11, buf13, primals_215, primals_216, 32, 7, grid=grid(32), stream=stream0)
        del primals_215
        del primals_216
        buf15 = reinterpret_tensor(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        buf696 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_6.run(buf3, buf10, buf11, primals_2, primals_3, buf15, buf696, 3211264, grid=grid(3211264), stream=stream0)
        del primals_3
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf16, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf17 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf16, buf17, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf18 = buf6; del buf6  # reuse
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf17, buf18, buf19, buf20, 25088, 128, grid=grid(25088), stream=stream0)
        buf21 = buf9; del buf9  # reuse
        buf22 = buf8; del buf8  # reuse
        buf23 = buf7; del buf7  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf18, buf19, buf20, buf21, buf22, buf23, 224, 112, grid=grid(224), stream=stream0)
        del buf18
        del buf19
        buf24 = buf11; del buf11  # reuse
        buf25 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf27 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf21, buf22, buf23, primals_218, primals_219, buf24, buf25, buf27, primals_218, primals_219, 32, 7, grid=grid(32), stream=stream0)
        del primals_218
        del primals_219
        buf28 = reinterpret_tensor(buf16, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf16  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_7.run(buf17, buf24, buf25, primals_4, primals_5, buf28, 3211264, grid=grid(3211264), stream=stream0)
        del primals_5
        buf29 = reinterpret_tensor(buf20, (8, 32, 1, 1, 98), (3136, 1, 25088, 25088, 32), 0); del buf20  # reuse
        # Source Nodes: [x_10, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_8.run(buf28, buf29, 25088, 128, grid=grid(25088), stream=stream0)
        buf30 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf31 = reinterpret_tensor(buf30, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf30  # reuse
        # Source Nodes: [x_10, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_9.run(buf31, buf29, 256, 98, grid=grid(256), stream=stream0)
        del buf29
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 8, 1, 1), (8, 1, 1, 1))
        buf33 = reinterpret_tensor(buf32, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf32  # reuse
        buf34 = empty_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_10.run(buf33, primals_106, buf34, 64, grid=grid(64), stream=stream0)
        del primals_106
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 32, 1, 1), (32, 1, 1, 1))
        buf36 = reinterpret_tensor(buf35, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf35  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(buf36, primals_108, 256, grid=grid(256), stream=stream0)
        del primals_108
        buf37 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_12.run(buf28, buf36, buf37, 3211264, grid=grid(3211264), stream=stream0)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf39 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf38, buf39, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf40 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf42 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf39, buf40, buf41, buf42, 12544, 128, grid=grid(12544), stream=stream0)
        buf43 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf40, buf41, buf42, buf43, buf44, buf45, 112, 112, grid=grid(112), stream=stream0)
        del buf40
        del buf41
        del buf42
        buf46 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf49 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf43, buf44, buf45, primals_221, primals_222, buf46, buf47, buf49, primals_221, primals_222, 16, 7, grid=grid(16), stream=stream0)
        del primals_221
        del primals_222
        buf50 = reinterpret_tensor(buf38, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf38  # reuse
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_17.run(buf39, buf46, buf47, primals_6, primals_7, buf50, 1605632, grid=grid(1605632), stream=stream0)
        del buf47
        del primals_7
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        buf52 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf51, buf52, 768, 12544, grid=grid(768, 12544), stream=stream0)
        buf53 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf52, buf53, buf54, buf55, 75264, 128, grid=grid(75264), stream=stream0)
        buf56 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf53, buf54, buf55, buf56, buf57, buf58, 672, 112, grid=grid(672), stream=stream0)
        buf59 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf62 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf56, buf57, buf58, primals_224, primals_225, buf59, buf60, buf62, primals_224, primals_225, 96, 7, grid=grid(96), stream=stream0)
        del primals_224
        del primals_225
        buf63 = reinterpret_tensor(buf51, (8, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf51  # reuse
        buf695 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_22.run(buf52, buf59, buf60, primals_8, primals_9, buf63, buf695, 9633792, grid=grid(9633792), stream=stream0)
        del primals_9
        buf64 = empty_strided((8, 96, 113, 113), (1225824, 1, 10848, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_23], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_23.run(buf63, buf64, 9806592, grid=grid(9806592), stream=stream0)
        del buf63
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_10, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf65, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf66 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf65, buf66, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf67 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf66, buf67, buf68, buf69, 18816, 128, grid=grid(18816), stream=stream0)
        buf70 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf67, buf68, buf69, buf70, buf71, buf72, 192, 98, grid=grid(192), stream=stream0)
        del buf67
        del buf68
        del buf69
        buf73 = buf60; del buf60  # reuse
        buf74 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf76 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf70, buf71, buf72, primals_227, primals_228, buf73, buf74, buf76, primals_227, primals_228, 96, 2, grid=grid(96), stream=stream0)
        del primals_227
        del primals_228
        buf77 = reinterpret_tensor(buf65, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf65  # reuse
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_28.run(buf66, buf73, buf74, primals_11, primals_12, buf77, 2408448, grid=grid(2408448), stream=stream0)
        del buf74
        del primals_12
        buf78 = empty_strided((8, 96, 1, 1, 25), (2400, 1, 19200, 19200, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_29.run(buf77, buf78, 19200, 126, grid=grid(19200), stream=stream0)
        buf79 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf80 = reinterpret_tensor(buf79, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf79  # reuse
        # Source Nodes: [x_28, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_30.run(buf80, buf78, 768, 25, grid=grid(768), stream=stream0)
        del buf78
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 4, 1, 1), (4, 1, 1, 1))
        buf82 = reinterpret_tensor(buf81, (8, 4, 1, 1), (4, 1, 4, 4), 0); del buf81  # reuse
        buf83 = reinterpret_tensor(buf25, (8, 4, 1, 1), (4, 1, 4, 4), 0); del buf25  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_31.run(buf82, primals_112, buf83, 32, grid=grid(32), stream=stream0)
        del primals_112
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 96, 1, 1), (96, 1, 1, 1))
        buf85 = reinterpret_tensor(buf84, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf84  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf85, primals_114, 768, grid=grid(768), stream=stream0)
        del primals_114
        buf86 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28, x_29], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_33.run(buf77, buf85, buf86, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf88 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf87, buf88, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf89 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf88, buf89, buf90, buf91, 4704, 128, grid=grid(4704), stream=stream0)
        buf92 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf89, buf90, buf91, buf92, buf93, buf94, 48, 98, grid=grid(48), stream=stream0)
        buf95 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf96 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf98 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf92, buf93, buf94, primals_230, primals_231, buf95, buf96, buf98, primals_230, primals_231, 24, 2, grid=grid(24), stream=stream0)
        del primals_230
        del primals_231
        buf99 = reinterpret_tensor(buf87, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf87  # reuse
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_38.run(buf88, buf95, buf96, primals_13, primals_14, buf99, 602112, grid=grid(602112), stream=stream0)
        del primals_14
        # Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf101 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf100, buf101, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf102 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf101, buf102, buf103, buf104, 28224, 128, grid=grid(28224), stream=stream0)
        buf105 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf106 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf107 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf102, buf103, buf104, buf105, buf106, buf107, 288, 98, grid=grid(288), stream=stream0)
        buf108 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf111 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf105, buf106, buf107, primals_233, primals_234, buf108, buf109, buf111, primals_233, primals_234, 144, 2, grid=grid(144), stream=stream0)
        del primals_233
        del primals_234
        buf113 = reinterpret_tensor(buf100, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf100  # reuse
        buf694 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_43.run(buf101, buf108, buf109, primals_15, primals_16, buf113, buf694, 3612672, grid=grid(3612672), stream=stream0)
        del primals_16
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf114, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf115 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf114, buf115, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf116 = buf104; del buf104  # reuse
        buf117 = buf103; del buf103  # reuse
        buf118 = buf102; del buf102  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf115, buf116, buf117, buf118, 28224, 128, grid=grid(28224), stream=stream0)
        buf119 = buf107; del buf107  # reuse
        buf120 = buf106; del buf106  # reuse
        buf121 = buf105; del buf105  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf116, buf117, buf118, buf119, buf120, buf121, 288, 98, grid=grid(288), stream=stream0)
        buf122 = buf109; del buf109  # reuse
        buf123 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf125 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf119, buf120, buf121, primals_236, primals_237, buf122, buf123, buf125, primals_236, primals_237, 144, 2, grid=grid(144), stream=stream0)
        del primals_236
        del primals_237
        buf126 = reinterpret_tensor(buf114, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf114  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_44.run(buf115, buf122, buf123, primals_17, primals_18, buf126, 3612672, grid=grid(3612672), stream=stream0)
        del primals_18
        buf127 = empty_strided((8, 144, 1, 1, 25), (3600, 1, 28800, 28800, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_45.run(buf126, buf127, 28800, 126, grid=grid(28800), stream=stream0)
        buf128 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf129 = reinterpret_tensor(buf128, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf128  # reuse
        # Source Nodes: [x_44, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_46.run(buf129, buf127, 1152, 25, grid=grid(1152), stream=stream0)
        del buf127
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 6, 1, 1), (6, 1, 1, 1))
        buf131 = reinterpret_tensor(buf130, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf130  # reuse
        buf132 = reinterpret_tensor(buf94, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf94  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_47.run(buf131, primals_119, buf132, 48, grid=grid(48), stream=stream0)
        del primals_119
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 144, 1, 1), (144, 1, 1, 1))
        buf134 = reinterpret_tensor(buf133, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf133  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf134, primals_121, 1152, grid=grid(1152), stream=stream0)
        del primals_121
        buf135 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44, x_45], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_49.run(buf126, buf134, buf135, 3612672, grid=grid(3612672), stream=stream0)
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf137 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf136, buf137, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf138 = buf91; del buf91  # reuse
        buf139 = buf90; del buf90  # reuse
        buf140 = buf89; del buf89  # reuse
        # Source Nodes: [x_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf137, buf138, buf139, buf140, 4704, 128, grid=grid(4704), stream=stream0)
        buf141 = buf93; del buf93  # reuse
        buf142 = buf92; del buf92  # reuse
        buf143 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf138, buf139, buf140, buf141, buf142, buf143, 48, 98, grid=grid(48), stream=stream0)
        del buf138
        del buf139
        del buf140
        buf144 = buf96; del buf96  # reuse
        buf145 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf147 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf141, buf142, buf143, primals_239, primals_240, buf144, buf145, buf147, primals_239, primals_240, 24, 2, grid=grid(24), stream=stream0)
        del buf141
        del buf142
        del primals_239
        del primals_240
        buf148 = reinterpret_tensor(buf136, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf136  # reuse
        # Source Nodes: [shortcut_3, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_50.run(buf137, buf144, buf145, primals_19, primals_20, buf99, buf148, 602112, grid=grid(602112), stream=stream0)
        del buf145
        del primals_20
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf150 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf149, buf150, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf151 = buf118; del buf118  # reuse
        buf152 = buf117; del buf117  # reuse
        buf153 = buf116; del buf116  # reuse
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf150, buf151, buf152, buf153, 28224, 128, grid=grid(28224), stream=stream0)
        buf154 = buf121; del buf121  # reuse
        buf155 = buf120; del buf120  # reuse
        buf156 = buf119; del buf119  # reuse
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf151, buf152, buf153, buf154, buf155, buf156, 288, 98, grid=grid(288), stream=stream0)
        del buf151
        del buf152
        del buf153
        buf157 = buf123; del buf123  # reuse
        buf158 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf160 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf154, buf155, buf156, primals_242, primals_243, buf157, buf158, buf160, primals_242, primals_243, 144, 2, grid=grid(144), stream=stream0)
        del buf154
        del buf155
        del buf156
        del primals_242
        del primals_243
        buf161 = reinterpret_tensor(buf149, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf149  # reuse
        buf693 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_51.run(buf150, buf157, buf158, primals_21, primals_22, buf161, buf693, 3612672, grid=grid(3612672), stream=stream0)
        del primals_22
        buf162 = empty_strided((8, 144, 59, 59), (501264, 1, 8496, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56, x_58], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_52.run(buf161, buf162, 4010112, grid=grid(4010112), stream=stream0)
        del buf161
        # Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_23, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf163, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf164 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf163, buf164, 1152, 784, grid=grid(1152, 784), stream=stream0)
        buf165 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        buf166 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf164, buf165, buf166, buf167, 7056, 128, grid=grid(7056), stream=stream0)
        buf168 = buf158; del buf158  # reuse
        buf169 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf171 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_55.run(buf165, buf166, buf167, primals_245, primals_246, buf168, buf169, buf171, primals_245, primals_246, 144, 49, grid=grid(144), stream=stream0)
        del buf165
        del buf166
        del buf167
        del primals_245
        del primals_246
        buf172 = reinterpret_tensor(buf163, (8, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf163  # reuse
        # Source Nodes: [x_60], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_56.run(buf164, buf168, buf169, primals_24, primals_25, buf172, 903168, grid=grid(903168), stream=stream0)
        del buf169
        del primals_25
        buf173 = empty_strided((8, 144, 1, 1, 7), (1008, 1, 8064, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_57.run(buf172, buf173, 8064, 112, grid=grid(8064), stream=stream0)
        buf174 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf175 = reinterpret_tensor(buf174, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf174  # reuse
        # Source Nodes: [x_63, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_58.run(buf175, buf173, 1152, 7, grid=grid(1152), stream=stream0)
        del buf173
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 6, 1, 1), (6, 1, 1, 1))
        buf177 = reinterpret_tensor(buf176, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf176  # reuse
        buf178 = reinterpret_tensor(buf143, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf143  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_47.run(buf177, primals_125, buf178, 48, grid=grid(48), stream=stream0)
        del primals_125
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 144, 1, 1), (144, 1, 1, 1))
        buf180 = reinterpret_tensor(buf179, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf179  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf180, primals_127, 1152, grid=grid(1152), stream=stream0)
        del primals_127
        buf181 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63, x_64], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_59.run(buf172, buf180, buf181, 903168, grid=grid(903168), stream=stream0)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf183 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf182, buf183, 320, 784, grid=grid(320, 784), stream=stream0)
        buf184 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf185 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf186 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf183, buf184, buf185, buf186, 1960, 128, grid=grid(1960), stream=stream0)
        buf187 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf190 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf184, buf185, buf186, primals_248, primals_249, buf187, buf188, buf190, primals_248, primals_249, 40, 49, grid=grid(40), stream=stream0)
        del primals_248
        del primals_249
        buf191 = reinterpret_tensor(buf182, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf182  # reuse
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_63.run(buf183, buf187, buf188, primals_26, primals_27, buf191, 250880, grid=grid(250880), stream=stream0)
        del primals_27
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 240, 28, 28), (188160, 784, 28, 1))
        buf193 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf192, buf193, 1920, 784, grid=grid(1920, 784), stream=stream0)
        buf194 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf193, buf194, buf195, buf196, 11760, 128, grid=grid(11760), stream=stream0)
        buf197 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf200 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf194, buf195, buf196, primals_251, primals_252, buf197, buf198, buf200, primals_251, primals_252, 240, 49, grid=grid(240), stream=stream0)
        del primals_251
        del primals_252
        buf202 = reinterpret_tensor(buf192, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf192  # reuse
        buf692 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_67.run(buf193, buf197, buf198, primals_28, primals_29, buf202, buf692, 1505280, grid=grid(1505280), stream=stream0)
        del primals_29
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_130, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf203, (8, 240, 28, 28), (188160, 784, 28, 1))
        buf204 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf203, buf204, 1920, 784, grid=grid(1920, 784), stream=stream0)
        buf205 = buf196; del buf196  # reuse
        buf206 = buf195; del buf195  # reuse
        buf207 = buf194; del buf194  # reuse
        # Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf204, buf205, buf206, buf207, 11760, 128, grid=grid(11760), stream=stream0)
        buf208 = buf198; del buf198  # reuse
        buf209 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf211 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf205, buf206, buf207, primals_254, primals_255, buf208, buf209, buf211, primals_254, primals_255, 240, 49, grid=grid(240), stream=stream0)
        del primals_254
        del primals_255
        buf212 = reinterpret_tensor(buf203, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf203  # reuse
        # Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_68.run(buf204, buf208, buf209, primals_30, primals_31, buf212, 1505280, grid=grid(1505280), stream=stream0)
        del primals_31
        buf213 = empty_strided((8, 240, 1, 1, 7), (1680, 1, 13440, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_69.run(buf212, buf213, 13440, 112, grid=grid(13440), stream=stream0)
        buf214 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf215 = reinterpret_tensor(buf214, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf214  # reuse
        # Source Nodes: [x_79, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_70.run(buf215, buf213, 1920, 7, grid=grid(1920), stream=stream0)
        del buf213
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 10, 1, 1), (10, 1, 1, 1))
        buf217 = reinterpret_tensor(buf216, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf216  # reuse
        buf218 = empty_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_71.run(buf217, primals_132, buf218, 80, grid=grid(80), stream=stream0)
        del primals_132
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 240, 1, 1), (240, 1, 1, 1))
        buf220 = reinterpret_tensor(buf219, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf219  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf220, primals_134, 1920, grid=grid(1920), stream=stream0)
        del primals_134
        buf221 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79, x_80], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_73.run(buf212, buf220, buf221, 1505280, grid=grid(1505280), stream=stream0)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf223 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf222, buf223, 320, 784, grid=grid(320, 784), stream=stream0)
        buf224 = buf186; del buf186  # reuse
        buf225 = buf185; del buf185  # reuse
        buf226 = buf184; del buf184  # reuse
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf223, buf224, buf225, buf226, 1960, 128, grid=grid(1960), stream=stream0)
        buf227 = buf188; del buf188  # reuse
        buf228 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf230 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf224, buf225, buf226, primals_257, primals_258, buf227, buf228, buf230, primals_257, primals_258, 40, 49, grid=grid(40), stream=stream0)
        del buf224
        del buf225
        del buf226
        del primals_257
        del primals_258
        buf231 = reinterpret_tensor(buf222, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf222  # reuse
        # Source Nodes: [shortcut_5, x_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_74.run(buf223, buf227, buf228, primals_32, primals_33, buf191, buf231, 250880, grid=grid(250880), stream=stream0)
        del buf228
        del primals_33
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 240, 28, 28), (188160, 784, 28, 1))
        buf233 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf232, buf233, 1920, 784, grid=grid(1920, 784), stream=stream0)
        buf234 = buf207; del buf207  # reuse
        buf235 = buf206; del buf206  # reuse
        buf236 = buf205; del buf205  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf233, buf234, buf235, buf236, 11760, 128, grid=grid(11760), stream=stream0)
        buf237 = buf209; del buf209  # reuse
        buf238 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf240 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf234, buf235, buf236, primals_260, primals_261, buf237, buf238, buf240, primals_260, primals_261, 240, 49, grid=grid(240), stream=stream0)
        del buf234
        del buf235
        del buf236
        del primals_260
        del primals_261
        buf241 = reinterpret_tensor(buf232, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf232  # reuse
        buf691 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_75.run(buf233, buf237, buf238, primals_34, primals_35, buf241, buf691, 1505280, grid=grid(1505280), stream=stream0)
        del primals_35
        buf242 = empty_strided((8, 240, 29, 29), (201840, 1, 6960, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91, x_93], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_76.run(buf241, buf242, 1614720, grid=grid(1614720), stream=stream0)
        del buf241
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_36, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf243, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf244 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf243, buf244, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf245 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf244, buf245, buf246, buf247, 3120, 121, grid=grid(3120), stream=stream0)
        buf248 = buf238; del buf238  # reuse
        buf249 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf251 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_79.run(buf245, buf246, buf247, primals_263, primals_264, buf248, buf249, buf251, primals_263, primals_264, 240, 13, grid=grid(240), stream=stream0)
        del buf245
        del buf246
        del buf247
        del primals_263
        del primals_264
        buf252 = reinterpret_tensor(buf243, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf243  # reuse
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_80.run(buf244, buf248, buf249, primals_37, primals_38, buf252, 376320, grid=grid(376320), stream=stream0)
        del buf249
        del primals_38
        buf253 = empty_strided((8, 240, 1, 1, 2), (480, 1, 3840, 3840, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_81.run(buf252, buf253, 3840, 98, grid=grid(3840), stream=stream0)
        buf254 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf255 = reinterpret_tensor(buf254, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf254  # reuse
        # Source Nodes: [x_98, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_82.run(buf255, buf253, 1920, 2, grid=grid(1920), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 10, 1, 1), (10, 1, 1, 1))
        buf257 = reinterpret_tensor(buf256, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf256  # reuse
        buf258 = empty_strided((8, 10, 1, 1), (10, 1, 10, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_71.run(buf257, primals_138, buf258, 80, grid=grid(80), stream=stream0)
        del primals_138
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 240, 1, 1), (240, 1, 1, 1))
        buf260 = reinterpret_tensor(buf259, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf259  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf260, primals_140, 1920, grid=grid(1920), stream=stream0)
        del primals_140
        buf261 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_98, x_99], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_83.run(buf252, buf260, buf261, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf263 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf262, buf263, 640, 196, grid=grid(640, 196), stream=stream0)
        buf264 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf266 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_85.run(buf263, buf264, buf265, buf266, 1040, 121, grid=grid(1040), stream=stream0)
        buf267 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf268 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf270 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_86.run(buf264, buf265, buf266, primals_266, primals_267, buf267, buf268, buf270, primals_266, primals_267, 80, 13, grid=grid(80), stream=stream0)
        del primals_266
        del primals_267
        buf271 = reinterpret_tensor(buf262, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf262  # reuse
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_87.run(buf263, buf267, buf268, primals_39, primals_40, buf271, 125440, grid=grid(125440), stream=stream0)
        del primals_40
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf273 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf272, buf273, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf274 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf276 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf273, buf274, buf275, buf276, 6240, 121, grid=grid(6240), stream=stream0)
        buf277 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf278 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf280 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf274, buf275, buf276, primals_269, primals_270, buf277, buf278, buf280, primals_269, primals_270, 480, 13, grid=grid(480), stream=stream0)
        del primals_269
        del primals_270
        buf282 = reinterpret_tensor(buf272, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf272  # reuse
        buf690 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91.run(buf273, buf277, buf278, primals_41, primals_42, buf282, buf690, 752640, grid=grid(752640), stream=stream0)
        del primals_42
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf283, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf284 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf283, buf284, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf285 = buf276; del buf276  # reuse
        buf286 = buf275; del buf275  # reuse
        buf287 = buf274; del buf274  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf284, buf285, buf286, buf287, 6240, 121, grid=grid(6240), stream=stream0)
        buf288 = buf278; del buf278  # reuse
        buf289 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf291 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf285, buf286, buf287, primals_272, primals_273, buf288, buf289, buf291, primals_272, primals_273, 480, 13, grid=grid(480), stream=stream0)
        del primals_272
        del primals_273
        buf292 = reinterpret_tensor(buf283, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf283  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf284, buf288, buf289, primals_43, primals_44, buf292, 752640, grid=grid(752640), stream=stream0)
        del primals_44
        buf293 = empty_strided((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_93.run(buf292, buf293, 7680, 98, grid=grid(7680), stream=stream0)
        buf294 = reinterpret_tensor(buf253, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf253  # reuse
        buf295 = reinterpret_tensor(buf294, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf294  # reuse
        # Source Nodes: [x_114, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_94.run(buf295, buf293, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 20, 1, 1), (20, 1, 1, 1))
        buf297 = reinterpret_tensor(buf296, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf296  # reuse
        buf298 = empty_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_95.run(buf297, primals_145, buf298, 160, grid=grid(160), stream=stream0)
        del primals_145
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 480, 1, 1), (480, 1, 1, 1))
        buf300 = reinterpret_tensor(buf299, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf299  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf300, primals_147, 3840, grid=grid(3840), stream=stream0)
        del primals_147
        buf301 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_114, x_115], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_97.run(buf292, buf300, buf301, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf303 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf302, buf303, 640, 196, grid=grid(640, 196), stream=stream0)
        buf304 = buf266; del buf266  # reuse
        buf305 = buf265; del buf265  # reuse
        buf306 = buf264; del buf264  # reuse
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_85.run(buf303, buf304, buf305, buf306, 1040, 121, grid=grid(1040), stream=stream0)
        buf307 = buf268; del buf268  # reuse
        buf308 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf310 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_86.run(buf304, buf305, buf306, primals_275, primals_276, buf307, buf308, buf310, primals_275, primals_276, 80, 13, grid=grid(80), stream=stream0)
        del primals_275
        del primals_276
        buf311 = reinterpret_tensor(buf302, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf302  # reuse
        # Source Nodes: [shortcut_7, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_98.run(buf303, buf307, buf308, primals_45, primals_46, buf271, buf311, 125440, grid=grid(125440), stream=stream0)
        del primals_46
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf313 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf312, buf313, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf314 = buf287; del buf287  # reuse
        buf315 = buf286; del buf286  # reuse
        buf316 = buf285; del buf285  # reuse
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf313, buf314, buf315, buf316, 6240, 121, grid=grid(6240), stream=stream0)
        buf317 = buf289; del buf289  # reuse
        buf318 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf320 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf314, buf315, buf316, primals_278, primals_279, buf317, buf318, buf320, primals_278, primals_279, 480, 13, grid=grid(480), stream=stream0)
        del primals_278
        del primals_279
        buf322 = reinterpret_tensor(buf312, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf312  # reuse
        buf689 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123, x_126], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91.run(buf313, buf317, buf318, primals_47, primals_48, buf322, buf689, 752640, grid=grid(752640), stream=stream0)
        del primals_48
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf323, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf324 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf323, buf324, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf325 = buf316; del buf316  # reuse
        buf326 = buf315; del buf315  # reuse
        buf327 = buf314; del buf314  # reuse
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf324, buf325, buf326, buf327, 6240, 121, grid=grid(6240), stream=stream0)
        buf328 = buf318; del buf318  # reuse
        buf329 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf331 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf325, buf326, buf327, primals_281, primals_282, buf328, buf329, buf331, primals_281, primals_282, 480, 13, grid=grid(480), stream=stream0)
        del primals_281
        del primals_282
        buf332 = reinterpret_tensor(buf323, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf323  # reuse
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf324, buf328, buf329, primals_49, primals_50, buf332, 752640, grid=grid(752640), stream=stream0)
        del primals_50
        buf333 = buf293; del buf293  # reuse
        # Source Nodes: [x_131, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_93.run(buf332, buf333, 7680, 98, grid=grid(7680), stream=stream0)
        buf334 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf335 = reinterpret_tensor(buf334, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf334  # reuse
        # Source Nodes: [x_131, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_94.run(buf335, buf333, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 20, 1, 1), (20, 1, 1, 1))
        buf337 = reinterpret_tensor(buf336, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf336  # reuse
        buf338 = empty_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_95.run(buf337, primals_152, buf338, 160, grid=grid(160), stream=stream0)
        del primals_152
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (8, 480, 1, 1), (480, 1, 1, 1))
        buf340 = reinterpret_tensor(buf339, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf339  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf340, primals_154, 3840, grid=grid(3840), stream=stream0)
        del primals_154
        buf341 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_131, x_132], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_97.run(buf332, buf340, buf341, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf343 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_84.run(buf342, buf343, 640, 196, grid=grid(640, 196), stream=stream0)
        buf344 = buf306; del buf306  # reuse
        buf345 = buf305; del buf305  # reuse
        buf346 = buf304; del buf304  # reuse
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_85.run(buf343, buf344, buf345, buf346, 1040, 121, grid=grid(1040), stream=stream0)
        buf347 = buf308; del buf308  # reuse
        buf348 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf350 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_86.run(buf344, buf345, buf346, primals_284, primals_285, buf347, buf348, buf350, primals_284, primals_285, 80, 13, grid=grid(80), stream=stream0)
        del buf344
        del buf345
        del buf346
        del primals_284
        del primals_285
        buf351 = reinterpret_tensor(buf342, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf342  # reuse
        # Source Nodes: [shortcut_8, x_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_98.run(buf343, buf347, buf348, primals_51, primals_52, buf311, buf351, 125440, grid=grid(125440), stream=stream0)
        del buf348
        del primals_52
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf353 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf352, buf353, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf354 = buf327; del buf327  # reuse
        buf355 = buf326; del buf326  # reuse
        buf356 = buf325; del buf325  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf353, buf354, buf355, buf356, 6240, 121, grid=grid(6240), stream=stream0)
        buf357 = buf329; del buf329  # reuse
        buf358 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf360 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf354, buf355, buf356, primals_287, primals_288, buf357, buf358, buf360, primals_287, primals_288, 480, 13, grid=grid(480), stream=stream0)
        del primals_287
        del primals_288
        buf362 = reinterpret_tensor(buf352, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf352  # reuse
        buf688 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_91.run(buf353, buf357, buf358, primals_53, primals_54, buf362, buf688, 752640, grid=grid(752640), stream=stream0)
        del primals_54
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_157, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf363, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf364 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_88.run(buf363, buf364, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf365 = buf356; del buf356  # reuse
        buf366 = buf355; del buf355  # reuse
        buf367 = buf354; del buf354  # reuse
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_89.run(buf364, buf365, buf366, buf367, 6240, 121, grid=grid(6240), stream=stream0)
        buf368 = buf358; del buf358  # reuse
        buf369 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf371 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_90.run(buf365, buf366, buf367, primals_290, primals_291, buf368, buf369, buf371, primals_290, primals_291, 480, 13, grid=grid(480), stream=stream0)
        del buf365
        del buf366
        del buf367
        del primals_290
        del primals_291
        buf372 = reinterpret_tensor(buf363, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf363  # reuse
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf364, buf368, buf369, primals_55, primals_56, buf372, 752640, grid=grid(752640), stream=stream0)
        del buf369
        del primals_56
        buf373 = buf333; del buf333  # reuse
        # Source Nodes: [x_148, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_93.run(buf372, buf373, 7680, 98, grid=grid(7680), stream=stream0)
        buf374 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf375 = reinterpret_tensor(buf374, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf374  # reuse
        # Source Nodes: [x_148, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_94.run(buf375, buf373, 3840, 2, grid=grid(3840), stream=stream0)
        del buf373
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 20, 1, 1), (20, 1, 1, 1))
        buf377 = reinterpret_tensor(buf376, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf376  # reuse
        buf378 = empty_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_95.run(buf377, primals_159, buf378, 160, grid=grid(160), stream=stream0)
        del primals_159
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 480, 1, 1), (480, 1, 1, 1))
        buf380 = reinterpret_tensor(buf379, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf379  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf380, primals_161, 3840, grid=grid(3840), stream=stream0)
        del primals_161
        buf381 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_148, x_149], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_97.run(buf372, buf380, buf381, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf383 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_99.run(buf382, buf383, 896, 196, grid=grid(896, 196), stream=stream0)
        buf384 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf385 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf386 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf383, buf384, buf385, buf386, 1456, 121, grid=grid(1456), stream=stream0)
        buf387 = reinterpret_tensor(buf45, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf45  # reuse
        buf388 = reinterpret_tensor(buf44, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf44  # reuse
        buf390 = reinterpret_tensor(buf43, (112, ), (1, ), 0); del buf43  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_101.run(buf384, buf385, buf386, primals_293, primals_294, buf387, buf388, buf390, primals_293, primals_294, 112, 13, grid=grid(112), stream=stream0)
        del primals_293
        del primals_294
        buf391 = reinterpret_tensor(buf382, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf382  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_102.run(buf383, buf387, buf388, primals_57, primals_58, buf391, 175616, grid=grid(175616), stream=stream0)
        del primals_58
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf393 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf392, buf393, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf394 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        buf395 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        buf396 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf393, buf394, buf395, buf396, 8736, 121, grid=grid(8736), stream=stream0)
        buf397 = reinterpret_tensor(buf58, (1, 672, 1, 1), (672, 1, 672, 672), 0); del buf58  # reuse
        buf398 = reinterpret_tensor(buf57, (1, 672, 1, 1), (672, 1, 672, 672), 0); del buf57  # reuse
        buf400 = reinterpret_tensor(buf56, (672, ), (1, ), 0); del buf56  # reuse
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf394, buf395, buf396, primals_296, primals_297, buf397, buf398, buf400, primals_296, primals_297, 672, 13, grid=grid(672), stream=stream0)
        del primals_296
        del primals_297
        buf402 = reinterpret_tensor(buf392, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf392  # reuse
        buf687 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_106.run(buf393, buf397, buf398, primals_59, primals_60, buf402, buf687, 1053696, grid=grid(1053696), stream=stream0)
        del primals_60
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_164, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf403, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf404 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf403, buf404, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf405 = buf396; del buf396  # reuse
        buf406 = buf395; del buf395  # reuse
        buf407 = buf394; del buf394  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf404, buf405, buf406, buf407, 8736, 121, grid=grid(8736), stream=stream0)
        buf408 = buf398; del buf398  # reuse
        buf409 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf411 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf405, buf406, buf407, primals_299, primals_300, buf408, buf409, buf411, primals_299, primals_300, 672, 13, grid=grid(672), stream=stream0)
        del primals_299
        del primals_300
        buf412 = reinterpret_tensor(buf403, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf403  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_107.run(buf404, buf408, buf409, primals_61, primals_62, buf412, 1053696, grid=grid(1053696), stream=stream0)
        del primals_62
        buf413 = empty_strided((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_108.run(buf412, buf413, 10752, 98, grid=grid(10752), stream=stream0)
        buf414 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf415 = reinterpret_tensor(buf414, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf414  # reuse
        # Source Nodes: [x_164, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_109.run(buf415, buf413, 5376, 2, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 28, 1, 1), (28, 1, 1, 1))
        buf417 = reinterpret_tensor(buf416, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf416  # reuse
        buf418 = reinterpret_tensor(buf23, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf23  # reuse
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_110.run(buf417, primals_166, buf418, 224, grid=grid(224), stream=stream0)
        del primals_166
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 672, 1, 1), (672, 1, 1, 1))
        buf420 = reinterpret_tensor(buf419, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf419  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_111.run(buf420, primals_168, 5376, grid=grid(5376), stream=stream0)
        del primals_168
        buf421 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_164, x_165], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_112.run(buf412, buf420, buf421, 1053696, grid=grid(1053696), stream=stream0)
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf423 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_99.run(buf422, buf423, 896, 196, grid=grid(896, 196), stream=stream0)
        buf424 = buf386; del buf386  # reuse
        buf425 = buf385; del buf385  # reuse
        buf426 = buf384; del buf384  # reuse
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf423, buf424, buf425, buf426, 1456, 121, grid=grid(1456), stream=stream0)
        buf427 = buf388; del buf388  # reuse
        buf428 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf430 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_101.run(buf424, buf425, buf426, primals_302, primals_303, buf427, buf428, buf430, primals_302, primals_303, 112, 13, grid=grid(112), stream=stream0)
        del primals_302
        del primals_303
        buf431 = reinterpret_tensor(buf422, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf422  # reuse
        # Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_113.run(buf423, buf427, buf428, primals_63, primals_64, buf391, buf431, 175616, grid=grid(175616), stream=stream0)
        del primals_64
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf433 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf432, buf433, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf434 = buf407; del buf407  # reuse
        buf435 = buf406; del buf406  # reuse
        buf436 = buf405; del buf405  # reuse
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf433, buf434, buf435, buf436, 8736, 121, grid=grid(8736), stream=stream0)
        buf437 = buf409; del buf409  # reuse
        buf438 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf440 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf434, buf435, buf436, primals_305, primals_306, buf437, buf438, buf440, primals_305, primals_306, 672, 13, grid=grid(672), stream=stream0)
        del primals_305
        del primals_306
        buf442 = reinterpret_tensor(buf432, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf432  # reuse
        buf686 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_106.run(buf433, buf437, buf438, primals_65, primals_66, buf442, buf686, 1053696, grid=grid(1053696), stream=stream0)
        del primals_66
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_171, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf443, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf444 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf443, buf444, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf445 = buf436; del buf436  # reuse
        buf446 = buf435; del buf435  # reuse
        buf447 = buf434; del buf434  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf444, buf445, buf446, buf447, 8736, 121, grid=grid(8736), stream=stream0)
        buf448 = buf438; del buf438  # reuse
        buf449 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf451 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf445, buf446, buf447, primals_308, primals_309, buf448, buf449, buf451, primals_308, primals_309, 672, 13, grid=grid(672), stream=stream0)
        del primals_308
        del primals_309
        buf452 = reinterpret_tensor(buf443, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf443  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_107.run(buf444, buf448, buf449, primals_67, primals_68, buf452, 1053696, grid=grid(1053696), stream=stream0)
        del primals_68
        buf453 = buf413; del buf413  # reuse
        # Source Nodes: [x_181, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_108.run(buf452, buf453, 10752, 98, grid=grid(10752), stream=stream0)
        buf454 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf455 = reinterpret_tensor(buf454, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf454  # reuse
        # Source Nodes: [x_181, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_109.run(buf455, buf453, 5376, 2, grid=grid(5376), stream=stream0)
        del buf453
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 28, 1, 1), (28, 1, 1, 1))
        buf457 = reinterpret_tensor(buf456, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf456  # reuse
        buf458 = reinterpret_tensor(buf22, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf22  # reuse
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_110.run(buf457, primals_173, buf458, 224, grid=grid(224), stream=stream0)
        del primals_173
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (8, 672, 1, 1), (672, 1, 1, 1))
        buf460 = reinterpret_tensor(buf459, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf459  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_111.run(buf460, primals_175, 5376, grid=grid(5376), stream=stream0)
        del primals_175
        buf461 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_181, x_182], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_112.run(buf452, buf460, buf461, 1053696, grid=grid(1053696), stream=stream0)
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf463 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_99.run(buf462, buf463, 896, 196, grid=grid(896, 196), stream=stream0)
        buf464 = buf426; del buf426  # reuse
        buf465 = buf425; del buf425  # reuse
        buf466 = buf424; del buf424  # reuse
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf463, buf464, buf465, buf466, 1456, 121, grid=grid(1456), stream=stream0)
        buf467 = buf428; del buf428  # reuse
        buf468 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf470 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_101.run(buf464, buf465, buf466, primals_311, primals_312, buf467, buf468, buf470, primals_311, primals_312, 112, 13, grid=grid(112), stream=stream0)
        del buf464
        del buf465
        del buf466
        del primals_311
        del primals_312
        buf471 = reinterpret_tensor(buf462, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf462  # reuse
        # Source Nodes: [shortcut_11, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_113.run(buf463, buf467, buf468, primals_69, primals_70, buf431, buf471, 175616, grid=grid(175616), stream=stream0)
        del buf468
        del primals_70
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf473 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf472, buf473, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf474 = buf447; del buf447  # reuse
        buf475 = buf446; del buf446  # reuse
        buf476 = buf445; del buf445  # reuse
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf473, buf474, buf475, buf476, 8736, 121, grid=grid(8736), stream=stream0)
        buf477 = buf449; del buf449  # reuse
        buf478 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf480 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf474, buf475, buf476, primals_314, primals_315, buf477, buf478, buf480, primals_314, primals_315, 672, 13, grid=grid(672), stream=stream0)
        del buf474
        del buf475
        del buf476
        del primals_314
        del primals_315
        buf481 = reinterpret_tensor(buf472, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf472  # reuse
        buf685 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_114.run(buf473, buf477, buf478, primals_71, primals_72, buf481, buf685, 1053696, grid=grid(1053696), stream=stream0)
        del primals_72
        buf482 = empty_strided((8, 672, 17, 17), (194208, 1, 11424, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193, x_195], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_115.run(buf481, buf482, 1553664, grid=grid(1553664), stream=stream0)
        del buf481
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_73, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf483, (8, 672, 7, 7), (32928, 49, 7, 1))
        buf484 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_116.run(buf483, buf484, 5376, 49, grid=grid(5376, 49), stream=stream0)
        buf485 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf486 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf487 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_117.run(buf484, buf485, buf486, buf487, 2688, 98, grid=grid(2688), stream=stream0)
        buf488 = buf478; del buf478  # reuse
        buf489 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf491 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_118.run(buf485, buf486, buf487, primals_317, primals_318, buf488, buf489, buf491, primals_317, primals_318, 672, 4, grid=grid(672), stream=stream0)
        del buf485
        del buf486
        del buf487
        del primals_317
        del primals_318
        buf492 = reinterpret_tensor(buf483, (8, 672, 7, 7), (32928, 1, 4704, 672), 0); del buf483  # reuse
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_119.run(buf484, buf488, buf489, primals_74, primals_75, buf492, 263424, grid=grid(263424), stream=stream0)
        del buf489
        del primals_75
        buf493 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf494 = reinterpret_tensor(buf493, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf493  # reuse
        # Source Nodes: [x_200, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_120.run(buf494, buf492, 5376, 49, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (8, 28, 1, 1), (28, 1, 1, 1))
        buf496 = reinterpret_tensor(buf495, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf495  # reuse
        buf497 = reinterpret_tensor(buf21, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf21  # reuse
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_110.run(buf496, primals_179, buf497, 224, grid=grid(224), stream=stream0)
        del primals_179
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (8, 672, 1, 1), (672, 1, 1, 1))
        buf499 = reinterpret_tensor(buf498, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf498  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_111.run(buf499, primals_181, 5376, grid=grid(5376), stream=stream0)
        del primals_181
        buf500 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200, x_201], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_121.run(buf492, buf499, buf500, 263424, grid=grid(263424), stream=stream0)
        # Source Nodes: [x_202], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf502 = reinterpret_tensor(buf55, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf55  # reuse
        # Source Nodes: [x_202], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf501, buf502, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf503 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf504 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf505 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_123.run(buf502, buf503, buf504, buf505, 768, 98, grid=grid(768), stream=stream0)
        buf506 = reinterpret_tensor(buf72, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf72  # reuse
        buf507 = reinterpret_tensor(buf71, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf71  # reuse
        buf509 = reinterpret_tensor(buf70, (192, ), (1, ), 0); del buf70  # reuse
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_124.run(buf503, buf504, buf505, primals_320, primals_321, buf506, buf507, buf509, primals_320, primals_321, 192, 4, grid=grid(192), stream=stream0)
        del primals_320
        del primals_321
        buf510 = reinterpret_tensor(buf501, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf501  # reuse
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_125.run(buf502, buf506, buf507, primals_76, primals_77, buf510, 75264, grid=grid(75264), stream=stream0)
        del primals_77
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf510, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf512 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf511, buf512, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf513 = empty_strided((1, 1152, 1, 1, 4), (4608, 1, 4608, 4608, 1152), device='cuda', dtype=torch.float32)
        buf514 = empty_strided((1, 1152, 1, 1, 4), (4608, 1, 4608, 4608, 1152), device='cuda', dtype=torch.float32)
        buf515 = empty_strided((1, 1152, 1, 1, 4), (4608, 1, 4608, 4608, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf512, buf513, buf514, buf515, 4608, 98, grid=grid(4608), stream=stream0)
        buf516 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf517 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf519 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf513, buf514, buf515, primals_323, primals_324, buf516, buf517, buf519, primals_323, primals_324, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_323
        del primals_324
        buf521 = reinterpret_tensor(buf511, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf511  # reuse
        buf684 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_129.run(buf512, buf516, buf517, primals_78, primals_79, buf521, buf684, 451584, grid=grid(451584), stream=stream0)
        del primals_79
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_184, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf522, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf523 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf522, buf523, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf524 = buf515; del buf515  # reuse
        buf525 = buf514; del buf514  # reuse
        buf526 = buf513; del buf513  # reuse
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf523, buf524, buf525, buf526, 4608, 98, grid=grid(4608), stream=stream0)
        buf527 = buf517; del buf517  # reuse
        buf528 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf530 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf524, buf525, buf526, primals_326, primals_327, buf527, buf528, buf530, primals_326, primals_327, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_326
        del primals_327
        buf531 = reinterpret_tensor(buf522, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf522  # reuse
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_130.run(buf523, buf527, buf528, primals_80, primals_81, buf531, 451584, grid=grid(451584), stream=stream0)
        del primals_81
        buf532 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf533 = reinterpret_tensor(buf532, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf532  # reuse
        # Source Nodes: [x_216, x_se_48], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_131.run(buf533, buf531, 9216, 49, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 48, 1, 1), (48, 1, 1, 1))
        buf535 = reinterpret_tensor(buf534, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf534  # reuse
        buf536 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_132.run(buf535, primals_186, buf536, 384, grid=grid(384), stream=stream0)
        del primals_186
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf537, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf538 = reinterpret_tensor(buf537, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf537  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf538, primals_188, 9216, grid=grid(9216), stream=stream0)
        del primals_188
        buf539 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_216, x_217], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_134.run(buf531, buf538, buf539, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf541 = reinterpret_tensor(buf54, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf54  # reuse
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf540, buf541, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf542 = buf505; del buf505  # reuse
        buf543 = buf504; del buf504  # reuse
        buf544 = buf503; del buf503  # reuse
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_123.run(buf541, buf542, buf543, buf544, 768, 98, grid=grid(768), stream=stream0)
        buf545 = buf507; del buf507  # reuse
        buf546 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf548 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_124.run(buf542, buf543, buf544, primals_329, primals_330, buf545, buf546, buf548, primals_329, primals_330, 192, 4, grid=grid(192), stream=stream0)
        del primals_329
        del primals_330
        buf549 = reinterpret_tensor(buf540, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf540  # reuse
        # Source Nodes: [shortcut_13, x_219], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_135.run(buf541, buf545, buf546, primals_82, primals_83, buf510, buf549, 75264, grid=grid(75264), stream=stream0)
        del primals_83
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf551 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf550, buf551, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf552 = buf526; del buf526  # reuse
        buf553 = buf525; del buf525  # reuse
        buf554 = buf524; del buf524  # reuse
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf551, buf552, buf553, buf554, 4608, 98, grid=grid(4608), stream=stream0)
        buf555 = buf528; del buf528  # reuse
        buf556 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf558 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf552, buf553, buf554, primals_332, primals_333, buf555, buf556, buf558, primals_332, primals_333, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_332
        del primals_333
        buf560 = reinterpret_tensor(buf550, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf550  # reuse
        buf683 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225, x_228], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_129.run(buf551, buf555, buf556, primals_84, primals_85, buf560, buf683, 451584, grid=grid(451584), stream=stream0)
        del primals_85
        # Source Nodes: [x_229], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf560, primals_191, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf561, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf562 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_229], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf561, buf562, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf563 = buf554; del buf554  # reuse
        buf564 = buf553; del buf553  # reuse
        buf565 = buf552; del buf552  # reuse
        # Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf562, buf563, buf564, buf565, 4608, 98, grid=grid(4608), stream=stream0)
        buf566 = buf556; del buf556  # reuse
        buf567 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf569 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf563, buf564, buf565, primals_335, primals_336, buf566, buf567, buf569, primals_335, primals_336, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_335
        del primals_336
        buf570 = reinterpret_tensor(buf561, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf561  # reuse
        # Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_130.run(buf562, buf566, buf567, primals_86, primals_87, buf570, 451584, grid=grid(451584), stream=stream0)
        del primals_87
        buf571 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf572 = reinterpret_tensor(buf571, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf571  # reuse
        # Source Nodes: [x_233, x_se_52], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_131.run(buf572, buf570, 9216, 49, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf573 = extern_kernels.convolution(buf572, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf573, (8, 48, 1, 1), (48, 1, 1, 1))
        buf574 = reinterpret_tensor(buf573, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf573  # reuse
        buf575 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_132.run(buf574, primals_193, buf575, 384, grid=grid(384), stream=stream0)
        del primals_193
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf577 = reinterpret_tensor(buf576, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf576  # reuse
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf577, primals_195, 9216, grid=grid(9216), stream=stream0)
        del primals_195
        buf578 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_233, x_234], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_134.run(buf570, buf577, buf578, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [x_235], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf580 = reinterpret_tensor(buf53, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf53  # reuse
        # Source Nodes: [x_235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf579, buf580, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf581 = buf544; del buf544  # reuse
        buf582 = buf543; del buf543  # reuse
        buf583 = buf542; del buf542  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_123.run(buf580, buf581, buf582, buf583, 768, 98, grid=grid(768), stream=stream0)
        buf584 = buf546; del buf546  # reuse
        buf585 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf587 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_124.run(buf581, buf582, buf583, primals_338, primals_339, buf584, buf585, buf587, primals_338, primals_339, 192, 4, grid=grid(192), stream=stream0)
        del primals_338
        del primals_339
        buf588 = reinterpret_tensor(buf579, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf579  # reuse
        # Source Nodes: [shortcut_14, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_135.run(buf580, buf584, buf585, primals_88, primals_89, buf549, buf588, 75264, grid=grid(75264), stream=stream0)
        del primals_89
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf590 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf589, buf590, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf591 = buf565; del buf565  # reuse
        buf592 = buf564; del buf564  # reuse
        buf593 = buf563; del buf563  # reuse
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf590, buf591, buf592, buf593, 4608, 98, grid=grid(4608), stream=stream0)
        buf594 = buf567; del buf567  # reuse
        buf595 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf597 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf591, buf592, buf593, primals_341, primals_342, buf594, buf595, buf597, primals_341, primals_342, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_341
        del primals_342
        buf599 = reinterpret_tensor(buf589, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf589  # reuse
        buf682 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242, x_245], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_129.run(buf590, buf594, buf595, primals_90, primals_91, buf599, buf682, 451584, grid=grid(451584), stream=stream0)
        del primals_91
        # Source Nodes: [x_246], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_198, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf600, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf601 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_246], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf600, buf601, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf602 = buf593; del buf593  # reuse
        buf603 = buf592; del buf592  # reuse
        buf604 = buf591; del buf591  # reuse
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf601, buf602, buf603, buf604, 4608, 98, grid=grid(4608), stream=stream0)
        buf605 = buf595; del buf595  # reuse
        buf606 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf608 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf602, buf603, buf604, primals_344, primals_345, buf605, buf606, buf608, primals_344, primals_345, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_344
        del primals_345
        buf609 = reinterpret_tensor(buf600, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf600  # reuse
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_130.run(buf601, buf605, buf606, primals_92, primals_93, buf609, 451584, grid=grid(451584), stream=stream0)
        del primals_93
        buf610 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf611 = reinterpret_tensor(buf610, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf610  # reuse
        # Source Nodes: [x_250, x_se_56], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_131.run(buf611, buf609, 9216, 49, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf611, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (8, 48, 1, 1), (48, 1, 1, 1))
        buf613 = reinterpret_tensor(buf612, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf612  # reuse
        buf614 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_132.run(buf613, primals_200, buf614, 384, grid=grid(384), stream=stream0)
        del primals_200
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf615 = extern_kernels.convolution(buf614, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf615, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf616 = reinterpret_tensor(buf615, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf615  # reuse
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf616, primals_202, 9216, grid=grid(9216), stream=stream0)
        del primals_202
        buf617 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_250, x_251], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_134.run(buf609, buf616, buf617, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf619 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf618, buf619, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf620 = buf583; del buf583  # reuse
        buf621 = buf582; del buf582  # reuse
        buf622 = buf581; del buf581  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_123.run(buf619, buf620, buf621, buf622, 768, 98, grid=grid(768), stream=stream0)
        buf623 = buf585; del buf585  # reuse
        buf624 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf626 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_124.run(buf620, buf621, buf622, primals_347, primals_348, buf623, buf624, buf626, primals_347, primals_348, 192, 4, grid=grid(192), stream=stream0)
        del buf620
        del buf621
        del buf622
        del primals_347
        del primals_348
        buf627 = reinterpret_tensor(buf618, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf618  # reuse
        # Source Nodes: [shortcut_15, x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_135.run(buf619, buf623, buf624, primals_94, primals_95, buf588, buf627, 75264, grid=grid(75264), stream=stream0)
        del buf624
        del primals_95
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        buf628 = extern_kernels.convolution(buf627, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf628, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf629 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf628, buf629, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf630 = buf604; del buf604  # reuse
        buf631 = buf603; del buf603  # reuse
        buf632 = buf602; del buf602  # reuse
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf629, buf630, buf631, buf632, 4608, 98, grid=grid(4608), stream=stream0)
        buf633 = buf606; del buf606  # reuse
        buf634 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf636 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf630, buf631, buf632, primals_350, primals_351, buf633, buf634, buf636, primals_350, primals_351, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_350
        del primals_351
        buf638 = reinterpret_tensor(buf628, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf628  # reuse
        buf681 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259, x_262], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_129.run(buf629, buf633, buf634, primals_96, primals_97, buf638, buf681, 451584, grid=grid(451584), stream=stream0)
        del primals_97
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf639, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf640 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_126.run(buf639, buf640, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf641 = buf632; del buf632  # reuse
        buf642 = buf631; del buf631  # reuse
        buf643 = buf630; del buf630  # reuse
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_127.run(buf640, buf641, buf642, buf643, 4608, 98, grid=grid(4608), stream=stream0)
        buf644 = buf634; del buf634  # reuse
        buf645 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf647 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf641, buf642, buf643, primals_353, primals_354, buf644, buf645, buf647, primals_353, primals_354, 1152, 4, grid=grid(1152), stream=stream0)
        del buf641
        del buf642
        del buf643
        del primals_353
        del primals_354
        buf648 = reinterpret_tensor(buf639, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf639  # reuse
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_130.run(buf640, buf644, buf645, primals_98, primals_99, buf648, 451584, grid=grid(451584), stream=stream0)
        del buf645
        del primals_99
        buf649 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf650 = reinterpret_tensor(buf649, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf649  # reuse
        # Source Nodes: [x_267, x_se_60], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_131.run(buf650, buf648, 9216, 49, grid=grid(9216), stream=stream0)
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (8, 48, 1, 1), (48, 1, 1, 1))
        buf652 = reinterpret_tensor(buf651, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf651  # reuse
        buf653 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_132.run(buf652, primals_207, buf653, 384, grid=grid(384), stream=stream0)
        del primals_207
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (8, 1152, 1, 1), (1152, 1, 1, 1))
        buf655 = reinterpret_tensor(buf654, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf654  # reuse
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf655, primals_209, 9216, grid=grid(9216), stream=stream0)
        del primals_209
        buf656 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_267, x_268], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_134.run(buf648, buf655, buf656, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf656, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf658 = empty_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_136.run(buf657, buf658, 2560, 49, grid=grid(2560, 49), stream=stream0)
        buf659 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf660 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf661 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_137.run(buf658, buf659, buf660, buf661, 1280, 98, grid=grid(1280), stream=stream0)
        buf662 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf663 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf665 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_138.run(buf659, buf660, buf661, primals_356, primals_357, buf662, buf663, buf665, primals_356, primals_357, 320, 4, grid=grid(320), stream=stream0)
        del primals_356
        del primals_357
        buf666 = reinterpret_tensor(buf657, (8, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf657  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_139.run(buf658, buf662, buf663, primals_100, primals_101, buf666, 125440, grid=grid(125440), stream=stream0)
        del buf663
        del primals_101
        # Source Nodes: [x_275], Original ATen: [aten.convolution]
        buf667 = extern_kernels.convolution(buf666, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (8, 1280, 7, 7), (62720, 49, 7, 1))
        buf668 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_140.run(buf667, buf668, 10240, 49, grid=grid(10240, 49), stream=stream0)
        buf669 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf670 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf671 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_141.run(buf668, buf669, buf670, buf671, 5120, 98, grid=grid(5120), stream=stream0)
        buf672 = reinterpret_tensor(buf661, (1, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf661  # reuse
        buf673 = reinterpret_tensor(buf660, (1, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf660  # reuse
        buf675 = reinterpret_tensor(buf659, (1280, ), (1, ), 0); del buf659  # reuse
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_142.run(buf669, buf670, buf671, primals_359, primals_360, buf672, buf673, buf675, primals_359, primals_360, 1280, 4, grid=grid(1280), stream=stream0)
        del buf669
        del buf670
        del buf671
        del primals_359
        del primals_360
        buf676 = reinterpret_tensor(buf667, (8, 1280, 7, 7), (62720, 1, 8960, 1280), 0); del buf667  # reuse
        buf680 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_143.run(buf668, buf672, buf673, primals_102, primals_103, buf676, buf680, 501760, grid=grid(501760), stream=stream0)
        del buf673
        del primals_103
        buf677 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf678 = reinterpret_tensor(buf677, (8, 1280), (1280, 1), 0); del buf677  # reuse
        # Source Nodes: [x_280, x_281, x_283], Original ATen: [aten.mean, aten.silu, aten.view]
        triton_per_fused_mean_silu_view_144.run(buf678, buf676, 10240, 49, grid=grid(10240), stream=stream0)
        del buf676
        buf679 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_213, buf678, reinterpret_tensor(primals_212, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf679)
        del primals_213
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_214, primals_214, 1, grid=grid(1), stream=stream0)
        del primals_214
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_217, primals_217, 1, grid=grid(1), stream=stream0)
        del primals_217
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_220, primals_220, 1, grid=grid(1), stream=stream0)
        del primals_220
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_223, primals_223, 1, grid=grid(1), stream=stream0)
        del primals_223
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_226, primals_226, 1, grid=grid(1), stream=stream0)
        del primals_226
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_229, primals_229, 1, grid=grid(1), stream=stream0)
        del primals_229
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_232, primals_232, 1, grid=grid(1), stream=stream0)
        del primals_232
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_235, primals_235, 1, grid=grid(1), stream=stream0)
        del primals_235
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_238, primals_238, 1, grid=grid(1), stream=stream0)
        del primals_238
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_241, primals_241, 1, grid=grid(1), stream=stream0)
        del primals_241
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_244, primals_244, 1, grid=grid(1), stream=stream0)
        del primals_244
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_247, primals_247, 1, grid=grid(1), stream=stream0)
        del primals_247
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_250, primals_250, 1, grid=grid(1), stream=stream0)
        del primals_250
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_253, primals_253, 1, grid=grid(1), stream=stream0)
        del primals_253
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_256, primals_256, 1, grid=grid(1), stream=stream0)
        del primals_256
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_259, primals_259, 1, grid=grid(1), stream=stream0)
        del primals_259
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_262, primals_262, 1, grid=grid(1), stream=stream0)
        del primals_262
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_265, primals_265, 1, grid=grid(1), stream=stream0)
        del primals_265
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_268, primals_268, 1, grid=grid(1), stream=stream0)
        del primals_268
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_271, primals_271, 1, grid=grid(1), stream=stream0)
        del primals_271
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_274, primals_274, 1, grid=grid(1), stream=stream0)
        del primals_274
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_277, primals_277, 1, grid=grid(1), stream=stream0)
        del primals_277
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_280, primals_280, 1, grid=grid(1), stream=stream0)
        del primals_280
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_283, primals_283, 1, grid=grid(1), stream=stream0)
        del primals_283
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_286, primals_286, 1, grid=grid(1), stream=stream0)
        del primals_286
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_289, primals_289, 1, grid=grid(1), stream=stream0)
        del primals_289
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_292, primals_292, 1, grid=grid(1), stream=stream0)
        del primals_292
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_313, primals_313, 1, grid=grid(1), stream=stream0)
        del primals_313
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_316, primals_316, 1, grid=grid(1), stream=stream0)
        del primals_316
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_319, primals_319, 1, grid=grid(1), stream=stream0)
        del primals_319
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_322, primals_322, 1, grid=grid(1), stream=stream0)
        del primals_322
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_325, primals_325, 1, grid=grid(1), stream=stream0)
        del primals_325
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_328, primals_328, 1, grid=grid(1), stream=stream0)
        del primals_328
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_331, primals_331, 1, grid=grid(1), stream=stream0)
        del primals_331
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_334, primals_334, 1, grid=grid(1), stream=stream0)
        del primals_334
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_337, primals_337, 1, grid=grid(1), stream=stream0)
        del primals_337
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_340, primals_340, 1, grid=grid(1), stream=stream0)
        del primals_340
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_343, primals_343, 1, grid=grid(1), stream=stream0)
        del primals_343
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_346, primals_346, 1, grid=grid(1), stream=stream0)
        del primals_346
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_349, primals_349, 1, grid=grid(1), stream=stream0)
        del primals_349
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_352, primals_352, 1, grid=grid(1), stream=stream0)
        del primals_352
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_355, primals_355, 1, grid=grid(1), stream=stream0)
        del primals_355
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_145.run(primals_358, primals_358, 1, grid=grid(1), stream=stream0)
        del primals_358
        return (buf679, buf0, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_126, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, buf1, buf3, buf13, buf15, buf17, buf27, buf28, buf31, buf33, buf34, buf36, buf37, buf39, buf49, buf50, buf52, buf62, buf64, buf66, buf76, buf77, buf80, buf82, buf83, buf85, buf86, buf88, buf98, buf99, buf101, buf111, buf113, buf115, buf125, buf126, buf129, buf131, buf132, buf134, buf135, buf137, buf147, buf148, buf150, buf160, buf162, buf164, buf171, buf172, buf175, buf177, buf178, buf180, buf181, buf183, buf190, buf191, buf193, buf200, buf202, buf204, buf211, buf212, buf215, buf217, buf218, buf220, buf221, buf223, buf230, buf231, buf233, buf240, buf242, buf244, buf251, buf252, buf255, buf257, buf258, buf260, buf261, buf263, buf270, buf271, buf273, buf280, buf282, buf284, buf291, buf292, buf295, buf297, buf298, buf300, buf301, buf303, buf310, buf311, buf313, buf320, buf322, buf324, buf331, buf332, buf335, buf337, buf338, buf340, buf341, buf343, buf350, buf351, buf353, buf360, buf362, buf364, buf371, buf372, buf375, buf377, buf378, buf380, buf381, buf383, buf390, buf391, buf393, buf400, buf402, buf404, buf411, buf412, buf415, buf417, buf418, buf420, buf421, buf423, buf430, buf431, buf433, buf440, buf442, buf444, buf451, buf452, buf455, buf457, buf458, buf460, buf461, buf463, buf470, buf471, buf473, buf480, buf482, buf484, buf491, buf492, buf494, buf496, buf497, buf499, buf500, buf502, buf509, buf510, buf512, buf519, buf521, buf523, buf530, buf531, buf533, buf535, buf536, buf538, buf539, buf541, buf548, buf549, buf551, buf558, buf560, buf562, buf569, buf570, buf572, buf574, buf575, buf577, buf578, buf580, buf587, buf588, buf590, buf597, buf599, buf601, buf608, buf609, buf611, buf613, buf614, buf616, buf617, buf619, buf626, buf627, buf629, buf636, buf638, buf640, buf647, buf648, buf650, buf652, buf653, buf655, buf656, buf658, buf665, buf666, buf668, buf675, buf678, reinterpret_tensor(primals_212, (1000, 1280), (1280, 1), 0), buf680, reinterpret_tensor(buf672, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf662, (1, 320, 1, 1), (320, 1, 1, 1), 0), reinterpret_tensor(buf644, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf681, reinterpret_tensor(buf633, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf623, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf605, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf682, reinterpret_tensor(buf594, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf584, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf566, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf683, reinterpret_tensor(buf555, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf545, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf527, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), buf684, reinterpret_tensor(buf516, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf506, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf488, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf685, reinterpret_tensor(buf477, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf467, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf448, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf686, reinterpret_tensor(buf437, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf427, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf408, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf687, reinterpret_tensor(buf397, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf368, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf688, reinterpret_tensor(buf357, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf347, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf328, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf689, reinterpret_tensor(buf317, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf307, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf288, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf690, reinterpret_tensor(buf277, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf248, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf691, reinterpret_tensor(buf237, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf227, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf208, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf692, reinterpret_tensor(buf197, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf168, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf693, reinterpret_tensor(buf157, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf144, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf122, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf694, reinterpret_tensor(buf108, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf95, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf73, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf695, reinterpret_tensor(buf59, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf46, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf696, reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_218 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_221 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_224 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_227 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_230 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_233 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_236 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_239 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_242 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_245 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_248 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_251 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_254 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_257 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_260 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_263 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_269 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_275 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_278 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_284 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_293 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_314 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_320 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_323 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_326 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_329 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_332 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_335 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_338 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_344 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_347 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_350 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_353 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_356 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_359 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_efficientnet_b0', benchmark_compiled_module)
