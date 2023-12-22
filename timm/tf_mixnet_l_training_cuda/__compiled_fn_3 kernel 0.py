
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtebfhuq7uuhmfk3j34jrxcgjypmuko42n6chqhd4ctzqlny4np.py
# Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# shortcut => relu
# x_2 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_6', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciuf2oldxss22ylhj7lvsmhm7323ub3bslg5wf26owzqwljs3ilr.py
# Source Nodes: [shortcut_1, x_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_1 => add_15
# x_13 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_add_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (y0 + (12544*x2) + (401408*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5ltbmtq6j5praupezdp6oiwwg76rehjngxhkeh37grptstxhs2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___conv_pw_0 => convolution_3
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
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
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y0) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahlnzom2occwx4cdrtixqmumlj56ubtcanmvdh6hjelmgrbgm4g.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___conv_pw_1 => convolution_4
triton_poi_fused_convolution_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': []},
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
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (200704 + x2 + (12544*y0) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7swxvmrmg5r5wrdpv32hb7i6tnikfi4ldimqviuwbrpjzdqwhe.py
# Source Nodes: [cat_81], Original ATen: [aten.cat]
# cat_81 => cat
triton_poi_fused_cat_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 192
    x2 = xindex
    y1 = (yindex // 192)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (12544*y0) + (1204224*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 192, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-1204224) + x2 + (12544*y0) + (1204224*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (192*x2) + (2408448*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cyt5pkizh4dbl6ife22mibgiladm2bdhgkl72lqdhmmjw5yhsjqv.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 196
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (37632*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3a/c3awonuqj63er7dd6dcm3sjc6v4shkt73qorzjqjmop6t44jtaqa.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (192*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (192*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (192*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxludooguxewjehao2esf4rvujbh3syghxmyksoivv6zrkjcxqqp.py
# Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
# x_20 => add_17, add_18, add_19, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/js/cjs5fc7sxubkhlyxn4ckkkslbapp5hpe33gt6ht3jxetj6rfmg44.py
# Source Nodes: [x_20, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_20 => add_17, add_20, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_23 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
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
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47drqxevyfpqixrfx7bxztgviwfs2mc4htfqxb22247rye3ntle.py
# Source Nodes: [x_25], Original ATen: [aten.constant_pad_nd]
# x_25 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6537728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7232) % 113
    x1 = (xindex // 64) % 113
    x0 = xindex % 64
    x3 = (xindex // 817216)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (192*x1) + (21504*x2) + (2408448*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x4), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3s7sbd6juguhmshrdztmgwmqxpxvy6zd335bxrmxfnjyregvlws.py
# Source Nodes: [x_27], Original ATen: [aten.constant_pad_nd]
# x_27 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6771200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7360) % 115
    x1 = (xindex // 64) % 115
    x0 = xindex % 64
    x3 = (xindex // 846400)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-21632) + x0 + (192*x1) + (21504*x2) + (2408448*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x6), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ee/ceehnbzqpfamhg3p27oy2legrds4shiczbxgvpsthx3ngwm2oh43.py
# Source Nodes: [x_29], Original ATen: [aten.constant_pad_nd]
# x_29 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7008768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7488) % 117
    x1 = (xindex // 64) % 117
    x0 = xindex % 64
    x3 = (xindex // 876096)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-43264) + x0 + (192*x1) + (21504*x2) + (2408448*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x6), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdljr2uuvs47rcb7netzudotkmuczjbu7oymblydxsebgp2qi7up.py
# Source Nodes: [cat_80], Original ATen: [aten.cat]
# cat_80 => cat_1
triton_poi_fused_cat_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 192
    x2 = xindex
    y1 = (yindex // 192)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (200704*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-200704) + x2 + (3136*y0) + (200704*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 192, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-401408) + x2 + (3136*y0) + (200704*y1)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (y0 + (192*x2) + (602112*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vlg22kssaf2dycnuarzo6yyxi627qyi7ihtqbr23kdnm76jziw.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => var_mean_4
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
    xnumel = 37632
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nt/cntpnwgzlgcu56b7ykjvcb6punw73cimwnns5wjm6dug6vvig72c.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => var_mean_4
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
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (18816*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (18816*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (192*r2) + (18816*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (192*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (192*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (192*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgxbhffffv2i7eawnpig6fakzypy23sbwb3jzgemzy5cccyvren.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => add_22, add_23, add_24, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/7n/c7naglgdojciixqkbqyl67evgaxu4awe5otaxe3bebltkejbk26k.py
# Source Nodes: [x_32, x_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_32 => add_22, add_25, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_35 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
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
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhyg4oc45nuyf4kcxcjk2263q4wrcox52yqgrvpwj4wmvww5vr4.py
# Source Nodes: [cat_79], Original ATen: [aten.cat]
# cat_79 => cat_2
triton_poi_fused_cat_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 40
    x2 = xindex
    y1 = (yindex // 40)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (62720*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-62720) + x2 + (3136*y0) + (62720*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (40*x2) + (125440*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co77c4d3bavvj6fezmo2kllkdbnbe34hl6pfc56b3cwtw76a6kfm.py
# Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
# x_39 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
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


# kernel path: /tmp/torchinductor_youkaichao/eh/cehusm2dsszvwyimyh7rzq7ebjbtiazy7munj27umhjubzqro3ps.py
# Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
# x_39 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 80
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
        tmp0 = tl.load(in_ptr0 + (x1 + (40*r2) + (3920*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (40*r2) + (3920*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (40*r2) + (3920*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (40*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (40*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (40*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdytntct7aiv4wagrdl2wmymhra474evxulacmtc5sbbhfgekder.py
# Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
# x_39 => add_27, add_28, add_29, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/ua/cua6ebxwcrd2mbkr43quzsfqc6ui6uyeq6rzzyuhe56vill5ugot.py
# Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
# x_39 => add_27, add_30, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
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


# kernel path: /tmp/torchinductor_youkaichao/cw/ccw7q3gr3wwcvvpumkouquyzemutz6ifywmzigbfqnu7cnhnxvhf.py
# Source Nodes: [cat_78], Original ATen: [aten.cat]
# cat_78 => cat_3
triton_poi_fused_cat_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 120
    x2 = xindex
    y1 = (yindex // 120)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (188160*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-188160) + x2 + (3136*y0) + (188160*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (120*x2) + (376320*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czhyninu3jyhhwlzsn3bzuooci5qjjiq2r5xwie66gizykwikfbz.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 23520
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2l/c2liw2zdsrykqgq3fnhcer5tgooqpbgen6u4vtyrj2y7wb3hb7xc.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
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
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (11760*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (120*r2) + (11760*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (120*r2) + (11760*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (120*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (120*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (120*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5wb5u6uftvonov6dylytnkoph6wzlufbuscjahbbfzlntd4srz.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => add_32, add_33, add_34, mul_43, mul_44, mul_45, mul_46, mul_47, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (120*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/y3/cy3tqt6oz57bl3yxz4adxagqrica5xfhhzf7laqwnugzdibk3jja.py
# Source Nodes: [x_45, x_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_45 => add_32, add_35, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_48 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdegsk5r2dvjebz6dxxyouhwhi2h45pzglcqqaadtspbgqrt5uvc.py
# Source Nodes: [x_49], Original ATen: [aten.convolution]
# x_49 => convolution_12
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (376320*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45wmldsmjvccp2kvcdf7u5prrly4lkaljbnjoq76hlerdphfk6q.py
# Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_50 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# x_53 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbu2jlwrzivmqml4vqucawpqrdz4qzpbwmplp7nvlpqyxfdzmky.py
# Source Nodes: [shortcut_3, x_57], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_46
# x_57 => add_42, add_45, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_add_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czv2rb3ohoxizoc7mi4s6iec5uyrwnd2bhq4wy4t4nac4r45aikl.py
# Source Nodes: [x_62], Original ATen: [aten.convolution]
# x_62 => convolution_15
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (752640*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwau6alx4pcb5cjv4cxkovb3tnnb2gatvbonuonufef6etnxdwm.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
# x_63 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47040
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


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvvxuprbxvagdcx4mkf7mn5oh47rnph2nszybclu3qca6kl7ppx.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
# x_63 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
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
        tmp0 = tl.load(in_ptr0 + (x1 + (240*r2) + (23520*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r2) + (23520*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (240*r2) + (23520*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (240*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (240*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (240*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vv2omog6xfwbtjhiij666jkwljkboszbnxe76bwrgg6ulsjorq.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
# x_63 => add_48, add_49, add_50, mul_64, mul_65, mul_66, mul_67, mul_68, rsqrt_9, squeeze_28, var_mean_9
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu54jr2muquhjhkwwlvmgsicusyig2xs3tvo2onp2zax66raij7i.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_63 => add_48, add_51, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
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


# kernel path: /tmp/torchinductor_youkaichao/wk/cwkso35roon5mbe656x2dmyw6mkzh4lsjdwcksoqbp3lrrlckinm.py
# Source Nodes: [x_68], Original ATen: [aten.constant_pad_nd]
# x_68 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1559520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3420) % 57
    x1 = (xindex // 60) % 57
    x0 = xindex % 60
    x3 = (xindex // 194940)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czveqdkea6plchdrnunzxcqgqn2tyucatewr57wtfmzjynvogrus.py
# Source Nodes: [x_70], Original ATen: [aten.constant_pad_nd]
# x_70 => constant_pad_nd_5
triton_poi_fused_constant_pad_nd_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1670880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3540) % 59
    x1 = (xindex // 60) % 59
    x0 = xindex % 60
    x3 = (xindex // 208860)
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
    tmp11 = tl.load(in_ptr0 + ((-13620) + x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nw2bwb7rugjjwzikwgiikq6bf7i44drf6nrd24yhfimgssrjcp.py
# Source Nodes: [x_72], Original ATen: [aten.constant_pad_nd]
# x_72 => constant_pad_nd_6
triton_poi_fused_constant_pad_nd_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1786080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3660) % 61
    x1 = (xindex // 60) % 61
    x0 = xindex % 60
    x3 = (xindex // 223260)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-27240) + x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx56kvbahb3p46hxfj5x6tolk4v7ggqrpm6265vq5lvufhjtpiof.py
# Source Nodes: [x_74], Original ATen: [aten.constant_pad_nd]
# x_74 => constant_pad_nd_7
triton_poi_fused_constant_pad_nd_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1905120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3780) % 63
    x1 = (xindex // 60) % 63
    x0 = xindex % 60
    x3 = (xindex // 238140)
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-40860) + x0 + (240*x1) + (13440*x2) + (752640*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucfmzph23bb3sax5dcf5wqz3zjtk6rzw4hqdfyry63a3ocb6smv.py
# Source Nodes: [cat_76], Original ATen: [aten.cat]
# cat_76 => cat_5
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 240
    x2 = xindex
    y1 = (yindex // 240)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (47040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-47040) + x2 + (784*y0) + (47040*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 180, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-94080) + x2 + (784*y0) + (47040*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 240, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-141120) + x2 + (784*y0) + (47040*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmrvxxuzgfq5zcq67zazs7bis4pm7dg6fpggpvcdtnymc2t5ik4.py
# Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
# x_77 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnxxbyw4skriegkz2to4s4xbwdorxujn57ys2gogwaiwmk3oj6u.py
# Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
# x_77 => add_53, add_54, add_55, mul_72, mul_73, mul_74, mul_75, mul_76, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/xo/cxohlarb6wvpt4nyjdm67x3u3wmc3gwpinwgjz4qk2pemtarbmmf.py
# Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
# x_77 => add_53, add_56, mul_71, mul_77, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcyxn5j2fixm26jvw5kpynrmfmstjur2pqovi7mj27c4h6xhwd3.py
# Source Nodes: [x_80, x_se], Original ATen: [aten.mean, aten.silu]
# x_80 => mul_78, sigmoid_1
# x_se => mean
triton_red_fused_mean_silu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_49', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rw/crwhn3ish2obkawp5idzmo6pwb3lp6d4lpv53fgscnuiy5h4bobl.py
# Source Nodes: [x_80, x_se], Original ATen: [aten.mean, aten.silu]
# x_80 => mul_78, sigmoid_1
# x_se => mean
triton_per_fused_mean_silu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_50', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmlrmup77bnfbxkfdt6bag3prb7vdtlo36in75plxhzwpws6phc.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
# x_se_1 => convolution_20
# x_se_2 => mul_79, sigmoid_2
triton_poi_fused_convolution_silu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_51', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/wy/cwypxbidgry74pdq3ocf3ygpdknj4d35jqzkbmobj6sj2mocjfea.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_21
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ak/cakqotaqi2aa7mo64tdcaje5xuivbxhkzwmu3jr5vmiqkdzcttem.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80, x_81], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
# x_80 => mul_78, sigmoid_1
# x_81 => mul_80
triton_poi_fused_mul_sigmoid_silu_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_53', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7k/c7kqruhx5nitzvpppwmxdl37wfqe2biip7ievhexo4zlhrq5tel3.py
# Source Nodes: [x_82], Original ATen: [aten.convolution]
# x_82 => convolution_22
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwocgvhjtud3q4aoqsn2gccjatgn6vabuwg2bwmt4mftxglqadx.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
# x_83 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2744
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (56*r2) + (7168*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbn3bxoi4c7rywamp4fnausm2qjlj2qnef5e7phd7qzl4farp3kc.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
# x_83 => add_58, add_59, add_60, mul_82, mul_83, mul_84, mul_85, mul_86, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (56*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q3ujgbqigp3rigle4l6xlilrxfc2txjbhfrmxwvi7t3eeff66t.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
# x_83 => add_58, add_61, mul_81, mul_87, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
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


# kernel path: /tmp/torchinductor_youkaichao/sj/csj7ziqhvfg2vhh2ex5ulyof3u5u367y6ptdbdxdxwl5epbgobdf.py
# Source Nodes: [cat_75], Original ATen: [aten.cat]
# cat_75 => cat_6
triton_poi_fused_cat_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (131712*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131712) + x2 + (784*y0) + (131712*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (336*x2) + (263424*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cav4yxxg7ybzgkt4qytwh4fd6lretywnee7d526ogggwh3ge5jlh.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
# x_89 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 336
    x1 = (xindex // 336)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (336*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrvkohhkimjkofrwbfuaiw7x4i4gqv6lflnkvyrtrerixys54t3.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
# x_89 => add_63, add_64, add_65, mul_89, mul_90, mul_91, mul_92, mul_93, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (336*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/in/cinx2mgrioaxt55r6xkg4lj56o4dkvnyvysxyitlx7g3poxnsv2p.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_89 => add_63, add_66, mul_88, mul_94, rsqrt_12, sub_12, var_mean_12
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
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


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bqb3z765kjcvl4h4jw5wfrkxh7hgzhcf6dkmgyxe5l5nxorio6.py
# Source Nodes: [x_92], Original ATen: [aten.silu]
# x_92 => mul_95, sigmoid_4
triton_poi_fused_silu_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cah47k54ahk75hbsbzratnkfbl566e4km3gve26jqx3mc7n2ccf5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 => convolution_25
triton_poi_fused_convolution_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1344
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 168
    y1 = (yindex // 168)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (168*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkjp2q5ehfa6jjl7instkasqaqbut427u52pbo2lezvfzt5nrxs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 => convolution_26
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
    ynumel = 1344
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 168
    y1 = (yindex // 168)
    tmp0 = tl.load(in_ptr0 + (131712 + x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (168*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpks4lqwgpgmqj6juv2rv7t7u6pxfnaqpcyook54kqgobfm3iirh.py
# Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
# x_95 => add_68, add_71, mul_102, mul_96, rsqrt_13, sub_13, var_mean_13
triton_poi_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
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


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxnsydxa2zdkolzxtznkfl7el5kihc7m7ekxsklwzye5vma2ugc.py
# Source Nodes: [x_98, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_98 => mul_103, sigmoid_5
# x_se_4 => mean_1
triton_red_fused_mean_silu_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 336
    x1 = (xindex // 336)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (336*r2) + (37632*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/calt36zrgymu3oudjieomtudtkhgsxm3yjkuluparwols4fp6qor.py
# Source Nodes: [x_98, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_98 => mul_103, sigmoid_5
# x_se_4 => mean_1
triton_per_fused_mean_silu_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_67', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 336
    x1 = (xindex // 336)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r2) + (2352*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozm7kk5lzz7hkv33474ww5rlie776i5zm3gaw6x3fko6j7vrkuu.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
# x_se_5 => convolution_27
# x_se_6 => mul_104, sigmoid_6
triton_poi_fused_convolution_silu_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_68', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a7n5rtuzg2q7a6mu7vwr7tsubjk5t33u2rjm47xia6yoqhtpu6.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_28
triton_poi_fused_convolution_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jne3rat7gsxd73jnjaoz53gpkksf23lgc4upyaw5me7thl6iov.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_98, x_99], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_7
# x_98 => mul_103, sigmoid_5
# x_99 => mul_105
triton_poi_fused_mul_sigmoid_silu_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5bmbvyu4pu4rsm5k5vmmzoa4n762ahnkgxqysu2lj2bm4ppnxl.py
# Source Nodes: [cat_73], Original ATen: [aten.cat]
# cat_73 => cat_8
triton_poi_fused_cat_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 56
    x2 = xindex
    y1 = (yindex // 56)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (21952*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 56, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-21952) + x2 + (784*y0) + (21952*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (56*x2) + (43904*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjl2lotzha6gn46223c7oycx6aovk7kcqdb7tjpwedvntxguffj.py
# Source Nodes: [shortcut_5, x_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_5 => add_77
# x_102 => add_73, add_76, mul_106, mul_112, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_add_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
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


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfi77dd6atcsvewqoypebj52uw6pq5s7fd5hjxwihb2nmadagcg.py
# Source Nodes: [x_147], Original ATen: [aten.convolution]
# x_147 => convolution_47
triton_poi_fused_convolution_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (336*x2) + (263424*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cst575s22emeilzzlxw2lzb3dpuy7oheyhquetarmpl4tbysoxfv.py
# Source Nodes: [x_153], Original ATen: [aten.constant_pad_nd]
# x_153 => constant_pad_nd_8
triton_poi_fused_constant_pad_nd_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 753536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3248) % 29
    x1 = (xindex // 112) % 29
    x0 = xindex % 112
    x3 = (xindex // 94192)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (336*x1) + (9408*x2) + (263424*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqh7p7s5k5y3jbywqd4qspbi5x2opk52estarmg46d72ch76je5.py
# Source Nodes: [x_155], Original ATen: [aten.constant_pad_nd]
# x_155 => constant_pad_nd_9
triton_poi_fused_constant_pad_nd_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 861056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3472) % 31
    x1 = (xindex // 112) % 31
    x0 = xindex % 112
    x3 = (xindex // 107632)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-9632) + x0 + (336*x1) + (9408*x2) + (263424*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhsg2okngahnplr2pudp4equy2z5cgzi5y3uvcbjjbpem3t4tir.py
# Source Nodes: [x_157], Original ATen: [aten.constant_pad_nd]
# x_157 => constant_pad_nd_10
triton_poi_fused_constant_pad_nd_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 975744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3696) % 33
    x1 = (xindex // 112) % 33
    x0 = xindex % 112
    x3 = (xindex // 121968)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-19264) + x0 + (336*x1) + (9408*x2) + (263424*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbq76dm2z526biboi24l7ignp224xghj57cydnlcm6yjovvop47g.py
# Source Nodes: [cat_66], Original ATen: [aten.cat]
# cat_66 => cat_15
triton_poi_fused_cat_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (21952*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 224, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-21952) + x2 + (196*y0) + (21952*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 336, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-43904) + x2 + (196*y0) + (21952*y1)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (y0 + (336*x2) + (65856*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz45cxz346z7ih5pgxl62sbaq24ivrt73buhfqggbkxpdwsqmqkz.py
# Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
# x_160 => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4368
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 336)
    x0 = xindex % 336
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
        tmp3 = tl.load(in_ptr0 + (x0 + (336*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckc5hjoy3ytns3bxb2feznni6naesfmuex6eageinmgt5wre75kx.py
# Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
# x_160 => add_116, add_117, add_118, mul_172, mul_173, mul_174, mul_175, mul_176, rsqrt_22, squeeze_67, var_mean_22
triton_per_fused__native_batch_norm_legit_functional_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_79', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (336*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ft/cfthorkqtmlutsjgj7shqjnnybhy3iv5l3pd6inypxyv6ky2fdnl.py
# Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
# x_160 => add_116, add_119, mul_171, mul_177, rsqrt_22, sub_22, var_mean_22
triton_poi_fused__native_batch_norm_legit_functional_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
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


# kernel path: /tmp/torchinductor_youkaichao/rs/crsbxqqmjsinp7rhawheqi6a47nu3wkspbbq36jpozkai3cdxpdl.py
# Source Nodes: [x_163, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_163 => mul_178, sigmoid_17
# x_se_16 => mean_4
triton_red_fused_mean_silu_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 336
    x1 = (xindex // 336)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (336*r2) + (32928*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxddlsgidshzzaljnotc2d5n5pejcn4aunrgxxftr3ahbcxv2rz.py
# Source Nodes: [x_163, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_163 => mul_178, sigmoid_17
# x_se_16 => mean_4
triton_per_fused_mean_silu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_82', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 336
    x1 = (xindex // 336)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r2) + (672*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/si/csi77drsh6nn47gck7gcbzwtl6jpdcsgxdnejmoq5rcmb7z7lyvr.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
# x_se_17 => convolution_51
# x_se_18 => mul_179, sigmoid_18
triton_poi_fused_convolution_silu_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpqgmdda3vxrkvdlr3yuh4g6s2zh6l6bdmx4wtxut6zswqvkr7r.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163, x_164], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
# x_163 => mul_178, sigmoid_17
# x_164 => mul_180
triton_poi_fused_mul_sigmoid_silu_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 336
    x2 = (xindex // 65856)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (336*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gl6hiji572pdg7ekaecdinp7yvxslnprlqd7jk47gdpcgx2fkk.py
# Source Nodes: [x_165], Original ATen: [aten.convolution]
# x_165 => convolution_53
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (20384*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqe7s5v7cgh2hghzb5fgxsem2rn4trqqsz2dkd3jdzjeliatfcm7.py
# Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
# x_166 => var_mean_23
triton_red_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 104)
    x0 = xindex % 104
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
        tmp3 = tl.load(in_ptr0 + (x0 + (104*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4a/c4akoikni54ieemaebxuin2mfxaepxsod6q3pkvvcbxjkqokdaxd.py
# Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
# x_166 => add_121, add_122, add_123, mul_182, mul_183, mul_184, mul_185, mul_186, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (104*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (104*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3e56mwjk2mlxl2utlxuon5vulxnhmqbb2fzyrert7vcetknol3.py
# Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
# x_166 => add_121, add_124, mul_181, mul_187, rsqrt_23, sub_23, var_mean_23
triton_poi_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 104
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


# kernel path: /tmp/torchinductor_youkaichao/dp/cdpyuleaqzk7ylbtdyk7xcdyspn7566v233sxilkbyl7vtwmiyzc.py
# Source Nodes: [cat_65], Original ATen: [aten.cat]
# cat_65 => cat_16
triton_poi_fused_cat_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 624
    x2 = xindex
    y1 = (yindex // 624)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (61152*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 624, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-61152) + x2 + (196*y0) + (61152*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (624*x2) + (122304*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqug4fiul2n44l26fcw2edukqartj3cnbavb3scpadd6m6lim6a.py
# Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
# x_172 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8112
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 624)
    x0 = xindex % 624
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
        tmp3 = tl.load(in_ptr0 + (x0 + (624*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/on/conuzf64p3brkjjwmnuiozwosnircubzg7mbumhmt5my7gg6rumx.py
# Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
# x_172 => add_126, add_127, add_128, mul_189, mul_190, mul_191, mul_192, mul_193, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_91', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 624
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (624*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (624*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (624*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kt/cktk6dlonuuxikpcwb2tilfsrwx25ueieq22s3ceqws3ort5snu3.py
# Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_172 => add_126, add_129, mul_188, mul_194, rsqrt_24, sub_24, var_mean_24
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 624
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


# kernel path: /tmp/torchinductor_youkaichao/aa/caa7clvotjjd2c44fl43kr4eoneff2hs3yt5exj3blkpftjuajhw.py
# Source Nodes: [x_175], Original ATen: [aten.silu]
# x_175 => mul_195, sigmoid_20
triton_poi_fused_silu_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czeomzsraq22gx22onpjek5ww2tf63uilj66f47xochnzucq65oi.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_0 => convolution_56
triton_poi_fused_convolution_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75kxnsf3oi7ntow7borvasqmww74nog2ylrhtxduvqkd2yij4t3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_1 => convolution_57
triton_poi_fused_convolution_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (30576 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6522edhtvddvyuxeafahw3wyl5konek5im6t33yk4ju5xy3rpy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_2 => convolution_58
triton_poi_fused_convolution_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kkqx2giohgliorjgwgvihe2oqfff5v4t6zbsycs3o3g55v2kya.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_3 => convolution_59
triton_poi_fused_convolution_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1248
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 156
    y1 = (yindex // 156)
    tmp0 = tl.load(in_ptr0 + (91728 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhxatowdbkmy75wj2vxsiaui5oqpfyuz56svylus76oeaoxkxh6.py
# Source Nodes: [cat_64], Original ATen: [aten.cat]
# cat_64 => cat_17
triton_poi_fused_cat_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 624
    x2 = xindex
    y1 = (yindex // 624)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 156, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (30576*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 312, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-30576) + x2 + (196*y0) + (30576*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 468, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-61152) + x2 + (196*y0) + (30576*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 624, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-91728) + x2 + (196*y0) + (30576*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tl.store(out_ptr0 + (y0 + (624*x2) + (122304*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpijf6uuh4uocnui44lefeha273rv5ktwszfflfpzbgw2krmanbg.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => add_131, add_134, mul_196, mul_202, rsqrt_25, sub_25, var_mean_25
triton_poi_fused__native_batch_norm_legit_functional_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 624
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


# kernel path: /tmp/torchinductor_youkaichao/4c/c4cnkbnrqjrqva3bumbx5g3sqfzt22tqi7zqdo3s463qhkxhbw6l.py
# Source Nodes: [x_181, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_181 => mul_203, sigmoid_21
# x_se_20 => mean_5
triton_red_fused_mean_silu_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_100', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 624
    x1 = (xindex // 624)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (624*r2) + (61152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd7jj4bjwajqazqilbi2h77r7gqzv4puazrfibc7t2yysb3y3uo5.py
# Source Nodes: [x_181, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_181 => mul_203, sigmoid_21
# x_se_20 => mean_5
triton_per_fused_mean_silu_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_101', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 624
    x1 = (xindex // 624)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (624*r2) + (1248*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2gg2zsv5fqk3pmqaaaatknhnlbeghybjzhmxjgnjo3krnlddfj.py
# Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
# x_se_21 => convolution_60
# x_se_22 => mul_204, sigmoid_22
triton_poi_fused_convolution_silu_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_102', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 26
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mg/cmgpvmrash46fehllhnbde2faqzufsqphk4m5seeajnybpqlaku7.py
# Source Nodes: [x_se_23], Original ATen: [aten.convolution]
# x_se_23 => convolution_61
triton_poi_fused_convolution_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_103', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 624
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7h6vm4373gayjschq2zkecnaleb3wgxzrqlx4s2fwcliwly3an.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_181, x_182], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_23
# x_181 => mul_203, sigmoid_21
# x_182 => mul_205
triton_poi_fused_mul_sigmoid_silu_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csgs2grm36bb56hippyth2nwxrdajrdmpjc6r3ofpuuww2lw6k6l.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 => convolution_62
triton_poi_fused_convolution_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2496
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crln6hxkqybotyiyciawsusbsevepsbnqzd5raoudozdogjidxwe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 => convolution_63
triton_poi_fused_convolution_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2496
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvql5qxmcodfljkfj637sfnmth4a6jdu7ajncdj5epv4aoevbs2.py
# Source Nodes: [cat_63], Original ATen: [aten.cat]
# cat_63 => cat_18
triton_poi_fused_cat_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 104
    x2 = xindex
    y1 = (yindex // 104)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 52, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (10192*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 104, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10192) + x2 + (196*y0) + (10192*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (104*x2) + (20384*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqkzlq75ip6csncecitemzerwlbvabwsxs44t4tylot4wn4z6xqd.py
# Source Nodes: [shortcut_9, x_185], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_9 => add_140
# x_185 => add_136, add_139, mul_206, mul_212, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_add_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 104
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


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbzini7hczzym3pt7n3q66ovbdjfvjpw2i2hmp6djirik3jbrve.py
# Source Nodes: [x_230], Original ATen: [aten.convolution]
# x_230 => convolution_84
triton_poi_fused_convolution_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (624*x2) + (122304*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazpok3ke3ytkgyrwetoq4ascvldrzopptwjsp5uw6z5xdqwb3xv.py
# Source Nodes: [x_231, x_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_231 => add_174, add_177, mul_263, mul_269, rsqrt_33, sub_33, var_mean_33
# x_234 => mul_270, sigmoid_32
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 624
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


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vvhdfhma23dkxkbms4gqxsi3cf2yroj3dqugntwcziolxttkql.py
# Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
# x_se_33 => convolution_86
# x_se_34 => mul_279, sigmoid_34
triton_poi_fused_convolution_silu_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_111', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 52
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdemannlvcnzuqxsfci7icwetd5pum65vwxajcggqe5mg6wuyq5r.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239, x_240], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
# x_239 => mul_278, sigmoid_33
# x_240 => mul_280
triton_poi_fused_mul_sigmoid_silu_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 978432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 624
    x2 = (xindex // 122304)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (624*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4j7ctvwtv3adzw7tate6e5uyaanvqgc6k45gmmrngvedagefwcr.py
# Source Nodes: [x_241], Original ATen: [aten.convolution]
# x_241 => convolution_88
triton_poi_fused_convolution_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (31360*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5dcd7riw2v3jjj27nezqth2ck3dckhhyl3jfsyvhwx6qqbp3fd.py
# Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
# x_242 => var_mean_35
triton_red_fused__native_batch_norm_legit_functional_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2080
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 160)
    x0 = xindex % 160
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
        tmp3 = tl.load(in_ptr0 + (x0 + (160*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ka/ckays3ibmxdobdy4vt4g4draymabrnrmbier2j4fctvr7nvubh5c.py
# Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
# x_242 => add_184, add_185, add_186, mul_282, mul_283, mul_284, mul_285, mul_286, rsqrt_35, squeeze_106, var_mean_35
triton_per_fused__native_batch_norm_legit_functional_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_115', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (160*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (160*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxtk4we6zqvkncsx7fk6nhmqs5rkm6annxunav5k5vg6af3mni5.py
# Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
# x_242 => add_184, add_187, mul_281, mul_287, rsqrt_35, sub_35, var_mean_35
triton_poi_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
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


# kernel path: /tmp/torchinductor_youkaichao/ln/cln6ipvpbkogug7jh3v5ljio3d62oufkxovaazyu6gavgd5smjbx.py
# Source Nodes: [cat_56], Original ATen: [aten.cat]
# cat_56 => cat_25
triton_poi_fused_cat_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_117', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 480
    x2 = xindex
    y1 = (yindex // 480)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (47040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-47040) + x2 + (196*y0) + (47040*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rm/crm6dyt7lwsj6b65zfciqhp36z575okufckjya3bvpbbmq6e37c5.py
# Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
# x_248 => var_mean_36
triton_red_fused__native_batch_norm_legit_functional_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_118', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pm/cpm7f4oqvtsqw2rohojm6e353qqgc6ktg6hci2dehrwhfurqwvib.py
# Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
# x_248 => add_189, add_190, add_191, mul_289, mul_290, mul_291, mul_292, mul_293, rsqrt_36, squeeze_109, var_mean_36
triton_per_fused__native_batch_norm_legit_functional_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_119', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhsz7bf6gx6dhvh7ck5mi2p4qtmax36jm364u6jn3lnwuohu4lo.py
# Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_248 => add_189, add_192, mul_288, mul_294, rsqrt_36, sub_36, var_mean_36
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr7kczkgdkwzmlit5pv4d2npmqfczwwc2fui34cqifenuqjxf5b.py
# Source Nodes: [x_251], Original ATen: [aten.silu]
# x_251 => mul_295, sigmoid_36
triton_poi_fused_silu_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_121', 'mutated_arg_names': []},
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
    y0 = yindex % 480
    y1 = (yindex // 480)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/cey4nni2nzhltw4su3fnnhlqhdetklx7g5zdcnvq3emfk4qhu5v4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_0 => convolution_91
triton_poi_fused_convolution_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_122', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd3ycf2ntbr3volu43habry2aftfjoimiyiynrkwrxuzw5msyt7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_1 => convolution_92
triton_poi_fused_convolution_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (23520 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7n6c27y7zuyw5sjdzmkf6wywxstg36ro6d277bydhqyx4nutnzl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_2 => convolution_93
triton_poi_fused_convolution_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bjgh5xnmbhud532reizfgaffjszorys4tlajrgaojxevlvzj2j.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_3 => convolution_94
triton_poi_fused_convolution_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_125', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (70560 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqan7wv6athwpppjfyfxwthg7dzdqo3xyi4ewkaga55pznhrcqy.py
# Source Nodes: [cat_55], Original ATen: [aten.cat]
# cat_55 => cat_26
triton_poi_fused_cat_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_126', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 480
    x2 = xindex
    y1 = (yindex // 480)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 120, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (23520*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 240, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-23520) + x2 + (196*y0) + (23520*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 360, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-47040) + x2 + (196*y0) + (23520*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 480, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-70560) + x2 + (196*y0) + (23520*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/coj6nwutptfjbew4pnjcyvhglwmngwdaek6xcjpx2quskkt77nxy.py
# Source Nodes: [x_254], Original ATen: [aten._native_batch_norm_legit_functional]
# x_254 => add_194, add_197, mul_296, mul_302, rsqrt_37, sub_37, var_mean_37
triton_poi_fused__native_batch_norm_legit_functional_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_127', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/po/cpoqxbygid3vr6mosvpz5t74r33jgm7tvc47q62ek7jy7nmginkn.py
# Source Nodes: [x_257, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_257 => mul_303, sigmoid_37
# x_se_36 => mean_9
triton_red_fused_mean_silu_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_128', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lu/clu64kt2on6inolstftr4yaf6ewr7ngplcjrivw2hjsqawivteoe.py
# Source Nodes: [x_257, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_257 => mul_303, sigmoid_37
# x_se_36 => mean_9
triton_per_fused_mean_silu_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_129', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/se/cseuq7uqmmsav4ff7nvlieqxgo4fn67evly6tlcbq2ag2oza6x6t.py
# Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
# x_se_37 => convolution_95
# x_se_38 => mul_304, sigmoid_38
triton_poi_fused_convolution_silu_130 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_130', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7qwu4etqgl4ebvoxnjbf7dbsfl5bb76er5wtck2geokxpjxc7b.py
# Source Nodes: [x_se_39], Original ATen: [aten.convolution]
# x_se_39 => convolution_96
triton_poi_fused_convolution_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_131', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ba/cbamrjklohmguauslpiq6bpujicpomydtupcubw3dzs55l6kx5eq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_257, x_258], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_257 => mul_303, sigmoid_37
# x_258 => mul_305
triton_poi_fused_mul_sigmoid_silu_132 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citnzvohvgx4d7uxhjz6njffsrictbtppm2amw7i3ai5qlmsfdmn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 => convolution_97
triton_poi_fused_convolution_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_133', 'mutated_arg_names': []},
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
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4g/c4graoxgv6xqzjssb3mzemwxfoif5ykjzukrrajlfpl3ee3k4u2p.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 => convolution_98
triton_poi_fused_convolution_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_134', 'mutated_arg_names': []},
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
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxodf4zxypesqveinqsx4i2tfacsuiuaf54vcahnipxg2uvbmwxa.py
# Source Nodes: [cat_54], Original ATen: [aten.cat]
# cat_54 => cat_27
triton_poi_fused_cat_135 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_135', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 160
    x2 = xindex
    y1 = (yindex // 160)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (15680*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-15680) + x2 + (196*y0) + (15680*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (160*x2) + (31360*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cin4lsz5eqp3b2bwflrqlqjl3liawuqyzp42ocjvev77zbr7u4s7.py
# Source Nodes: [shortcut_13, x_261], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_13 => add_203
# x_261 => add_199, add_202, mul_306, mul_312, rsqrt_38, sub_38, var_mean_38
triton_poi_fused__native_batch_norm_legit_functional_add_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
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


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhsdmykdfuzuudj4e42xft6yyec5mfmhe3mvuhrgq2bfy2z7siu.py
# Source Nodes: [x_306], Original ATen: [aten.convolution]
# x_306 => convolution_119
triton_poi_fused_convolution_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_137', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (960*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinmhol5qlxycwu6rtfcaloh5xnmqgzbmx4li4jnjhntppvuwkld.py
# Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
# x_307 => var_mean_45
triton_red_fused__native_batch_norm_legit_functional_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_138', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12480
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 960)
    x0 = xindex % 960
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
        tmp3 = tl.load(in_ptr0 + (x0 + (960*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/63/c635dtj2yyclxef66axjv2vypmyvvaa6dyt3cw6vr6vtfgjyvh66.py
# Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
# x_307 => add_237, add_238, add_239, mul_364, mul_365, mul_366, mul_367, mul_368, rsqrt_45, squeeze_136, var_mean_45
triton_per_fused__native_batch_norm_legit_functional_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_139', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (960*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4a/c4aecrnqmu6yvqhuy4f52bevqj63nhym6knesfso4onj35ug3ibj.py
# Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_307 => add_237, add_240, mul_363, mul_369, rsqrt_45, sub_45, var_mean_45
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_140', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qs/cqs7vbcc4by72ir46j55cjalj3373ajxwesho7cakyjhizm6kgvs.py
# Source Nodes: [x_312], Original ATen: [aten.constant_pad_nd]
# x_312 => constant_pad_nd_11
triton_poi_fused_constant_pad_nd_141 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_141', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3600) % 15
    x1 = (xindex // 240) % 15
    x0 = xindex % 240
    x3 = (xindex // 54000)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 14, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4k6vxpqanifk5ivafdc55d6u4ce2cnd4lkrs2c56l7iawauasyz.py
# Source Nodes: [x_314], Original ATen: [aten.constant_pad_nd]
# x_314 => constant_pad_nd_12
triton_poi_fused_constant_pad_nd_142 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_142', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 554880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4080) % 17
    x1 = (xindex // 240) % 17
    x0 = xindex % 240
    x3 = (xindex // 69360)
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
    tmp11 = tl.load(in_ptr0 + ((-14160) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddbbbrddmsukes6yyd2s3sycht6452bp4txmqsvt33irgsunle5.py
# Source Nodes: [x_316], Original ATen: [aten.constant_pad_nd]
# x_316 => constant_pad_nd_13
triton_poi_fused_constant_pad_nd_143 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_143', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 693120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4560) % 19
    x1 = (xindex // 240) % 19
    x0 = xindex % 240
    x3 = (xindex // 86640)
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-28320) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5y5iutx6m6nnlhpcnxwsyq4dibhhs2nueeonw4ifjwkmvx6jjgn.py
# Source Nodes: [x_318], Original ATen: [aten.constant_pad_nd]
# x_318 => constant_pad_nd_14
triton_poi_fused_constant_pad_nd_144 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_144', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 846720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5040) % 21
    x1 = (xindex // 240) % 21
    x0 = xindex % 240
    x3 = (xindex // 105840)
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-42480) + x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csapcxe4qksrym47kduqs2mhnsq6nwsglkbk6ikrafaapnoxtael.py
# Source Nodes: [cat_47], Original ATen: [aten.cat]
# cat_47 => cat_34
triton_poi_fused_cat_145 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_145', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 960
    x2 = xindex
    y1 = (yindex // 960)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (11760*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-11760) + x2 + (49*y0) + (11760*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 720, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-23520) + x2 + (49*y0) + (11760*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 960, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-35280) + x2 + (49*y0) + (11760*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjt2l3ovklgjzcmqz7sorjqjykp7csxqgpod5jnomjnoguej4uet.py
# Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
# x_321 => var_mean_46
triton_red_fused__native_batch_norm_legit_functional_146 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4o65mca4kmuibq7fr2jytux3zpo3ml5yu7cg4t45nmcwniddiw.py
# Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
# x_321 => add_242, add_243, add_244, mul_372, mul_373, mul_374, mul_375, mul_376, rsqrt_46, squeeze_139, var_mean_46
triton_per_fused__native_batch_norm_legit_functional_147 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_147', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (960*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjiws5b4zrjg3rapcxegqwtegmrh2esb6b6wvxhmhmjlrx224dbk.py
# Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
# x_321 => add_242, add_245, mul_371, mul_377, rsqrt_46, sub_46, var_mean_46
triton_poi_fused__native_batch_norm_legit_functional_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_148', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
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


# kernel path: /tmp/torchinductor_youkaichao/qs/cqsfwqhaqfchvby4ov2qb2bqkj2kwpoxqlg6jbhf2s6pxkek2fyc.py
# Source Nodes: [x_324, x_se_48], Original ATen: [aten.mean, aten.silu]
# x_324 => mul_378, sigmoid_49
# x_se_48 => mean_12
triton_per_fused_mean_silu_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_149', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyl6jiwtid3cjlamxy62ryw3lklkv27ubxgz6awiol42uc7btsz.py
# Source Nodes: [x_se_51], Original ATen: [aten.convolution]
# x_se_51 => convolution_125
triton_poi_fused_convolution_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_150', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzruvqfc4py7afvsios72azyjjwroafnxknwg44ozuwnub5jmr2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324, x_325], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
# x_324 => mul_378, sigmoid_49
# x_325 => mul_380
triton_poi_fused_mul_sigmoid_silu_151 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_151', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 960
    x2 = (xindex // 47040)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (960*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnw57sp5hudbrpyh7nrsput6myw4i2mzak63ga3q57epc5kvhkp3.py
# Source Nodes: [x_326], Original ATen: [aten.convolution]
# x_326 => convolution_126
triton_poi_fused_convolution_152 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 264
    y1 = (yindex // 264)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (264*x2) + (12936*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7dyreqbwi764llrhm3lflkzlak5wltwbwtrp3zlpcljacmf3mu.py
# Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
# x_327 => var_mean_47
triton_red_fused__native_batch_norm_legit_functional_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_153', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1056
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 264
    x1 = (xindex // 264)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (264*r2) + (25872*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ev/cevekfx7pzw32kkitiozouzzujozdorgxab7e4kjlwwguw6q35sa.py
# Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
# x_327 => add_247, add_248, add_249, mul_382, mul_383, mul_384, mul_385, mul_386, rsqrt_47, squeeze_142, var_mean_47
triton_per_fused__native_batch_norm_legit_functional_154 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_154', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 264
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (264*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (264*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (264*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gbkcsz57ohzdi7m2ftf6gpthq3d2ycthmtsuhsom4xhojiv643.py
# Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
# x_327 => add_247, add_250, mul_381, mul_387, rsqrt_47, sub_47, var_mean_47
triton_poi_fused__native_batch_norm_legit_functional_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_155', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 103488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 264
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


# kernel path: /tmp/torchinductor_youkaichao/sm/csmk76kvbdruyjdv5mmafz4bnp23yomczgdpd5uu7bjy3byb7fsw.py
# Source Nodes: [x_331], Original ATen: [aten.convolution]
# x_331 => convolution_127
triton_poi_fused_convolution_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_156', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1584
    y1 = (yindex // 1584)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1584*x2) + (77616*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvljptzu7z4fvmvdkkecqgythi2lctagdukpcibm7clc4qitkvve.py
# Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional]
# x_332 => var_mean_48
triton_red_fused__native_batch_norm_legit_functional_157 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_157', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6336
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1584
    x1 = (xindex // 1584)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1584*r2) + (155232*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ql/cqlt3qqbz3xurx657h7kknbjrlyrzufich6inwpgude3636ex3hu.py
# Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional]
# x_332 => add_252, add_253, add_254, mul_389, mul_390, mul_391, mul_392, mul_393, rsqrt_48, squeeze_145, var_mean_48
triton_per_fused__native_batch_norm_legit_functional_158 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_158', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1584
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1584*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1584*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1584*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/p3/cp326ywllxv5nwhmfgisqkjwhyizj2zt5uzo2slfnw7c3ju5wukw.py
# Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_332 => add_252, add_255, mul_388, mul_394, rsqrt_48, sub_48, var_mean_48
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_159', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 620928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1584
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
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dpi3eiphykdrbkq4cdj5rwwhttjhkhueep25kizpvv2okdfn7r.py
# Source Nodes: [x_335], Original ATen: [aten.silu]
# x_335 => mul_395, sigmoid_52
triton_poi_fused_silu_160 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_160', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1584
    y1 = (yindex // 1584)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1584*x2) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmuh6fg6k57mvsre6vdlhvbvdqb6tfwpkss4gjkbzz5p3t52vuhb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_0 => convolution_128
triton_poi_fused_convolution_161 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_161', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6crfosnc7fzdiiu6bajznh2xj3tspvkjvxhvvhzbebovrpggiz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_1 => convolution_129
triton_poi_fused_convolution_162 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_162', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (19404 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7fp2pqv65hxbhla54glracghcgfxolprsac6jtj6m5ykkufid4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_2 => convolution_130
triton_poi_fused_convolution_163 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_163', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmphb23dgwzlnpmc74p26ntdwtanxvqyeli7ujac5jhkcbjrjmrd.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_3 => convolution_131
triton_poi_fused_convolution_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_164', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 396
    y1 = (yindex // 396)
    tmp0 = tl.load(in_ptr0 + (58212 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hsbz2su6vc676o6hrqcqi3wneil3ldibiihqyreprqu5qlg3j2.py
# Source Nodes: [cat_46], Original ATen: [aten.cat]
# cat_46 => cat_35
triton_poi_fused_cat_165 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_165', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1584
    x2 = xindex
    y1 = (yindex // 1584)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 396, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (19404*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 792, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-19404) + x2 + (49*y0) + (19404*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 1188, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-38808) + x2 + (49*y0) + (19404*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 1584, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-58212) + x2 + (49*y0) + (19404*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tl.store(out_ptr0 + (y0 + (1584*x2) + (77616*y1)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curqvwavwv6tijjji2l2rmbyh6ksmzadqty3fnqi3u5tjabvcz4u.py
# Source Nodes: [x_338], Original ATen: [aten._native_batch_norm_legit_functional]
# x_338 => add_257, add_260, mul_396, mul_402, rsqrt_49, sub_49, var_mean_49
triton_poi_fused__native_batch_norm_legit_functional_166 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_166', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 620928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1584
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dzywra6tv3373bd3yc6kspi37q5hfcok5gxwwdaq7fqfcsxhuv.py
# Source Nodes: [x_341, x_se_52], Original ATen: [aten.mean, aten.silu]
# x_341 => mul_403, sigmoid_53
# x_se_52 => mean_13
triton_per_fused_mean_silu_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_167', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12672
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1584
    x1 = (xindex // 1584)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1584*r2) + (77616*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5exhgkwmagv7mgtym7kcbq6stascqi2ybzu7cru7bpwmthbznq.py
# Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
# x_se_53 => convolution_132
# x_se_54 => mul_404, sigmoid_54
triton_poi_fused_convolution_silu_168 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_168', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 132
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crtl4uvtgusz6k3lyo6puhallno2zm5u2fwj2djmoqtgxt2paxxz.py
# Source Nodes: [x_se_55], Original ATen: [aten.convolution]
# x_se_55 => convolution_133
triton_poi_fused_convolution_169 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_169', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1584
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd763yvrjez3kl7bbbvsszwmncmrc2msrkzviq7smb5ps4xdqd55.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_341, x_342], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_55
# x_341 => mul_403, sigmoid_53
# x_342 => mul_405
triton_poi_fused_mul_sigmoid_silu_170 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_170', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1584
    y1 = (yindex // 1584)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1584*x2) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxyhsmkhoasgr6gslw447hojeeig3ojdnrom6biawodh2g5bvxf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 => convolution_134
triton_poi_fused_convolution_171 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_171', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6336
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 792
    y1 = (yindex // 792)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4iqqxtw4cxln6zbtfpr2fotad3mrxgjiji7eem53rjcfq4shsa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 => convolution_135
triton_poi_fused_convolution_172 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_172', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6336
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 792
    y1 = (yindex // 792)
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/cls5jgiceqsr5s5576l2q4yasdnxmxs67tapngqnqi56l3asglko.py
# Source Nodes: [cat_45], Original ATen: [aten.cat]
# cat_45 => cat_36
triton_poi_fused_cat_173 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_173', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 264
    x2 = xindex
    y1 = (yindex // 264)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 132, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (6468*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 264, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-6468) + x2 + (49*y0) + (6468*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (y0 + (264*x2) + (12936*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5zzltieovmebamz4pdld634v5gj6szcpzynj3ic4dhl4wqzeoa.py
# Source Nodes: [shortcut_17, x_345], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_17 => add_266
# x_345 => add_262, add_265, mul_406, mul_412, rsqrt_50, sub_50, var_mean_50
triton_poi_fused__native_batch_norm_legit_functional_add_174 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_174', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 103488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 264
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


# kernel path: /tmp/torchinductor_youkaichao/6u/c6uqid6jpzzs7mxfs7sniiy4i5pe7qwf6qz3fw3lcnlzi4afr4vk.py
# Source Nodes: [x_389], Original ATen: [aten.convolution]
# x_389 => convolution_154
triton_poi_fused_convolution_175 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_175', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1536*x2) + (75264*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jr/cjr3ya7gm4ybqmht4k7joiat5b4itqdem7h4m4l533gcq4vzufow.py
# Source Nodes: [x_390], Original ATen: [aten._native_batch_norm_legit_functional]
# x_390 => var_mean_57
triton_red_fused__native_batch_norm_legit_functional_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_176', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (150528*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iskekc55cuakws534vry6cg6e6owqekvokhawpuuncaavmk7xc.py
# Source Nodes: [x_390], Original ATen: [aten._native_batch_norm_legit_functional]
# x_390 => add_300, add_301, add_302, mul_464, mul_465, mul_466, mul_467, mul_468, rsqrt_57, squeeze_172, var_mean_57
triton_per_fused__native_batch_norm_legit_functional_177 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_177', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/32/c32jdmtpplpffddo63qqplryqz3qxjav5t33j64iyksvtsl67muk.py
# Source Nodes: [x_390, x_394], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_390 => add_300, add_303, mul_463, mul_469, rsqrt_57, sub_57, var_mean_57
# x_394 => relu_6
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_178 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_178', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmukezcav5jnnubuhugfu4zie54lgr5fiupjpfacaqvdkvsk4zs.py
# Source Nodes: [x_395, x_397], Original ATen: [aten.mean, aten.view]
# x_395 => mean_16
# x_397 => view
triton_per_fused_mean_view_179 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_179', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (75264*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cch765bbyvt64ndwr7cbc2zv6olo5wsymmes6qjhtryhvd3cznhh.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_180 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_180', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_12, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (40, ), (1, ))
    assert_size_stride(primals_16, (40, ), (1, ))
    assert_size_stride(primals_17, (120, ), (1, ))
    assert_size_stride(primals_18, (120, ), (1, ))
    assert_size_stride(primals_19, (120, ), (1, ))
    assert_size_stride(primals_20, (120, ), (1, ))
    assert_size_stride(primals_21, (40, ), (1, ))
    assert_size_stride(primals_22, (40, ), (1, ))
    assert_size_stride(primals_23, (240, ), (1, ))
    assert_size_stride(primals_24, (240, ), (1, ))
    assert_size_stride(primals_25, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_26, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_27, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_28, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_29, (240, ), (1, ))
    assert_size_stride(primals_30, (240, ), (1, ))
    assert_size_stride(primals_31, (56, ), (1, ))
    assert_size_stride(primals_32, (56, ), (1, ))
    assert_size_stride(primals_33, (336, ), (1, ))
    assert_size_stride(primals_34, (336, ), (1, ))
    assert_size_stride(primals_35, (336, ), (1, ))
    assert_size_stride(primals_36, (336, ), (1, ))
    assert_size_stride(primals_37, (56, ), (1, ))
    assert_size_stride(primals_38, (56, ), (1, ))
    assert_size_stride(primals_39, (336, ), (1, ))
    assert_size_stride(primals_40, (336, ), (1, ))
    assert_size_stride(primals_41, (336, ), (1, ))
    assert_size_stride(primals_42, (336, ), (1, ))
    assert_size_stride(primals_43, (56, ), (1, ))
    assert_size_stride(primals_44, (56, ), (1, ))
    assert_size_stride(primals_45, (336, ), (1, ))
    assert_size_stride(primals_46, (336, ), (1, ))
    assert_size_stride(primals_47, (336, ), (1, ))
    assert_size_stride(primals_48, (336, ), (1, ))
    assert_size_stride(primals_49, (56, ), (1, ))
    assert_size_stride(primals_50, (56, ), (1, ))
    assert_size_stride(primals_51, (336, ), (1, ))
    assert_size_stride(primals_52, (336, ), (1, ))
    assert_size_stride(primals_53, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_54, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_55, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_56, (336, ), (1, ))
    assert_size_stride(primals_57, (336, ), (1, ))
    assert_size_stride(primals_58, (104, ), (1, ))
    assert_size_stride(primals_59, (104, ), (1, ))
    assert_size_stride(primals_60, (624, ), (1, ))
    assert_size_stride(primals_61, (624, ), (1, ))
    assert_size_stride(primals_62, (624, ), (1, ))
    assert_size_stride(primals_63, (624, ), (1, ))
    assert_size_stride(primals_64, (104, ), (1, ))
    assert_size_stride(primals_65, (104, ), (1, ))
    assert_size_stride(primals_66, (624, ), (1, ))
    assert_size_stride(primals_67, (624, ), (1, ))
    assert_size_stride(primals_68, (624, ), (1, ))
    assert_size_stride(primals_69, (624, ), (1, ))
    assert_size_stride(primals_70, (104, ), (1, ))
    assert_size_stride(primals_71, (104, ), (1, ))
    assert_size_stride(primals_72, (624, ), (1, ))
    assert_size_stride(primals_73, (624, ), (1, ))
    assert_size_stride(primals_74, (624, ), (1, ))
    assert_size_stride(primals_75, (624, ), (1, ))
    assert_size_stride(primals_76, (104, ), (1, ))
    assert_size_stride(primals_77, (104, ), (1, ))
    assert_size_stride(primals_78, (624, ), (1, ))
    assert_size_stride(primals_79, (624, ), (1, ))
    assert_size_stride(primals_80, (624, ), (1, ))
    assert_size_stride(primals_81, (624, ), (1, ))
    assert_size_stride(primals_82, (160, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_84, (480, ), (1, ))
    assert_size_stride(primals_85, (480, ), (1, ))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_87, (480, ), (1, ))
    assert_size_stride(primals_88, (160, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_90, (480, ), (1, ))
    assert_size_stride(primals_91, (480, ), (1, ))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_93, (480, ), (1, ))
    assert_size_stride(primals_94, (160, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_96, (480, ), (1, ))
    assert_size_stride(primals_97, (480, ), (1, ))
    assert_size_stride(primals_98, (480, ), (1, ))
    assert_size_stride(primals_99, (480, ), (1, ))
    assert_size_stride(primals_100, (160, ), (1, ))
    assert_size_stride(primals_101, (160, ), (1, ))
    assert_size_stride(primals_102, (960, ), (1, ))
    assert_size_stride(primals_103, (960, ), (1, ))
    assert_size_stride(primals_104, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_107, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_108, (960, ), (1, ))
    assert_size_stride(primals_109, (960, ), (1, ))
    assert_size_stride(primals_110, (264, ), (1, ))
    assert_size_stride(primals_111, (264, ), (1, ))
    assert_size_stride(primals_112, (1584, ), (1, ))
    assert_size_stride(primals_113, (1584, ), (1, ))
    assert_size_stride(primals_114, (1584, ), (1, ))
    assert_size_stride(primals_115, (1584, ), (1, ))
    assert_size_stride(primals_116, (264, ), (1, ))
    assert_size_stride(primals_117, (264, ), (1, ))
    assert_size_stride(primals_118, (1584, ), (1, ))
    assert_size_stride(primals_119, (1584, ), (1, ))
    assert_size_stride(primals_120, (1584, ), (1, ))
    assert_size_stride(primals_121, (1584, ), (1, ))
    assert_size_stride(primals_122, (264, ), (1, ))
    assert_size_stride(primals_123, (264, ), (1, ))
    assert_size_stride(primals_124, (1584, ), (1, ))
    assert_size_stride(primals_125, (1584, ), (1, ))
    assert_size_stride(primals_126, (1584, ), (1, ))
    assert_size_stride(primals_127, (1584, ), (1, ))
    assert_size_stride(primals_128, (264, ), (1, ))
    assert_size_stride(primals_129, (264, ), (1, ))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_131, (1536, ), (1, ))
    assert_size_stride(primals_132, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_135, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_136, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_137, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_138, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_139, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_140, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_141, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_142, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_143, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_144, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_145, (20, ), (1, ))
    assert_size_stride(primals_146, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_147, (240, ), (1, ))
    assert_size_stride(primals_148, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_149, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_150, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_151, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_154, (28, ), (1, ))
    assert_size_stride(primals_155, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_156, (336, ), (1, ))
    assert_size_stride(primals_157, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_158, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_159, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_160, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_161, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_163, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_164, (28, ), (1, ))
    assert_size_stride(primals_165, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_166, (336, ), (1, ))
    assert_size_stride(primals_167, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_168, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_169, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_170, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_171, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_172, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_173, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_174, (28, ), (1, ))
    assert_size_stride(primals_175, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_176, (336, ), (1, ))
    assert_size_stride(primals_177, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_178, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_179, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_180, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_181, (14, ), (1, ))
    assert_size_stride(primals_182, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_183, (336, ), (1, ))
    assert_size_stride(primals_184, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_185, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_186, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_187, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_188, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_189, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_190, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_191, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_192, (26, ), (1, ))
    assert_size_stride(primals_193, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_194, (624, ), (1, ))
    assert_size_stride(primals_195, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_196, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_197, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_198, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_199, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_201, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_202, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_203, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_204, (26, ), (1, ))
    assert_size_stride(primals_205, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_206, (624, ), (1, ))
    assert_size_stride(primals_207, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_208, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_209, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_210, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_211, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_212, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_213, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_214, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_215, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_216, (26, ), (1, ))
    assert_size_stride(primals_217, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_218, (624, ), (1, ))
    assert_size_stride(primals_219, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_220, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_221, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_222, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_223, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_224, (52, ), (1, ))
    assert_size_stride(primals_225, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_226, (624, ), (1, ))
    assert_size_stride(primals_227, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_228, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_229, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_230, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_231, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_232, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_233, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_234, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_235, (80, ), (1, ))
    assert_size_stride(primals_236, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_237, (480, ), (1, ))
    assert_size_stride(primals_238, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_239, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_240, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_241, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_242, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_244, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_245, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_246, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_247, (80, ), (1, ))
    assert_size_stride(primals_248, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_249, (480, ), (1, ))
    assert_size_stride(primals_250, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_251, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_252, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_253, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_254, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_255, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_256, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_257, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_258, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_259, (80, ), (1, ))
    assert_size_stride(primals_260, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_261, (480, ), (1, ))
    assert_size_stride(primals_262, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_263, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_264, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_265, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_266, (80, ), (1, ))
    assert_size_stride(primals_267, (960, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_268, (960, ), (1, ))
    assert_size_stride(primals_269, (264, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_270, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_271, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_272, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_273, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_274, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_275, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_276, (132, ), (1, ))
    assert_size_stride(primals_277, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_278, (1584, ), (1, ))
    assert_size_stride(primals_279, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_280, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_281, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_282, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_283, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_284, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_285, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_286, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_287, (132, ), (1, ))
    assert_size_stride(primals_288, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_289, (1584, ), (1, ))
    assert_size_stride(primals_290, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_291, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_292, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_293, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_294, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_295, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_296, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_297, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_298, (132, ), (1, ))
    assert_size_stride(primals_299, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_300, (1584, ), (1, ))
    assert_size_stride(primals_301, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_302, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_303, (1536, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_304, (1000, 1536), (1536, 1))
    assert_size_stride(primals_305, (1000, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (32, ), (1, ))
    assert_size_stride(primals_308, (32, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (32, ), (1, ))
    assert_size_stride(primals_311, (32, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (32, ), (1, ))
    assert_size_stride(primals_314, (32, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (192, ), (1, ))
    assert_size_stride(primals_317, (192, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (192, ), (1, ))
    assert_size_stride(primals_320, (192, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (40, ), (1, ))
    assert_size_stride(primals_323, (40, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (120, ), (1, ))
    assert_size_stride(primals_326, (120, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (120, ), (1, ))
    assert_size_stride(primals_329, (120, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (40, ), (1, ))
    assert_size_stride(primals_332, (40, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (240, ), (1, ))
    assert_size_stride(primals_335, (240, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (240, ), (1, ))
    assert_size_stride(primals_338, (240, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (56, ), (1, ))
    assert_size_stride(primals_341, (56, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (336, ), (1, ))
    assert_size_stride(primals_344, (336, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (336, ), (1, ))
    assert_size_stride(primals_347, (336, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (56, ), (1, ))
    assert_size_stride(primals_350, (56, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (336, ), (1, ))
    assert_size_stride(primals_353, (336, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (336, ), (1, ))
    assert_size_stride(primals_356, (336, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (56, ), (1, ))
    assert_size_stride(primals_359, (56, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (336, ), (1, ))
    assert_size_stride(primals_362, (336, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (336, ), (1, ))
    assert_size_stride(primals_365, (336, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (56, ), (1, ))
    assert_size_stride(primals_368, (56, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (336, ), (1, ))
    assert_size_stride(primals_371, (336, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (336, ), (1, ))
    assert_size_stride(primals_374, (336, ), (1, ))
    assert_size_stride(primals_375, (), ())
    assert_size_stride(primals_376, (104, ), (1, ))
    assert_size_stride(primals_377, (104, ), (1, ))
    assert_size_stride(primals_378, (), ())
    assert_size_stride(primals_379, (624, ), (1, ))
    assert_size_stride(primals_380, (624, ), (1, ))
    assert_size_stride(primals_381, (), ())
    assert_size_stride(primals_382, (624, ), (1, ))
    assert_size_stride(primals_383, (624, ), (1, ))
    assert_size_stride(primals_384, (), ())
    assert_size_stride(primals_385, (104, ), (1, ))
    assert_size_stride(primals_386, (104, ), (1, ))
    assert_size_stride(primals_387, (), ())
    assert_size_stride(primals_388, (624, ), (1, ))
    assert_size_stride(primals_389, (624, ), (1, ))
    assert_size_stride(primals_390, (), ())
    assert_size_stride(primals_391, (624, ), (1, ))
    assert_size_stride(primals_392, (624, ), (1, ))
    assert_size_stride(primals_393, (), ())
    assert_size_stride(primals_394, (104, ), (1, ))
    assert_size_stride(primals_395, (104, ), (1, ))
    assert_size_stride(primals_396, (), ())
    assert_size_stride(primals_397, (624, ), (1, ))
    assert_size_stride(primals_398, (624, ), (1, ))
    assert_size_stride(primals_399, (), ())
    assert_size_stride(primals_400, (624, ), (1, ))
    assert_size_stride(primals_401, (624, ), (1, ))
    assert_size_stride(primals_402, (), ())
    assert_size_stride(primals_403, (104, ), (1, ))
    assert_size_stride(primals_404, (104, ), (1, ))
    assert_size_stride(primals_405, (), ())
    assert_size_stride(primals_406, (624, ), (1, ))
    assert_size_stride(primals_407, (624, ), (1, ))
    assert_size_stride(primals_408, (), ())
    assert_size_stride(primals_409, (624, ), (1, ))
    assert_size_stride(primals_410, (624, ), (1, ))
    assert_size_stride(primals_411, (), ())
    assert_size_stride(primals_412, (160, ), (1, ))
    assert_size_stride(primals_413, (160, ), (1, ))
    assert_size_stride(primals_414, (), ())
    assert_size_stride(primals_415, (480, ), (1, ))
    assert_size_stride(primals_416, (480, ), (1, ))
    assert_size_stride(primals_417, (), ())
    assert_size_stride(primals_418, (480, ), (1, ))
    assert_size_stride(primals_419, (480, ), (1, ))
    assert_size_stride(primals_420, (), ())
    assert_size_stride(primals_421, (160, ), (1, ))
    assert_size_stride(primals_422, (160, ), (1, ))
    assert_size_stride(primals_423, (), ())
    assert_size_stride(primals_424, (480, ), (1, ))
    assert_size_stride(primals_425, (480, ), (1, ))
    assert_size_stride(primals_426, (), ())
    assert_size_stride(primals_427, (480, ), (1, ))
    assert_size_stride(primals_428, (480, ), (1, ))
    assert_size_stride(primals_429, (), ())
    assert_size_stride(primals_430, (160, ), (1, ))
    assert_size_stride(primals_431, (160, ), (1, ))
    assert_size_stride(primals_432, (), ())
    assert_size_stride(primals_433, (480, ), (1, ))
    assert_size_stride(primals_434, (480, ), (1, ))
    assert_size_stride(primals_435, (), ())
    assert_size_stride(primals_436, (480, ), (1, ))
    assert_size_stride(primals_437, (480, ), (1, ))
    assert_size_stride(primals_438, (), ())
    assert_size_stride(primals_439, (160, ), (1, ))
    assert_size_stride(primals_440, (160, ), (1, ))
    assert_size_stride(primals_441, (), ())
    assert_size_stride(primals_442, (960, ), (1, ))
    assert_size_stride(primals_443, (960, ), (1, ))
    assert_size_stride(primals_444, (), ())
    assert_size_stride(primals_445, (960, ), (1, ))
    assert_size_stride(primals_446, (960, ), (1, ))
    assert_size_stride(primals_447, (), ())
    assert_size_stride(primals_448, (264, ), (1, ))
    assert_size_stride(primals_449, (264, ), (1, ))
    assert_size_stride(primals_450, (), ())
    assert_size_stride(primals_451, (1584, ), (1, ))
    assert_size_stride(primals_452, (1584, ), (1, ))
    assert_size_stride(primals_453, (), ())
    assert_size_stride(primals_454, (1584, ), (1, ))
    assert_size_stride(primals_455, (1584, ), (1, ))
    assert_size_stride(primals_456, (), ())
    assert_size_stride(primals_457, (264, ), (1, ))
    assert_size_stride(primals_458, (264, ), (1, ))
    assert_size_stride(primals_459, (), ())
    assert_size_stride(primals_460, (1584, ), (1, ))
    assert_size_stride(primals_461, (1584, ), (1, ))
    assert_size_stride(primals_462, (), ())
    assert_size_stride(primals_463, (1584, ), (1, ))
    assert_size_stride(primals_464, (1584, ), (1, ))
    assert_size_stride(primals_465, (), ())
    assert_size_stride(primals_466, (264, ), (1, ))
    assert_size_stride(primals_467, (264, ), (1, ))
    assert_size_stride(primals_468, (), ())
    assert_size_stride(primals_469, (1584, ), (1, ))
    assert_size_stride(primals_470, (1584, ), (1, ))
    assert_size_stride(primals_471, (), ())
    assert_size_stride(primals_472, (1584, ), (1, ))
    assert_size_stride(primals_473, (1584, ), (1, ))
    assert_size_stride(primals_474, (), ())
    assert_size_stride(primals_475, (264, ), (1, ))
    assert_size_stride(primals_476, (264, ), (1, ))
    assert_size_stride(primals_477, (), ())
    assert_size_stride(primals_478, (1536, ), (1, ))
    assert_size_stride(primals_479, (1536, ), (1, ))
    assert_size_stride(primals_480, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_1.run(primals_480, buf1, 24, 50625, grid=grid(24, 50625), stream=stream0)
        del primals_480
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
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_307, primals_308, buf10, buf11, buf13, primals_307, primals_308, 32, 7, grid=grid(32), stream=stream0)
        del primals_307
        del primals_308
        buf14 = reinterpret_tensor(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        # Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf3, buf10, buf11, primals_2, primals_3, buf14, 3211264, grid=grid(3211264), stream=stream0)
        del primals_3
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf15, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf16 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf15, buf16, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf17 = buf6; del buf6  # reuse
        buf18 = buf5; del buf5  # reuse
        buf19 = buf4; del buf4  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf16, buf17, buf18, buf19, 25088, 128, grid=grid(25088), stream=stream0)
        buf20 = buf9; del buf9  # reuse
        buf21 = buf8; del buf8  # reuse
        buf22 = buf7; del buf7  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf17, buf18, buf19, buf20, buf21, buf22, 224, 112, grid=grid(224), stream=stream0)
        buf23 = buf11; del buf11  # reuse
        buf24 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf26 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf20, buf21, buf22, primals_310, primals_311, buf23, buf24, buf26, primals_310, primals_311, 32, 7, grid=grid(32), stream=stream0)
        del primals_310
        del primals_311
        buf27 = reinterpret_tensor(buf15, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf15  # reuse
        # Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf16, buf23, buf24, primals_4, primals_5, buf27, 3211264, grid=grid(3211264), stream=stream0)
        del primals_5
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf29 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf28, buf29, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf30 = buf19; del buf19  # reuse
        buf31 = buf18; del buf18  # reuse
        buf32 = buf17; del buf17  # reuse
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf29, buf30, buf31, buf32, 25088, 128, grid=grid(25088), stream=stream0)
        buf33 = buf22; del buf22  # reuse
        buf34 = buf21; del buf21  # reuse
        buf35 = buf20; del buf20  # reuse
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf30, buf31, buf32, buf33, buf34, buf35, 224, 112, grid=grid(224), stream=stream0)
        del buf30
        del buf31
        del buf32
        buf36 = buf24; del buf24  # reuse
        buf37 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf39 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf33, buf34, buf35, primals_313, primals_314, buf36, buf37, buf39, primals_313, primals_314, 32, 7, grid=grid(32), stream=stream0)
        del primals_313
        del primals_314
        buf40 = buf28; del buf28  # reuse
        # Source Nodes: [shortcut_1, x_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_7.run(buf29, buf36, buf37, primals_6, primals_7, buf14, buf40, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del buf37
        del primals_7
        buf41 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf40, buf41, 128, 12544, grid=grid(128, 12544), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        buf43 = buf41; del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf40, buf43, 128, 12544, grid=grid(128, 12544), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del buf43
        buf45 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf42, buf44, buf45, 1536, 12544, grid=grid(1536, 12544), stream=stream0)
        del buf42
        del buf44
        buf46 = empty_strided((1, 192, 1, 1, 512), (98304, 1, 98304, 98304, 192), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 192, 1, 1, 512), (98304, 1, 98304, 98304, 192), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((1, 192, 1, 1, 512), (98304, 1, 98304, 98304, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf45, buf46, buf47, buf48, 98304, 196, grid=grid(98304), stream=stream0)
        buf49 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_12.run(buf46, buf47, buf48, buf49, buf50, buf51, 768, 128, grid=grid(768), stream=stream0)
        del buf46
        del buf47
        del buf48
        buf52 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf55 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_13.run(buf49, buf50, buf51, primals_316, primals_317, buf52, buf53, buf55, primals_316, primals_317, 192, 4, grid=grid(192), stream=stream0)
        del buf49
        del buf50
        del buf51
        del primals_316
        del primals_317
        buf56 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        buf934 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_20, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_14.run(buf45, buf52, buf53, primals_8, primals_9, buf56, buf934, 19267584, grid=grid(19267584), stream=stream0)
        del primals_9
        buf57 = empty_strided((8, 64, 113, 113), (817216, 1, 7232, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_15.run(buf56, buf57, 6537728, grid=grid(6537728), stream=stream0)
        # Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_10, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf58, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf59 = empty_strided((8, 64, 115, 115), (846400, 1, 7360, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf56, buf59, 6771200, grid=grid(6771200), stream=stream0)
        # Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_11, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf60, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf61 = empty_strided((8, 64, 117, 117), (876096, 1, 7488, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf56, buf61, 7008768, grid=grid(7008768), stream=stream0)
        del buf56
        # Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_12, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf62, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf63 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_18.run(buf58, buf60, buf62, buf63, 1536, 3136, grid=grid(1536, 3136), stream=stream0)
        del buf58
        del buf60
        del buf62
        buf64 = empty_strided((1, 192, 1, 1, 196), (37632, 1, 37632, 37632, 192), device='cuda', dtype=torch.float32)
        buf65 = empty_strided((1, 192, 1, 1, 196), (37632, 1, 37632, 37632, 192), device='cuda', dtype=torch.float32)
        buf66 = empty_strided((1, 192, 1, 1, 196), (37632, 1, 37632, 37632, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf63, buf64, buf65, buf66, 37632, 128, grid=grid(37632), stream=stream0)
        buf67 = empty_strided((1, 192, 1, 1, 2), (384, 1, 384, 384, 192), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((1, 192, 1, 1, 2), (384, 1, 384, 384, 192), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((1, 192, 1, 1, 2), (384, 1, 384, 384, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf64, buf65, buf66, buf67, buf68, buf69, 384, 98, grid=grid(384), stream=stream0)
        del buf64
        del buf65
        del buf66
        buf70 = buf53; del buf53  # reuse
        buf71 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf73 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf67, buf68, buf69, primals_319, primals_320, buf70, buf71, buf73, primals_319, primals_320, 192, 2, grid=grid(192), stream=stream0)
        del buf67
        del buf68
        del buf69
        del primals_319
        del primals_320
        buf74 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda', dtype=torch.float32)
        buf933 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_32, x_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_22.run(buf63, buf70, buf71, primals_13, primals_14, buf74, buf933, 4816896, grid=grid(4816896), stream=stream0)
        del buf71
        del primals_14
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 96, 56, 56), (602112, 1, 10752, 192), 0), primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 20, 56, 56), (62720, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 96, 56, 56), (602112, 1, 10752, 192), 96), primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 20, 56, 56), (62720, 3136, 56, 1))
        buf77 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf75, buf76, buf77, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del buf75
        del buf76
        buf78 = empty_strided((1, 40, 1, 1, 196), (7840, 1, 7840, 7840, 40), device='cuda', dtype=torch.float32)
        buf79 = empty_strided((1, 40, 1, 1, 196), (7840, 1, 7840, 7840, 40), device='cuda', dtype=torch.float32)
        buf80 = empty_strided((1, 40, 1, 1, 196), (7840, 1, 7840, 7840, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf77, buf78, buf79, buf80, 7840, 128, grid=grid(7840), stream=stream0)
        buf81 = empty_strided((1, 40, 1, 1, 2), (80, 1, 80, 80, 40), device='cuda', dtype=torch.float32)
        buf82 = empty_strided((1, 40, 1, 1, 2), (80, 1, 80, 80, 40), device='cuda', dtype=torch.float32)
        buf83 = empty_strided((1, 40, 1, 1, 2), (80, 1, 80, 80, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf78, buf79, buf80, buf81, buf82, buf83, 80, 98, grid=grid(80), stream=stream0)
        buf84 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf87 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf81, buf82, buf83, primals_322, primals_323, buf84, buf85, buf87, primals_322, primals_323, 40, 2, grid=grid(40), stream=stream0)
        del primals_322
        del primals_323
        buf88 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_27.run(buf77, buf84, buf85, primals_15, primals_16, buf88, 1003520, grid=grid(1003520), stream=stream0)
        del primals_16
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(reinterpret_tensor(buf88, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 60, 56, 56), (188160, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(reinterpret_tensor(buf88, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 60, 56, 56), (188160, 3136, 56, 1))
        buf91 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_78], Original ATen: [aten.cat]
        triton_poi_fused_cat_28.run(buf89, buf90, buf91, 960, 3136, grid=grid(960, 3136), stream=stream0)
        buf92 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf91, buf92, buf93, buf94, 23520, 128, grid=grid(23520), stream=stream0)
        buf95 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        buf96 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        buf97 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf92, buf93, buf94, buf95, buf96, buf97, 240, 98, grid=grid(240), stream=stream0)
        buf98 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf101 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf95, buf96, buf97, primals_325, primals_326, buf98, buf99, buf101, primals_325, primals_326, 120, 2, grid=grid(120), stream=stream0)
        del primals_325
        del primals_326
        buf102 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf91, buf98, buf99, primals_17, primals_18, buf102, 3010560, grid=grid(3010560), stream=stream0)
        del primals_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf103, (8, 120, 56, 56), (376320, 3136, 56, 1))
        buf104 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf103, buf104, 960, 3136, grid=grid(960, 3136), stream=stream0)
        buf105 = buf94; del buf94  # reuse
        buf106 = buf93; del buf93  # reuse
        buf107 = buf92; del buf92  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf104, buf105, buf106, buf107, 23520, 128, grid=grid(23520), stream=stream0)
        buf108 = buf97; del buf97  # reuse
        buf109 = buf96; del buf96  # reuse
        buf110 = buf95; del buf95  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf105, buf106, buf107, buf108, buf109, buf110, 240, 98, grid=grid(240), stream=stream0)
        del buf105
        del buf106
        del buf107
        buf111 = buf99; del buf99  # reuse
        buf112 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf114 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf108, buf109, buf110, primals_328, primals_329, buf111, buf112, buf114, primals_328, primals_329, 120, 2, grid=grid(120), stream=stream0)
        del primals_328
        del primals_329
        buf115 = reinterpret_tensor(buf103, (8, 120, 56, 56), (376320, 1, 6720, 120), 0); del buf103  # reuse
        buf932 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_34.run(buf104, buf111, buf112, primals_19, primals_20, buf115, buf932, 3010560, grid=grid(3010560), stream=stream0)
        del buf112
        del primals_20
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(reinterpret_tensor(buf115, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 20, 56, 56), (62720, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(reinterpret_tensor(buf115, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 20, 56, 56), (62720, 3136, 56, 1))
        buf118 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_77], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf116, buf117, buf118, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del buf116
        del buf117
        buf119 = buf80; del buf80  # reuse
        buf120 = buf79; del buf79  # reuse
        buf121 = buf78; del buf78  # reuse
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf118, buf119, buf120, buf121, 7840, 128, grid=grid(7840), stream=stream0)
        buf122 = buf83; del buf83  # reuse
        buf123 = buf82; del buf82  # reuse
        buf124 = buf81; del buf81  # reuse
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf119, buf120, buf121, buf122, buf123, buf124, 80, 98, grid=grid(80), stream=stream0)
        del buf119
        del buf120
        del buf121
        buf125 = buf85; del buf85  # reuse
        buf126 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf128 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf122, buf123, buf124, primals_331, primals_332, buf125, buf126, buf128, primals_331, primals_332, 40, 2, grid=grid(40), stream=stream0)
        del buf122
        del buf123
        del buf124
        del primals_331
        del primals_332
        buf129 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3, x_57], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_35.run(buf118, buf125, buf126, primals_21, primals_22, buf88, buf129, 1003520, grid=grid(1003520), stream=stream0)
        del buf126
        del primals_22
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 240, 56, 56), (752640, 3136, 56, 1))
        buf131 = empty_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf130, buf131, 1920, 3136, grid=grid(1920, 3136), stream=stream0)
        buf132 = empty_strided((1, 240, 1, 1, 196), (47040, 1, 47040, 47040, 240), device='cuda', dtype=torch.float32)
        buf133 = empty_strided((1, 240, 1, 1, 196), (47040, 1, 47040, 47040, 240), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((1, 240, 1, 1, 196), (47040, 1, 47040, 47040, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf131, buf132, buf133, buf134, 47040, 128, grid=grid(47040), stream=stream0)
        buf135 = empty_strided((1, 240, 1, 1, 2), (480, 1, 480, 480, 240), device='cuda', dtype=torch.float32)
        buf136 = empty_strided((1, 240, 1, 1, 2), (480, 1, 480, 480, 240), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((1, 240, 1, 1, 2), (480, 1, 480, 480, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf132, buf133, buf134, buf135, buf136, buf137, 480, 98, grid=grid(480), stream=stream0)
        del buf132
        del buf133
        del buf134
        buf138 = reinterpret_tensor(buf110, (1, 240, 1, 1), (240, 1, 240, 240), 0); del buf110  # reuse
        buf139 = reinterpret_tensor(buf109, (1, 240, 1, 1), (240, 1, 240, 240), 0); del buf109  # reuse
        buf141 = reinterpret_tensor(buf108, (240, ), (1, ), 0); del buf108  # reuse
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf135, buf136, buf137, primals_334, primals_335, buf138, buf139, buf141, primals_334, primals_335, 240, 2, grid=grid(240), stream=stream0)
        del primals_334
        del primals_335
        buf142 = reinterpret_tensor(buf130, (8, 240, 56, 56), (752640, 1, 13440, 240), 0); del buf130  # reuse
        buf931 = empty_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_40.run(buf131, buf138, buf139, primals_23, primals_24, buf142, buf931, 6021120, grid=grid(6021120), stream=stream0)
        del primals_24
        buf143 = empty_strided((8, 60, 57, 57), (194940, 1, 3420, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_41.run(buf142, buf143, 1559520, grid=grid(1559520), stream=stream0)
        # Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_25, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf144, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf145 = empty_strided((8, 60, 59, 59), (208860, 1, 3540, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_42.run(buf142, buf145, 1670880, grid=grid(1670880), stream=stream0)
        # Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_26, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf146, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf147 = empty_strided((8, 60, 61, 61), (223260, 1, 3660, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_43.run(buf142, buf147, 1786080, grid=grid(1786080), stream=stream0)
        # Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_27, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf148, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf149 = empty_strided((8, 60, 63, 63), (238140, 1, 3780, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_44.run(buf142, buf149, 1905120, grid=grid(1905120), stream=stream0)
        del buf142
        # Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_28, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf150, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf151 = reinterpret_tensor(buf90, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf90  # reuse
        # Source Nodes: [cat_76], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf144, buf146, buf148, buf150, buf151, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del buf144
        del buf146
        del buf148
        del buf150
        buf152 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf153 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf154 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf151, buf152, buf153, buf154, 11760, 128, grid=grid(11760), stream=stream0)
        buf155 = buf139; del buf139  # reuse
        buf156 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf158 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf152, buf153, buf154, primals_337, primals_338, buf155, buf156, buf158, primals_337, primals_338, 240, 49, grid=grid(240), stream=stream0)
        del buf152
        del buf153
        del buf154
        del primals_337
        del primals_338
        buf159 = reinterpret_tensor(buf89, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf89  # reuse
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_48.run(buf151, buf155, buf156, primals_29, primals_30, buf159, 1505280, grid=grid(1505280), stream=stream0)
        del buf156
        del primals_30
        buf160 = empty_strided((8, 240, 1, 1, 7), (1680, 1, 13440, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_49.run(buf159, buf160, 13440, 112, grid=grid(13440), stream=stream0)
        buf161 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf162 = reinterpret_tensor(buf161, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf161  # reuse
        # Source Nodes: [x_80, x_se], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_50.run(buf162, buf160, 1920, 7, grid=grid(1920), stream=stream0)
        del buf160
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 20, 1, 1), (20, 1, 1, 1))
        buf164 = reinterpret_tensor(buf163, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf163  # reuse
        buf165 = empty_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_51.run(buf164, primals_145, buf165, 160, grid=grid(160), stream=stream0)
        del primals_145
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 240, 1, 1), (240, 1, 1, 1))
        buf167 = reinterpret_tensor(buf166, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf166  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf167, primals_147, 1920, grid=grid(1920), stream=stream0)
        del primals_147
        buf168 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80, x_81], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_53.run(buf159, buf167, buf168, 1505280, grid=grid(1505280), stream=stream0)
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 56, 28, 28), (43904, 784, 28, 1))
        buf170 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf169, buf170, 448, 784, grid=grid(448, 784), stream=stream0)
        buf171 = empty_strided((1, 56, 1, 1, 49), (2744, 1, 2744, 2744, 56), device='cuda', dtype=torch.float32)
        buf172 = empty_strided((1, 56, 1, 1, 49), (2744, 1, 2744, 2744, 56), device='cuda', dtype=torch.float32)
        buf173 = empty_strided((1, 56, 1, 1, 49), (2744, 1, 2744, 2744, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf170, buf171, buf172, buf173, 2744, 128, grid=grid(2744), stream=stream0)
        buf174 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf175 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf177 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf171, buf172, buf173, primals_340, primals_341, buf174, buf175, buf177, primals_340, primals_341, 56, 49, grid=grid(56), stream=stream0)
        del primals_340
        del primals_341
        buf178 = reinterpret_tensor(buf169, (8, 56, 28, 28), (43904, 1, 1568, 56), 0); del buf169  # reuse
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_57.run(buf170, buf174, buf175, primals_31, primals_32, buf178, 351232, grid=grid(351232), stream=stream0)
        del primals_32
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(reinterpret_tensor(buf178, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 168, 28, 28), (131712, 784, 28, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(reinterpret_tensor(buf178, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf181 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_75], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf179, buf180, buf181, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf179
        buf182 = empty_strided((1, 336, 1, 1, 49), (16464, 1, 16464, 16464, 336), device='cuda', dtype=torch.float32)
        buf183 = empty_strided((1, 336, 1, 1, 49), (16464, 1, 16464, 16464, 336), device='cuda', dtype=torch.float32)
        buf184 = empty_strided((1, 336, 1, 1, 49), (16464, 1, 16464, 16464, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf181, buf182, buf183, buf184, 16464, 128, grid=grid(16464), stream=stream0)
        buf185 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf186 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf188 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf182, buf183, buf184, primals_343, primals_344, buf185, buf186, buf188, primals_343, primals_344, 336, 49, grid=grid(336), stream=stream0)
        del primals_343
        del primals_344
        buf189 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        buf930 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_61.run(buf181, buf185, buf186, primals_33, primals_34, buf189, buf930, 2107392, grid=grid(2107392), stream=stream0)
        del primals_34
        buf190 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.silu]
        triton_poi_fused_silu_62.run(buf189, buf190, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf191 = reinterpret_tensor(buf180, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf180  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf190, buf191, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf192, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf193 = buf191; del buf191  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf190, buf193, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_152, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf194, (8, 168, 28, 28), (131712, 784, 28, 1))
        del buf193
        buf195 = buf189; del buf189  # reuse
        # Source Nodes: [cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf192, buf194, buf195, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf192
        buf196 = buf184; del buf184  # reuse
        buf197 = buf183; del buf183  # reuse
        buf198 = buf182; del buf182  # reuse
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf195, buf196, buf197, buf198, 16464, 128, grid=grid(16464), stream=stream0)
        buf199 = buf186; del buf186  # reuse
        buf200 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf202 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf196, buf197, buf198, primals_346, primals_347, buf199, buf200, buf202, primals_346, primals_347, 336, 49, grid=grid(336), stream=stream0)
        del primals_346
        del primals_347
        buf203 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_65.run(buf195, buf199, buf200, primals_35, primals_36, buf203, 2107392, grid=grid(2107392), stream=stream0)
        del primals_36
        buf204 = empty_strided((8, 336, 1, 1, 7), (2352, 1, 18816, 18816, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_66.run(buf203, buf204, 18816, 112, grid=grid(18816), stream=stream0)
        buf205 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf206 = reinterpret_tensor(buf205, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf205  # reuse
        # Source Nodes: [x_98, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_67.run(buf206, buf204, 2688, 7, grid=grid(2688), stream=stream0)
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 28, 1, 1), (28, 1, 1, 1))
        buf208 = reinterpret_tensor(buf207, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf207  # reuse
        buf209 = reinterpret_tensor(buf35, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf35  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_68.run(buf208, primals_154, buf209, 224, grid=grid(224), stream=stream0)
        del primals_154
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 336, 1, 1), (336, 1, 1, 1))
        buf211 = reinterpret_tensor(buf210, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf210  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf211, primals_156, 2688, grid=grid(2688), stream=stream0)
        del primals_156
        buf212 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_98, x_99], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_70.run(buf203, buf211, buf212, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf213 = reinterpret_tensor(buf194, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf194  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf212, buf213, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf215 = buf213; del buf213  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf212, buf215, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf215
        buf217 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_73], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf214, buf216, buf217, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf214
        del buf216
        buf218 = buf173; del buf173  # reuse
        buf219 = buf172; del buf172  # reuse
        buf220 = buf171; del buf171  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf217, buf218, buf219, buf220, 2744, 128, grid=grid(2744), stream=stream0)
        buf221 = buf175; del buf175  # reuse
        buf222 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf224 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf218, buf219, buf220, primals_349, primals_350, buf221, buf222, buf224, primals_349, primals_350, 56, 49, grid=grid(56), stream=stream0)
        del primals_349
        del primals_350
        buf225 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5, x_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_72.run(buf217, buf221, buf222, primals_37, primals_38, buf178, buf225, 351232, grid=grid(351232), stream=stream0)
        del primals_38
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(reinterpret_tensor(buf225, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 168, 28, 28), (131712, 784, 28, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(reinterpret_tensor(buf225, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf228 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_72], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf226, buf227, buf228, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf226
        buf229 = buf198; del buf198  # reuse
        buf230 = buf197; del buf197  # reuse
        buf231 = buf196; del buf196  # reuse
        # Source Nodes: [x_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf228, buf229, buf230, buf231, 16464, 128, grid=grid(16464), stream=stream0)
        buf232 = buf200; del buf200  # reuse
        buf233 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf235 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf229, buf230, buf231, primals_352, primals_353, buf232, buf233, buf235, primals_352, primals_353, 336, 49, grid=grid(336), stream=stream0)
        del primals_352
        del primals_353
        buf236 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        buf929 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_61.run(buf228, buf232, buf233, primals_39, primals_40, buf236, buf929, 2107392, grid=grid(2107392), stream=stream0)
        del primals_40
        buf237 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.silu]
        triton_poi_fused_silu_62.run(buf236, buf237, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf238 = reinterpret_tensor(buf227, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf227  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf237, buf238, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf239, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf240 = buf238; del buf238  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf237, buf240, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_162, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf241, (8, 168, 28, 28), (131712, 784, 28, 1))
        del buf240
        buf242 = buf236; del buf236  # reuse
        # Source Nodes: [cat_71], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf239, buf241, buf242, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf239
        buf243 = buf231; del buf231  # reuse
        buf244 = buf230; del buf230  # reuse
        buf245 = buf229; del buf229  # reuse
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf242, buf243, buf244, buf245, 16464, 128, grid=grid(16464), stream=stream0)
        buf246 = buf233; del buf233  # reuse
        buf247 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf249 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf243, buf244, buf245, primals_355, primals_356, buf246, buf247, buf249, primals_355, primals_356, 336, 49, grid=grid(336), stream=stream0)
        del primals_355
        del primals_356
        buf250 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_65.run(buf242, buf246, buf247, primals_41, primals_42, buf250, 2107392, grid=grid(2107392), stream=stream0)
        del primals_42
        buf251 = buf204; del buf204  # reuse
        # Source Nodes: [x_118, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_66.run(buf250, buf251, 18816, 112, grid=grid(18816), stream=stream0)
        buf252 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf253 = reinterpret_tensor(buf252, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf252  # reuse
        # Source Nodes: [x_118, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_67.run(buf253, buf251, 2688, 7, grid=grid(2688), stream=stream0)
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 28, 1, 1), (28, 1, 1, 1))
        buf255 = reinterpret_tensor(buf254, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf254  # reuse
        buf256 = reinterpret_tensor(buf34, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf34  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_68.run(buf255, primals_164, buf256, 224, grid=grid(224), stream=stream0)
        del primals_164
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 336, 1, 1), (336, 1, 1, 1))
        buf258 = reinterpret_tensor(buf257, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf257  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf258, primals_166, 2688, grid=grid(2688), stream=stream0)
        del primals_166
        buf259 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_118, x_119], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_70.run(buf250, buf258, buf259, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf260 = reinterpret_tensor(buf241, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf241  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf259, buf260, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf262 = buf260; del buf260  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf259, buf262, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf262
        buf264 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_70], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf261, buf263, buf264, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf261
        del buf263
        buf265 = buf220; del buf220  # reuse
        buf266 = buf219; del buf219  # reuse
        buf267 = buf218; del buf218  # reuse
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf264, buf265, buf266, buf267, 2744, 128, grid=grid(2744), stream=stream0)
        buf268 = buf222; del buf222  # reuse
        buf269 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf271 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf265, buf266, buf267, primals_358, primals_359, buf268, buf269, buf271, primals_358, primals_359, 56, 49, grid=grid(56), stream=stream0)
        del primals_358
        del primals_359
        buf272 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6, x_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_72.run(buf264, buf268, buf269, primals_43, primals_44, buf225, buf272, 351232, grid=grid(351232), stream=stream0)
        del primals_44
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(reinterpret_tensor(buf272, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 168, 28, 28), (131712, 784, 28, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(reinterpret_tensor(buf272, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf275 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_69], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf273, buf274, buf275, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf273
        buf276 = buf245; del buf245  # reuse
        buf277 = buf244; del buf244  # reuse
        buf278 = buf243; del buf243  # reuse
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf275, buf276, buf277, buf278, 16464, 128, grid=grid(16464), stream=stream0)
        buf279 = buf247; del buf247  # reuse
        buf280 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf282 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf276, buf277, buf278, primals_361, primals_362, buf279, buf280, buf282, primals_361, primals_362, 336, 49, grid=grid(336), stream=stream0)
        del primals_361
        del primals_362
        buf283 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        buf928 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_61.run(buf275, buf279, buf280, primals_45, primals_46, buf283, buf928, 2107392, grid=grid(2107392), stream=stream0)
        del primals_46
        buf284 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.silu]
        triton_poi_fused_silu_62.run(buf283, buf284, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf285 = reinterpret_tensor(buf274, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf274  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf284, buf285, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf286, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf287 = buf285; del buf285  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf284, buf287, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_172, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf288, (8, 168, 28, 28), (131712, 784, 28, 1))
        del buf287
        buf289 = buf283; del buf283  # reuse
        # Source Nodes: [cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf286, buf288, buf289, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf286
        buf290 = buf278; del buf278  # reuse
        buf291 = buf277; del buf277  # reuse
        buf292 = buf276; del buf276  # reuse
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf289, buf290, buf291, buf292, 16464, 128, grid=grid(16464), stream=stream0)
        buf293 = buf280; del buf280  # reuse
        buf294 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf296 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf290, buf291, buf292, primals_364, primals_365, buf293, buf294, buf296, primals_364, primals_365, 336, 49, grid=grid(336), stream=stream0)
        del primals_364
        del primals_365
        buf297 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_65.run(buf289, buf293, buf294, primals_47, primals_48, buf297, 2107392, grid=grid(2107392), stream=stream0)
        del primals_48
        buf298 = buf251; del buf251  # reuse
        # Source Nodes: [x_138, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_66.run(buf297, buf298, 18816, 112, grid=grid(18816), stream=stream0)
        buf299 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf300 = reinterpret_tensor(buf299, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf299  # reuse
        # Source Nodes: [x_138, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_67.run(buf300, buf298, 2688, 7, grid=grid(2688), stream=stream0)
        del buf298
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 28, 1, 1), (28, 1, 1, 1))
        buf302 = reinterpret_tensor(buf301, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf301  # reuse
        buf303 = reinterpret_tensor(buf33, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf33  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_68.run(buf302, primals_174, buf303, 224, grid=grid(224), stream=stream0)
        del primals_174
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 336, 1, 1), (336, 1, 1, 1))
        buf305 = reinterpret_tensor(buf304, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf304  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf305, primals_176, 2688, grid=grid(2688), stream=stream0)
        del primals_176
        buf306 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_138, x_139], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_70.run(buf297, buf305, buf306, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf307 = reinterpret_tensor(buf288, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf288  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf306, buf307, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf309 = buf307; del buf307  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf306, buf309, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf309
        buf311 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_67], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf308, buf310, buf311, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf308
        del buf310
        buf312 = buf267; del buf267  # reuse
        buf313 = buf266; del buf266  # reuse
        buf314 = buf265; del buf265  # reuse
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf311, buf312, buf313, buf314, 2744, 128, grid=grid(2744), stream=stream0)
        buf315 = buf269; del buf269  # reuse
        buf316 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf318 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf312, buf313, buf314, primals_367, primals_368, buf315, buf316, buf318, primals_367, primals_368, 56, 49, grid=grid(56), stream=stream0)
        del buf312
        del buf313
        del buf314
        del primals_367
        del primals_368
        buf319 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_7, x_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_72.run(buf311, buf315, buf316, primals_49, primals_50, buf272, buf319, 351232, grid=grid(351232), stream=stream0)
        del buf316
        del primals_50
        # Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 336, 28, 28), (263424, 784, 28, 1))
        buf321 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf320, buf321, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf322 = buf292; del buf292  # reuse
        buf323 = buf291; del buf291  # reuse
        buf324 = buf290; del buf290  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf321, buf322, buf323, buf324, 16464, 128, grid=grid(16464), stream=stream0)
        buf325 = buf294; del buf294  # reuse
        buf326 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf328 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf322, buf323, buf324, primals_370, primals_371, buf325, buf326, buf328, primals_370, primals_371, 336, 49, grid=grid(336), stream=stream0)
        del buf322
        del buf323
        del buf324
        del primals_370
        del primals_371
        buf329 = reinterpret_tensor(buf320, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf320  # reuse
        buf927 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_61.run(buf321, buf325, buf326, primals_51, primals_52, buf329, buf927, 2107392, grid=grid(2107392), stream=stream0)
        del primals_52
        buf330 = empty_strided((8, 112, 29, 29), (94192, 1, 3248, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_74.run(buf329, buf330, 753536, grid=grid(753536), stream=stream0)
        # Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_53, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf331, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf332 = empty_strided((8, 112, 31, 31), (107632, 1, 3472, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_75.run(buf329, buf332, 861056, grid=grid(861056), stream=stream0)
        # Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_54, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf333, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf334 = empty_strided((8, 112, 33, 33), (121968, 1, 3696, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_76.run(buf329, buf334, 975744, grid=grid(975744), stream=stream0)
        del buf329
        # Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, primals_55, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf335, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf336 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_77.run(buf331, buf333, buf335, buf336, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del buf331
        del buf333
        del buf335
        buf337 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf338 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf339 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf336, buf337, buf338, buf339, 4368, 121, grid=grid(4368), stream=stream0)
        buf340 = buf326; del buf326  # reuse
        buf341 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf343 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_79.run(buf337, buf338, buf339, primals_373, primals_374, buf340, buf341, buf343, primals_373, primals_374, 336, 13, grid=grid(336), stream=stream0)
        del buf337
        del buf338
        del buf339
        del primals_373
        del primals_374
        buf344 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_80.run(buf336, buf340, buf341, primals_56, primals_57, buf344, 526848, grid=grid(526848), stream=stream0)
        del buf341
        del primals_57
        buf345 = empty_strided((8, 336, 1, 1, 2), (672, 1, 5376, 5376, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_81.run(buf344, buf345, 5376, 98, grid=grid(5376), stream=stream0)
        buf346 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf347 = reinterpret_tensor(buf346, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf346  # reuse
        # Source Nodes: [x_163, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_82.run(buf347, buf345, 2688, 2, grid=grid(2688), stream=stream0)
        del buf345
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 14, 1, 1), (14, 1, 1, 1))
        buf349 = reinterpret_tensor(buf348, (8, 14, 1, 1), (14, 1, 14, 14), 0); del buf348  # reuse
        buf350 = empty_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_83.run(buf349, primals_181, buf350, 112, grid=grid(112), stream=stream0)
        del primals_181
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 336, 1, 1), (336, 1, 1, 1))
        buf352 = reinterpret_tensor(buf351, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf351  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf352, primals_183, 2688, grid=grid(2688), stream=stream0)
        del primals_183
        buf353 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163, x_164], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_84.run(buf344, buf352, buf353, 526848, grid=grid(526848), stream=stream0)
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf355 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf354, buf355, 832, 196, grid=grid(832, 196), stream=stream0)
        buf356 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        buf357 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        buf358 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf355, buf356, buf357, buf358, 1352, 121, grid=grid(1352), stream=stream0)
        buf359 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf360 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf362 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf356, buf357, buf358, primals_376, primals_377, buf359, buf360, buf362, primals_376, primals_377, 104, 13, grid=grid(104), stream=stream0)
        del primals_376
        del primals_377
        buf363 = reinterpret_tensor(buf354, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf354  # reuse
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_88.run(buf355, buf359, buf360, primals_58, primals_59, buf363, 163072, grid=grid(163072), stream=stream0)
        del primals_59
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(reinterpret_tensor(buf363, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 312, 14, 14), (61152, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(reinterpret_tensor(buf363, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 312, 14, 14), (61152, 196, 14, 1))
        buf366 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_89.run(buf364, buf365, buf366, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf364
        buf367 = empty_strided((1, 624, 1, 1, 13), (8112, 1, 8112, 8112, 624), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((1, 624, 1, 1, 13), (8112, 1, 8112, 8112, 624), device='cuda', dtype=torch.float32)
        buf369 = empty_strided((1, 624, 1, 1, 13), (8112, 1, 8112, 8112, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf366, buf367, buf368, buf369, 8112, 121, grid=grid(8112), stream=stream0)
        buf370 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf371 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf373 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf367, buf368, buf369, primals_379, primals_380, buf370, buf371, buf373, primals_379, primals_380, 624, 13, grid=grid(624), stream=stream0)
        del primals_379
        del primals_380
        buf374 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        buf926 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_92.run(buf366, buf370, buf371, primals_60, primals_61, buf374, buf926, 978432, grid=grid(978432), stream=stream0)
        del primals_61
        buf375 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten.silu]
        triton_poi_fused_silu_93.run(buf374, buf375, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf376 = empty_strided((8, 156, 14, 14), (30576, 1, 2184, 156), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf375, buf376, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf377, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf378 = buf376; del buf376  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf375, buf378, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_188, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf379, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf380 = buf378; del buf378  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf375, buf380, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_189, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf381, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf382 = buf380; del buf380  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf375, buf382, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_190, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf383, (8, 156, 14, 14), (30576, 196, 14, 1))
        del buf382
        buf384 = buf374; del buf374  # reuse
        # Source Nodes: [cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_98.run(buf377, buf379, buf381, buf383, buf384, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf377
        del buf379
        del buf381
        buf385 = buf369; del buf369  # reuse
        buf386 = buf368; del buf368  # reuse
        buf387 = buf367; del buf367  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf384, buf385, buf386, buf387, 8112, 121, grid=grid(8112), stream=stream0)
        buf388 = buf371; del buf371  # reuse
        buf389 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf391 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf385, buf386, buf387, primals_382, primals_383, buf388, buf389, buf391, primals_382, primals_383, 624, 13, grid=grid(624), stream=stream0)
        del primals_382
        del primals_383
        buf392 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_99.run(buf384, buf388, buf389, primals_62, primals_63, buf392, 978432, grid=grid(978432), stream=stream0)
        del primals_63
        buf393 = empty_strided((8, 624, 1, 1, 2), (1248, 1, 9984, 9984, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_100.run(buf392, buf393, 9984, 98, grid=grid(9984), stream=stream0)
        buf394 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf395 = reinterpret_tensor(buf394, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf394  # reuse
        # Source Nodes: [x_181, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_101.run(buf395, buf393, 4992, 2, grid=grid(4992), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 26, 1, 1), (26, 1, 1, 1))
        buf397 = reinterpret_tensor(buf396, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf396  # reuse
        buf398 = empty_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_102.run(buf397, primals_192, buf398, 208, grid=grid(208), stream=stream0)
        del primals_192
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 624, 1, 1), (624, 1, 1, 1))
        buf400 = reinterpret_tensor(buf399, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf399  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf400, primals_194, 4992, grid=grid(4992), stream=stream0)
        del primals_194
        buf401 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_181, x_182], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_104.run(buf392, buf400, buf401, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf402 = reinterpret_tensor(buf365, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf365  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_105.run(buf401, buf402, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf403 = extern_kernels.convolution(buf402, primals_195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 52, 14, 14), (10192, 196, 14, 1))
        buf404 = buf402; del buf402  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf401, buf404, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 52, 14, 14), (10192, 196, 14, 1))
        del buf404
        buf406 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf403, buf405, buf406, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf403
        del buf405
        buf407 = buf358; del buf358  # reuse
        buf408 = buf357; del buf357  # reuse
        buf409 = buf356; del buf356  # reuse
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf406, buf407, buf408, buf409, 1352, 121, grid=grid(1352), stream=stream0)
        buf410 = buf360; del buf360  # reuse
        buf411 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf413 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf407, buf408, buf409, primals_385, primals_386, buf410, buf411, buf413, primals_385, primals_386, 104, 13, grid=grid(104), stream=stream0)
        del primals_385
        del primals_386
        buf414 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_185], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_108.run(buf406, buf410, buf411, primals_64, primals_65, buf363, buf414, 163072, grid=grid(163072), stream=stream0)
        del primals_65
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(reinterpret_tensor(buf414, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 312, 14, 14), (61152, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(reinterpret_tensor(buf414, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 312, 14, 14), (61152, 196, 14, 1))
        buf417 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_89.run(buf415, buf416, buf417, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf415
        buf418 = buf387; del buf387  # reuse
        buf419 = buf386; del buf386  # reuse
        buf420 = buf385; del buf385  # reuse
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf417, buf418, buf419, buf420, 8112, 121, grid=grid(8112), stream=stream0)
        buf421 = buf389; del buf389  # reuse
        buf422 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf424 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf418, buf419, buf420, primals_388, primals_389, buf421, buf422, buf424, primals_388, primals_389, 624, 13, grid=grid(624), stream=stream0)
        del primals_388
        del primals_389
        buf425 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        buf925 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_92.run(buf417, buf421, buf422, primals_66, primals_67, buf425, buf925, 978432, grid=grid(978432), stream=stream0)
        del primals_67
        buf426 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195], Original ATen: [aten.silu]
        triton_poi_fused_silu_93.run(buf425, buf426, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf427 = reinterpret_tensor(buf383, (8, 156, 14, 14), (30576, 1, 2184, 156), 0); del buf383  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf426, buf427, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf428, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf429 = buf427; del buf427  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf426, buf429, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_200, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf430, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf431 = buf429; del buf429  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf426, buf431, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_201, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf432, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf433 = buf431; del buf431  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf426, buf433, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_202, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf434, (8, 156, 14, 14), (30576, 196, 14, 1))
        del buf433
        buf435 = buf425; del buf425  # reuse
        # Source Nodes: [cat_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_98.run(buf428, buf430, buf432, buf434, buf435, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf428
        del buf430
        del buf432
        buf436 = buf420; del buf420  # reuse
        buf437 = buf419; del buf419  # reuse
        buf438 = buf418; del buf418  # reuse
        # Source Nodes: [x_198], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf435, buf436, buf437, buf438, 8112, 121, grid=grid(8112), stream=stream0)
        buf439 = buf422; del buf422  # reuse
        buf440 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf442 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf436, buf437, buf438, primals_391, primals_392, buf439, buf440, buf442, primals_391, primals_392, 624, 13, grid=grid(624), stream=stream0)
        del primals_391
        del primals_392
        buf443 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_99.run(buf435, buf439, buf440, primals_68, primals_69, buf443, 978432, grid=grid(978432), stream=stream0)
        del primals_69
        buf444 = buf393; del buf393  # reuse
        # Source Nodes: [x_201, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_100.run(buf443, buf444, 9984, 98, grid=grid(9984), stream=stream0)
        buf445 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf446 = reinterpret_tensor(buf445, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf445  # reuse
        # Source Nodes: [x_201, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_101.run(buf446, buf444, 4992, 2, grid=grid(4992), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (8, 26, 1, 1), (26, 1, 1, 1))
        buf448 = reinterpret_tensor(buf447, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf447  # reuse
        buf449 = empty_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_102.run(buf448, primals_204, buf449, 208, grid=grid(208), stream=stream0)
        del primals_204
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (8, 624, 1, 1), (624, 1, 1, 1))
        buf451 = reinterpret_tensor(buf450, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf450  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf451, primals_206, 4992, grid=grid(4992), stream=stream0)
        del primals_206
        buf452 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_201, x_202], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_104.run(buf443, buf451, buf452, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf453 = reinterpret_tensor(buf416, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf416  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_105.run(buf452, buf453, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (8, 52, 14, 14), (10192, 196, 14, 1))
        buf455 = buf453; del buf453  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf452, buf455, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 52, 14, 14), (10192, 196, 14, 1))
        del buf455
        buf457 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf454, buf456, buf457, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf454
        del buf456
        buf458 = buf409; del buf409  # reuse
        buf459 = buf408; del buf408  # reuse
        buf460 = buf407; del buf407  # reuse
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf457, buf458, buf459, buf460, 1352, 121, grid=grid(1352), stream=stream0)
        buf461 = buf411; del buf411  # reuse
        buf462 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf464 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf458, buf459, buf460, primals_394, primals_395, buf461, buf462, buf464, primals_394, primals_395, 104, 13, grid=grid(104), stream=stream0)
        del primals_394
        del primals_395
        buf465 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10, x_205], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_108.run(buf457, buf461, buf462, primals_70, primals_71, buf414, buf465, 163072, grid=grid(163072), stream=stream0)
        del primals_71
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(reinterpret_tensor(buf465, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), primals_209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 312, 14, 14), (61152, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(reinterpret_tensor(buf465, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (8, 312, 14, 14), (61152, 196, 14, 1))
        buf468 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_59], Original ATen: [aten.cat]
        triton_poi_fused_cat_89.run(buf466, buf467, buf468, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf466
        buf469 = buf438; del buf438  # reuse
        buf470 = buf437; del buf437  # reuse
        buf471 = buf436; del buf436  # reuse
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf468, buf469, buf470, buf471, 8112, 121, grid=grid(8112), stream=stream0)
        buf472 = buf440; del buf440  # reuse
        buf473 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf475 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf469, buf470, buf471, primals_397, primals_398, buf472, buf473, buf475, primals_397, primals_398, 624, 13, grid=grid(624), stream=stream0)
        del primals_397
        del primals_398
        buf476 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        buf924 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_92.run(buf468, buf472, buf473, primals_72, primals_73, buf476, buf924, 978432, grid=grid(978432), stream=stream0)
        del primals_73
        buf477 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.silu]
        triton_poi_fused_silu_93.run(buf476, buf477, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf478 = reinterpret_tensor(buf434, (8, 156, 14, 14), (30576, 1, 2184, 156), 0); del buf434  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf477, buf478, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf479, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf480 = buf478; del buf478  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf477, buf480, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_212, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf481, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf482 = buf480; del buf480  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf477, buf482, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_213, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf483, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf484 = buf482; del buf482  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf477, buf484, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf484, primals_214, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf485, (8, 156, 14, 14), (30576, 196, 14, 1))
        del buf484
        buf486 = buf476; del buf476  # reuse
        # Source Nodes: [cat_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_98.run(buf479, buf481, buf483, buf485, buf486, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf479
        del buf481
        del buf483
        del buf485
        buf487 = buf471; del buf471  # reuse
        buf488 = buf470; del buf470  # reuse
        buf489 = buf469; del buf469  # reuse
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf486, buf487, buf488, buf489, 8112, 121, grid=grid(8112), stream=stream0)
        buf490 = buf473; del buf473  # reuse
        buf491 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf493 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf487, buf488, buf489, primals_400, primals_401, buf490, buf491, buf493, primals_400, primals_401, 624, 13, grid=grid(624), stream=stream0)
        del primals_400
        del primals_401
        buf494 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_99.run(buf486, buf490, buf491, primals_74, primals_75, buf494, 978432, grid=grid(978432), stream=stream0)
        del primals_75
        buf495 = buf444; del buf444  # reuse
        # Source Nodes: [x_221, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_100.run(buf494, buf495, 9984, 98, grid=grid(9984), stream=stream0)
        buf496 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf497 = reinterpret_tensor(buf496, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf496  # reuse
        # Source Nodes: [x_221, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_101.run(buf497, buf495, 4992, 2, grid=grid(4992), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (8, 26, 1, 1), (26, 1, 1, 1))
        buf499 = reinterpret_tensor(buf498, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf498  # reuse
        buf500 = empty_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_102.run(buf499, primals_216, buf500, 208, grid=grid(208), stream=stream0)
        del primals_216
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (8, 624, 1, 1), (624, 1, 1, 1))
        buf502 = reinterpret_tensor(buf501, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf501  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf502, primals_218, 4992, grid=grid(4992), stream=stream0)
        del primals_218
        buf503 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_221, x_222], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_104.run(buf494, buf502, buf503, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf504 = reinterpret_tensor(buf467, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf467  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_105.run(buf503, buf504, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (8, 52, 14, 14), (10192, 196, 14, 1))
        buf506 = buf504; del buf504  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf503, buf506, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (8, 52, 14, 14), (10192, 196, 14, 1))
        del buf506
        buf508 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_57], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf505, buf507, buf508, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf505
        del buf507
        buf509 = buf460; del buf460  # reuse
        buf510 = buf459; del buf459  # reuse
        buf511 = buf458; del buf458  # reuse
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf508, buf509, buf510, buf511, 1352, 121, grid=grid(1352), stream=stream0)
        buf512 = buf462; del buf462  # reuse
        buf513 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf515 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf509, buf510, buf511, primals_403, primals_404, buf512, buf513, buf515, primals_403, primals_404, 104, 13, grid=grid(104), stream=stream0)
        del buf509
        del buf510
        del buf511
        del primals_403
        del primals_404
        buf516 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_11, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_108.run(buf508, buf512, buf513, primals_76, primals_77, buf465, buf516, 163072, grid=grid(163072), stream=stream0)
        del buf513
        del primals_77
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 624, 14, 14), (122304, 196, 14, 1))
        buf518 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_109.run(buf517, buf518, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf519 = buf489; del buf489  # reuse
        buf520 = buf488; del buf488  # reuse
        buf521 = buf487; del buf487  # reuse
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf518, buf519, buf520, buf521, 8112, 121, grid=grid(8112), stream=stream0)
        buf522 = buf491; del buf491  # reuse
        buf523 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf525 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf519, buf520, buf521, primals_406, primals_407, buf522, buf523, buf525, primals_406, primals_407, 624, 13, grid=grid(624), stream=stream0)
        del primals_406
        del primals_407
        buf527 = reinterpret_tensor(buf517, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf517  # reuse
        buf923 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231, x_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_110.run(buf518, buf522, buf523, primals_78, primals_79, buf527, buf923, 978432, grid=grid(978432), stream=stream0)
        del primals_79
        # Source Nodes: [x_235], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=624, bias=None)
        assert_size_stride(buf528, (8, 624, 14, 14), (122304, 196, 14, 1))
        buf529 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_109.run(buf528, buf529, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf530 = buf521; del buf521  # reuse
        buf531 = buf520; del buf520  # reuse
        buf532 = buf519; del buf519  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf529, buf530, buf531, buf532, 8112, 121, grid=grid(8112), stream=stream0)
        buf533 = buf523; del buf523  # reuse
        buf534 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf536 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf530, buf531, buf532, primals_409, primals_410, buf533, buf534, buf536, primals_409, primals_410, 624, 13, grid=grid(624), stream=stream0)
        del buf530
        del buf531
        del buf532
        del primals_409
        del primals_410
        buf537 = reinterpret_tensor(buf528, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf528  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_99.run(buf529, buf533, buf534, primals_80, primals_81, buf537, 978432, grid=grid(978432), stream=stream0)
        del buf534
        del primals_81
        buf538 = buf495; del buf495  # reuse
        # Source Nodes: [x_239, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_100.run(buf537, buf538, 9984, 98, grid=grid(9984), stream=stream0)
        buf539 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf540 = reinterpret_tensor(buf539, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf539  # reuse
        # Source Nodes: [x_239, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_101.run(buf540, buf538, 4992, 2, grid=grid(4992), stream=stream0)
        del buf538
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (8, 52, 1, 1), (52, 1, 1, 1))
        buf542 = reinterpret_tensor(buf541, (8, 52, 1, 1), (52, 1, 52, 52), 0); del buf541  # reuse
        buf543 = empty_strided((8, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_111.run(buf542, primals_224, buf543, 416, grid=grid(416), stream=stream0)
        del primals_224
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (8, 624, 1, 1), (624, 1, 1, 1))
        buf545 = reinterpret_tensor(buf544, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf544  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf545, primals_226, 4992, grid=grid(4992), stream=stream0)
        del primals_226
        buf546 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239, x_240], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_112.run(buf537, buf545, buf546, 978432, grid=grid(978432), stream=stream0)
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (8, 160, 14, 14), (31360, 196, 14, 1))
        buf548 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_113.run(buf547, buf548, 1280, 196, grid=grid(1280, 196), stream=stream0)
        buf549 = empty_strided((1, 160, 1, 1, 13), (2080, 1, 2080, 2080, 160), device='cuda', dtype=torch.float32)
        buf550 = empty_strided((1, 160, 1, 1, 13), (2080, 1, 2080, 2080, 160), device='cuda', dtype=torch.float32)
        buf551 = empty_strided((1, 160, 1, 1, 13), (2080, 1, 2080, 2080, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_114.run(buf548, buf549, buf550, buf551, 2080, 121, grid=grid(2080), stream=stream0)
        buf552 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf553 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf555 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_115.run(buf549, buf550, buf551, primals_412, primals_413, buf552, buf553, buf555, primals_412, primals_413, 160, 13, grid=grid(160), stream=stream0)
        del primals_412
        del primals_413
        buf556 = reinterpret_tensor(buf547, (8, 160, 14, 14), (31360, 1, 2240, 160), 0); del buf547  # reuse
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_116.run(buf548, buf552, buf553, primals_82, primals_83, buf556, 250880, grid=grid(250880), stream=stream0)
        del primals_83
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(reinterpret_tensor(buf556, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (8, 240, 14, 14), (47040, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(reinterpret_tensor(buf556, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf559 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_117.run(buf557, buf558, buf559, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf557
        buf560 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf561 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf562 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf559, buf560, buf561, buf562, 6240, 121, grid=grid(6240), stream=stream0)
        buf563 = reinterpret_tensor(buf137, (1, 480, 1, 1), (480, 1, 480, 480), 0); del buf137  # reuse
        buf564 = reinterpret_tensor(buf136, (1, 480, 1, 1), (480, 1, 480, 480), 0); del buf136  # reuse
        buf566 = reinterpret_tensor(buf135, (480, ), (1, ), 0); del buf135  # reuse
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf560, buf561, buf562, primals_415, primals_416, buf563, buf564, buf566, primals_415, primals_416, 480, 13, grid=grid(480), stream=stream0)
        del primals_415
        del primals_416
        buf567 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf922 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_120.run(buf559, buf563, buf564, primals_84, primals_85, buf567, buf922, 752640, grid=grid(752640), stream=stream0)
        del primals_85
        buf568 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251], Original ATen: [aten.silu]
        triton_poi_fused_silu_121.run(buf567, buf568, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf569 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf568, buf569, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf570, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf571 = buf569; del buf569  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_123.run(buf568, buf571, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_231, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf572, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf573 = buf571; del buf571  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf568, buf573, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_232, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf574, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf575 = buf573; del buf573  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_125.run(buf568, buf575, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_233, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf576, (8, 120, 14, 14), (23520, 196, 14, 1))
        del buf575
        buf577 = buf567; del buf567  # reuse
        # Source Nodes: [cat_55], Original ATen: [aten.cat]
        triton_poi_fused_cat_126.run(buf570, buf572, buf574, buf576, buf577, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf570
        del buf572
        del buf574
        buf578 = buf562; del buf562  # reuse
        buf579 = buf561; del buf561  # reuse
        buf580 = buf560; del buf560  # reuse
        # Source Nodes: [x_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf577, buf578, buf579, buf580, 6240, 121, grid=grid(6240), stream=stream0)
        buf581 = buf564; del buf564  # reuse
        buf582 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf584 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf578, buf579, buf580, primals_418, primals_419, buf581, buf582, buf584, primals_418, primals_419, 480, 13, grid=grid(480), stream=stream0)
        del primals_418
        del primals_419
        buf585 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_127.run(buf577, buf581, buf582, primals_86, primals_87, buf585, 752640, grid=grid(752640), stream=stream0)
        del primals_87
        buf586 = empty_strided((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_128.run(buf585, buf586, 7680, 98, grid=grid(7680), stream=stream0)
        buf587 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf588 = reinterpret_tensor(buf587, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf587  # reuse
        # Source Nodes: [x_257, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_129.run(buf588, buf586, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (8, 80, 1, 1), (80, 1, 1, 1))
        buf590 = reinterpret_tensor(buf589, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf589  # reuse
        buf591 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_130.run(buf590, primals_235, buf591, 640, grid=grid(640), stream=stream0)
        del primals_235
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf592 = extern_kernels.convolution(buf591, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (8, 480, 1, 1), (480, 1, 1, 1))
        buf593 = reinterpret_tensor(buf592, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf592  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_131.run(buf593, primals_237, 3840, grid=grid(3840), stream=stream0)
        del primals_237
        buf594 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_257, x_258], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_132.run(buf585, buf593, buf594, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf595 = reinterpret_tensor(buf558, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf558  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf594, buf595, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf597 = buf595; del buf595  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_134.run(buf594, buf597, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(buf597, primals_239, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (8, 80, 14, 14), (15680, 196, 14, 1))
        del buf597
        buf599 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_135.run(buf596, buf598, buf599, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf596
        del buf598
        buf600 = buf551; del buf551  # reuse
        buf601 = buf550; del buf550  # reuse
        buf602 = buf549; del buf549  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_114.run(buf599, buf600, buf601, buf602, 2080, 121, grid=grid(2080), stream=stream0)
        buf603 = buf553; del buf553  # reuse
        buf604 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf606 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_115.run(buf600, buf601, buf602, primals_421, primals_422, buf603, buf604, buf606, primals_421, primals_422, 160, 13, grid=grid(160), stream=stream0)
        del primals_421
        del primals_422
        buf607 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_13, x_261], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_136.run(buf599, buf603, buf604, primals_88, primals_89, buf556, buf607, 250880, grid=grid(250880), stream=stream0)
        del primals_89
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(reinterpret_tensor(buf607, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (8, 240, 14, 14), (47040, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(reinterpret_tensor(buf607, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf609, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf610 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_117.run(buf608, buf609, buf610, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf608
        buf611 = buf580; del buf580  # reuse
        buf612 = buf579; del buf579  # reuse
        buf613 = buf578; del buf578  # reuse
        # Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf610, buf611, buf612, buf613, 6240, 121, grid=grid(6240), stream=stream0)
        buf614 = buf582; del buf582  # reuse
        buf615 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf617 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf611, buf612, buf613, primals_424, primals_425, buf614, buf615, buf617, primals_424, primals_425, 480, 13, grid=grid(480), stream=stream0)
        del primals_424
        del primals_425
        buf618 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf921 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_120.run(buf610, buf614, buf615, primals_90, primals_91, buf618, buf921, 752640, grid=grid(752640), stream=stream0)
        del primals_91
        buf619 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten.silu]
        triton_poi_fused_silu_121.run(buf618, buf619, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf620 = reinterpret_tensor(buf576, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf576  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf619, buf620, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf621 = extern_kernels.convolution(buf620, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf621, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf622 = buf620; del buf620  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_123.run(buf619, buf622, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, primals_243, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf623, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf624 = buf622; del buf622  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf619, buf624, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf624, primals_244, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf625, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf626 = buf624; del buf624  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_125.run(buf619, buf626, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, primals_245, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf627, (8, 120, 14, 14), (23520, 196, 14, 1))
        del buf626
        buf628 = buf618; del buf618  # reuse
        # Source Nodes: [cat_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_126.run(buf621, buf623, buf625, buf627, buf628, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf621
        del buf623
        del buf625
        buf629 = buf613; del buf613  # reuse
        buf630 = buf612; del buf612  # reuse
        buf631 = buf611; del buf611  # reuse
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf628, buf629, buf630, buf631, 6240, 121, grid=grid(6240), stream=stream0)
        buf632 = buf615; del buf615  # reuse
        buf633 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf635 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf629, buf630, buf631, primals_427, primals_428, buf632, buf633, buf635, primals_427, primals_428, 480, 13, grid=grid(480), stream=stream0)
        del primals_427
        del primals_428
        buf636 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_127.run(buf628, buf632, buf633, primals_92, primals_93, buf636, 752640, grid=grid(752640), stream=stream0)
        del primals_93
        buf637 = buf586; del buf586  # reuse
        # Source Nodes: [x_277, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_128.run(buf636, buf637, 7680, 98, grid=grid(7680), stream=stream0)
        buf638 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf639 = reinterpret_tensor(buf638, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf638  # reuse
        # Source Nodes: [x_277, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_129.run(buf639, buf637, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf640 = extern_kernels.convolution(buf639, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf640, (8, 80, 1, 1), (80, 1, 1, 1))
        buf641 = reinterpret_tensor(buf640, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf640  # reuse
        buf642 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_130.run(buf641, primals_247, buf642, 640, grid=grid(640), stream=stream0)
        del primals_247
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (8, 480, 1, 1), (480, 1, 1, 1))
        buf644 = reinterpret_tensor(buf643, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf643  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_131.run(buf644, primals_249, 3840, grid=grid(3840), stream=stream0)
        del primals_249
        buf645 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_277, x_278], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_132.run(buf636, buf644, buf645, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf646 = reinterpret_tensor(buf609, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf609  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf645, buf646, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf647 = extern_kernels.convolution(buf646, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf647, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf648 = buf646; del buf646  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_134.run(buf645, buf648, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf650 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_51], Original ATen: [aten.cat]
        triton_poi_fused_cat_135.run(buf647, buf649, buf650, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf647
        del buf649
        buf651 = buf602; del buf602  # reuse
        buf652 = buf601; del buf601  # reuse
        buf653 = buf600; del buf600  # reuse
        # Source Nodes: [x_281], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_114.run(buf650, buf651, buf652, buf653, 2080, 121, grid=grid(2080), stream=stream0)
        buf654 = buf604; del buf604  # reuse
        buf655 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf657 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_281], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_115.run(buf651, buf652, buf653, primals_430, primals_431, buf654, buf655, buf657, primals_430, primals_431, 160, 13, grid=grid(160), stream=stream0)
        del primals_430
        del primals_431
        buf658 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_14, x_281], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_136.run(buf650, buf654, buf655, primals_94, primals_95, buf607, buf658, 250880, grid=grid(250880), stream=stream0)
        del primals_95
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(reinterpret_tensor(buf658, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (8, 240, 14, 14), (47040, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(reinterpret_tensor(buf658, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf660, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf661 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_50], Original ATen: [aten.cat]
        triton_poi_fused_cat_117.run(buf659, buf660, buf661, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf662 = buf631; del buf631  # reuse
        buf663 = buf630; del buf630  # reuse
        buf664 = buf629; del buf629  # reuse
        # Source Nodes: [x_288], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf661, buf662, buf663, buf664, 6240, 121, grid=grid(6240), stream=stream0)
        buf665 = buf633; del buf633  # reuse
        buf666 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf668 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_288], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf662, buf663, buf664, primals_433, primals_434, buf665, buf666, buf668, primals_433, primals_434, 480, 13, grid=grid(480), stream=stream0)
        del primals_433
        del primals_434
        buf669 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf920 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_288], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_120.run(buf661, buf665, buf666, primals_96, primals_97, buf669, buf920, 752640, grid=grid(752640), stream=stream0)
        del primals_97
        buf670 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten.silu]
        triton_poi_fused_silu_121.run(buf669, buf670, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf671 = reinterpret_tensor(buf627, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf627  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf670, buf671, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf672 = extern_kernels.convolution(buf671, primals_254, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf672, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf673 = buf671; del buf671  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_123.run(buf670, buf673, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, primals_255, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf674, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf675 = buf673; del buf673  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf670, buf675, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, primals_256, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf676, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf677 = buf675; del buf675  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_125.run(buf670, buf677, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf678 = extern_kernels.convolution(buf677, primals_257, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf678, (8, 120, 14, 14), (23520, 196, 14, 1))
        del buf677
        buf679 = buf669; del buf669  # reuse
        # Source Nodes: [cat_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_126.run(buf672, buf674, buf676, buf678, buf679, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf672
        del buf674
        del buf676
        del buf678
        buf680 = buf664; del buf664  # reuse
        buf681 = buf663; del buf663  # reuse
        buf682 = buf662; del buf662  # reuse
        # Source Nodes: [x_294], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf679, buf680, buf681, buf682, 6240, 121, grid=grid(6240), stream=stream0)
        buf683 = buf666; del buf666  # reuse
        buf684 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf686 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf680, buf681, buf682, primals_436, primals_437, buf683, buf684, buf686, primals_436, primals_437, 480, 13, grid=grid(480), stream=stream0)
        del buf680
        del buf681
        del buf682
        del primals_436
        del primals_437
        buf687 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_127.run(buf679, buf683, buf684, primals_98, primals_99, buf687, 752640, grid=grid(752640), stream=stream0)
        del buf684
        del primals_99
        buf688 = buf637; del buf637  # reuse
        # Source Nodes: [x_297, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_128.run(buf687, buf688, 7680, 98, grid=grid(7680), stream=stream0)
        buf689 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf690 = reinterpret_tensor(buf689, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf689  # reuse
        # Source Nodes: [x_297, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_129.run(buf690, buf688, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf691 = extern_kernels.convolution(buf690, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (8, 80, 1, 1), (80, 1, 1, 1))
        buf692 = reinterpret_tensor(buf691, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf691  # reuse
        buf693 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_130.run(buf692, primals_259, buf693, 640, grid=grid(640), stream=stream0)
        del primals_259
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf694 = extern_kernels.convolution(buf693, primals_260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf694, (8, 480, 1, 1), (480, 1, 1, 1))
        buf695 = reinterpret_tensor(buf694, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf694  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_131.run(buf695, primals_261, 3840, grid=grid(3840), stream=stream0)
        del primals_261
        buf696 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_297, x_298], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_132.run(buf687, buf695, buf696, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf697 = reinterpret_tensor(buf660, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf660  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_133.run(buf696, buf697, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf698 = extern_kernels.convolution(buf697, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf698, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf699 = buf697; del buf697  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_134.run(buf696, buf699, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf701 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_135.run(buf698, buf700, buf701, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf698
        del buf700
        buf702 = buf653; del buf653  # reuse
        buf703 = buf652; del buf652  # reuse
        buf704 = buf651; del buf651  # reuse
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_114.run(buf701, buf702, buf703, buf704, 2080, 121, grid=grid(2080), stream=stream0)
        buf705 = buf655; del buf655  # reuse
        buf706 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf708 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_115.run(buf702, buf703, buf704, primals_439, primals_440, buf705, buf706, buf708, primals_439, primals_440, 160, 13, grid=grid(160), stream=stream0)
        del buf702
        del buf703
        del buf704
        del primals_439
        del primals_440
        buf709 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_15, x_301], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_136.run(buf701, buf705, buf706, primals_100, primals_101, buf658, buf709, 250880, grid=grid(250880), stream=stream0)
        del buf706
        del primals_101
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf710 = extern_kernels.convolution(buf709, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf710, (8, 960, 14, 14), (188160, 196, 14, 1))
        buf711 = empty_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_137.run(buf710, buf711, 7680, 196, grid=grid(7680, 196), stream=stream0)
        buf712 = empty_strided((1, 960, 1, 1, 13), (12480, 1, 12480, 12480, 960), device='cuda', dtype=torch.float32)
        buf713 = empty_strided((1, 960, 1, 1, 13), (12480, 1, 12480, 12480, 960), device='cuda', dtype=torch.float32)
        buf714 = empty_strided((1, 960, 1, 1, 13), (12480, 1, 12480, 12480, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_138.run(buf711, buf712, buf713, buf714, 12480, 121, grid=grid(12480), stream=stream0)
        buf715 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf716 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf718 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_139.run(buf712, buf713, buf714, primals_442, primals_443, buf715, buf716, buf718, primals_442, primals_443, 960, 13, grid=grid(960), stream=stream0)
        del buf712
        del buf713
        del buf714
        del primals_442
        del primals_443
        buf719 = reinterpret_tensor(buf710, (8, 960, 14, 14), (188160, 1, 13440, 960), 0); del buf710  # reuse
        buf919 = empty_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_140.run(buf711, buf715, buf716, primals_102, primals_103, buf719, buf919, 1505280, grid=grid(1505280), stream=stream0)
        del primals_103
        buf720 = empty_strided((8, 240, 15, 15), (54000, 1, 3600, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_141.run(buf719, buf720, 432000, grid=grid(432000), stream=stream0)
        # Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf721 = extern_kernels.convolution(buf720, primals_104, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf721, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf722 = empty_strided((8, 240, 17, 17), (69360, 1, 4080, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_142.run(buf719, buf722, 554880, grid=grid(554880), stream=stream0)
        # Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf723 = extern_kernels.convolution(buf722, primals_105, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf723, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf724 = empty_strided((8, 240, 19, 19), (86640, 1, 4560, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_143.run(buf719, buf724, 693120, grid=grid(693120), stream=stream0)
        # Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, primals_106, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf725, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf726 = empty_strided((8, 240, 21, 21), (105840, 1, 5040, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_318], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_144.run(buf719, buf726, 846720, grid=grid(846720), stream=stream0)
        del buf719
        # Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf727 = extern_kernels.convolution(buf726, primals_107, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf727, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf728 = reinterpret_tensor(buf699, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf699  # reuse
        # Source Nodes: [cat_47], Original ATen: [aten.cat]
        triton_poi_fused_cat_145.run(buf721, buf723, buf725, buf727, buf728, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del buf721
        del buf723
        del buf725
        del buf727
        buf729 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf730 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf731 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_146.run(buf728, buf729, buf730, buf731, 3840, 98, grid=grid(3840), stream=stream0)
        buf732 = buf716; del buf716  # reuse
        buf733 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf735 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_147.run(buf729, buf730, buf731, primals_445, primals_446, buf732, buf733, buf735, primals_445, primals_446, 960, 4, grid=grid(960), stream=stream0)
        del buf729
        del buf730
        del buf731
        del primals_445
        del primals_446
        buf736 = reinterpret_tensor(buf659, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf659  # reuse
        # Source Nodes: [x_321], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_148.run(buf728, buf732, buf733, primals_108, primals_109, buf736, 376320, grid=grid(376320), stream=stream0)
        del buf733
        del primals_109
        buf737 = reinterpret_tensor(buf688, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf688  # reuse
        buf738 = reinterpret_tensor(buf737, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf737  # reuse
        # Source Nodes: [x_324, x_se_48], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_149.run(buf738, buf736, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf739 = extern_kernels.convolution(buf738, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf739, (8, 80, 1, 1), (80, 1, 1, 1))
        buf740 = reinterpret_tensor(buf739, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf739  # reuse
        buf741 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_130.run(buf740, primals_266, buf741, 640, grid=grid(640), stream=stream0)
        del primals_266
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf742 = extern_kernels.convolution(buf741, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf742, (8, 960, 1, 1), (960, 1, 1, 1))
        buf743 = reinterpret_tensor(buf742, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf742  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_150.run(buf743, primals_268, 7680, grid=grid(7680), stream=stream0)
        del primals_268
        buf744 = reinterpret_tensor(buf648, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf648  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324, x_325], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_151.run(buf736, buf743, buf744, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf745 = extern_kernels.convolution(buf744, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf745, (8, 264, 7, 7), (12936, 49, 7, 1))
        buf746 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_152.run(buf745, buf746, 2112, 49, grid=grid(2112, 49), stream=stream0)
        buf747 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        buf748 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        buf749 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_153.run(buf746, buf747, buf748, buf749, 1056, 98, grid=grid(1056), stream=stream0)
        buf750 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf751 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf753 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_154.run(buf747, buf748, buf749, primals_448, primals_449, buf750, buf751, buf753, primals_448, primals_449, 264, 4, grid=grid(264), stream=stream0)
        del primals_448
        del primals_449
        buf754 = reinterpret_tensor(buf745, (8, 264, 7, 7), (12936, 1, 1848, 264), 0); del buf745  # reuse
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_155.run(buf746, buf750, buf751, primals_110, primals_111, buf754, 103488, grid=grid(103488), stream=stream0)
        del primals_111
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf755 = extern_kernels.convolution(buf754, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf755, (8, 1584, 7, 7), (77616, 49, 7, 1))
        buf756 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_156.run(buf755, buf756, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf757 = empty_strided((1, 1584, 1, 1, 4), (6336, 1, 6336, 6336, 1584), device='cuda', dtype=torch.float32)
        buf758 = empty_strided((1, 1584, 1, 1, 4), (6336, 1, 6336, 6336, 1584), device='cuda', dtype=torch.float32)
        buf759 = empty_strided((1, 1584, 1, 1, 4), (6336, 1, 6336, 6336, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_157.run(buf756, buf757, buf758, buf759, 6336, 98, grid=grid(6336), stream=stream0)
        buf760 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf761 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf763 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_158.run(buf757, buf758, buf759, primals_451, primals_452, buf760, buf761, buf763, primals_451, primals_452, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_451
        del primals_452
        buf764 = reinterpret_tensor(buf755, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf755  # reuse
        buf918 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_159.run(buf756, buf760, buf761, primals_112, primals_113, buf764, buf918, 620928, grid=grid(620928), stream=stream0)
        del primals_113
        buf765 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_335], Original ATen: [aten.silu]
        triton_poi_fused_silu_160.run(buf764, buf765, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf766 = empty_strided((8, 396, 7, 7), (19404, 1, 2772, 396), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_161.run(buf765, buf766, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf767 = extern_kernels.convolution(buf766, primals_271, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf767, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf768 = buf766; del buf766  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_162.run(buf765, buf768, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf769 = extern_kernels.convolution(buf768, primals_272, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf769, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf770 = buf768; del buf768  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_163.run(buf765, buf770, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf771 = extern_kernels.convolution(buf770, primals_273, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf771, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf772 = buf770; del buf770  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_164.run(buf765, buf772, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf773 = extern_kernels.convolution(buf772, primals_274, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf773, (8, 396, 7, 7), (19404, 49, 7, 1))
        del buf772
        buf774 = buf764; del buf764  # reuse
        # Source Nodes: [cat_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_165.run(buf767, buf769, buf771, buf773, buf774, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf767
        del buf769
        del buf771
        buf775 = buf759; del buf759  # reuse
        buf776 = buf758; del buf758  # reuse
        buf777 = buf757; del buf757  # reuse
        # Source Nodes: [x_338], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_157.run(buf774, buf775, buf776, buf777, 6336, 98, grid=grid(6336), stream=stream0)
        buf778 = buf761; del buf761  # reuse
        buf779 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf781 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_338], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_158.run(buf775, buf776, buf777, primals_454, primals_455, buf778, buf779, buf781, primals_454, primals_455, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_454
        del primals_455
        buf782 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_338], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_166.run(buf774, buf778, buf779, primals_114, primals_115, buf782, 620928, grid=grid(620928), stream=stream0)
        del primals_115
        buf783 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf784 = reinterpret_tensor(buf783, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf783  # reuse
        # Source Nodes: [x_341, x_se_52], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_167.run(buf784, buf782, 12672, 49, grid=grid(12672), stream=stream0)
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf785 = extern_kernels.convolution(buf784, primals_275, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf785, (8, 132, 1, 1), (132, 1, 1, 1))
        buf786 = reinterpret_tensor(buf785, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf785  # reuse
        buf787 = reinterpret_tensor(buf749, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf749  # reuse
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_168.run(buf786, primals_276, buf787, 1056, grid=grid(1056), stream=stream0)
        del primals_276
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf788 = extern_kernels.convolution(buf787, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf788, (8, 1584, 1, 1), (1584, 1, 1, 1))
        buf789 = reinterpret_tensor(buf788, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf788  # reuse
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_169.run(buf789, primals_278, 12672, grid=grid(12672), stream=stream0)
        del primals_278
        buf790 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_341, x_342], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_170.run(buf782, buf789, buf790, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf791 = empty_strided((8, 792, 7, 7), (38808, 1, 5544, 792), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_171.run(buf790, buf791, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf792 = extern_kernels.convolution(buf791, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf792, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf793 = buf791; del buf791  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_172.run(buf790, buf793, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf794 = extern_kernels.convolution(buf793, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf794, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf795 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_45], Original ATen: [aten.cat]
        triton_poi_fused_cat_173.run(buf792, buf794, buf795, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf792
        del buf794
        buf796 = buf748; del buf748  # reuse
        buf797 = buf747; del buf747  # reuse
        buf798 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_345], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_153.run(buf795, buf796, buf797, buf798, 1056, 98, grid=grid(1056), stream=stream0)
        buf799 = buf751; del buf751  # reuse
        buf800 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf802 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_345], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_154.run(buf796, buf797, buf798, primals_457, primals_458, buf799, buf800, buf802, primals_457, primals_458, 264, 4, grid=grid(264), stream=stream0)
        del primals_457
        del primals_458
        buf803 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17, x_345], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_174.run(buf795, buf799, buf800, primals_116, primals_117, buf754, buf803, 103488, grid=grid(103488), stream=stream0)
        del primals_117
        # Source Nodes: [x_350], Original ATen: [aten.convolution]
        buf804 = extern_kernels.convolution(buf803, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf804, (8, 1584, 7, 7), (77616, 49, 7, 1))
        buf805 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_156.run(buf804, buf805, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf806 = buf777; del buf777  # reuse
        buf807 = buf776; del buf776  # reuse
        buf808 = buf775; del buf775  # reuse
        # Source Nodes: [x_351], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_157.run(buf805, buf806, buf807, buf808, 6336, 98, grid=grid(6336), stream=stream0)
        buf809 = buf779; del buf779  # reuse
        buf810 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf812 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_351], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_158.run(buf806, buf807, buf808, primals_460, primals_461, buf809, buf810, buf812, primals_460, primals_461, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_460
        del primals_461
        buf813 = reinterpret_tensor(buf804, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf804  # reuse
        buf917 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_351], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_159.run(buf805, buf809, buf810, primals_118, primals_119, buf813, buf917, 620928, grid=grid(620928), stream=stream0)
        del primals_119
        buf814 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten.silu]
        triton_poi_fused_silu_160.run(buf813, buf814, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf815 = reinterpret_tensor(buf773, (8, 396, 7, 7), (19404, 1, 2772, 396), 0); del buf773  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_161.run(buf814, buf815, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf816 = extern_kernels.convolution(buf815, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf816, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf817 = buf815; del buf815  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_162.run(buf814, buf817, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf818 = extern_kernels.convolution(buf817, primals_283, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf818, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf819 = buf817; del buf817  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_163.run(buf814, buf819, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf820 = extern_kernels.convolution(buf819, primals_284, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf820, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf821 = buf819; del buf819  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_164.run(buf814, buf821, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf822 = extern_kernels.convolution(buf821, primals_285, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf822, (8, 396, 7, 7), (19404, 49, 7, 1))
        del buf821
        buf823 = buf813; del buf813  # reuse
        # Source Nodes: [cat_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_165.run(buf816, buf818, buf820, buf822, buf823, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf816
        del buf818
        del buf820
        buf824 = buf808; del buf808  # reuse
        buf825 = buf807; del buf807  # reuse
        buf826 = buf806; del buf806  # reuse
        # Source Nodes: [x_357], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_157.run(buf823, buf824, buf825, buf826, 6336, 98, grid=grid(6336), stream=stream0)
        buf827 = buf810; del buf810  # reuse
        buf828 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf830 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_357], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_158.run(buf824, buf825, buf826, primals_463, primals_464, buf827, buf828, buf830, primals_463, primals_464, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_463
        del primals_464
        buf831 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_357], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_166.run(buf823, buf827, buf828, primals_120, primals_121, buf831, 620928, grid=grid(620928), stream=stream0)
        del primals_121
        buf832 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf833 = reinterpret_tensor(buf832, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf832  # reuse
        # Source Nodes: [x_360, x_se_56], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_167.run(buf833, buf831, 12672, 49, grid=grid(12672), stream=stream0)
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf834 = extern_kernels.convolution(buf833, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf834, (8, 132, 1, 1), (132, 1, 1, 1))
        buf835 = reinterpret_tensor(buf834, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf834  # reuse
        buf836 = reinterpret_tensor(buf798, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf798  # reuse
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_168.run(buf835, primals_287, buf836, 1056, grid=grid(1056), stream=stream0)
        del primals_287
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf837 = extern_kernels.convolution(buf836, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf837, (8, 1584, 1, 1), (1584, 1, 1, 1))
        buf838 = reinterpret_tensor(buf837, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf837  # reuse
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_169.run(buf838, primals_289, 12672, grid=grid(12672), stream=stream0)
        del primals_289
        buf839 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_360, x_361], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_170.run(buf831, buf838, buf839, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf840 = buf793; del buf793  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_171.run(buf839, buf840, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf841 = extern_kernels.convolution(buf840, primals_290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf841, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf842 = buf840; del buf840  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_172.run(buf839, buf842, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf843 = extern_kernels.convolution(buf842, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf843, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf844 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_173.run(buf841, buf843, buf844, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf841
        del buf843
        buf845 = buf797; del buf797  # reuse
        buf846 = buf796; del buf796  # reuse
        buf847 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_153.run(buf844, buf845, buf846, buf847, 1056, 98, grid=grid(1056), stream=stream0)
        buf848 = buf800; del buf800  # reuse
        buf849 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf851 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_154.run(buf845, buf846, buf847, primals_466, primals_467, buf848, buf849, buf851, primals_466, primals_467, 264, 4, grid=grid(264), stream=stream0)
        del primals_466
        del primals_467
        buf852 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_18, x_364], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_174.run(buf844, buf848, buf849, primals_122, primals_123, buf803, buf852, 103488, grid=grid(103488), stream=stream0)
        del primals_123
        # Source Nodes: [x_369], Original ATen: [aten.convolution]
        buf853 = extern_kernels.convolution(buf852, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf853, (8, 1584, 7, 7), (77616, 49, 7, 1))
        buf854 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_369], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_156.run(buf853, buf854, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf855 = buf826; del buf826  # reuse
        buf856 = buf825; del buf825  # reuse
        buf857 = buf824; del buf824  # reuse
        # Source Nodes: [x_370], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_157.run(buf854, buf855, buf856, buf857, 6336, 98, grid=grid(6336), stream=stream0)
        buf858 = buf828; del buf828  # reuse
        buf859 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf861 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_158.run(buf855, buf856, buf857, primals_469, primals_470, buf858, buf859, buf861, primals_469, primals_470, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_469
        del primals_470
        buf862 = reinterpret_tensor(buf853, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf853  # reuse
        buf916 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_159.run(buf854, buf858, buf859, primals_124, primals_125, buf862, buf916, 620928, grid=grid(620928), stream=stream0)
        del primals_125
        buf863 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_373], Original ATen: [aten.silu]
        triton_poi_fused_silu_160.run(buf862, buf863, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf864 = reinterpret_tensor(buf822, (8, 396, 7, 7), (19404, 1, 2772, 396), 0); del buf822  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_161.run(buf863, buf864, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf865 = extern_kernels.convolution(buf864, primals_293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf865, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf866 = buf864; del buf864  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_162.run(buf863, buf866, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf867 = extern_kernels.convolution(buf866, primals_294, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf867, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf868 = buf866; del buf866  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_163.run(buf863, buf868, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf869 = extern_kernels.convolution(buf868, primals_295, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf869, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf870 = buf868; del buf868  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_164.run(buf863, buf870, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf871 = extern_kernels.convolution(buf870, primals_296, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf871, (8, 396, 7, 7), (19404, 49, 7, 1))
        del buf870
        buf872 = buf862; del buf862  # reuse
        # Source Nodes: [cat_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_165.run(buf865, buf867, buf869, buf871, buf872, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf865
        del buf867
        del buf869
        del buf871
        buf873 = buf857; del buf857  # reuse
        buf874 = buf856; del buf856  # reuse
        buf875 = buf855; del buf855  # reuse
        # Source Nodes: [x_376], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_157.run(buf872, buf873, buf874, buf875, 6336, 98, grid=grid(6336), stream=stream0)
        buf876 = buf859; del buf859  # reuse
        buf877 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf879 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_158.run(buf873, buf874, buf875, primals_472, primals_473, buf876, buf877, buf879, primals_472, primals_473, 1584, 4, grid=grid(1584), stream=stream0)
        del buf873
        del buf874
        del buf875
        del primals_472
        del primals_473
        buf880 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_166.run(buf872, buf876, buf877, primals_126, primals_127, buf880, 620928, grid=grid(620928), stream=stream0)
        del buf877
        del primals_127
        buf881 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf882 = reinterpret_tensor(buf881, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf881  # reuse
        # Source Nodes: [x_379, x_se_60], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_167.run(buf882, buf880, 12672, 49, grid=grid(12672), stream=stream0)
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 132, 1, 1), (132, 1, 1, 1))
        buf884 = reinterpret_tensor(buf883, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf883  # reuse
        buf885 = reinterpret_tensor(buf847, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf847  # reuse
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_168.run(buf884, primals_298, buf885, 1056, grid=grid(1056), stream=stream0)
        del primals_298
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf886 = extern_kernels.convolution(buf885, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf886, (8, 1584, 1, 1), (1584, 1, 1, 1))
        buf887 = reinterpret_tensor(buf886, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf886  # reuse
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_169.run(buf887, primals_300, 12672, grid=grid(12672), stream=stream0)
        del primals_300
        buf888 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_379, x_380], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_170.run(buf880, buf887, buf888, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf889 = buf842; del buf842  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_171.run(buf888, buf889, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf890 = extern_kernels.convolution(buf889, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf890, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf891 = buf889; del buf889  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_172.run(buf888, buf891, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf892 = extern_kernels.convolution(buf891, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf892, (8, 132, 7, 7), (6468, 49, 7, 1))
        del buf891
        buf893 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_173.run(buf890, buf892, buf893, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf890
        del buf892
        buf894 = buf846; del buf846  # reuse
        buf895 = buf845; del buf845  # reuse
        buf896 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_153.run(buf893, buf894, buf895, buf896, 1056, 98, grid=grid(1056), stream=stream0)
        buf897 = buf849; del buf849  # reuse
        buf898 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf900 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_154.run(buf894, buf895, buf896, primals_475, primals_476, buf897, buf898, buf900, primals_475, primals_476, 264, 4, grid=grid(264), stream=stream0)
        del buf894
        del buf895
        del buf896
        del primals_475
        del primals_476
        buf901 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_383, x_388], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_174.run(buf893, buf897, buf898, primals_128, primals_129, buf852, buf901, 103488, grid=grid(103488), stream=stream0)
        del buf898
        del primals_129
        # Source Nodes: [x_389], Original ATen: [aten.convolution]
        buf902 = extern_kernels.convolution(buf901, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf902, (8, 1536, 7, 7), (75264, 49, 7, 1))
        buf903 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_389], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_175.run(buf902, buf903, 12288, 49, grid=grid(12288, 49), stream=stream0)
        buf904 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        buf905 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        buf906 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_390], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_176.run(buf903, buf904, buf905, buf906, 6144, 98, grid=grid(6144), stream=stream0)
        buf907 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf908 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf910 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_390], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_177.run(buf904, buf905, buf906, primals_478, primals_479, buf907, buf908, buf910, primals_478, primals_479, 1536, 4, grid=grid(1536), stream=stream0)
        del buf904
        del buf905
        del buf906
        del primals_478
        del primals_479
        buf911 = reinterpret_tensor(buf902, (8, 1536, 7, 7), (75264, 1, 10752, 1536), 0); del buf902  # reuse
        buf915 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_390, x_394], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_178.run(buf903, buf907, buf908, primals_130, primals_131, buf911, buf915, 602112, grid=grid(602112), stream=stream0)
        del buf908
        del primals_131
        buf912 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf913 = reinterpret_tensor(buf912, (8, 1536), (1536, 1), 0); del buf912  # reuse
        # Source Nodes: [x_395, x_397], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_179.run(buf913, buf911, 12288, 49, grid=grid(12288), stream=stream0)
        del buf911
        buf914 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_305, buf913, reinterpret_tensor(primals_304, (1536, 1000), (1, 1536), 0), alpha=1, beta=1, out=buf914)
        del primals_305
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_345, primals_345, 1, grid=grid(1), stream=stream0)
        del primals_345
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_348, primals_348, 1, grid=grid(1), stream=stream0)
        del primals_348
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_351, primals_351, 1, grid=grid(1), stream=stream0)
        del primals_351
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_354, primals_354, 1, grid=grid(1), stream=stream0)
        del primals_354
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_357, primals_357, 1, grid=grid(1), stream=stream0)
        del primals_357
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_360, primals_360, 1, grid=grid(1), stream=stream0)
        del primals_360
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_363, primals_363, 1, grid=grid(1), stream=stream0)
        del primals_363
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_366, primals_366, 1, grid=grid(1), stream=stream0)
        del primals_366
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_369, primals_369, 1, grid=grid(1), stream=stream0)
        del primals_369
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_372, primals_372, 1, grid=grid(1), stream=stream0)
        del primals_372
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_375, primals_375, 1, grid=grid(1), stream=stream0)
        del primals_375
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_378, primals_378, 1, grid=grid(1), stream=stream0)
        del primals_378
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_381, primals_381, 1, grid=grid(1), stream=stream0)
        del primals_381
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_384, primals_384, 1, grid=grid(1), stream=stream0)
        del primals_384
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_387, primals_387, 1, grid=grid(1), stream=stream0)
        del primals_387
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_390, primals_390, 1, grid=grid(1), stream=stream0)
        del primals_390
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_393, primals_393, 1, grid=grid(1), stream=stream0)
        del primals_393
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_396, primals_396, 1, grid=grid(1), stream=stream0)
        del primals_396
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_399, primals_399, 1, grid=grid(1), stream=stream0)
        del primals_399
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_402, primals_402, 1, grid=grid(1), stream=stream0)
        del primals_402
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_405, primals_405, 1, grid=grid(1), stream=stream0)
        del primals_405
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_408, primals_408, 1, grid=grid(1), stream=stream0)
        del primals_408
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_411, primals_411, 1, grid=grid(1), stream=stream0)
        del primals_411
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_414, primals_414, 1, grid=grid(1), stream=stream0)
        del primals_414
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_417, primals_417, 1, grid=grid(1), stream=stream0)
        del primals_417
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_420, primals_420, 1, grid=grid(1), stream=stream0)
        del primals_420
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_423, primals_423, 1, grid=grid(1), stream=stream0)
        del primals_423
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_426, primals_426, 1, grid=grid(1), stream=stream0)
        del primals_426
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_429, primals_429, 1, grid=grid(1), stream=stream0)
        del primals_429
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_432, primals_432, 1, grid=grid(1), stream=stream0)
        del primals_432
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_435, primals_435, 1, grid=grid(1), stream=stream0)
        del primals_435
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_438, primals_438, 1, grid=grid(1), stream=stream0)
        del primals_438
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_441, primals_441, 1, grid=grid(1), stream=stream0)
        del primals_441
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_444, primals_444, 1, grid=grid(1), stream=stream0)
        del primals_444
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_447, primals_447, 1, grid=grid(1), stream=stream0)
        del primals_447
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_450, primals_450, 1, grid=grid(1), stream=stream0)
        del primals_450
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_453, primals_453, 1, grid=grid(1), stream=stream0)
        del primals_453
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_456, primals_456, 1, grid=grid(1), stream=stream0)
        del primals_456
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_459, primals_459, 1, grid=grid(1), stream=stream0)
        del primals_459
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_462, primals_462, 1, grid=grid(1), stream=stream0)
        del primals_462
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_465, primals_465, 1, grid=grid(1), stream=stream0)
        del primals_465
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_468, primals_468, 1, grid=grid(1), stream=stream0)
        del primals_468
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_471, primals_471, 1, grid=grid(1), stream=stream0)
        del primals_471
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_474, primals_474, 1, grid=grid(1), stream=stream0)
        del primals_474
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_180.run(primals_477, primals_477, 1, grid=grid(1), stream=stream0)
        del primals_477
        return (buf914, buf0, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_12, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_54, primals_55, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, buf1, buf3, buf13, buf14, buf16, buf26, buf27, buf29, buf39, reinterpret_tensor(buf40, (8, 16, 112, 112), (401408, 12544, 112, 1), 0), reinterpret_tensor(buf40, (8, 16, 112, 112), (401408, 12544, 112, 1), 200704), buf45, buf55, buf57, buf59, buf61, buf63, buf73, reinterpret_tensor(buf74, (8, 96, 56, 56), (602112, 1, 10752, 192), 0), reinterpret_tensor(buf74, (8, 96, 56, 56), (602112, 1, 10752, 192), 96), buf77, buf87, reinterpret_tensor(buf88, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), reinterpret_tensor(buf88, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), buf91, buf101, buf102, buf104, buf114, reinterpret_tensor(buf115, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), reinterpret_tensor(buf115, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), buf118, buf128, buf129, buf131, buf141, buf143, buf145, buf147, buf149, buf151, buf158, buf159, buf162, buf164, buf165, buf167, buf168, buf170, buf177, reinterpret_tensor(buf178, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), reinterpret_tensor(buf178, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), buf181, buf188, reinterpret_tensor(buf190, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf190, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf195, buf202, buf203, buf206, buf208, buf209, buf211, reinterpret_tensor(buf212, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf212, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf217, buf224, reinterpret_tensor(buf225, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), reinterpret_tensor(buf225, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), buf228, buf235, reinterpret_tensor(buf237, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf237, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf242, buf249, buf250, buf253, buf255, buf256, buf258, reinterpret_tensor(buf259, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf259, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf264, buf271, reinterpret_tensor(buf272, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), reinterpret_tensor(buf272, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), buf275, buf282, reinterpret_tensor(buf284, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf284, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf289, buf296, buf297, buf300, buf302, buf303, buf305, reinterpret_tensor(buf306, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf306, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf311, buf318, buf319, buf321, buf328, buf330, buf332, buf334, buf336, buf343, buf344, buf347, buf349, buf350, buf352, buf353, buf355, buf362, reinterpret_tensor(buf363, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), reinterpret_tensor(buf363, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), buf366, buf373, reinterpret_tensor(buf375, (8, 156, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf375, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), reinterpret_tensor(buf375, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), reinterpret_tensor(buf375, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), buf384, buf391, buf392, buf395, buf397, buf398, buf400, reinterpret_tensor(buf401, (8, 312, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf401, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), buf406, buf413, reinterpret_tensor(buf414, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), reinterpret_tensor(buf414, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), buf417, buf424, reinterpret_tensor(buf426, (8, 156, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf426, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), reinterpret_tensor(buf426, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), reinterpret_tensor(buf426, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), buf435, buf442, buf443, buf446, buf448, buf449, buf451, reinterpret_tensor(buf452, (8, 312, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf452, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), buf457, buf464, reinterpret_tensor(buf465, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), reinterpret_tensor(buf465, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), buf468, buf475, reinterpret_tensor(buf477, (8, 156, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf477, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), reinterpret_tensor(buf477, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), reinterpret_tensor(buf477, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), buf486, buf493, buf494, buf497, buf499, buf500, buf502, reinterpret_tensor(buf503, (8, 312, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf503, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), buf508, buf515, buf516, buf518, buf525, buf527, buf529, buf536, buf537, buf540, buf542, buf543, buf545, buf546, buf548, buf555, reinterpret_tensor(buf556, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), reinterpret_tensor(buf556, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), buf559, buf566, reinterpret_tensor(buf568, (8, 120, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf568, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), reinterpret_tensor(buf568, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), reinterpret_tensor(buf568, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), buf577, buf584, buf585, buf588, buf590, buf591, buf593, reinterpret_tensor(buf594, (8, 240, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf594, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), buf599, buf606, reinterpret_tensor(buf607, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), reinterpret_tensor(buf607, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), buf610, buf617, reinterpret_tensor(buf619, (8, 120, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf619, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), reinterpret_tensor(buf619, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), reinterpret_tensor(buf619, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), buf628, buf635, buf636, buf639, buf641, buf642, buf644, reinterpret_tensor(buf645, (8, 240, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf645, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), buf650, buf657, reinterpret_tensor(buf658, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), reinterpret_tensor(buf658, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), buf661, buf668, reinterpret_tensor(buf670, (8, 120, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf670, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), reinterpret_tensor(buf670, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), reinterpret_tensor(buf670, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), buf679, buf686, buf687, buf690, buf692, buf693, buf695, reinterpret_tensor(buf696, (8, 240, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf696, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), buf701, buf708, buf709, buf711, buf718, buf720, buf722, buf724, buf726, buf728, buf735, buf736, buf738, buf740, buf741, buf743, buf744, buf746, buf753, buf754, buf756, buf763, reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), buf774, buf781, buf782, buf784, buf786, buf787, buf789, reinterpret_tensor(buf790, (8, 792, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf790, (8, 792, 7, 7), (77616, 49, 7, 1), 38808), buf795, buf802, buf803, buf805, buf812, reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), buf823, buf830, buf831, buf833, buf835, buf836, buf838, reinterpret_tensor(buf839, (8, 792, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf839, (8, 792, 7, 7), (77616, 49, 7, 1), 38808), buf844, buf851, buf852, buf854, buf861, reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), buf872, buf879, buf880, buf882, buf884, buf885, buf887, reinterpret_tensor(buf888, (8, 792, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf888, (8, 792, 7, 7), (77616, 49, 7, 1), 38808), buf893, buf900, buf901, buf903, buf910, buf913, reinterpret_tensor(primals_304, (1000, 1536), (1536, 1), 0), buf915, reinterpret_tensor(buf907, (1, 1536, 1, 1), (1536, 1, 1, 1), 0), reinterpret_tensor(buf897, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf876, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), buf916, reinterpret_tensor(buf858, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), reinterpret_tensor(buf848, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf827, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), buf917, reinterpret_tensor(buf809, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), reinterpret_tensor(buf799, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf778, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), buf918, reinterpret_tensor(buf760, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), reinterpret_tensor(buf750, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf732, (1, 960, 1, 1), (960, 1, 1, 1), 0), buf919, reinterpret_tensor(buf715, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf705, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf683, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf920, reinterpret_tensor(buf665, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf654, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf632, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf921, reinterpret_tensor(buf614, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf603, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf581, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf922, reinterpret_tensor(buf563, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf552, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf533, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf923, reinterpret_tensor(buf522, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf512, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf490, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf924, reinterpret_tensor(buf472, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf461, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf439, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf925, reinterpret_tensor(buf421, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf410, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf388, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf926, reinterpret_tensor(buf370, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf359, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf340, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf927, reinterpret_tensor(buf325, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf293, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf928, reinterpret_tensor(buf279, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf268, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf246, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf929, reinterpret_tensor(buf232, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf221, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf199, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf930, reinterpret_tensor(buf185, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf174, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf155, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf931, reinterpret_tensor(buf138, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf125, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf932, reinterpret_tensor(buf111, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf98, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf84, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf933, reinterpret_tensor(buf70, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf934, reinterpret_tensor(buf52, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf36, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf23, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((264, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1536, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1000, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_319 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_325 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_328 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_349 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_352 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_355 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_358 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_364 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_367 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_370 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_373 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_376 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_379 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_382 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_385 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_388 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_391 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_394 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_397 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_400 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_403 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_406 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_409 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_412 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_415 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_418 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_421 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_424 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_427 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_430 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_433 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_436 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_439 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_442 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_445 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_448 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_451 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_454 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_457 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_460 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_463 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_466 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_469 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_472 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_475 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_478 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_mixnet_l', benchmark_compiled_module)
