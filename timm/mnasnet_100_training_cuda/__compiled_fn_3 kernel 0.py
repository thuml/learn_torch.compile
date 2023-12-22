
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckowufij3paig7x67vavgowelk2fsgz6chglyo5lz3fpnubxaftt.py
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
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clk32p3dallahhj3b3au642oeuehx5abnbgpaj4hxrbmnnlj5ws4.py
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


# kernel path: /tmp/torchinductor_youkaichao/zp/czpbhsqa4bl64yzbi53idd7q6sif56ryd5svk75hqtg6lhfdapz6.py
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/zy/czyzqrzvhjblomppazsrzqjlhvumbd7b75l6uxtbevueyy5ue6l6.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# shortcut => relu
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmvogclisrh7vnusk3z2hqzansor3uzlthoxli7pjsq637772mn.py
# Source Nodes: [x_11], Original ATen: [aten.convolution]
# x_11 => convolution_2
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpskxxvym2ot4bfayyv2tiwg2m3jgbutwbwxkgir36weqwcgh2b.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/uz/cuznr5pnwhsteyql7y3qstt2bcprx7i6wmsxqbnvnnzget4knlxf.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lo/clo7nrxhd2ynknaqnte5mi6imdd3vp25ioaaq5gtokpxbxe4iokt.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5jv44h4c4inlt3o6uy6gnvz5eq75jq2r5l32cvaaobvz3w2buw.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
# x_12 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77nretaif6g67svuxfqtbqpok2zrduidmqx6npeidje3dgunf5g.py
# Source Nodes: [x_16], Original ATen: [aten.convolution]
# x_16 => convolution_3
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (602112*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2yvqyspav7z3o5fdeq44qjyxmbhpi5ffyvoxbhmla7ylxyy2ff.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7pup6jomuojz5gwicehncrbzxtubdqngh7tizrcordw3yp3vgc.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
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
        tmp0 = tl.load(in_ptr0 + (x1 + (48*r2) + (5376*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (48*r2) + (5376*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (48*r2) + (5376*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (48*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (48*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (48*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrifjxxs6yluiq33nbkimwkfcesfokbrwjby2h6nbkyp3wk4aja.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (48*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbwwpc4i6wq3nhldgc4nauqvta65vtpk4sdpcorlhdfd6xolfuy.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_17 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_20 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccu5sod4gkmxk7opdhhfu5osrjxexcgrivz6sqk7xc3pufw7nl2z.py
# Source Nodes: [x_21], Original ATen: [aten.convolution]
# x_21 => convolution_4
triton_poi_fused_convolution_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44c2pklxczfcztceskm3c52jfwybxoi476zhflfahv3msicf4ap.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nuc7zvbr4tvmk6opdlhxkmkpo66lqdbcq7zywqpxp6ms4h5cwz.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (x1 + (48*r2) + (4704*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (48*r2) + (4704*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (48*r2) + (4704*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (48*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (48*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (48*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxq4upjguqfiq6kznmf3jcllclqa6leh2fxerdjlyngly2hwa7s.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
# x_22 => add_21, add_22, add_23, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (48*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/27/c27uoi5fwla6gohwti3risjdfv3c2nvb2pjrofszkbt7v6e3hpeg.py
# Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_22 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_25 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvbemyg7r4ina5czrsqinjatus4rsyigq36zeduqlw6jrg3aogm.py
# Source Nodes: [x_27], Original ATen: [aten.convolution]
# x_27 => convolution_5
triton_poi_fused_convolution_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqj3vkaile7cnanrkjk5bl5ydpcukk4mwtfd2bwjx47msta77sm.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtxshsygjumd2xtksgejdgr67tqchhc3zujsesoklj6xa2d4cfm.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fe/cfe2ov5xsg2jmm4pp22w7fxqdulkrfcyqfjmyebycmcjd75oxdfw.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => add_26, add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/cf/ccf5pz4pu6cyu6yw3tayfiqedlyelxcswn66tmutx365t4wfzhed.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csrymyhosrtabm22h7zqju4vslo76blh3zew7e6ahpwqhcw543ez.py
# Source Nodes: [x_32], Original ATen: [aten.convolution]
# x_32 => convolution_6
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (225792*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckoepdj5dfh4bmewlda2niwpswwjwzfiw4h7czsjh5yfivdgb7zd.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14112
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xzpmxfmlastabws4i26ecbzy7me2ll3i5bobveukpj6njwqu5r.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
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
        tmp0 = tl.load(in_ptr0 + (x1 + (72*r2) + (7056*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (72*r2) + (7056*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (72*r2) + (7056*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (72*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (72*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (72*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccunks44r7re443e6au3gvaoy6uzsflzu3a5ln55wpkqkaftf3vs.py
# Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
# x_33 => add_31, add_32, add_33, mul_43, mul_44, mul_45, mul_46, mul_47, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (72*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ys/cysobzwebmmxmz22fdcwsvfs5bwiqavjlbyvtqooly5uimj2lpkl.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_33 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_36 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sd/csdnu5ry27a75idn633qyllc4hwzwsik25hdnach7kj4cret7gnc.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_45
# x_44 => add_41, add_44, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_add_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_32', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/ctedqvlo5sjkm7qnnxm33qqqsv7zsnv3fqvqm6gnzu43tbdfi45m.py
# Source Nodes: [x_71], Original ATen: [aten.convolution]
# x_71 => convolution_13
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (56448*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5fygwpqc7xh6zfmtgex6eopkunn3yrvpwtxazgqznzcupwbb77.py
# Source Nodes: [x_72], Original ATen: [aten._native_batch_norm_legit_functional]
# x_72 => var_mean_13
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3528
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rv/crvmzrbah774oiwzy7qzwxfpzc5ya6wy3v6fvmf2bdv6ecinmxzt.py
# Source Nodes: [x_72], Original ATen: [aten._native_batch_norm_legit_functional]
# x_72 => add_68, add_69, add_70, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_per_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (72*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/5p/c5py7oyvolvke7a3eb2ko33f4gzhyaq3cw2jybpb3mr5xcbts65j.py
# Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_72 => add_68, add_71, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# x_75 => relu_9
triton_poi_fused__native_batch_norm_legit_functional_relu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6aiv72f43t6rhuhgvvbfudaghvtkrrwixu7ycvc75mipoccak6.py
# Source Nodes: [x_77], Original ATen: [aten.convolution]
# x_77 => convolution_14
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzpp54kanuqrvqbkoyelowkld75pstel3igew63ldjdfymuzr5h.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
# x_78 => var_mean_14
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5rzamtyjfgdnu3vjqvkpax2ekgydohaln5vi73amfh5exau4om.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
# x_78 => add_73, add_74, add_75, mul_100, mul_101, mul_102, mul_103, mul_99, rsqrt_14, squeeze_43, var_mean_14
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/zd/czdmies4mf6f5jiz6kg7dvcleummdrhidvqrqc5wnhytthfaiv7r.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
# x_78 => add_73, add_76, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdnzwmyvjc4jldht3ct72wqeof674a4gv3k43yiz3bgqwkapxbb.py
# Source Nodes: [x_82], Original ATen: [aten.convolution]
# x_82 => convolution_15
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5v7e6hybuatv5movo6rv4ehmnok4xwzr4tgiyegrat7yepwhqx.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
# x_83 => var_mean_15
triton_red_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5880
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


# kernel path: /tmp/torchinductor_youkaichao/ki/ckirlxlijvp7iwksahd2ncemwmfjefdzqwtotziwn626xyvmd7rp.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
# x_83 => add_78, add_79, add_80, mul_106, mul_107, mul_108, mul_109, mul_110, rsqrt_15, squeeze_46, var_mean_15
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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
    tmp16 = 6272.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/jx/cjx3ubmwmc5j4xcykomzydy5dwdcx5mzbpa5ireo23c6klxx42bp.py
# Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_83 => add_78, add_81, mul_105, mul_111, rsqrt_15, sub_15, var_mean_15
# x_86 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_relu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaembzwe433f47ufxgjh3iwmshwd7tkfs72tblwd4dnca26h6hf.py
# Source Nodes: [shortcut_6, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_6 => add_92
# x_94 => add_88, add_91, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_add_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_45', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clrhmevgppfoyjlwusv2pp54smwoj6ykar6jleykun5wnorp344k.py
# Source Nodes: [x_116], Original ATen: [aten.convolution]
# x_116 => convolution_21
triton_poi_fused_convolution_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_46', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4cyglucowv6zbl7do3fsxx7ngicmr7iedtnwzqneau34diyajd.py
# Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
# x_117 => var_mean_21
triton_red_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4o/c4ozwycr4wvnapvpvsywzf2vcbktiauf2vwgfdc6hx5ubevd63bb.py
# Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
# x_117 => add_110, add_111, add_112, mul_148, mul_149, mul_150, mul_151, mul_152, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ub/cubg47chftz6cdor3bb5nvjssfifopfrvlgabc5uc6qwryvb7abk.py
# Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_117 => add_110, add_113, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# x_120 => relu_14
triton_poi_fused__native_batch_norm_legit_functional_relu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_49', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3wo6g5vxr5rpzvmhnbmu5wwxuitbyuroijkjptonae6fq6vqnk.py
# Source Nodes: [x_121], Original ATen: [aten.convolution]
# x_121 => convolution_22
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sr/csreeqll5tsahdrcvfks2rc5jkt344dr5hcwocfq3gqmmkafdrmc.py
# Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
# x_122 => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chp3ryukakwgmmmtlh3rqrzafamzyffphuvesbdtn7tx4fmfr6xz.py
# Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
# x_122 => add_115, add_116, add_117, mul_155, mul_156, mul_157, mul_158, mul_159, rsqrt_22, squeeze_67, var_mean_22
triton_per_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/wo/cwopwfjrmgmtpyfga7mtuu6uqp4cyb4soxtv6ftalpvf4cjgohe3.py
# Source Nodes: [x_122, x_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_122 => add_115, add_118, mul_154, mul_160, rsqrt_22, sub_22, var_mean_22
# x_125 => relu_15
triton_poi_fused__native_batch_norm_legit_functional_relu_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_53', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tbcyftkjnfc57k7vq2slo342h5kgaj2jy7rpzaimqsvbl3po54.py
# Source Nodes: [x_127], Original ATen: [aten.convolution]
# x_127 => convolution_23
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcyywxtpzdnbfwi7un3g7nr7fogspcv544fqvuqxx42xtdcznov.py
# Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
# x_128 => var_mean_23
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hi/chimhslv5rbjtfua5tppkbbmb7kowrlmnpw55h2xylyrepvveyau.py
# Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
# x_128 => add_120, add_121, add_122, mul_162, mul_163, mul_164, mul_165, mul_166, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/s2/cs24gyor4zvfzvoxshkozsnydyzkdoeyljzcxu2zopz4o6lgsaey.py
# Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
# x_128 => add_120, add_123, mul_161, mul_167, rsqrt_23, sub_23, var_mean_23
triton_poi_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlzpddfsnnw3hay3ceopf7jh6sgkrydge7vrjrcw6m66xw2hvcc.py
# Source Nodes: [x_132], Original ATen: [aten.convolution]
# x_132 => convolution_24
triton_poi_fused_convolution_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjwqnx3ht54hyr47cnkwwhdrprvk7men5cqsrrudbjhb6izwod2.py
# Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
# x_133 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4p2ymmiqxmxzefqv7l3r43izeregwf7ypj6nt4tldymjxmh6fx.py
# Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
# x_133 => add_125, add_126, add_127, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/id/cidakdpmr2kgmeetytb5fpwb2ktfkrjg7z43xhqat2a6kdx2b3nw.py
# Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_133 => add_125, add_128, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# x_136 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_61', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxgnpskxchc7fvdeng23kjqfpwawzsznqdf4332stat5lucmvni.py
# Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_9 => add_139
# x_144 => add_135, add_138, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_add_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_62', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ak/cakzcdhfgakdyhyilnjb4q34yytoqfjzye3gx2rug2p2czhnavqu.py
# Source Nodes: [x_177], Original ATen: [aten.convolution]
# x_177 => convolution_32
triton_poi_fused_convolution_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (18816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/cke5yd35tnjyx42yqgnyozyf3etb6gete75pe647vqtdaske3fbw.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => var_mean_32
triton_red_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1248
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96)
    x0 = xindex % 96
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
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/km/ckmwxd7jkzkueinlixfbb6n3cc6csjjtixk2bcgk5ueicaeyleoo.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => add_167, add_168, add_169, mul_225, mul_226, mul_227, mul_228, mul_229, rsqrt_32, squeeze_97, var_mean_32
triton_per_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dcxyghrfymyvlqvo5pl6ldnffmvirdvnvgaz4xtuyqrhw4p35a.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => add_167, add_170, mul_224, mul_230, rsqrt_32, sub_32, var_mean_32
triton_poi_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn73fg7fl37jwa2vmuapdzeqb75zte34vve2q3xibkw4ojtdzgt2.py
# Source Nodes: [x_182], Original ATen: [aten.convolution]
# x_182 => convolution_33
triton_poi_fused_convolution_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (576*x2) + (112896*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzytpo3zxt3heoqnxaokw4o3bjw5nvrxrv25qgtz26bbhg5rnsv.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
# x_183 => var_mean_33
triton_red_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7488
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 576)
    x0 = xindex % 576
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
        tmp3 = tl.load(in_ptr0 + (x0 + (576*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pug7el5asnrzgqfw3iqrb7zq2kdtqrfq7ulczxvkstmvcdh7yl.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
# x_183 => add_172, add_173, add_174, mul_232, mul_233, mul_234, mul_235, mul_236, rsqrt_33, squeeze_100, var_mean_33
triton_per_fused__native_batch_norm_legit_functional_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_69', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (576*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7vsdoduoqlpax5i2dxrbhwtlpl3wbavs4ka4wkj5xcoi7zgzwa.py
# Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_183 => add_172, add_175, mul_231, mul_237, rsqrt_33, sub_33, var_mean_33
# x_186 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfq34dqf6e5tugcuxfjf5f7d5f65vxn6rs2t6ys54slkou2vns4t.py
# Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_12 => add_186
# x_194 => add_182, add_185, mul_245, mul_251, rsqrt_35, sub_35, var_mean_35
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
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lzerxr2ztjiwckmur5vzp6kqh3tkv2hzcr5lckj4qj5zok4sud.py
# Source Nodes: [x_204], Original ATen: [aten.convolution]
# x_204 => convolution_37
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (576*x2) + (28224*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtsrkyrghhfmsv56xifw3hgzvqrrg2dxbmyjus24cmjlacnxppy.py
# Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
# x_205 => var_mean_37
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (56448*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5o/c5ozi6pe5wcznyqtgblezsy3xurdgk5hn4lcdcgk55aqi6ytjtxm.py
# Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
# x_205 => add_193, add_194, add_195, mul_260, mul_261, mul_262, mul_263, mul_264, rsqrt_37, squeeze_112, var_mean_37
triton_per_fused__native_batch_norm_legit_functional_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_74', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (576*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzoaw7yc2wgg5z4hqyxlf7vhx2nuanruqedsabpej36nbzkuvcj.py
# Source Nodes: [x_205, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_205 => add_193, add_196, mul_259, mul_265, rsqrt_37, sub_37, var_mean_37
# x_208 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_relu_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3u/c3u7ehr33jmzaloanpaksp32hwgcx3qi2qb3nnewmt7bkrz3bkp7.py
# Source Nodes: [x_210], Original ATen: [aten.convolution]
# x_210 => convolution_38
triton_poi_fused_convolution_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_76', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hg/chgiyfuyadkyyp7xouwth4kvu2q7gtc2fefdqrj6ibb5ukelcysv.py
# Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
# x_211 => var_mean_38
triton_red_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ky/ckym23tm4g3ohghxtn4kfd4sckod4brh27n3n6hngtp5xx4a6cco.py
# Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
# x_211 => add_198, add_199, add_200, mul_267, mul_268, mul_269, mul_270, mul_271, rsqrt_38, squeeze_115, var_mean_38
triton_per_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chpnvalovoo6bzhif3taefmvyyjic236v7vhre5ubxl6nk7qomgb.py
# Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
# x_211 => add_198, add_201, mul_266, mul_272, rsqrt_38, sub_38, var_mean_38
triton_poi_fused__native_batch_norm_legit_functional_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_79', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwemrdyl55sg2rhcfiosgj5wnbqnxglm7fyvlqx5aqdyqv5txwcv.py
# Source Nodes: [x_215], Original ATen: [aten.convolution]
# x_215 => convolution_39
triton_poi_fused_convolution_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_80', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdimaiw32evc7hidalruhcmyodemn5ba57i3xmysyu5yiy7c4lj.py
# Source Nodes: [x_216], Original ATen: [aten._native_batch_norm_legit_functional]
# x_216 => var_mean_39
triton_red_fused__native_batch_norm_legit_functional_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7uv4rf4l4bjgefrn36eqy5uryvsgg3gf4tmjplvhw7jyd7xyzv.py
# Source Nodes: [x_216], Original ATen: [aten._native_batch_norm_legit_functional]
# x_216 => add_203, add_204, add_205, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
triton_per_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/zj/czjyvhagvvhkgx7rqmabenktszjdqy3ew643bs6drw34a655w4gu.py
# Source Nodes: [x_216, x_219], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_216 => add_203, add_206, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# x_219 => relu_26
triton_poi_fused__native_batch_norm_legit_functional_relu_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_83', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nf/cnf4b4t5kvbzob6qduc7s25vq4zuj5osmwixvjq5mc6xwghqgcso.py
# Source Nodes: [shortcut_14, x_227], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_14 => add_217
# x_227 => add_213, add_216, mul_287, mul_293, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_add_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_84', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/coffnchnz34qzikuxlddh4vxbpw3h62ryeamuzazbym6fiy7urbb.py
# Source Nodes: [x_277], Original ATen: [aten.convolution]
# x_277 => convolution_50
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfey26wekz7tproqh3cr4kbejeaurjy472blznzsj5oufgf3qk5.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => var_mean_50
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4g/c4gsp4dggzw7irh6buukwlgz2zvcatsheyqrhrv6nixeer2ksjya.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => add_261, add_262, add_263, mul_351, mul_352, mul_353, mul_354, mul_355, rsqrt_50, squeeze_151, var_mean_50
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/6i/c6idojjihyc7ioumrmd4sqwmw36vawlwhrtthckbuiyznk2j6xnx.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => add_261, add_264, mul_350, mul_356, rsqrt_50, sub_50, var_mean_50
triton_poi_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crfmbxm73xpb5c5jwtauccyb3q3fu25eazopbto6l5uyt7engsfn.py
# Source Nodes: [x_283], Original ATen: [aten.convolution]
# x_283 => convolution_51
triton_poi_fused_convolution_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_89', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/io/ciolrwtsng544w5mahx63vwdz3hoo42jcg4vhu2piix2uyrr4633.py
# Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
# x_284 => var_mean_51
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzeusnu22z7sgib5ffsjzzuggy3s5lixtwgnqizc5gpsydi63k2.py
# Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
# x_284 => add_266, add_267, add_268, mul_358, mul_359, mul_360, mul_361, mul_362, rsqrt_51, squeeze_154, var_mean_51
triton_per_fused__native_batch_norm_legit_functional_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_91', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp18 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/7m/c7m5qdgxc4smyfucd3qpls5sgunippnn2gjxfjbiorr3xfn53mjj.py
# Source Nodes: [x_284, x_288], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_284 => add_266, add_269, mul_357, mul_363, rsqrt_51, sub_51, var_mean_51
# x_288 => relu_34
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_92', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6besfgdamdhx6gvbzx2oldsbk7o7vrrcoffrvibodlacyvtapi.py
# Source Nodes: [x_289, x_291], Original ATen: [aten.mean, aten.view]
# x_289 => mean
# x_291 => view
triton_per_fused_mean_view_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_93', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2iqksjizqc7wgt5mw2jrlowbm73xpsfxtq5bwfe7t6xqjububw.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_94', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (48, ), (1, ))
    assert_size_stride(primals_8, (48, ), (1, ))
    assert_size_stride(primals_9, (48, ), (1, ))
    assert_size_stride(primals_10, (48, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (72, ), (1, ))
    assert_size_stride(primals_14, (72, ), (1, ))
    assert_size_stride(primals_15, (72, ), (1, ))
    assert_size_stride(primals_16, (72, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (72, ), (1, ))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_21, (72, ), (1, ))
    assert_size_stride(primals_22, (72, ), (1, ))
    assert_size_stride(primals_23, (24, ), (1, ))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (72, ), (1, ))
    assert_size_stride(primals_26, (72, ), (1, ))
    assert_size_stride(primals_27, (72, ), (1, ))
    assert_size_stride(primals_28, (72, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_30, (40, ), (1, ))
    assert_size_stride(primals_31, (120, ), (1, ))
    assert_size_stride(primals_32, (120, ), (1, ))
    assert_size_stride(primals_33, (120, ), (1, ))
    assert_size_stride(primals_34, (120, ), (1, ))
    assert_size_stride(primals_35, (40, ), (1, ))
    assert_size_stride(primals_36, (40, ), (1, ))
    assert_size_stride(primals_37, (120, ), (1, ))
    assert_size_stride(primals_38, (120, ), (1, ))
    assert_size_stride(primals_39, (120, ), (1, ))
    assert_size_stride(primals_40, (120, ), (1, ))
    assert_size_stride(primals_41, (40, ), (1, ))
    assert_size_stride(primals_42, (40, ), (1, ))
    assert_size_stride(primals_43, (240, ), (1, ))
    assert_size_stride(primals_44, (240, ), (1, ))
    assert_size_stride(primals_45, (240, ), (1, ))
    assert_size_stride(primals_46, (240, ), (1, ))
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
    assert_size_stride(primals_59, (80, ), (1, ))
    assert_size_stride(primals_60, (80, ), (1, ))
    assert_size_stride(primals_61, (480, ), (1, ))
    assert_size_stride(primals_62, (480, ), (1, ))
    assert_size_stride(primals_63, (480, ), (1, ))
    assert_size_stride(primals_64, (480, ), (1, ))
    assert_size_stride(primals_65, (96, ), (1, ))
    assert_size_stride(primals_66, (96, ), (1, ))
    assert_size_stride(primals_67, (576, ), (1, ))
    assert_size_stride(primals_68, (576, ), (1, ))
    assert_size_stride(primals_69, (576, ), (1, ))
    assert_size_stride(primals_70, (576, ), (1, ))
    assert_size_stride(primals_71, (96, ), (1, ))
    assert_size_stride(primals_72, (96, ), (1, ))
    assert_size_stride(primals_73, (576, ), (1, ))
    assert_size_stride(primals_74, (576, ), (1, ))
    assert_size_stride(primals_75, (576, ), (1, ))
    assert_size_stride(primals_76, (576, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_78, (192, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_80, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_82, (1152, ), (1, ))
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
    assert_size_stride(primals_101, (320, ), (1, ))
    assert_size_stride(primals_102, (320, ), (1, ))
    assert_size_stride(primals_103, (1280, ), (1, ))
    assert_size_stride(primals_104, (1280, ), (1, ))
    assert_size_stride(primals_105, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_106, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_108, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_109, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_111, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_112, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_114, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_115, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_116, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_117, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_118, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_119, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_120, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_121, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_123, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_124, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_125, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_126, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_127, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_128, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_129, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_130, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_131, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_132, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_133, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_134, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_135, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_136, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_138, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_139, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_141, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_142, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_143, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_144, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_145, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_147, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_148, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_149, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_150, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_151, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_152, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_153, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_154, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_156, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_157, (1000, 1280), (1280, 1))
    assert_size_stride(primals_158, (1000, ), (1, ))
    assert_size_stride(primals_159, (), ())
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (32, ), (1, ))
    assert_size_stride(primals_162, (), ())
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_164, (32, ), (1, ))
    assert_size_stride(primals_165, (), ())
    assert_size_stride(primals_166, (16, ), (1, ))
    assert_size_stride(primals_167, (16, ), (1, ))
    assert_size_stride(primals_168, (), ())
    assert_size_stride(primals_169, (48, ), (1, ))
    assert_size_stride(primals_170, (48, ), (1, ))
    assert_size_stride(primals_171, (), ())
    assert_size_stride(primals_172, (48, ), (1, ))
    assert_size_stride(primals_173, (48, ), (1, ))
    assert_size_stride(primals_174, (), ())
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (24, ), (1, ))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (72, ), (1, ))
    assert_size_stride(primals_179, (72, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (72, ), (1, ))
    assert_size_stride(primals_182, (72, ), (1, ))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_185, (24, ), (1, ))
    assert_size_stride(primals_186, (), ())
    assert_size_stride(primals_187, (72, ), (1, ))
    assert_size_stride(primals_188, (72, ), (1, ))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (72, ), (1, ))
    assert_size_stride(primals_191, (72, ), (1, ))
    assert_size_stride(primals_192, (), ())
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_194, (24, ), (1, ))
    assert_size_stride(primals_195, (), ())
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (72, ), (1, ))
    assert_size_stride(primals_198, (), ())
    assert_size_stride(primals_199, (72, ), (1, ))
    assert_size_stride(primals_200, (72, ), (1, ))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (40, ), (1, ))
    assert_size_stride(primals_203, (40, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (120, ), (1, ))
    assert_size_stride(primals_206, (120, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (120, ), (1, ))
    assert_size_stride(primals_209, (120, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (40, ), (1, ))
    assert_size_stride(primals_212, (40, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_215, (120, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (120, ), (1, ))
    assert_size_stride(primals_218, (120, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (40, ), (1, ))
    assert_size_stride(primals_221, (40, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (240, ), (1, ))
    assert_size_stride(primals_224, (240, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (240, ), (1, ))
    assert_size_stride(primals_227, (240, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (80, ), (1, ))
    assert_size_stride(primals_230, (80, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (480, ), (1, ))
    assert_size_stride(primals_233, (480, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_236, (480, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_239, (80, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (480, ), (1, ))
    assert_size_stride(primals_242, (480, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_245, (480, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (80, ), (1, ))
    assert_size_stride(primals_248, (80, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_251, (480, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_254, (480, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (96, ), (1, ))
    assert_size_stride(primals_257, (96, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (576, ), (1, ))
    assert_size_stride(primals_260, (576, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (576, ), (1, ))
    assert_size_stride(primals_263, (576, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_266, (96, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (576, ), (1, ))
    assert_size_stride(primals_269, (576, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (576, ), (1, ))
    assert_size_stride(primals_272, (576, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_275, (192, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (1152, ), (1, ))
    assert_size_stride(primals_278, (1152, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (1152, ), (1, ))
    assert_size_stride(primals_281, (1152, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (192, ), (1, ))
    assert_size_stride(primals_284, (192, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_287, (1152, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_290, (1152, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (192, ), (1, ))
    assert_size_stride(primals_293, (192, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_296, (1152, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_299, (1152, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_302, (192, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_305, (1152, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_308, (1152, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (320, ), (1, ))
    assert_size_stride(primals_311, (320, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (1280, ), (1, ))
    assert_size_stride(primals_314, (1280, ), (1, ))
    assert_size_stride(primals_315, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_105, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_105
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_315, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_315
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf3 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf4 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 32, 1, 1, 784), (25088, 1, 25088, 25088, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf3, buf4, buf5, buf6, 25088, 128, grid=grid(25088), stream=stream0)
        buf7 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1, 32, 1, 1, 7), (224, 1, 224, 224, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf4, buf5, buf6, buf7, buf8, buf9, 224, 112, grid=grid(224), stream=stream0)
        buf10 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf13 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_160, primals_161, buf10, buf11, buf13, primals_160, primals_161, 32, 7, grid=grid(32), stream=stream0)
        del primals_160
        del primals_161
        buf14 = reinterpret_tensor(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf3, buf10, buf11, primals_1, primals_2, buf14, 3211264, grid=grid(3211264), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf15, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf16 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf15, buf16, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf17 = buf6; del buf6  # reuse
        buf18 = buf5; del buf5  # reuse
        buf19 = buf4; del buf4  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf16, buf17, buf18, buf19, 25088, 128, grid=grid(25088), stream=stream0)
        buf20 = buf9; del buf9  # reuse
        buf21 = buf8; del buf8  # reuse
        buf22 = buf7; del buf7  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf17, buf18, buf19, buf20, buf21, buf22, 224, 112, grid=grid(224), stream=stream0)
        del buf17
        del buf18
        del buf19
        buf23 = buf11; del buf11  # reuse
        buf24 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf26 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf20, buf21, buf22, primals_163, primals_164, buf23, buf24, buf26, primals_163, primals_164, 32, 7, grid=grid(32), stream=stream0)
        del buf20
        del buf21
        del buf22
        del primals_163
        del primals_164
        buf27 = reinterpret_tensor(buf15, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf15  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf16, buf23, buf24, primals_3, primals_4, buf27, 3211264, grid=grid(3211264), stream=stream0)
        del buf24
        del primals_4
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf29 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf28, buf29, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf30 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf32 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf29, buf30, buf31, buf32, 12544, 128, grid=grid(12544), stream=stream0)
        buf33 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf34 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf30, buf31, buf32, buf33, buf34, buf35, 112, 112, grid=grid(112), stream=stream0)
        del buf30
        del buf31
        del buf32
        buf36 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf39 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf33, buf34, buf35, primals_166, primals_167, buf36, buf37, buf39, primals_166, primals_167, 16, 7, grid=grid(16), stream=stream0)
        del buf33
        del buf34
        del buf35
        del primals_166
        del primals_167
        buf40 = reinterpret_tensor(buf28, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf28  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_11.run(buf29, buf36, buf37, primals_5, primals_6, buf40, 1605632, grid=grid(1605632), stream=stream0)
        del buf37
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 48, 112, 112), (602112, 12544, 112, 1))
        buf42 = empty_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf41, buf42, 384, 12544, grid=grid(384, 12544), stream=stream0)
        buf43 = empty_strided((1, 48, 1, 1, 784), (37632, 1, 37632, 37632, 48), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 48, 1, 1, 784), (37632, 1, 37632, 37632, 48), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((1, 48, 1, 1, 784), (37632, 1, 37632, 37632, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf42, buf43, buf44, buf45, 37632, 128, grid=grid(37632), stream=stream0)
        buf46 = empty_strided((1, 48, 1, 1, 7), (336, 1, 336, 336, 48), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 48, 1, 1, 7), (336, 1, 336, 336, 48), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((1, 48, 1, 1, 7), (336, 1, 336, 336, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf43, buf44, buf45, buf46, buf47, buf48, 336, 112, grid=grid(336), stream=stream0)
        del buf43
        del buf44
        del buf45
        buf49 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf52 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf46, buf47, buf48, primals_169, primals_170, buf49, buf50, buf52, primals_169, primals_170, 48, 7, grid=grid(48), stream=stream0)
        del buf46
        del buf47
        del buf48
        del primals_169
        del primals_170
        buf53 = reinterpret_tensor(buf41, (8, 48, 112, 112), (602112, 1, 5376, 48), 0); del buf41  # reuse
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf42, buf49, buf50, primals_7, primals_8, buf53, 4816896, grid=grid(4816896), stream=stream0)
        del primals_8
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_109, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf54, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf55 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf54, buf55, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf56 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf55, buf56, buf57, buf58, 9408, 128, grid=grid(9408), stream=stream0)
        buf59 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf56, buf57, buf58, buf59, buf60, buf61, 96, 98, grid=grid(96), stream=stream0)
        del buf56
        del buf57
        del buf58
        buf62 = buf50; del buf50  # reuse
        buf63 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf65 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf59, buf60, buf61, primals_172, primals_173, buf62, buf63, buf65, primals_172, primals_173, 48, 2, grid=grid(48), stream=stream0)
        del primals_172
        del primals_173
        buf66 = reinterpret_tensor(buf54, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf54  # reuse
        # Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_21.run(buf55, buf62, buf63, primals_9, primals_10, buf66, 1204224, grid=grid(1204224), stream=stream0)
        del primals_10
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf68 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf67, buf68, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf69 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf68, buf69, buf70, buf71, 4704, 128, grid=grid(4704), stream=stream0)
        buf72 = reinterpret_tensor(buf63, (1, 24, 1, 1, 2), (48, 1, 48, 48, 24), 0); del buf63  # reuse
        buf73 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf69, buf70, buf71, buf72, buf73, buf74, 48, 98, grid=grid(48), stream=stream0)
        buf75 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf78 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf72, buf73, buf74, primals_175, primals_176, buf75, buf76, buf78, primals_175, primals_176, 24, 2, grid=grid(24), stream=stream0)
        del primals_175
        del primals_176
        buf79 = reinterpret_tensor(buf67, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf67  # reuse
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_26.run(buf68, buf75, buf76, primals_11, primals_12, buf79, 602112, grid=grid(602112), stream=stream0)
        del primals_12
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf81 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf80, buf81, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf82 = empty_strided((1, 72, 1, 1, 196), (14112, 1, 14112, 14112, 72), device='cuda', dtype=torch.float32)
        buf83 = empty_strided((1, 72, 1, 1, 196), (14112, 1, 14112, 14112, 72), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((1, 72, 1, 1, 196), (14112, 1, 14112, 14112, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf81, buf82, buf83, buf84, 14112, 128, grid=grid(14112), stream=stream0)
        buf85 = empty_strided((1, 72, 1, 1, 2), (144, 1, 144, 144, 72), device='cuda', dtype=torch.float32)
        buf86 = empty_strided((1, 72, 1, 1, 2), (144, 1, 144, 144, 72), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((1, 72, 1, 1, 2), (144, 1, 144, 144, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf82, buf83, buf84, buf85, buf86, buf87, 144, 98, grid=grid(144), stream=stream0)
        buf88 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf89 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf91 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf85, buf86, buf87, primals_178, primals_179, buf88, buf89, buf91, primals_178, primals_179, 72, 2, grid=grid(72), stream=stream0)
        del primals_178
        del primals_179
        buf92 = reinterpret_tensor(buf80, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf80  # reuse
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf81, buf88, buf89, primals_13, primals_14, buf92, 1806336, grid=grid(1806336), stream=stream0)
        del primals_14
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf93, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf94 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf93, buf94, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf95 = buf84; del buf84  # reuse
        buf96 = buf83; del buf83  # reuse
        buf97 = buf82; del buf82  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf94, buf95, buf96, buf97, 14112, 128, grid=grid(14112), stream=stream0)
        buf98 = buf87; del buf87  # reuse
        buf99 = buf86; del buf86  # reuse
        buf100 = buf85; del buf85  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf95, buf96, buf97, buf98, buf99, buf100, 144, 98, grid=grid(144), stream=stream0)
        buf101 = buf89; del buf89  # reuse
        buf102 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf104 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf98, buf99, buf100, primals_181, primals_182, buf101, buf102, buf104, primals_181, primals_182, 72, 2, grid=grid(72), stream=stream0)
        del primals_181
        del primals_182
        buf105 = reinterpret_tensor(buf93, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf93  # reuse
        # Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf94, buf101, buf102, primals_15, primals_16, buf105, 1806336, grid=grid(1806336), stream=stream0)
        del primals_16
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf107 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf106, buf107, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf108 = buf71; del buf71  # reuse
        buf109 = buf70; del buf70  # reuse
        buf110 = buf69; del buf69  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf107, buf108, buf109, buf110, 4704, 128, grid=grid(4704), stream=stream0)
        buf111 = buf74; del buf74  # reuse
        buf112 = buf73; del buf73  # reuse
        buf113 = buf72; del buf72  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf108, buf109, buf110, buf111, buf112, buf113, 48, 98, grid=grid(48), stream=stream0)
        buf114 = buf76; del buf76  # reuse
        buf115 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf117 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf111, buf112, buf113, primals_184, primals_185, buf114, buf115, buf117, primals_184, primals_185, 24, 2, grid=grid(24), stream=stream0)
        del primals_184
        del primals_185
        buf118 = reinterpret_tensor(buf106, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf106  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_32.run(buf107, buf114, buf115, primals_17, primals_18, buf79, buf118, 602112, grid=grid(602112), stream=stream0)
        del primals_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf120 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf119, buf120, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf121 = buf97; del buf97  # reuse
        buf122 = buf96; del buf96  # reuse
        buf123 = buf95; del buf95  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf120, buf121, buf122, buf123, 14112, 128, grid=grid(14112), stream=stream0)
        buf124 = buf99; del buf99  # reuse
        buf125 = buf98; del buf98  # reuse
        buf126 = buf100; del buf100  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf121, buf122, buf123, buf124, buf125, buf126, 144, 98, grid=grid(144), stream=stream0)
        buf127 = buf102; del buf102  # reuse
        buf128 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf130 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf124, buf125, buf126, primals_187, primals_188, buf127, buf128, buf130, primals_187, primals_188, 72, 2, grid=grid(72), stream=stream0)
        del primals_187
        del primals_188
        buf131 = reinterpret_tensor(buf119, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf119  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf120, buf127, buf128, primals_19, primals_20, buf131, 1806336, grid=grid(1806336), stream=stream0)
        del primals_20
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf132, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf133 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf132, buf133, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf134 = buf123; del buf123  # reuse
        buf135 = buf122; del buf122  # reuse
        buf136 = buf121; del buf121  # reuse
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf133, buf134, buf135, buf136, 14112, 128, grid=grid(14112), stream=stream0)
        buf137 = buf126; del buf126  # reuse
        buf138 = buf125; del buf125  # reuse
        buf139 = buf124; del buf124  # reuse
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf134, buf135, buf136, buf137, buf138, buf139, 144, 98, grid=grid(144), stream=stream0)
        buf140 = buf128; del buf128  # reuse
        buf141 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf143 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf137, buf138, buf139, primals_190, primals_191, buf140, buf141, buf143, primals_190, primals_191, 72, 2, grid=grid(72), stream=stream0)
        del primals_190
        del primals_191
        buf144 = reinterpret_tensor(buf132, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf132  # reuse
        # Source Nodes: [x_55, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf133, buf140, buf141, primals_21, primals_22, buf144, 1806336, grid=grid(1806336), stream=stream0)
        del primals_22
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf146 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf145, buf146, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf147 = buf110; del buf110  # reuse
        buf148 = buf109; del buf109  # reuse
        buf149 = buf108; del buf108  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf146, buf147, buf148, buf149, 4704, 128, grid=grid(4704), stream=stream0)
        buf150 = buf113; del buf113  # reuse
        buf151 = buf112; del buf112  # reuse
        buf152 = buf111; del buf111  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf147, buf148, buf149, buf150, buf151, buf152, 48, 98, grid=grid(48), stream=stream0)
        del buf147
        del buf148
        del buf149
        buf153 = buf115; del buf115  # reuse
        buf154 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf156 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf150, buf151, buf152, primals_193, primals_194, buf153, buf154, buf156, primals_193, primals_194, 24, 2, grid=grid(24), stream=stream0)
        del buf150
        del buf151
        del buf152
        del primals_193
        del primals_194
        buf157 = reinterpret_tensor(buf145, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf145  # reuse
        # Source Nodes: [shortcut_4, x_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_32.run(buf146, buf153, buf154, primals_23, primals_24, buf118, buf157, 602112, grid=grid(602112), stream=stream0)
        del buf154
        del primals_24
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf159 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf158, buf159, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf160 = buf136; del buf136  # reuse
        buf161 = buf135; del buf135  # reuse
        buf162 = buf134; del buf134  # reuse
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf159, buf160, buf161, buf162, 14112, 128, grid=grid(14112), stream=stream0)
        buf163 = buf139; del buf139  # reuse
        buf164 = buf138; del buf138  # reuse
        buf165 = buf137; del buf137  # reuse
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf160, buf161, buf162, buf163, buf164, buf165, 144, 98, grid=grid(144), stream=stream0)
        del buf160
        del buf161
        del buf162
        buf166 = buf141; del buf141  # reuse
        buf167 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf169 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf163, buf164, buf165, primals_196, primals_197, buf166, buf167, buf169, primals_196, primals_197, 72, 2, grid=grid(72), stream=stream0)
        del buf163
        del buf164
        del buf165
        del primals_196
        del primals_197
        buf170 = reinterpret_tensor(buf158, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf158  # reuse
        # Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_31.run(buf159, buf166, buf167, primals_25, primals_26, buf170, 1806336, grid=grid(1806336), stream=stream0)
        del primals_26
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_118, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf171, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf172 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf171, buf172, 576, 784, grid=grid(576, 784), stream=stream0)
        buf173 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        buf175 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf172, buf173, buf174, buf175, 3528, 128, grid=grid(3528), stream=stream0)
        buf176 = buf167; del buf167  # reuse
        buf177 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf179 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf173, buf174, buf175, primals_199, primals_200, buf176, buf177, buf179, primals_199, primals_200, 72, 49, grid=grid(72), stream=stream0)
        del buf173
        del buf174
        del buf175
        del primals_199
        del primals_200
        buf180 = reinterpret_tensor(buf171, (8, 72, 28, 28), (56448, 1, 2016, 72), 0); del buf171  # reuse
        # Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_36.run(buf172, buf176, buf177, primals_27, primals_28, buf180, 451584, grid=grid(451584), stream=stream0)
        del buf177
        del primals_28
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf182 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf181, buf182, 320, 784, grid=grid(320, 784), stream=stream0)
        buf183 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf184 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf185 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf182, buf183, buf184, buf185, 1960, 128, grid=grid(1960), stream=stream0)
        buf186 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf187 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf189 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf183, buf184, buf185, primals_202, primals_203, buf186, buf187, buf189, primals_202, primals_203, 40, 49, grid=grid(40), stream=stream0)
        del primals_202
        del primals_203
        buf190 = reinterpret_tensor(buf181, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf181  # reuse
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_40.run(buf182, buf186, buf187, primals_29, primals_30, buf190, 250880, grid=grid(250880), stream=stream0)
        del primals_30
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf192 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf191, buf192, 960, 784, grid=grid(960, 784), stream=stream0)
        buf193 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf192, buf193, buf194, buf195, 5880, 128, grid=grid(5880), stream=stream0)
        buf196 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf197 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf199 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf193, buf194, buf195, primals_205, primals_206, buf196, buf197, buf199, primals_205, primals_206, 120, 49, grid=grid(120), stream=stream0)
        del primals_205
        del primals_206
        buf200 = reinterpret_tensor(buf191, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf191  # reuse
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf192, buf196, buf197, primals_31, primals_32, buf200, 752640, grid=grid(752640), stream=stream0)
        del primals_32
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_121, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf201, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf202 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf201, buf202, 960, 784, grid=grid(960, 784), stream=stream0)
        buf203 = buf195; del buf195  # reuse
        buf204 = buf194; del buf194  # reuse
        buf205 = buf193; del buf193  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf202, buf203, buf204, buf205, 5880, 128, grid=grid(5880), stream=stream0)
        buf206 = buf197; del buf197  # reuse
        buf207 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf209 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf203, buf204, buf205, primals_208, primals_209, buf206, buf207, buf209, primals_208, primals_209, 120, 49, grid=grid(120), stream=stream0)
        del primals_208
        del primals_209
        buf210 = reinterpret_tensor(buf201, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf201  # reuse
        # Source Nodes: [x_88, x_91], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf202, buf206, buf207, primals_33, primals_34, buf210, 752640, grid=grid(752640), stream=stream0)
        del primals_34
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf212 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf211, buf212, 320, 784, grid=grid(320, 784), stream=stream0)
        buf213 = buf185; del buf185  # reuse
        buf214 = buf184; del buf184  # reuse
        buf215 = buf183; del buf183  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf212, buf213, buf214, buf215, 1960, 128, grid=grid(1960), stream=stream0)
        buf216 = buf187; del buf187  # reuse
        buf217 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf219 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf213, buf214, buf215, primals_211, primals_212, buf216, buf217, buf219, primals_211, primals_212, 40, 49, grid=grid(40), stream=stream0)
        del primals_211
        del primals_212
        buf220 = reinterpret_tensor(buf211, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf211  # reuse
        # Source Nodes: [shortcut_6, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_45.run(buf212, buf216, buf217, primals_35, primals_36, buf190, buf220, 250880, grid=grid(250880), stream=stream0)
        del primals_36
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf222 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf221, buf222, 960, 784, grid=grid(960, 784), stream=stream0)
        buf223 = buf205; del buf205  # reuse
        buf224 = buf204; del buf204  # reuse
        buf225 = buf203; del buf203  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf222, buf223, buf224, buf225, 5880, 128, grid=grid(5880), stream=stream0)
        buf226 = buf207; del buf207  # reuse
        buf227 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf229 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf223, buf224, buf225, primals_214, primals_215, buf226, buf227, buf229, primals_214, primals_215, 120, 49, grid=grid(120), stream=stream0)
        del primals_214
        del primals_215
        buf230 = reinterpret_tensor(buf221, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf221  # reuse
        # Source Nodes: [x_100, x_103], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf222, buf226, buf227, primals_37, primals_38, buf230, 752640, grid=grid(752640), stream=stream0)
        del primals_38
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_124, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf231, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf232 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf231, buf232, 960, 784, grid=grid(960, 784), stream=stream0)
        buf233 = buf225; del buf225  # reuse
        buf234 = buf224; del buf224  # reuse
        buf235 = buf223; del buf223  # reuse
        # Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf232, buf233, buf234, buf235, 5880, 128, grid=grid(5880), stream=stream0)
        buf236 = buf227; del buf227  # reuse
        buf237 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf239 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf233, buf234, buf235, primals_217, primals_218, buf236, buf237, buf239, primals_217, primals_218, 120, 49, grid=grid(120), stream=stream0)
        del buf233
        del buf234
        del buf235
        del primals_217
        del primals_218
        buf240 = reinterpret_tensor(buf231, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf231  # reuse
        # Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf232, buf236, buf237, primals_39, primals_40, buf240, 752640, grid=grid(752640), stream=stream0)
        del buf237
        del primals_40
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf242 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf241, buf242, 320, 784, grid=grid(320, 784), stream=stream0)
        buf243 = buf215; del buf215  # reuse
        buf244 = buf214; del buf214  # reuse
        buf245 = buf213; del buf213  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf242, buf243, buf244, buf245, 1960, 128, grid=grid(1960), stream=stream0)
        buf246 = buf217; del buf217  # reuse
        buf247 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf249 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf243, buf244, buf245, primals_220, primals_221, buf246, buf247, buf249, primals_220, primals_221, 40, 49, grid=grid(40), stream=stream0)
        del buf243
        del buf244
        del buf245
        del primals_220
        del primals_221
        buf250 = reinterpret_tensor(buf241, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf241  # reuse
        # Source Nodes: [shortcut_7, x_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_45.run(buf242, buf246, buf247, primals_41, primals_42, buf220, buf250, 250880, grid=grid(250880), stream=stream0)
        del buf247
        del primals_42
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 240, 28, 28), (188160, 784, 28, 1))
        buf252 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_46.run(buf251, buf252, 1920, 784, grid=grid(1920, 784), stream=stream0)
        buf253 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf254 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf255 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf252, buf253, buf254, buf255, 11760, 128, grid=grid(11760), stream=stream0)
        buf256 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf257 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf259 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_48.run(buf253, buf254, buf255, primals_223, primals_224, buf256, buf257, buf259, primals_223, primals_224, 240, 49, grid=grid(240), stream=stream0)
        del buf253
        del buf254
        del buf255
        del primals_223
        del primals_224
        buf260 = reinterpret_tensor(buf251, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf251  # reuse
        # Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_49.run(buf252, buf256, buf257, primals_43, primals_44, buf260, 1505280, grid=grid(1505280), stream=stream0)
        del primals_44
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_127, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf261, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf262 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf261, buf262, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf263 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf262, buf263, buf264, buf265, 3120, 121, grid=grid(3120), stream=stream0)
        buf266 = buf257; del buf257  # reuse
        buf267 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf269 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf263, buf264, buf265, primals_226, primals_227, buf266, buf267, buf269, primals_226, primals_227, 240, 13, grid=grid(240), stream=stream0)
        del buf263
        del buf264
        del buf265
        del primals_226
        del primals_227
        buf270 = reinterpret_tensor(buf261, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf261  # reuse
        # Source Nodes: [x_122, x_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf262, buf266, buf267, primals_45, primals_46, buf270, 376320, grid=grid(376320), stream=stream0)
        del buf267
        del primals_46
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf272 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf271, buf272, 640, 196, grid=grid(640, 196), stream=stream0)
        buf273 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf274 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf272, buf273, buf274, buf275, 1040, 121, grid=grid(1040), stream=stream0)
        buf276 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf277 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf279 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf273, buf274, buf275, primals_229, primals_230, buf276, buf277, buf279, primals_229, primals_230, 80, 13, grid=grid(80), stream=stream0)
        del primals_229
        del primals_230
        buf280 = reinterpret_tensor(buf271, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf271  # reuse
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_57.run(buf272, buf276, buf277, primals_47, primals_48, buf280, 125440, grid=grid(125440), stream=stream0)
        del primals_48
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf282 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf281, buf282, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf283 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf285 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf282, buf283, buf284, buf285, 6240, 121, grid=grid(6240), stream=stream0)
        buf286 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf287 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf289 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf283, buf284, buf285, primals_232, primals_233, buf286, buf287, buf289, primals_232, primals_233, 480, 13, grid=grid(480), stream=stream0)
        del primals_232
        del primals_233
        buf290 = reinterpret_tensor(buf281, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf281  # reuse
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf282, buf286, buf287, primals_49, primals_50, buf290, 752640, grid=grid(752640), stream=stream0)
        del primals_50
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_130, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf291, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf292 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf291, buf292, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf293 = buf285; del buf285  # reuse
        buf294 = buf284; del buf284  # reuse
        buf295 = buf283; del buf283  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf292, buf293, buf294, buf295, 6240, 121, grid=grid(6240), stream=stream0)
        buf296 = buf287; del buf287  # reuse
        buf297 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf299 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf293, buf294, buf295, primals_235, primals_236, buf296, buf297, buf299, primals_235, primals_236, 480, 13, grid=grid(480), stream=stream0)
        del primals_235
        del primals_236
        buf300 = reinterpret_tensor(buf291, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf291  # reuse
        # Source Nodes: [x_138, x_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf292, buf296, buf297, primals_51, primals_52, buf300, 752640, grid=grid(752640), stream=stream0)
        del primals_52
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf302 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf301, buf302, 640, 196, grid=grid(640, 196), stream=stream0)
        buf303 = buf275; del buf275  # reuse
        buf304 = buf274; del buf274  # reuse
        buf305 = buf273; del buf273  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf302, buf303, buf304, buf305, 1040, 121, grid=grid(1040), stream=stream0)
        buf306 = buf277; del buf277  # reuse
        buf307 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf309 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf303, buf304, buf305, primals_238, primals_239, buf306, buf307, buf309, primals_238, primals_239, 80, 13, grid=grid(80), stream=stream0)
        del primals_238
        del primals_239
        buf310 = reinterpret_tensor(buf301, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf301  # reuse
        # Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_62.run(buf302, buf306, buf307, primals_53, primals_54, buf280, buf310, 125440, grid=grid(125440), stream=stream0)
        del primals_54
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf312 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf311, buf312, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf313 = buf295; del buf295  # reuse
        buf314 = buf294; del buf294  # reuse
        buf315 = buf293; del buf293  # reuse
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf312, buf313, buf314, buf315, 6240, 121, grid=grid(6240), stream=stream0)
        buf316 = buf297; del buf297  # reuse
        buf317 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf319 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf313, buf314, buf315, primals_241, primals_242, buf316, buf317, buf319, primals_241, primals_242, 480, 13, grid=grid(480), stream=stream0)
        del primals_241
        del primals_242
        buf320 = reinterpret_tensor(buf311, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf311  # reuse
        # Source Nodes: [x_150, x_153], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf312, buf316, buf317, primals_55, primals_56, buf320, 752640, grid=grid(752640), stream=stream0)
        del primals_56
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, primals_133, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf321, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf322 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf321, buf322, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf323 = buf315; del buf315  # reuse
        buf324 = buf314; del buf314  # reuse
        buf325 = buf313; del buf313  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf322, buf323, buf324, buf325, 6240, 121, grid=grid(6240), stream=stream0)
        buf326 = buf317; del buf317  # reuse
        buf327 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf329 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf323, buf324, buf325, primals_244, primals_245, buf326, buf327, buf329, primals_244, primals_245, 480, 13, grid=grid(480), stream=stream0)
        del primals_244
        del primals_245
        buf330 = reinterpret_tensor(buf321, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf321  # reuse
        # Source Nodes: [x_155, x_158], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf322, buf326, buf327, primals_57, primals_58, buf330, 752640, grid=grid(752640), stream=stream0)
        del primals_58
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf332 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf331, buf332, 640, 196, grid=grid(640, 196), stream=stream0)
        buf333 = buf305; del buf305  # reuse
        buf334 = buf304; del buf304  # reuse
        buf335 = buf303; del buf303  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf332, buf333, buf334, buf335, 1040, 121, grid=grid(1040), stream=stream0)
        buf336 = buf307; del buf307  # reuse
        buf337 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf339 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf333, buf334, buf335, primals_247, primals_248, buf336, buf337, buf339, primals_247, primals_248, 80, 13, grid=grid(80), stream=stream0)
        del buf333
        del buf334
        del buf335
        del primals_247
        del primals_248
        buf340 = reinterpret_tensor(buf331, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf331  # reuse
        # Source Nodes: [shortcut_10, x_161], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_62.run(buf332, buf336, buf337, primals_59, primals_60, buf310, buf340, 125440, grid=grid(125440), stream=stream0)
        del buf337
        del primals_60
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf342 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf341, buf342, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf343 = buf325; del buf325  # reuse
        buf344 = buf324; del buf324  # reuse
        buf345 = buf323; del buf323  # reuse
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf342, buf343, buf344, buf345, 6240, 121, grid=grid(6240), stream=stream0)
        buf346 = buf327; del buf327  # reuse
        buf347 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf349 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf343, buf344, buf345, primals_250, primals_251, buf346, buf347, buf349, primals_250, primals_251, 480, 13, grid=grid(480), stream=stream0)
        del primals_250
        del primals_251
        buf350 = reinterpret_tensor(buf341, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf341  # reuse
        # Source Nodes: [x_167, x_170], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf342, buf346, buf347, primals_61, primals_62, buf350, 752640, grid=grid(752640), stream=stream0)
        del primals_62
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf351, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf352 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_58.run(buf351, buf352, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf353 = buf345; del buf345  # reuse
        buf354 = buf344; del buf344  # reuse
        buf355 = buf343; del buf343  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf352, buf353, buf354, buf355, 6240, 121, grid=grid(6240), stream=stream0)
        buf356 = buf347; del buf347  # reuse
        buf357 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf359 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_60.run(buf353, buf354, buf355, primals_253, primals_254, buf356, buf357, buf359, primals_253, primals_254, 480, 13, grid=grid(480), stream=stream0)
        del buf353
        del buf354
        del buf355
        del primals_253
        del primals_254
        buf360 = reinterpret_tensor(buf351, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf351  # reuse
        # Source Nodes: [x_172, x_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf352, buf356, buf357, primals_63, primals_64, buf360, 752640, grid=grid(752640), stream=stream0)
        del buf357
        del primals_64
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 96, 14, 14), (18816, 196, 14, 1))
        buf362 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf361, buf362, 768, 196, grid=grid(768, 196), stream=stream0)
        buf363 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        buf364 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        buf365 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf362, buf363, buf364, buf365, 1248, 121, grid=grid(1248), stream=stream0)
        buf366 = reinterpret_tensor(buf61, (1, 96, 1, 1), (96, 1, 96, 96), 0); del buf61  # reuse
        buf367 = reinterpret_tensor(buf60, (1, 96, 1, 1), (96, 1, 96, 96), 0); del buf60  # reuse
        buf369 = reinterpret_tensor(buf59, (96, ), (1, ), 0); del buf59  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf363, buf364, buf365, primals_256, primals_257, buf366, buf367, buf369, primals_256, primals_257, 96, 13, grid=grid(96), stream=stream0)
        del primals_256
        del primals_257
        buf370 = reinterpret_tensor(buf361, (8, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf361  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_66.run(buf362, buf366, buf367, primals_65, primals_66, buf370, 150528, grid=grid(150528), stream=stream0)
        del primals_66
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf372 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf371, buf372, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf373 = empty_strided((1, 576, 1, 1, 13), (7488, 1, 7488, 7488, 576), device='cuda', dtype=torch.float32)
        buf374 = empty_strided((1, 576, 1, 1, 13), (7488, 1, 7488, 7488, 576), device='cuda', dtype=torch.float32)
        buf375 = empty_strided((1, 576, 1, 1, 13), (7488, 1, 7488, 7488, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf372, buf373, buf374, buf375, 7488, 121, grid=grid(7488), stream=stream0)
        buf376 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf377 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf379 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf373, buf374, buf375, primals_259, primals_260, buf376, buf377, buf379, primals_259, primals_260, 576, 13, grid=grid(576), stream=stream0)
        del primals_259
        del primals_260
        buf380 = reinterpret_tensor(buf371, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf371  # reuse
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_70.run(buf372, buf376, buf377, primals_67, primals_68, buf380, 903168, grid=grid(903168), stream=stream0)
        del primals_68
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf381, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf382 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf381, buf382, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf383 = buf375; del buf375  # reuse
        buf384 = buf374; del buf374  # reuse
        buf385 = buf373; del buf373  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf382, buf383, buf384, buf385, 7488, 121, grid=grid(7488), stream=stream0)
        buf386 = buf377; del buf377  # reuse
        buf387 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf389 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf383, buf384, buf385, primals_262, primals_263, buf386, buf387, buf389, primals_262, primals_263, 576, 13, grid=grid(576), stream=stream0)
        del primals_262
        del primals_263
        buf390 = reinterpret_tensor(buf381, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf381  # reuse
        # Source Nodes: [x_188, x_191], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_70.run(buf382, buf386, buf387, primals_69, primals_70, buf390, 903168, grid=grid(903168), stream=stream0)
        del primals_70
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (8, 96, 14, 14), (18816, 196, 14, 1))
        buf392 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf391, buf392, 768, 196, grid=grid(768, 196), stream=stream0)
        buf393 = buf365; del buf365  # reuse
        buf394 = buf364; del buf364  # reuse
        buf395 = buf363; del buf363  # reuse
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf392, buf393, buf394, buf395, 1248, 121, grid=grid(1248), stream=stream0)
        buf396 = buf367; del buf367  # reuse
        buf397 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf399 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf393, buf394, buf395, primals_265, primals_266, buf396, buf397, buf399, primals_265, primals_266, 96, 13, grid=grid(96), stream=stream0)
        del buf393
        del buf394
        del buf395
        del primals_265
        del primals_266
        buf400 = reinterpret_tensor(buf391, (8, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf391  # reuse
        # Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_71.run(buf392, buf396, buf397, primals_71, primals_72, buf370, buf400, 150528, grid=grid(150528), stream=stream0)
        del buf397
        del primals_72
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf402 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf401, buf402, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf403 = buf385; del buf385  # reuse
        buf404 = buf384; del buf384  # reuse
        buf405 = buf383; del buf383  # reuse
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf402, buf403, buf404, buf405, 7488, 121, grid=grid(7488), stream=stream0)
        buf406 = buf387; del buf387  # reuse
        buf407 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf409 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_69.run(buf403, buf404, buf405, primals_268, primals_269, buf406, buf407, buf409, primals_268, primals_269, 576, 13, grid=grid(576), stream=stream0)
        del buf403
        del buf404
        del buf405
        del primals_268
        del primals_269
        buf410 = reinterpret_tensor(buf401, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf401  # reuse
        # Source Nodes: [x_200, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_70.run(buf402, buf406, buf407, primals_73, primals_74, buf410, 903168, grid=grid(903168), stream=stream0)
        del primals_74
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_142, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf411, (8, 576, 7, 7), (28224, 49, 7, 1))
        buf412 = empty_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf411, buf412, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf413 = empty_strided((1, 576, 1, 1, 4), (2304, 1, 2304, 2304, 576), device='cuda', dtype=torch.float32)
        buf414 = empty_strided((1, 576, 1, 1, 4), (2304, 1, 2304, 2304, 576), device='cuda', dtype=torch.float32)
        buf415 = empty_strided((1, 576, 1, 1, 4), (2304, 1, 2304, 2304, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf412, buf413, buf414, buf415, 2304, 98, grid=grid(2304), stream=stream0)
        buf416 = buf407; del buf407  # reuse
        buf417 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf419 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf413, buf414, buf415, primals_271, primals_272, buf416, buf417, buf419, primals_271, primals_272, 576, 4, grid=grid(576), stream=stream0)
        del buf413
        del buf414
        del buf415
        del primals_271
        del primals_272
        buf420 = reinterpret_tensor(buf411, (8, 576, 7, 7), (28224, 1, 4032, 576), 0); del buf411  # reuse
        # Source Nodes: [x_205, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_75.run(buf412, buf416, buf417, primals_75, primals_76, buf420, 225792, grid=grid(225792), stream=stream0)
        del buf417
        del primals_76
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf422 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf421, buf422, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf423 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf424 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf425 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf422, buf423, buf424, buf425, 768, 98, grid=grid(768), stream=stream0)
        buf426 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf427 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf429 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf423, buf424, buf425, primals_274, primals_275, buf426, buf427, buf429, primals_274, primals_275, 192, 4, grid=grid(192), stream=stream0)
        del primals_274
        del primals_275
        buf430 = reinterpret_tensor(buf421, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf421  # reuse
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_79.run(buf422, buf426, buf427, primals_77, primals_78, buf430, 75264, grid=grid(75264), stream=stream0)
        del primals_78
        # Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf432 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf431, buf432, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf433 = empty_strided((1, 1152, 1, 1, 4), (4608, 1, 4608, 4608, 1152), device='cuda', dtype=torch.float32)
        buf434 = empty_strided((1, 1152, 1, 1, 4), (4608, 1, 4608, 4608, 1152), device='cuda', dtype=torch.float32)
        buf435 = empty_strided((1, 1152, 1, 1, 4), (4608, 1, 4608, 4608, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf432, buf433, buf434, buf435, 4608, 98, grid=grid(4608), stream=stream0)
        buf436 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf437 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf439 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf433, buf434, buf435, primals_277, primals_278, buf436, buf437, buf439, primals_277, primals_278, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_277
        del primals_278
        buf440 = reinterpret_tensor(buf431, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf431  # reuse
        # Source Nodes: [x_216, x_219], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf432, buf436, buf437, primals_79, primals_80, buf440, 451584, grid=grid(451584), stream=stream0)
        del primals_80
        # Source Nodes: [x_220], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_145, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf441, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf442 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf441, buf442, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf443 = buf435; del buf435  # reuse
        buf444 = buf434; del buf434  # reuse
        buf445 = buf433; del buf433  # reuse
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf442, buf443, buf444, buf445, 4608, 98, grid=grid(4608), stream=stream0)
        buf446 = buf437; del buf437  # reuse
        buf447 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf449 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf443, buf444, buf445, primals_280, primals_281, buf446, buf447, buf449, primals_280, primals_281, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_280
        del primals_281
        buf450 = reinterpret_tensor(buf441, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf441  # reuse
        # Source Nodes: [x_221, x_224], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf442, buf446, buf447, primals_81, primals_82, buf450, 451584, grid=grid(451584), stream=stream0)
        del primals_82
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf452 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf451, buf452, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf453 = buf425; del buf425  # reuse
        buf454 = buf424; del buf424  # reuse
        buf455 = buf423; del buf423  # reuse
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf452, buf453, buf454, buf455, 768, 98, grid=grid(768), stream=stream0)
        buf456 = buf427; del buf427  # reuse
        buf457 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf459 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf453, buf454, buf455, primals_283, primals_284, buf456, buf457, buf459, primals_283, primals_284, 192, 4, grid=grid(192), stream=stream0)
        del primals_283
        del primals_284
        buf460 = reinterpret_tensor(buf451, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf451  # reuse
        # Source Nodes: [shortcut_14, x_227], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_84.run(buf452, buf456, buf457, primals_83, primals_84, buf430, buf460, 75264, grid=grid(75264), stream=stream0)
        del primals_84
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf462 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf461, buf462, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf463 = buf445; del buf445  # reuse
        buf464 = buf444; del buf444  # reuse
        buf465 = buf443; del buf443  # reuse
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf462, buf463, buf464, buf465, 4608, 98, grid=grid(4608), stream=stream0)
        buf466 = buf447; del buf447  # reuse
        buf467 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf469 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf463, buf464, buf465, primals_286, primals_287, buf466, buf467, buf469, primals_286, primals_287, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_286
        del primals_287
        buf470 = reinterpret_tensor(buf461, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf461  # reuse
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf462, buf466, buf467, primals_85, primals_86, buf470, 451584, grid=grid(451584), stream=stream0)
        del primals_86
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, primals_148, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf471, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf472 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf471, buf472, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf473 = buf465; del buf465  # reuse
        buf474 = buf464; del buf464  # reuse
        buf475 = buf463; del buf463  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf472, buf473, buf474, buf475, 4608, 98, grid=grid(4608), stream=stream0)
        buf476 = buf467; del buf467  # reuse
        buf477 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf479 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf473, buf474, buf475, primals_289, primals_290, buf476, buf477, buf479, primals_289, primals_290, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_289
        del primals_290
        buf480 = reinterpret_tensor(buf471, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf471  # reuse
        # Source Nodes: [x_238, x_241], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf472, buf476, buf477, primals_87, primals_88, buf480, 451584, grid=grid(451584), stream=stream0)
        del primals_88
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf482 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf481, buf482, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf483 = buf455; del buf455  # reuse
        buf484 = buf454; del buf454  # reuse
        buf485 = buf453; del buf453  # reuse
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf482, buf483, buf484, buf485, 768, 98, grid=grid(768), stream=stream0)
        buf486 = buf457; del buf457  # reuse
        buf487 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf489 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf483, buf484, buf485, primals_292, primals_293, buf486, buf487, buf489, primals_292, primals_293, 192, 4, grid=grid(192), stream=stream0)
        del primals_292
        del primals_293
        buf490 = reinterpret_tensor(buf481, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf481  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_84.run(buf482, buf486, buf487, primals_89, primals_90, buf460, buf490, 75264, grid=grid(75264), stream=stream0)
        del primals_90
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf492 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf491, buf492, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf493 = buf475; del buf475  # reuse
        buf494 = buf474; del buf474  # reuse
        buf495 = buf473; del buf473  # reuse
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf492, buf493, buf494, buf495, 4608, 98, grid=grid(4608), stream=stream0)
        buf496 = buf477; del buf477  # reuse
        buf497 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf499 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf493, buf494, buf495, primals_295, primals_296, buf496, buf497, buf499, primals_295, primals_296, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_295
        del primals_296
        buf500 = reinterpret_tensor(buf491, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf491  # reuse
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf492, buf496, buf497, primals_91, primals_92, buf500, 451584, grid=grid(451584), stream=stream0)
        del primals_92
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_151, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf501, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf502 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf501, buf502, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf503 = buf495; del buf495  # reuse
        buf504 = buf494; del buf494  # reuse
        buf505 = buf493; del buf493  # reuse
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf502, buf503, buf504, buf505, 4608, 98, grid=grid(4608), stream=stream0)
        buf506 = buf497; del buf497  # reuse
        buf507 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf509 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf503, buf504, buf505, primals_298, primals_299, buf506, buf507, buf509, primals_298, primals_299, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_298
        del primals_299
        buf510 = reinterpret_tensor(buf501, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf501  # reuse
        # Source Nodes: [x_255, x_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf502, buf506, buf507, primals_93, primals_94, buf510, 451584, grid=grid(451584), stream=stream0)
        del primals_94
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf510, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf512 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf511, buf512, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf513 = buf485; del buf485  # reuse
        buf514 = buf484; del buf484  # reuse
        buf515 = buf483; del buf483  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf512, buf513, buf514, buf515, 768, 98, grid=grid(768), stream=stream0)
        buf516 = buf487; del buf487  # reuse
        buf517 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf519 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf513, buf514, buf515, primals_301, primals_302, buf516, buf517, buf519, primals_301, primals_302, 192, 4, grid=grid(192), stream=stream0)
        del buf513
        del buf514
        del buf515
        del primals_301
        del primals_302
        buf520 = reinterpret_tensor(buf511, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf511  # reuse
        # Source Nodes: [shortcut_16, x_261], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_84.run(buf512, buf516, buf517, primals_95, primals_96, buf490, buf520, 75264, grid=grid(75264), stream=stream0)
        del buf517
        del primals_96
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf521 = extern_kernels.convolution(buf520, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf521, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf522 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf521, buf522, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf523 = buf505; del buf505  # reuse
        buf524 = buf504; del buf504  # reuse
        buf525 = buf503; del buf503  # reuse
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf522, buf523, buf524, buf525, 4608, 98, grid=grid(4608), stream=stream0)
        buf526 = buf507; del buf507  # reuse
        buf527 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf529 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf523, buf524, buf525, primals_304, primals_305, buf526, buf527, buf529, primals_304, primals_305, 1152, 4, grid=grid(1152), stream=stream0)
        del primals_304
        del primals_305
        buf530 = reinterpret_tensor(buf521, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf521  # reuse
        # Source Nodes: [x_267, x_270], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf522, buf526, buf527, primals_97, primals_98, buf530, 451584, grid=grid(451584), stream=stream0)
        del primals_98
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf531, (8, 1152, 7, 7), (56448, 49, 7, 1))
        buf532 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf531, buf532, 9216, 49, grid=grid(9216, 49), stream=stream0)
        buf533 = buf525; del buf525  # reuse
        buf534 = buf524; del buf524  # reuse
        buf535 = buf523; del buf523  # reuse
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf532, buf533, buf534, buf535, 4608, 98, grid=grid(4608), stream=stream0)
        buf536 = buf527; del buf527  # reuse
        buf537 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf539 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf533, buf534, buf535, primals_307, primals_308, buf536, buf537, buf539, primals_307, primals_308, 1152, 4, grid=grid(1152), stream=stream0)
        del buf533
        del buf534
        del buf535
        del primals_307
        del primals_308
        buf540 = reinterpret_tensor(buf531, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf531  # reuse
        # Source Nodes: [x_272, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_83.run(buf532, buf536, buf537, primals_99, primals_100, buf540, 451584, grid=grid(451584), stream=stream0)
        del buf537
        del primals_100
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf542 = empty_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf541, buf542, 2560, 49, grid=grid(2560, 49), stream=stream0)
        buf543 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf544 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf545 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf542, buf543, buf544, buf545, 1280, 98, grid=grid(1280), stream=stream0)
        buf546 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf547 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf549 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf543, buf544, buf545, primals_310, primals_311, buf546, buf547, buf549, primals_310, primals_311, 320, 4, grid=grid(320), stream=stream0)
        del primals_310
        del primals_311
        buf550 = reinterpret_tensor(buf541, (8, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf541  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_88.run(buf542, buf546, buf547, primals_101, primals_102, buf550, 125440, grid=grid(125440), stream=stream0)
        del buf547
        del primals_102
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (8, 1280, 7, 7), (62720, 49, 7, 1))
        buf552 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf551, buf552, 10240, 49, grid=grid(10240, 49), stream=stream0)
        buf553 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf554 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf555 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf552, buf553, buf554, buf555, 5120, 98, grid=grid(5120), stream=stream0)
        buf556 = reinterpret_tensor(buf545, (1, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf545  # reuse
        buf557 = reinterpret_tensor(buf544, (1, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf544  # reuse
        buf559 = reinterpret_tensor(buf543, (1280, ), (1, ), 0); del buf543  # reuse
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf553, buf554, buf555, primals_313, primals_314, buf556, buf557, buf559, primals_313, primals_314, 1280, 4, grid=grid(1280), stream=stream0)
        del buf553
        del buf554
        del buf555
        del primals_313
        del primals_314
        buf560 = reinterpret_tensor(buf551, (8, 1280, 7, 7), (62720, 1, 8960, 1280), 0); del buf551  # reuse
        buf564 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_284, x_288], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_92.run(buf552, buf556, buf557, primals_103, primals_104, buf560, buf564, 501760, grid=grid(501760), stream=stream0)
        del buf557
        del primals_104
        buf561 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf562 = reinterpret_tensor(buf561, (8, 1280), (1280, 1), 0); del buf561  # reuse
        # Source Nodes: [x_289, x_291], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_93.run(buf562, buf560, 10240, 49, grid=grid(10240), stream=stream0)
        del buf560
        buf563 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf562, reinterpret_tensor(primals_157, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf563)
        del primals_158
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_159, primals_159, 1, grid=grid(1), stream=stream0)
        del primals_159
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_162, primals_162, 1, grid=grid(1), stream=stream0)
        del primals_162
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_165, primals_165, 1, grid=grid(1), stream=stream0)
        del primals_165
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_168, primals_168, 1, grid=grid(1), stream=stream0)
        del primals_168
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_171, primals_171, 1, grid=grid(1), stream=stream0)
        del primals_171
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_174, primals_174, 1, grid=grid(1), stream=stream0)
        del primals_174
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_177, primals_177, 1, grid=grid(1), stream=stream0)
        del primals_177
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_180, primals_180, 1, grid=grid(1), stream=stream0)
        del primals_180
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_183, primals_183, 1, grid=grid(1), stream=stream0)
        del primals_183
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_186, primals_186, 1, grid=grid(1), stream=stream0)
        del primals_186
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_189, primals_189, 1, grid=grid(1), stream=stream0)
        del primals_189
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_192, primals_192, 1, grid=grid(1), stream=stream0)
        del primals_192
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_195, primals_195, 1, grid=grid(1), stream=stream0)
        del primals_195
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_198, primals_198, 1, grid=grid(1), stream=stream0)
        del primals_198
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_201, primals_201, 1, grid=grid(1), stream=stream0)
        del primals_201
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_204, primals_204, 1, grid=grid(1), stream=stream0)
        del primals_204
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_207, primals_207, 1, grid=grid(1), stream=stream0)
        del primals_207
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_210, primals_210, 1, grid=grid(1), stream=stream0)
        del primals_210
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_213, primals_213, 1, grid=grid(1), stream=stream0)
        del primals_213
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_216, primals_216, 1, grid=grid(1), stream=stream0)
        del primals_216
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_219, primals_219, 1, grid=grid(1), stream=stream0)
        del primals_219
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_222, primals_222, 1, grid=grid(1), stream=stream0)
        del primals_222
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_94.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        return (buf563, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, buf0, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, buf1, buf3, buf13, buf14, buf16, buf26, buf27, buf29, buf39, buf40, buf42, buf52, buf53, buf55, buf65, buf66, buf68, buf78, buf79, buf81, buf91, buf92, buf94, buf104, buf105, buf107, buf117, buf118, buf120, buf130, buf131, buf133, buf143, buf144, buf146, buf156, buf157, buf159, buf169, buf170, buf172, buf179, buf180, buf182, buf189, buf190, buf192, buf199, buf200, buf202, buf209, buf210, buf212, buf219, buf220, buf222, buf229, buf230, buf232, buf239, buf240, buf242, buf249, buf250, buf252, buf259, buf260, buf262, buf269, buf270, buf272, buf279, buf280, buf282, buf289, buf290, buf292, buf299, buf300, buf302, buf309, buf310, buf312, buf319, buf320, buf322, buf329, buf330, buf332, buf339, buf340, buf342, buf349, buf350, buf352, buf359, buf360, buf362, buf369, buf370, buf372, buf379, buf380, buf382, buf389, buf390, buf392, buf399, buf400, buf402, buf409, buf410, buf412, buf419, buf420, buf422, buf429, buf430, buf432, buf439, buf440, buf442, buf449, buf450, buf452, buf459, buf460, buf462, buf469, buf470, buf472, buf479, buf480, buf482, buf489, buf490, buf492, buf499, buf500, buf502, buf509, buf510, buf512, buf519, buf520, buf522, buf529, buf530, buf532, buf539, buf540, buf542, buf549, buf550, buf552, buf559, buf562, reinterpret_tensor(primals_157, (1000, 1280), (1280, 1), 0), buf564, reinterpret_tensor(buf556, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf546, (1, 320, 1, 1), (320, 1, 1, 1), 0), reinterpret_tensor(buf536, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf526, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf516, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf506, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf496, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf486, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf476, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf466, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf456, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf446, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf436, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf426, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf416, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf396, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf386, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf376, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf366, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf356, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf346, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf336, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf326, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf316, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf306, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf296, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf286, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf276, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf266, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf256, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf246, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf236, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf226, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf216, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf206, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf196, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf186, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf176, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf166, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf140, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf127, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf114, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf101, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf88, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf62, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf49, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf36, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf23, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_59 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_101 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_166 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_169 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_172 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_178 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_181 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_184 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_187 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_190 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_193 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_196 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_199 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet_100', benchmark_compiled_module)
