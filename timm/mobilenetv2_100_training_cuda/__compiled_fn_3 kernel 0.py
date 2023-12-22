
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


# kernel path: /tmp/torchinductor_youkaichao/o7/co7smob6jwlmhsrd3tsksn33ns6hzlv6ehhh3lhu745ptvd5qm3g.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# shortcut => clamp_max, clamp_min
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_6', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
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


# kernel path: /tmp/torchinductor_youkaichao/hu/chu3icemn5kgsg5m5ffimkjahp2vbrfsycb7sr77zc7ba7fy4tl2.py
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
    size_hints=[1024, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6b/c6b3qywrlcqx6taytastilz2fmy2wp76qxnmj7woxbjt43hgtx5i.py
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
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5g2fpgdhqieeghuikrpdlyrne4a4kfsnfuwswb6ql36hzx7tbs.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hy/chy5r4ipts5w7cxlxfupgktl56u6k77xfmfntpastkrb2ssjb7as.py
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
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkpp4srpoojxdej5zeuzwns7a3a3nprbl4ymis2nloz67nfw5te.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_17 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_20 => clamp_max_2, clamp_min_2
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cnctqf24chm6f4jd4zhjvmogx65f4zgivbqjmchbvi5cdyiop3xq.py
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
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xt/cxtvvmvugd37ymfiiixrj2gzrdr3dehggvkkmvetq74otk7wzhtt.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmyyp5mnw2vq7cntv7pr4ezsgllesaqec3kkyhkxofrs2l24zu5.py
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
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwylslyvndohk7sqtsnevkj76cwbch7p3nr2kjy5qgrigfjp3eg.py
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
    size_hints=[128, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/j5/cj53esav6zy2tnbh4ytbiwtictoylilh7t2pnvzas5635uax4wec.py
# Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_22 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_25 => clamp_max_3, clamp_min_3
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
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


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6krbghmcvzh6az57pjfzl2fskk2a24jcau6whonoomrbp5rqlb.py
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
    size_hints=[2048, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wxeclmkrztvew3vcknooaqbt3l6ykqifnjs2ocdu7y2kw6ytx4.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckoiavv3iwk7nngkspa7mwq6o2swurmitz7qtwqzxqhd7p5ux2lw.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4u/c4ucucrpankrhlfgb3abpjpceydhjwclwk5uof2z6er4xsurdtbx.py
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
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jldsb2diir6nb4muidd6mh44mwsjl2rpauuk6psknad33j4lx7.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_33 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_36 => clamp_max_4, clamp_min_4
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_31', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
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


# kernel path: /tmp/torchinductor_youkaichao/7n/c7n22hylrsnrwotw4yfieiyxpuloeshqfiawxr75vuihnsroecnk.py
# Source Nodes: [x_54], Original ATen: [aten.convolution]
# x_54 => convolution_10
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zu/czuyrb6h35vsnvvf4plrslztybkomddevungi5scoq3cjeshpe7j.py
# Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
# x_55 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pf/cpf4qwuedu72uwlba4p7jw2u3p5aqldtcmxrtrg7vciupwhvcqri.py
# Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
# x_55 => add_52, add_53, add_54, mul_71, mul_72, mul_73, mul_74, mul_75, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/lq/clqkzuacnszwnepktgeom6racuizwn42wafvctdcksda56cnlrox.py
# Source Nodes: [x_55, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_55 => add_52, add_55, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
# x_58 => clamp_max_7, clamp_min_7
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfv4b4rz3ogxzmjn2kelvss2p3i3joycs676pltrzqh2xnst6od4.py
# Source Nodes: [x_60], Original ATen: [aten.convolution]
# x_60 => convolution_11
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (25088*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyvid3h4je6jbcbxvdrvsjibzkxd7hb2l7tonbdrtfvfrkdlatn.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
# x_61 => var_mean_11
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zovyufxixd2wgmbon7dyjppk7kyrgbimawylpdshdfqgd4ey37.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
# x_61 => add_57, add_58, add_59, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/wo/cwou54u5khkge2aoh5dttcbvnfyszkcejucc4txc3m5iv6tvkqmv.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
# x_61 => add_57, add_60, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
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
    xnumel = 200704
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
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cew6m7offwk3rxd6hfbgymr55v64n4egjn5r4jddd66ztywwsiqx.py
# Source Nodes: [x_65], Original ATen: [aten.convolution]
# x_65 => convolution_12
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cju3zkz3hpojmvett7z65n2nzwrn4duuis6owuhxjw6mdloipl4x.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
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


# kernel path: /tmp/torchinductor_youkaichao/rg/crgu3gjmd7zd6sodyydwfyiidpf4k6gx4wtumilam46lnhwixoyw.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
# x_66 => add_62, add_63, add_64, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6snifwjkusn5lbkxsvakxbiauypp2muknxlqbotnvuiswp4bpq.py
# Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_66 => add_62, add_65, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# x_69 => clamp_max_8, clamp_min_8
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
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
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/crepjj6kwa623g7bkglrpngcxcsac7hrysul7t4n3xbyq6kqmf2j.py
# Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_5 => add_76
# x_77 => add_72, add_75, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
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
    xnumel = 200704
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3lhczdj7yblncpihiqqzbk4wo5xchvbx4faiv3so4fbdcpii5q.py
# Source Nodes: [x_104], Original ATen: [aten.convolution]
# x_104 => convolution_19
triton_poi_fused_convolution_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (37632*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxsyx7kr3ybfupvbvzmfwp64vozxtflnfnqabktdsrxc6pif6y6n.py
# Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
# x_105 => var_mean_19
triton_red_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2496
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
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
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vo/cvoj633lkfychdk4rrgcjmodlpk523nsjdtvw2cw7l6gc7pmmweq.py
# Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
# x_105 => add_100, add_101, add_99, mul_134, mul_135, mul_136, mul_137, mul_138, rsqrt_19, squeeze_58, var_mean_19
triton_per_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/uu/cuuhlqgmgpj6sboxhhrd5i7prl6q25wd3yhmmnkbrd33aog7qayy.py
# Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_105 => add_102, add_99, mul_133, mul_139, rsqrt_19, sub_19, var_mean_19
# x_108 => clamp_max_13, clamp_min_13
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
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
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czk4iudar2t2tiqukxntjwafzolntfdjxhjkxn6lvcgya2ofsxmg.py
# Source Nodes: [x_110], Original ATen: [aten.convolution]
# x_110 => convolution_20
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (12544*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7uqm57ecvjsnqd4ptlohcvnuw3v2ndqc7ho6flsqwkxyw6777h6.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
# x_111 => var_mean_20
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
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
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vy/cvyj6iufn4oftbcxe42dz2cse4464pydvsg6br4bkgiq26rlph5n.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
# x_111 => add_104, add_105, add_106, mul_141, mul_142, mul_143, mul_144, mul_145, rsqrt_20, squeeze_61, var_mean_20
triton_per_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxajqn7sw6ctw3ep3pzax7fq3y5sqalojwoskmxcmm3wvasyji6z.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
# x_111 => add_104, add_107, mul_140, mul_146, rsqrt_20, sub_20, var_mean_20
triton_poi_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
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
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jy/cjydi6roa6z236tch65zcdtbyodc23vcwkgw42ydel65sv4wzsaj.py
# Source Nodes: [x_115], Original ATen: [aten.convolution]
# x_115 => convolution_21
triton_poi_fused_convolution_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (75264*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmgcqxtxdtulsq7rencqrzw5lkbslxm5mtjfss5qr3qhrj3y5gt.py
# Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
# x_116 => var_mean_21
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
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
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5y/c5yejtaofoldmitvnn5ncw3slifnj5k6knzkx23wl5yfox2lqwbt.py
# Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
# x_116 => add_109, add_110, add_111, mul_148, mul_149, mul_150, mul_151, mul_152, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eo/ceoanve5ahxrzsuv3k4vtz6kqeht33sc3oigenjl5zaw4bjoxtxe.py
# Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_116 => add_109, add_112, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# x_119 => clamp_max_14, clamp_min_14
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
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
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/csc62n4hd3rw4blwwa3jbet4ywjaaa5mnkkhqo52s4blvbgpuelt.py
# Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_8 => add_123
# x_127 => add_119, add_122, mul_161, mul_167, rsqrt_23, sub_23, var_mean_23
triton_poi_fused__native_batch_norm_legit_functional_add_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27dd5j7ccyfpe6fvkhn5yxczekyjywem5fk76oum6od6qayl57l.py
# Source Nodes: [x_177], Original ATen: [aten.convolution]
# x_177 => convolution_32
triton_poi_fused_convolution_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_59', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ld/cld3sd4ecu3d2tgeq3odbwhrm7nt6rehbs5mtuymld7efnczfrom.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => var_mean_32
triton_red_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ue/cuez7y3lry2szk5ccnqmte74cv7qimykn6spnyplnvdyrq7ncldy.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => add_167, add_168, add_169, mul_225, mul_226, mul_227, mul_228, mul_229, rsqrt_32, squeeze_97, var_mean_32
triton_per_fused__native_batch_norm_legit_functional_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7puopztwn3him34b62hzqrjh6hj43lbcqowlbz32xbzgbqjvkk.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
# x_178 => add_167, add_170, mul_224, mul_230, rsqrt_32, sub_32, var_mean_32
triton_poi_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfrje7wgyjjpa7z4ken5benzzfapf2qxln3e55oxroy6mx4uu34.py
# Source Nodes: [x_182], Original ATen: [aten.convolution]
# x_182 => convolution_33
triton_poi_fused_convolution_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_63', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ir/ciria3xk64hnfqvf5tvazpzl6nqtjpnzirucjsppfajfmxh373ix.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
# x_183 => var_mean_33
triton_red_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zm/czmxmaxemarbtjhfao6p6ruuj3tcrcleln22xoqxrv3sq443mjeb.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
# x_183 => add_172, add_173, add_174, mul_232, mul_233, mul_234, mul_235, mul_236, rsqrt_33, squeeze_100, var_mean_33
triton_per_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3kpcylbnu5ie74gsvdgh7djlo4z2zkoi264io6tg4u77zzujqr.py
# Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_183 => add_172, add_175, mul_231, mul_237, rsqrt_33, sub_33, var_mean_33
# x_186 => clamp_max_22, clamp_min_22
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcabww4if7w4vypv2to4ayxle6fqtoqzckjadaxyyzuwbiyj7fm.py
# Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_12 => add_186
# x_194 => add_182, add_185, mul_245, mul_251, rsqrt_35, sub_35, var_mean_35
triton_poi_fused__native_batch_norm_legit_functional_add_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_67', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qu/cqub3gg3yz276bnvfzv34arvwugqqdtn644s4lrei225dx5k6x55.py
# Source Nodes: [x_221], Original ATen: [aten.convolution]
# x_221 => convolution_40
triton_poi_fused_convolution_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_68', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sh/cshyyejmzciraziyj7g7hgn5t346lkbjxyfbcgggdwj5x2dvz76c.py
# Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
# x_222 => var_mean_40
triton_red_fused__native_batch_norm_legit_functional_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_69', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ht/chtjn2hjqlqrpolul4rexjfpdvqavaubke3qup74j5uwe5tff6yq.py
# Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
# x_222 => add_209, add_210, add_211, mul_281, mul_282, mul_283, mul_284, mul_285, rsqrt_40, squeeze_121, var_mean_40
triton_per_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_70', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyh2tjce7o4w4wgraorffjkav2o44gvdjk2pxve4k4mzjtfkiwxk.py
# Source Nodes: [x_222, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_222 => add_209, add_212, mul_280, mul_286, rsqrt_40, sub_40, var_mean_40
# x_225 => clamp_max_27, clamp_min_27
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zy3neoqrsh3ivwiavhysg2axfhgqh34x2qzxbx2eoejqps4wc3.py
# Source Nodes: [x_227], Original ATen: [aten.convolution]
# x_227 => convolution_41
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (7840*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citrsixqpzk4dioggshph7wdteynivdt3ng737qjeh2nn3sz454k.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => var_mean_41
triton_red_fused__native_batch_norm_legit_functional_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (15680*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ix/cix4zlevr7b4aeb2xybjoa4c5wzj43xescqcxyihzibk5o6ztmlw.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => add_214, add_215, add_216, mul_288, mul_289, mul_290, mul_291, mul_292, rsqrt_41, squeeze_124, var_mean_41
triton_per_fused__native_batch_norm_legit_functional_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_74', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/um/cum74a6xweaohtj3jb6kh2zu5ar4s46wh42syn5vd7pyir3ws65y.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
# x_228 => add_214, add_217, mul_287, mul_293, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
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


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmkcn6b5pp43dvfum3gztvurgwikze6u3pgp4w2iisfragrmumr.py
# Source Nodes: [x_232], Original ATen: [aten.convolution]
# x_232 => convolution_42
triton_poi_fused_convolution_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfntnlpp2vid7d6jfomoytdrweifu2c4sqo5yidgaan5z5luihfe.py
# Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
# x_233 => var_mean_42
triton_red_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zl/czlc2dninrlianot7w2i53pz4lmxjudmm6ysgkh75xpxzgnhd2qc.py
# Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
# x_233 => add_219, add_220, add_221, mul_295, mul_296, mul_297, mul_298, mul_299, rsqrt_42, squeeze_127, var_mean_42
triton_per_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqzac5rdf56q3xqeg7bekpd57mdnfdhsfnm6lbu5cdnikfo4nrv.py
# Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
# x_233 => add_219, add_222, mul_294, mul_300, rsqrt_42, sub_42, var_mean_42
# x_236 => clamp_max_28, clamp_min_28
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp13 <= tmp14
    tmp19 = tmp13 >= tmp16
    tmp20 = tmp18 | tmp19
    tl.store(out_ptr1 + (x2), tmp17, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsrza4sfzlzkvxfq2wnq5tzgzc6rbinylldxg5kw6rivdcnt6ug.py
# Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_15 => add_233
# x_244 => add_229, add_232, mul_308, mul_314, rsqrt_44, sub_44, var_mean_44
triton_poi_fused__native_batch_norm_legit_functional_add_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
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


# kernel path: /tmp/torchinductor_youkaichao/iu/ciul7k53bwtvc42rrc4hbwtfuoun3br4ld3b5wlan7in4mr3dgca.py
# Source Nodes: [x_277], Original ATen: [aten.convolution]
# x_277 => convolution_50
triton_poi_fused_convolution_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_81', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ye/cyer47ruo7yjq6sfq6ne6nbpodd2zpp2ltzjzt3jsaimy7zgvy7r.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => var_mean_50
triton_red_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsxghgz4wmdjylmginlhm75tumw3c2gvznwsougemrj5yc6ccbj.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => add_261, add_262, add_263, mul_351, mul_352, mul_353, mul_354, mul_355, rsqrt_50, squeeze_151, var_mean_50
triton_per_fused__native_batch_norm_legit_functional_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_83', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ls/clsszmabo47w6wdyiq7ee55dx2t4hrso5a3ilmgltvpyzk5jt73e.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
# x_278 => add_261, add_264, mul_350, mul_356, rsqrt_50, sub_50, var_mean_50
triton_poi_fused__native_batch_norm_legit_functional_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fz/cfz4lkynw4xla2wkinbmdyplqehc6hth5doeojlpjnne6b2v3sdi.py
# Source Nodes: [x_283], Original ATen: [aten.convolution]
# x_283 => convolution_51
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtg555pjv46nbi64b4vw5ywqmb7hgddajcdaujydg2fgrcjvadz.py
# Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
# x_284 => var_mean_51
triton_red_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rxz3sa7tw4k4lapg3yu7aebzkyjw7o3chgzj4fpmprx3wnepwg.py
# Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
# x_284 => add_266, add_267, add_268, mul_358, mul_359, mul_360, mul_361, mul_362, rsqrt_51, squeeze_154, var_mean_51
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2kpfso55yjiojlflrgn2qxszh4m5v4krose7rfhwiafqkqbmzz.py
# Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward]
# x_284 => add_266, add_269, mul_357, mul_363, rsqrt_51, sub_51, var_mean_51
triton_poi_fused__native_batch_norm_legit_functional_hardtanh_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardtanh_backward_88', 'mutated_arg_names': []},
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
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp16 = 6.0
    tmp17 = tmp13 >= tmp16
    tmp18 = tmp15 | tmp17
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfgz2ouuhvh37ojxhrpaapt535w6x4hzip5viq677omw3yq7dis.py
# Source Nodes: [x_288, x_289, x_291], Original ATen: [aten.hardtanh, aten.mean, aten.view]
# x_288 => clamp_max_34, clamp_min_34
# x_289 => mean
# x_291 => view
triton_per_fused_hardtanh_mean_view_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_mean_view_89', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 6.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbb6e6adgcxbc47dq4ucilc4rc7n2non6zs7tfnvs6n6ehndqav.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_90', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_44, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, ), (1, ))
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
    assert_size_stride(primals_77, (96, ), (1, ))
    assert_size_stride(primals_78, (96, ), (1, ))
    assert_size_stride(primals_79, (576, ), (1, ))
    assert_size_stride(primals_80, (576, ), (1, ))
    assert_size_stride(primals_81, (576, ), (1, ))
    assert_size_stride(primals_82, (576, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_84, (160, ), (1, ))
    assert_size_stride(primals_85, (960, ), (1, ))
    assert_size_stride(primals_86, (960, ), (1, ))
    assert_size_stride(primals_87, (960, ), (1, ))
    assert_size_stride(primals_88, (960, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_90, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_92, (960, ), (1, ))
    assert_size_stride(primals_93, (960, ), (1, ))
    assert_size_stride(primals_94, (960, ), (1, ))
    assert_size_stride(primals_95, (160, ), (1, ))
    assert_size_stride(primals_96, (160, ), (1, ))
    assert_size_stride(primals_97, (960, ), (1, ))
    assert_size_stride(primals_98, (960, ), (1, ))
    assert_size_stride(primals_99, (960, ), (1, ))
    assert_size_stride(primals_100, (960, ), (1, ))
    assert_size_stride(primals_101, (320, ), (1, ))
    assert_size_stride(primals_102, (320, ), (1, ))
    assert_size_stride(primals_103, (1280, ), (1, ))
    assert_size_stride(primals_104, (1280, ), (1, ))
    assert_size_stride(primals_105, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_106, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_108, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_109, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_111, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_112, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_114, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_115, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_116, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_117, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_118, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_120, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_121, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_122, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_123, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_124, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_126, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_127, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_129, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_130, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_132, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_133, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_135, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_136, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_138, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_139, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_141, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_142, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_144, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_145, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (160, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_147, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_148, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_150, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_151, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_153, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_154, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (320, 960, 1, 1), (960, 1, 1, 1))
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
    assert_size_stride(primals_169, (96, ), (1, ))
    assert_size_stride(primals_170, (96, ), (1, ))
    assert_size_stride(primals_171, (), ())
    assert_size_stride(primals_172, (96, ), (1, ))
    assert_size_stride(primals_173, (96, ), (1, ))
    assert_size_stride(primals_174, (), ())
    assert_size_stride(primals_175, (24, ), (1, ))
    assert_size_stride(primals_176, (24, ), (1, ))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (144, ), (1, ))
    assert_size_stride(primals_179, (144, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (144, ), (1, ))
    assert_size_stride(primals_182, (144, ), (1, ))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (24, ), (1, ))
    assert_size_stride(primals_185, (24, ), (1, ))
    assert_size_stride(primals_186, (), ())
    assert_size_stride(primals_187, (144, ), (1, ))
    assert_size_stride(primals_188, (144, ), (1, ))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (144, ), (1, ))
    assert_size_stride(primals_191, (144, ), (1, ))
    assert_size_stride(primals_192, (), ())
    assert_size_stride(primals_193, (32, ), (1, ))
    assert_size_stride(primals_194, (32, ), (1, ))
    assert_size_stride(primals_195, (), ())
    assert_size_stride(primals_196, (192, ), (1, ))
    assert_size_stride(primals_197, (192, ), (1, ))
    assert_size_stride(primals_198, (), ())
    assert_size_stride(primals_199, (192, ), (1, ))
    assert_size_stride(primals_200, (192, ), (1, ))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (32, ), (1, ))
    assert_size_stride(primals_203, (32, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (192, ), (1, ))
    assert_size_stride(primals_206, (192, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (192, ), (1, ))
    assert_size_stride(primals_209, (192, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (32, ), (1, ))
    assert_size_stride(primals_212, (32, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (192, ), (1, ))
    assert_size_stride(primals_215, (192, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (192, ), (1, ))
    assert_size_stride(primals_218, (192, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (384, ), (1, ))
    assert_size_stride(primals_233, (384, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_236, (384, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (384, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (64, ), (1, ))
    assert_size_stride(primals_248, (64, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (384, ), (1, ))
    assert_size_stride(primals_251, (384, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_254, (384, ), (1, ))
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
    assert_size_stride(primals_274, (96, ), (1, ))
    assert_size_stride(primals_275, (96, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (576, ), (1, ))
    assert_size_stride(primals_278, (576, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (576, ), (1, ))
    assert_size_stride(primals_281, (576, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (160, ), (1, ))
    assert_size_stride(primals_284, (160, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (960, ), (1, ))
    assert_size_stride(primals_287, (960, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (960, ), (1, ))
    assert_size_stride(primals_290, (960, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (160, ), (1, ))
    assert_size_stride(primals_293, (160, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (960, ), (1, ))
    assert_size_stride(primals_299, (960, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (160, ), (1, ))
    assert_size_stride(primals_302, (160, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (960, ), (1, ))
    assert_size_stride(primals_308, (960, ), (1, ))
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
        buf15 = reinterpret_tensor(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        buf623 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.bool)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_6.run(buf3, buf10, buf11, primals_1, primals_2, buf15, buf623, 3211264, grid=grid(3211264), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf16, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf17 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf16, buf17, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf18 = buf6; del buf6  # reuse
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf17, buf18, buf19, buf20, 25088, 128, grid=grid(25088), stream=stream0)
        buf21 = buf9; del buf9  # reuse
        buf22 = buf8; del buf8  # reuse
        buf23 = buf7; del buf7  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf18, buf19, buf20, buf21, buf22, buf23, 224, 112, grid=grid(224), stream=stream0)
        del buf18
        del buf19
        del buf20
        buf24 = buf11; del buf11  # reuse
        buf25 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf27 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf21, buf22, buf23, primals_163, primals_164, buf24, buf25, buf27, primals_163, primals_164, 32, 7, grid=grid(32), stream=stream0)
        del buf21
        del buf22
        del buf23
        del primals_163
        del primals_164
        buf29 = reinterpret_tensor(buf16, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf16  # reuse
        buf622 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_6.run(buf17, buf24, buf25, primals_3, primals_4, buf29, buf622, 3211264, grid=grid(3211264), stream=stream0)
        del primals_4
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf31 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf30, buf31, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf32 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf34 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf31, buf32, buf33, buf34, 12544, 128, grid=grid(12544), stream=stream0)
        buf35 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf32, buf33, buf34, buf35, buf36, buf37, 112, 112, grid=grid(112), stream=stream0)
        del buf32
        del buf33
        del buf34
        buf38 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf41 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf35, buf36, buf37, primals_166, primals_167, buf38, buf39, buf41, primals_166, primals_167, 16, 7, grid=grid(16), stream=stream0)
        del buf35
        del buf36
        del buf37
        del primals_166
        del primals_167
        buf42 = reinterpret_tensor(buf30, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf30  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_11.run(buf31, buf38, buf39, primals_5, primals_6, buf42, 1605632, grid=grid(1605632), stream=stream0)
        del buf39
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        buf44 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf43, buf44, 768, 12544, grid=grid(768, 12544), stream=stream0)
        buf45 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf44, buf45, buf46, buf47, 75264, 128, grid=grid(75264), stream=stream0)
        buf48 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf45, buf46, buf47, buf48, buf49, buf50, 672, 112, grid=grid(672), stream=stream0)
        del buf45
        del buf46
        del buf47
        buf51 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf54 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf48, buf49, buf50, primals_169, primals_170, buf51, buf52, buf54, primals_169, primals_170, 96, 7, grid=grid(96), stream=stream0)
        del buf48
        del buf49
        del buf50
        del primals_169
        del primals_170
        buf56 = reinterpret_tensor(buf43, (8, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf43  # reuse
        buf621 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_16.run(buf44, buf51, buf52, primals_7, primals_8, buf56, buf621, 9633792, grid=grid(9633792), stream=stream0)
        del primals_8
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_109, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf57, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf58 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf57, buf58, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf59 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf58, buf59, buf60, buf61, 18816, 128, grid=grid(18816), stream=stream0)
        buf62 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf64 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf59, buf60, buf61, buf62, buf63, buf64, 192, 98, grid=grid(192), stream=stream0)
        del buf59
        del buf60
        del buf61
        buf65 = buf52; del buf52  # reuse
        buf66 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf68 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_20.run(buf62, buf63, buf64, primals_172, primals_173, buf65, buf66, buf68, primals_172, primals_173, 96, 2, grid=grid(96), stream=stream0)
        del primals_172
        del primals_173
        buf70 = reinterpret_tensor(buf57, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf57  # reuse
        buf620 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_21.run(buf58, buf65, buf66, primals_9, primals_10, buf70, buf620, 2408448, grid=grid(2408448), stream=stream0)
        del primals_10
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf72 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf71, buf72, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf73 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf72, buf73, buf74, buf75, 4704, 128, grid=grid(4704), stream=stream0)
        buf76 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf77 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf73, buf74, buf75, buf76, buf77, buf78, 48, 98, grid=grid(48), stream=stream0)
        buf79 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf80 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf82 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf76, buf77, buf78, primals_175, primals_176, buf79, buf80, buf82, primals_175, primals_176, 24, 2, grid=grid(24), stream=stream0)
        del primals_175
        del primals_176
        buf83 = reinterpret_tensor(buf71, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf71  # reuse
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_26.run(buf72, buf79, buf80, primals_11, primals_12, buf83, 602112, grid=grid(602112), stream=stream0)
        del primals_12
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf85 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf84, buf85, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf86 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf85, buf86, buf87, buf88, 28224, 128, grid=grid(28224), stream=stream0)
        buf89 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf86, buf87, buf88, buf89, buf90, buf91, 288, 98, grid=grid(288), stream=stream0)
        buf92 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf95 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf89, buf90, buf91, primals_178, primals_179, buf92, buf93, buf95, primals_178, primals_179, 144, 2, grid=grid(144), stream=stream0)
        del primals_178
        del primals_179
        buf97 = reinterpret_tensor(buf84, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf84  # reuse
        buf619 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_31.run(buf85, buf92, buf93, primals_13, primals_14, buf97, buf619, 3612672, grid=grid(3612672), stream=stream0)
        del primals_14
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf98, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf99 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf98, buf99, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf100 = buf88; del buf88  # reuse
        buf101 = buf87; del buf87  # reuse
        buf102 = buf86; del buf86  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf99, buf100, buf101, buf102, 28224, 128, grid=grid(28224), stream=stream0)
        buf103 = buf91; del buf91  # reuse
        buf104 = buf90; del buf90  # reuse
        buf105 = buf89; del buf89  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf100, buf101, buf102, buf103, buf104, buf105, 288, 98, grid=grid(288), stream=stream0)
        buf106 = buf93; del buf93  # reuse
        buf107 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf109 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf103, buf104, buf105, primals_181, primals_182, buf106, buf107, buf109, primals_181, primals_182, 144, 2, grid=grid(144), stream=stream0)
        del primals_181
        del primals_182
        buf111 = reinterpret_tensor(buf98, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf98  # reuse
        buf618 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_31.run(buf99, buf106, buf107, primals_15, primals_16, buf111, buf618, 3612672, grid=grid(3612672), stream=stream0)
        del primals_16
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf113 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_22.run(buf112, buf113, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf114 = buf75; del buf75  # reuse
        buf115 = buf74; del buf74  # reuse
        buf116 = buf73; del buf73  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf113, buf114, buf115, buf116, 4704, 128, grid=grid(4704), stream=stream0)
        buf117 = buf78; del buf78  # reuse
        buf118 = buf77; del buf77  # reuse
        buf119 = buf76; del buf76  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf114, buf115, buf116, buf117, buf118, buf119, 48, 98, grid=grid(48), stream=stream0)
        del buf114
        del buf115
        del buf116
        buf120 = buf80; del buf80  # reuse
        buf121 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf123 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf117, buf118, buf119, primals_184, primals_185, buf120, buf121, buf123, primals_184, primals_185, 24, 2, grid=grid(24), stream=stream0)
        del buf117
        del buf118
        del buf119
        del primals_184
        del primals_185
        buf124 = reinterpret_tensor(buf112, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf112  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_32.run(buf113, buf120, buf121, primals_17, primals_18, buf83, buf124, 602112, grid=grid(602112), stream=stream0)
        del buf121
        del primals_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf126 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf125, buf126, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf127 = buf102; del buf102  # reuse
        buf128 = buf101; del buf101  # reuse
        buf129 = buf100; del buf100  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf126, buf127, buf128, buf129, 28224, 128, grid=grid(28224), stream=stream0)
        buf130 = buf105; del buf105  # reuse
        buf131 = buf104; del buf104  # reuse
        buf132 = buf103; del buf103  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_29.run(buf127, buf128, buf129, buf130, buf131, buf132, 288, 98, grid=grid(288), stream=stream0)
        del buf127
        del buf128
        del buf129
        buf133 = buf107; del buf107  # reuse
        buf134 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf136 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_30.run(buf130, buf131, buf132, primals_187, primals_188, buf133, buf134, buf136, primals_187, primals_188, 144, 2, grid=grid(144), stream=stream0)
        del buf130
        del buf131
        del buf132
        del primals_187
        del primals_188
        buf138 = reinterpret_tensor(buf125, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf125  # reuse
        buf617 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_31.run(buf126, buf133, buf134, primals_19, primals_20, buf138, buf617, 3612672, grid=grid(3612672), stream=stream0)
        del primals_20
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_115, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf139, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf140 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf139, buf140, 1152, 784, grid=grid(1152, 784), stream=stream0)
        buf141 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        buf143 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf140, buf141, buf142, buf143, 7056, 128, grid=grid(7056), stream=stream0)
        buf144 = buf134; del buf134  # reuse
        buf145 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf147 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf141, buf142, buf143, primals_190, primals_191, buf144, buf145, buf147, primals_190, primals_191, 144, 49, grid=grid(144), stream=stream0)
        del buf141
        del buf142
        del buf143
        del primals_190
        del primals_191
        buf149 = reinterpret_tensor(buf139, (8, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf139  # reuse
        buf616 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_55, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_36.run(buf140, buf144, buf145, primals_21, primals_22, buf149, buf616, 903168, grid=grid(903168), stream=stream0)
        del buf145
        del primals_22
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf151 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf150, buf151, 256, 784, grid=grid(256, 784), stream=stream0)
        buf152 = empty_strided((1, 32, 1, 1, 49), (1568, 1, 1568, 1568, 32), device='cuda', dtype=torch.float32)
        buf153 = empty_strided((1, 32, 1, 1, 49), (1568, 1, 1568, 1568, 32), device='cuda', dtype=torch.float32)
        buf154 = empty_strided((1, 32, 1, 1, 49), (1568, 1, 1568, 1568, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf151, buf152, buf153, buf154, 1568, 128, grid=grid(1568), stream=stream0)
        buf155 = buf25; del buf25  # reuse
        buf156 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf158 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf152, buf153, buf154, primals_193, primals_194, buf155, buf156, buf158, primals_193, primals_194, 32, 49, grid=grid(32), stream=stream0)
        del primals_193
        del primals_194
        buf159 = reinterpret_tensor(buf150, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf150  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_40.run(buf151, buf155, buf156, primals_23, primals_24, buf159, 200704, grid=grid(200704), stream=stream0)
        del primals_24
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf161 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf160, buf161, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf162 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        buf163 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf161, buf162, buf163, buf164, 9408, 128, grid=grid(9408), stream=stream0)
        buf165 = reinterpret_tensor(buf64, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf64  # reuse
        buf166 = reinterpret_tensor(buf63, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf63  # reuse
        buf168 = reinterpret_tensor(buf62, (192, ), (1, ), 0); del buf62  # reuse
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf162, buf163, buf164, primals_196, primals_197, buf165, buf166, buf168, primals_196, primals_197, 192, 49, grid=grid(192), stream=stream0)
        del primals_196
        del primals_197
        buf170 = reinterpret_tensor(buf160, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf160  # reuse
        buf615 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44.run(buf161, buf165, buf166, primals_25, primals_26, buf170, buf615, 1204224, grid=grid(1204224), stream=stream0)
        del primals_26
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf171, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf172 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf171, buf172, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf173 = buf164; del buf164  # reuse
        buf174 = buf163; del buf163  # reuse
        buf175 = buf162; del buf162  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf172, buf173, buf174, buf175, 9408, 128, grid=grid(9408), stream=stream0)
        buf176 = buf166; del buf166  # reuse
        buf177 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf179 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf173, buf174, buf175, primals_199, primals_200, buf176, buf177, buf179, primals_199, primals_200, 192, 49, grid=grid(192), stream=stream0)
        del primals_199
        del primals_200
        buf181 = reinterpret_tensor(buf171, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf171  # reuse
        buf614 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_71, x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44.run(buf172, buf176, buf177, primals_27, primals_28, buf181, buf614, 1204224, grid=grid(1204224), stream=stream0)
        del primals_28
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf183 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf182, buf183, 256, 784, grid=grid(256, 784), stream=stream0)
        buf184 = buf154; del buf154  # reuse
        buf185 = buf153; del buf153  # reuse
        buf186 = buf152; del buf152  # reuse
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf183, buf184, buf185, buf186, 1568, 128, grid=grid(1568), stream=stream0)
        buf187 = buf156; del buf156  # reuse
        buf188 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf190 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf184, buf185, buf186, primals_202, primals_203, buf187, buf188, buf190, primals_202, primals_203, 32, 49, grid=grid(32), stream=stream0)
        del primals_202
        del primals_203
        buf191 = reinterpret_tensor(buf182, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf182  # reuse
        # Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_45.run(buf183, buf187, buf188, primals_29, primals_30, buf159, buf191, 200704, grid=grid(200704), stream=stream0)
        del primals_30
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf193 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf192, buf193, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf194 = buf175; del buf175  # reuse
        buf195 = buf174; del buf174  # reuse
        buf196 = buf173; del buf173  # reuse
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf193, buf194, buf195, buf196, 9408, 128, grid=grid(9408), stream=stream0)
        buf197 = buf177; del buf177  # reuse
        buf198 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf200 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf194, buf195, buf196, primals_205, primals_206, buf197, buf198, buf200, primals_205, primals_206, 192, 49, grid=grid(192), stream=stream0)
        del primals_205
        del primals_206
        buf202 = reinterpret_tensor(buf192, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf192  # reuse
        buf613 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44.run(buf193, buf197, buf198, primals_31, primals_32, buf202, buf613, 1204224, grid=grid(1204224), stream=stream0)
        del primals_32
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_121, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf203, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf204 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf203, buf204, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf205 = buf196; del buf196  # reuse
        buf206 = buf195; del buf195  # reuse
        buf207 = buf194; del buf194  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf204, buf205, buf206, buf207, 9408, 128, grid=grid(9408), stream=stream0)
        buf208 = buf198; del buf198  # reuse
        buf209 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf211 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf205, buf206, buf207, primals_208, primals_209, buf208, buf209, buf211, primals_208, primals_209, 192, 49, grid=grid(192), stream=stream0)
        del primals_208
        del primals_209
        buf213 = reinterpret_tensor(buf203, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf203  # reuse
        buf612 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_88, x_91], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44.run(buf204, buf208, buf209, primals_33, primals_34, buf213, buf612, 1204224, grid=grid(1204224), stream=stream0)
        del primals_34
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf215 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf214, buf215, 256, 784, grid=grid(256, 784), stream=stream0)
        buf216 = buf186; del buf186  # reuse
        buf217 = buf185; del buf185  # reuse
        buf218 = buf184; del buf184  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf215, buf216, buf217, buf218, 1568, 128, grid=grid(1568), stream=stream0)
        buf219 = buf188; del buf188  # reuse
        buf220 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf222 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf216, buf217, buf218, primals_211, primals_212, buf219, buf220, buf222, primals_211, primals_212, 32, 49, grid=grid(32), stream=stream0)
        del buf216
        del buf217
        del buf218
        del primals_211
        del primals_212
        buf223 = reinterpret_tensor(buf214, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf214  # reuse
        # Source Nodes: [shortcut_6, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_45.run(buf215, buf219, buf220, primals_35, primals_36, buf191, buf223, 200704, grid=grid(200704), stream=stream0)
        del buf220
        del primals_36
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf225 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf224, buf225, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf226 = buf207; del buf207  # reuse
        buf227 = buf206; del buf206  # reuse
        buf228 = buf205; del buf205  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf225, buf226, buf227, buf228, 9408, 128, grid=grid(9408), stream=stream0)
        buf229 = buf209; del buf209  # reuse
        buf230 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf232 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf226, buf227, buf228, primals_214, primals_215, buf229, buf230, buf232, primals_214, primals_215, 192, 49, grid=grid(192), stream=stream0)
        del buf226
        del buf227
        del buf228
        del primals_214
        del primals_215
        buf234 = reinterpret_tensor(buf224, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf224  # reuse
        buf611 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_100, x_103], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_44.run(buf225, buf229, buf230, primals_37, primals_38, buf234, buf611, 1204224, grid=grid(1204224), stream=stream0)
        del primals_38
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf235, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf236 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_46.run(buf235, buf236, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf237 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        buf239 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf236, buf237, buf238, buf239, 2496, 121, grid=grid(2496), stream=stream0)
        buf240 = buf230; del buf230  # reuse
        buf241 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf243 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_48.run(buf237, buf238, buf239, primals_217, primals_218, buf240, buf241, buf243, primals_217, primals_218, 192, 13, grid=grid(192), stream=stream0)
        del buf237
        del buf238
        del buf239
        del primals_217
        del primals_218
        buf245 = reinterpret_tensor(buf235, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf235  # reuse
        buf610 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_49.run(buf236, buf240, buf241, primals_39, primals_40, buf245, buf610, 301056, grid=grid(301056), stream=stream0)
        del buf241
        del primals_40
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf247 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf246, buf247, 512, 196, grid=grid(512, 196), stream=stream0)
        buf248 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        buf249 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        buf250 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf247, buf248, buf249, buf250, 832, 121, grid=grid(832), stream=stream0)
        buf251 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf252 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf254 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf248, buf249, buf250, primals_220, primals_221, buf251, buf252, buf254, primals_220, primals_221, 64, 13, grid=grid(64), stream=stream0)
        del primals_220
        del primals_221
        buf255 = reinterpret_tensor(buf246, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf246  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_53.run(buf247, buf251, buf252, primals_41, primals_42, buf255, 100352, grid=grid(100352), stream=stream0)
        del primals_42
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf257 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf256, buf257, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf258 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        buf259 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        buf260 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf257, buf258, buf259, buf260, 4992, 121, grid=grid(4992), stream=stream0)
        buf261 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf262 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf264 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf258, buf259, buf260, primals_223, primals_224, buf261, buf262, buf264, primals_223, primals_224, 384, 13, grid=grid(384), stream=stream0)
        del primals_223
        del primals_224
        buf266 = reinterpret_tensor(buf256, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf256  # reuse
        buf609 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf257, buf261, buf262, primals_43, primals_44, buf266, buf609, 602112, grid=grid(602112), stream=stream0)
        del primals_44
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf267, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf268 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf267, buf268, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf269 = buf260; del buf260  # reuse
        buf270 = buf259; del buf259  # reuse
        buf271 = buf258; del buf258  # reuse
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf268, buf269, buf270, buf271, 4992, 121, grid=grid(4992), stream=stream0)
        buf272 = buf262; del buf262  # reuse
        buf273 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf275 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf269, buf270, buf271, primals_226, primals_227, buf272, buf273, buf275, primals_226, primals_227, 384, 13, grid=grid(384), stream=stream0)
        del primals_226
        del primals_227
        buf277 = reinterpret_tensor(buf267, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf267  # reuse
        buf608 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_121, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf268, buf272, buf273, primals_45, primals_46, buf277, buf608, 602112, grid=grid(602112), stream=stream0)
        del primals_46
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf279 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf278, buf279, 512, 196, grid=grid(512, 196), stream=stream0)
        buf280 = buf250; del buf250  # reuse
        buf281 = buf249; del buf249  # reuse
        buf282 = buf248; del buf248  # reuse
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf279, buf280, buf281, buf282, 832, 121, grid=grid(832), stream=stream0)
        buf283 = buf252; del buf252  # reuse
        buf284 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf286 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf280, buf281, buf282, primals_229, primals_230, buf283, buf284, buf286, primals_229, primals_230, 64, 13, grid=grid(64), stream=stream0)
        del primals_229
        del primals_230
        buf287 = reinterpret_tensor(buf278, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf278  # reuse
        # Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_58.run(buf279, buf283, buf284, primals_47, primals_48, buf255, buf287, 100352, grid=grid(100352), stream=stream0)
        del primals_48
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf289 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf288, buf289, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf290 = buf271; del buf271  # reuse
        buf291 = buf270; del buf270  # reuse
        buf292 = buf269; del buf269  # reuse
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf289, buf290, buf291, buf292, 4992, 121, grid=grid(4992), stream=stream0)
        buf293 = buf273; del buf273  # reuse
        buf294 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf296 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf290, buf291, buf292, primals_232, primals_233, buf293, buf294, buf296, primals_232, primals_233, 384, 13, grid=grid(384), stream=stream0)
        del primals_232
        del primals_233
        buf298 = reinterpret_tensor(buf288, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf288  # reuse
        buf607 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf289, buf293, buf294, primals_49, primals_50, buf298, buf607, 602112, grid=grid(602112), stream=stream0)
        del primals_50
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf299, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf300 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf299, buf300, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf301 = buf292; del buf292  # reuse
        buf302 = buf291; del buf291  # reuse
        buf303 = buf290; del buf290  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf300, buf301, buf302, buf303, 4992, 121, grid=grid(4992), stream=stream0)
        buf304 = buf294; del buf294  # reuse
        buf305 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf307 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf301, buf302, buf303, primals_235, primals_236, buf304, buf305, buf307, primals_235, primals_236, 384, 13, grid=grid(384), stream=stream0)
        del primals_235
        del primals_236
        buf309 = reinterpret_tensor(buf299, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf299  # reuse
        buf606 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_138, x_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf300, buf304, buf305, primals_51, primals_52, buf309, buf606, 602112, grid=grid(602112), stream=stream0)
        del primals_52
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf311 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf310, buf311, 512, 196, grid=grid(512, 196), stream=stream0)
        buf312 = buf282; del buf282  # reuse
        buf313 = buf281; del buf281  # reuse
        buf314 = buf280; del buf280  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf311, buf312, buf313, buf314, 832, 121, grid=grid(832), stream=stream0)
        buf315 = buf284; del buf284  # reuse
        buf316 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf318 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf312, buf313, buf314, primals_238, primals_239, buf315, buf316, buf318, primals_238, primals_239, 64, 13, grid=grid(64), stream=stream0)
        del primals_238
        del primals_239
        buf319 = reinterpret_tensor(buf310, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf310  # reuse
        # Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_58.run(buf311, buf315, buf316, primals_53, primals_54, buf287, buf319, 100352, grid=grid(100352), stream=stream0)
        del primals_54
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf321 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf320, buf321, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf322 = buf303; del buf303  # reuse
        buf323 = buf302; del buf302  # reuse
        buf324 = buf301; del buf301  # reuse
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf321, buf322, buf323, buf324, 4992, 121, grid=grid(4992), stream=stream0)
        buf325 = buf305; del buf305  # reuse
        buf326 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf328 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf322, buf323, buf324, primals_241, primals_242, buf325, buf326, buf328, primals_241, primals_242, 384, 13, grid=grid(384), stream=stream0)
        del primals_241
        del primals_242
        buf330 = reinterpret_tensor(buf320, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf320  # reuse
        buf605 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_150, x_153], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf321, buf325, buf326, primals_55, primals_56, buf330, buf605, 602112, grid=grid(602112), stream=stream0)
        del primals_56
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf331, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf332 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf331, buf332, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf333 = buf324; del buf324  # reuse
        buf334 = buf323; del buf323  # reuse
        buf335 = buf322; del buf322  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf332, buf333, buf334, buf335, 4992, 121, grid=grid(4992), stream=stream0)
        buf336 = buf326; del buf326  # reuse
        buf337 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf339 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf333, buf334, buf335, primals_244, primals_245, buf336, buf337, buf339, primals_244, primals_245, 384, 13, grid=grid(384), stream=stream0)
        del primals_244
        del primals_245
        buf341 = reinterpret_tensor(buf331, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf331  # reuse
        buf604 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_155, x_158], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf332, buf336, buf337, primals_57, primals_58, buf341, buf604, 602112, grid=grid(602112), stream=stream0)
        del primals_58
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf343 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf342, buf343, 512, 196, grid=grid(512, 196), stream=stream0)
        buf344 = buf314; del buf314  # reuse
        buf345 = buf313; del buf313  # reuse
        buf346 = buf312; del buf312  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf343, buf344, buf345, buf346, 832, 121, grid=grid(832), stream=stream0)
        buf347 = buf316; del buf316  # reuse
        buf348 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf350 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_52.run(buf344, buf345, buf346, primals_247, primals_248, buf347, buf348, buf350, primals_247, primals_248, 64, 13, grid=grid(64), stream=stream0)
        del buf344
        del buf345
        del buf346
        del primals_247
        del primals_248
        buf351 = reinterpret_tensor(buf342, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf342  # reuse
        # Source Nodes: [shortcut_10, x_161], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_58.run(buf343, buf347, buf348, primals_59, primals_60, buf319, buf351, 100352, grid=grid(100352), stream=stream0)
        del buf348
        del primals_60
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf353 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf352, buf353, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf354 = buf335; del buf335  # reuse
        buf355 = buf334; del buf334  # reuse
        buf356 = buf333; del buf333  # reuse
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf353, buf354, buf355, buf356, 4992, 121, grid=grid(4992), stream=stream0)
        buf357 = buf337; del buf337  # reuse
        buf358 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf360 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf354, buf355, buf356, primals_250, primals_251, buf357, buf358, buf360, primals_250, primals_251, 384, 13, grid=grid(384), stream=stream0)
        del primals_250
        del primals_251
        buf362 = reinterpret_tensor(buf352, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf352  # reuse
        buf603 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_167, x_170], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf353, buf357, buf358, primals_61, primals_62, buf362, buf603, 602112, grid=grid(602112), stream=stream0)
        del primals_62
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf363, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf364 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_54.run(buf363, buf364, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf365 = buf356; del buf356  # reuse
        buf366 = buf355; del buf355  # reuse
        buf367 = buf354; del buf354  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf364, buf365, buf366, buf367, 4992, 121, grid=grid(4992), stream=stream0)
        buf368 = buf358; del buf358  # reuse
        buf369 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf371 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_56.run(buf365, buf366, buf367, primals_253, primals_254, buf368, buf369, buf371, primals_253, primals_254, 384, 13, grid=grid(384), stream=stream0)
        del buf365
        del buf366
        del buf367
        del primals_253
        del primals_254
        buf373 = reinterpret_tensor(buf363, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf363  # reuse
        buf602 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_172, x_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_57.run(buf364, buf368, buf369, primals_63, primals_64, buf373, buf602, 602112, grid=grid(602112), stream=stream0)
        del buf369
        del primals_64
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 96, 14, 14), (18816, 196, 14, 1))
        buf375 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf374, buf375, 768, 196, grid=grid(768, 196), stream=stream0)
        buf376 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        buf377 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        buf378 = empty_strided((1, 96, 1, 1, 13), (1248, 1, 1248, 1248, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf375, buf376, buf377, buf378, 1248, 121, grid=grid(1248), stream=stream0)
        buf379 = buf66; del buf66  # reuse
        buf380 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf382 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf376, buf377, buf378, primals_256, primals_257, buf379, buf380, buf382, primals_256, primals_257, 96, 13, grid=grid(96), stream=stream0)
        del primals_256
        del primals_257
        buf383 = reinterpret_tensor(buf374, (8, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf374  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_62.run(buf375, buf379, buf380, primals_65, primals_66, buf383, 150528, grid=grid(150528), stream=stream0)
        del primals_66
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf383, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf385 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf384, buf385, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf386 = empty_strided((1, 576, 1, 1, 13), (7488, 1, 7488, 7488, 576), device='cuda', dtype=torch.float32)
        buf387 = empty_strided((1, 576, 1, 1, 13), (7488, 1, 7488, 7488, 576), device='cuda', dtype=torch.float32)
        buf388 = empty_strided((1, 576, 1, 1, 13), (7488, 1, 7488, 7488, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf385, buf386, buf387, buf388, 7488, 121, grid=grid(7488), stream=stream0)
        buf389 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf390 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf392 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf386, buf387, buf388, primals_259, primals_260, buf389, buf390, buf392, primals_259, primals_260, 576, 13, grid=grid(576), stream=stream0)
        del primals_259
        del primals_260
        buf394 = reinterpret_tensor(buf384, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf384  # reuse
        buf601 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66.run(buf385, buf389, buf390, primals_67, primals_68, buf394, buf601, 903168, grid=grid(903168), stream=stream0)
        del primals_68
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf395, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf396 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf395, buf396, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf397 = buf388; del buf388  # reuse
        buf398 = buf387; del buf387  # reuse
        buf399 = buf386; del buf386  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf396, buf397, buf398, buf399, 7488, 121, grid=grid(7488), stream=stream0)
        buf400 = buf390; del buf390  # reuse
        buf401 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf403 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf397, buf398, buf399, primals_262, primals_263, buf400, buf401, buf403, primals_262, primals_263, 576, 13, grid=grid(576), stream=stream0)
        del primals_262
        del primals_263
        buf405 = reinterpret_tensor(buf395, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf395  # reuse
        buf600 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_188, x_191], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66.run(buf396, buf400, buf401, primals_69, primals_70, buf405, buf600, 903168, grid=grid(903168), stream=stream0)
        del primals_70
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf405, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (8, 96, 14, 14), (18816, 196, 14, 1))
        buf407 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf406, buf407, 768, 196, grid=grid(768, 196), stream=stream0)
        buf408 = buf378; del buf378  # reuse
        buf409 = buf377; del buf377  # reuse
        buf410 = buf376; del buf376  # reuse
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf407, buf408, buf409, buf410, 1248, 121, grid=grid(1248), stream=stream0)
        buf411 = buf380; del buf380  # reuse
        buf412 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf414 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf408, buf409, buf410, primals_265, primals_266, buf411, buf412, buf414, primals_265, primals_266, 96, 13, grid=grid(96), stream=stream0)
        del primals_265
        del primals_266
        buf415 = reinterpret_tensor(buf406, (8, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf406  # reuse
        # Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_67.run(buf407, buf411, buf412, primals_71, primals_72, buf383, buf415, 150528, grid=grid(150528), stream=stream0)
        del primals_72
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf417 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf416, buf417, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf418 = buf399; del buf399  # reuse
        buf419 = buf398; del buf398  # reuse
        buf420 = buf397; del buf397  # reuse
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf417, buf418, buf419, buf420, 7488, 121, grid=grid(7488), stream=stream0)
        buf421 = buf401; del buf401  # reuse
        buf422 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf424 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf418, buf419, buf420, primals_268, primals_269, buf421, buf422, buf424, primals_268, primals_269, 576, 13, grid=grid(576), stream=stream0)
        del primals_268
        del primals_269
        buf426 = reinterpret_tensor(buf416, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf416  # reuse
        buf599 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_200, x_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66.run(buf417, buf421, buf422, primals_73, primals_74, buf426, buf599, 903168, grid=grid(903168), stream=stream0)
        del primals_74
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf427, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf428 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf427, buf428, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf429 = buf420; del buf420  # reuse
        buf430 = buf419; del buf419  # reuse
        buf431 = buf418; del buf418  # reuse
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf428, buf429, buf430, buf431, 7488, 121, grid=grid(7488), stream=stream0)
        buf432 = buf422; del buf422  # reuse
        buf433 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf435 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf429, buf430, buf431, primals_271, primals_272, buf432, buf433, buf435, primals_271, primals_272, 576, 13, grid=grid(576), stream=stream0)
        del primals_271
        del primals_272
        buf437 = reinterpret_tensor(buf427, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf427  # reuse
        buf598 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_205, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66.run(buf428, buf432, buf433, primals_75, primals_76, buf437, buf598, 903168, grid=grid(903168), stream=stream0)
        del primals_76
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (8, 96, 14, 14), (18816, 196, 14, 1))
        buf439 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_59.run(buf438, buf439, 768, 196, grid=grid(768, 196), stream=stream0)
        buf440 = buf410; del buf410  # reuse
        buf441 = buf409; del buf409  # reuse
        buf442 = buf408; del buf408  # reuse
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf439, buf440, buf441, buf442, 1248, 121, grid=grid(1248), stream=stream0)
        buf443 = buf412; del buf412  # reuse
        buf444 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf446 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_61.run(buf440, buf441, buf442, primals_274, primals_275, buf443, buf444, buf446, primals_274, primals_275, 96, 13, grid=grid(96), stream=stream0)
        del buf440
        del buf441
        del buf442
        del primals_274
        del primals_275
        buf447 = reinterpret_tensor(buf438, (8, 96, 14, 14), (18816, 1, 1344, 96), 0); del buf438  # reuse
        # Source Nodes: [shortcut_13, x_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_67.run(buf439, buf443, buf444, primals_77, primals_78, buf415, buf447, 150528, grid=grid(150528), stream=stream0)
        del buf444
        del primals_78
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 576, 14, 14), (112896, 196, 14, 1))
        buf449 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf448, buf449, 4608, 196, grid=grid(4608, 196), stream=stream0)
        buf450 = buf431; del buf431  # reuse
        buf451 = buf430; del buf430  # reuse
        buf452 = buf429; del buf429  # reuse
        # Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf449, buf450, buf451, buf452, 7488, 121, grid=grid(7488), stream=stream0)
        buf453 = buf433; del buf433  # reuse
        buf454 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf456 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf450, buf451, buf452, primals_277, primals_278, buf453, buf454, buf456, primals_277, primals_278, 576, 13, grid=grid(576), stream=stream0)
        del buf450
        del buf451
        del buf452
        del primals_277
        del primals_278
        buf458 = reinterpret_tensor(buf448, (8, 576, 14, 14), (112896, 1, 8064, 576), 0); del buf448  # reuse
        buf597 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_66.run(buf449, buf453, buf454, primals_79, primals_80, buf458, buf597, 903168, grid=grid(903168), stream=stream0)
        del primals_80
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, primals_145, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf459, (8, 576, 7, 7), (28224, 49, 7, 1))
        buf460 = empty_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_68.run(buf459, buf460, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf461 = empty_strided((1, 576, 1, 1, 4), (2304, 1, 2304, 2304, 576), device='cuda', dtype=torch.float32)
        buf462 = empty_strided((1, 576, 1, 1, 4), (2304, 1, 2304, 2304, 576), device='cuda', dtype=torch.float32)
        buf463 = empty_strided((1, 576, 1, 1, 4), (2304, 1, 2304, 2304, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_69.run(buf460, buf461, buf462, buf463, 2304, 98, grid=grid(2304), stream=stream0)
        buf464 = buf454; del buf454  # reuse
        buf465 = empty_strided((1, 576, 1, 1), (576, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf467 = empty((576, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_70.run(buf461, buf462, buf463, primals_280, primals_281, buf464, buf465, buf467, primals_280, primals_281, 576, 4, grid=grid(576), stream=stream0)
        del buf461
        del buf462
        del buf463
        del primals_280
        del primals_281
        buf469 = reinterpret_tensor(buf459, (8, 576, 7, 7), (28224, 1, 4032, 576), 0); del buf459  # reuse
        buf596 = empty_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_222, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_71.run(buf460, buf464, buf465, primals_81, primals_82, buf469, buf596, 225792, grid=grid(225792), stream=stream0)
        del buf465
        del primals_82
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf471 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf470, buf471, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf472 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        buf473 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        buf474 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf471, buf472, buf473, buf474, 640, 98, grid=grid(640), stream=stream0)
        buf475 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf476 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf478 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf472, buf473, buf474, primals_283, primals_284, buf475, buf476, buf478, primals_283, primals_284, 160, 4, grid=grid(160), stream=stream0)
        del primals_283
        del primals_284
        buf479 = reinterpret_tensor(buf470, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf470  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_75.run(buf471, buf475, buf476, primals_83, primals_84, buf479, 62720, grid=grid(62720), stream=stream0)
        del primals_84
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf481 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf480, buf481, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf482 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf483 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf484 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf481, buf482, buf483, buf484, 3840, 98, grid=grid(3840), stream=stream0)
        buf485 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf486 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf488 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf482, buf483, buf484, primals_286, primals_287, buf485, buf486, buf488, primals_286, primals_287, 960, 4, grid=grid(960), stream=stream0)
        del primals_286
        del primals_287
        buf490 = reinterpret_tensor(buf480, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf480  # reuse
        buf595 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79.run(buf481, buf485, buf486, primals_85, primals_86, buf490, buf595, 376320, grid=grid(376320), stream=stream0)
        del primals_86
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf491, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf492 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf491, buf492, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf493 = buf484; del buf484  # reuse
        buf494 = buf483; del buf483  # reuse
        buf495 = buf482; del buf482  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf492, buf493, buf494, buf495, 3840, 98, grid=grid(3840), stream=stream0)
        buf496 = buf486; del buf486  # reuse
        buf497 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf499 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf493, buf494, buf495, primals_289, primals_290, buf496, buf497, buf499, primals_289, primals_290, 960, 4, grid=grid(960), stream=stream0)
        del primals_289
        del primals_290
        buf501 = reinterpret_tensor(buf491, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf491  # reuse
        buf594 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_238, x_241], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79.run(buf492, buf496, buf497, primals_87, primals_88, buf501, buf594, 376320, grid=grid(376320), stream=stream0)
        del primals_88
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf503 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf502, buf503, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf504 = buf474; del buf474  # reuse
        buf505 = buf473; del buf473  # reuse
        buf506 = buf472; del buf472  # reuse
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf503, buf504, buf505, buf506, 640, 98, grid=grid(640), stream=stream0)
        buf507 = buf476; del buf476  # reuse
        buf508 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf510 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf504, buf505, buf506, primals_292, primals_293, buf507, buf508, buf510, primals_292, primals_293, 160, 4, grid=grid(160), stream=stream0)
        del primals_292
        del primals_293
        buf511 = reinterpret_tensor(buf502, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf502  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf503, buf507, buf508, primals_89, primals_90, buf479, buf511, 62720, grid=grid(62720), stream=stream0)
        del primals_90
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf513 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf512, buf513, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf514 = buf495; del buf495  # reuse
        buf515 = buf494; del buf494  # reuse
        buf516 = buf493; del buf493  # reuse
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf513, buf514, buf515, buf516, 3840, 98, grid=grid(3840), stream=stream0)
        buf517 = buf497; del buf497  # reuse
        buf518 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf520 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf514, buf515, buf516, primals_295, primals_296, buf517, buf518, buf520, primals_295, primals_296, 960, 4, grid=grid(960), stream=stream0)
        del primals_295
        del primals_296
        buf522 = reinterpret_tensor(buf512, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf512  # reuse
        buf593 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79.run(buf513, buf517, buf518, primals_91, primals_92, buf522, buf593, 376320, grid=grid(376320), stream=stream0)
        del primals_92
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf523, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf524 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf523, buf524, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf525 = buf516; del buf516  # reuse
        buf526 = buf515; del buf515  # reuse
        buf527 = buf514; del buf514  # reuse
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf524, buf525, buf526, buf527, 3840, 98, grid=grid(3840), stream=stream0)
        buf528 = buf518; del buf518  # reuse
        buf529 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf531 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf525, buf526, buf527, primals_298, primals_299, buf528, buf529, buf531, primals_298, primals_299, 960, 4, grid=grid(960), stream=stream0)
        del primals_298
        del primals_299
        buf533 = reinterpret_tensor(buf523, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf523  # reuse
        buf592 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_255, x_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79.run(buf524, buf528, buf529, primals_93, primals_94, buf533, buf592, 376320, grid=grid(376320), stream=stream0)
        del primals_94
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf535 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf534, buf535, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf536 = buf506; del buf506  # reuse
        buf537 = buf505; del buf505  # reuse
        buf538 = buf504; del buf504  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_73.run(buf535, buf536, buf537, buf538, 640, 98, grid=grid(640), stream=stream0)
        buf539 = buf508; del buf508  # reuse
        buf540 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf542 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_74.run(buf536, buf537, buf538, primals_301, primals_302, buf539, buf540, buf542, primals_301, primals_302, 160, 4, grid=grid(160), stream=stream0)
        del buf536
        del buf537
        del buf538
        del primals_301
        del primals_302
        buf543 = reinterpret_tensor(buf534, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf534  # reuse
        # Source Nodes: [shortcut_16, x_261], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf535, buf539, buf540, primals_95, primals_96, buf511, buf543, 62720, grid=grid(62720), stream=stream0)
        del buf540
        del primals_96
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf544, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf545 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf544, buf545, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf546 = buf527; del buf527  # reuse
        buf547 = buf526; del buf526  # reuse
        buf548 = buf525; del buf525  # reuse
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf545, buf546, buf547, buf548, 3840, 98, grid=grid(3840), stream=stream0)
        buf549 = buf529; del buf529  # reuse
        buf550 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf552 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf546, buf547, buf548, primals_304, primals_305, buf549, buf550, buf552, primals_304, primals_305, 960, 4, grid=grid(960), stream=stream0)
        del primals_304
        del primals_305
        buf554 = reinterpret_tensor(buf544, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf544  # reuse
        buf591 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_267, x_270], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79.run(buf545, buf549, buf550, primals_97, primals_98, buf554, buf591, 376320, grid=grid(376320), stream=stream0)
        del primals_98
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf554, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf555, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf556 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_76.run(buf555, buf556, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf557 = buf548; del buf548  # reuse
        buf558 = buf547; del buf547  # reuse
        buf559 = buf546; del buf546  # reuse
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf556, buf557, buf558, buf559, 3840, 98, grid=grid(3840), stream=stream0)
        buf560 = buf550; del buf550  # reuse
        buf561 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf563 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_78.run(buf557, buf558, buf559, primals_307, primals_308, buf560, buf561, buf563, primals_307, primals_308, 960, 4, grid=grid(960), stream=stream0)
        del buf557
        del buf558
        del buf559
        del primals_307
        del primals_308
        buf565 = reinterpret_tensor(buf555, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf555  # reuse
        buf590 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_272, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_hardtanh_backward_79.run(buf556, buf560, buf561, primals_99, primals_100, buf565, buf590, 376320, grid=grid(376320), stream=stream0)
        del buf561
        del primals_100
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 320, 7, 7), (15680, 49, 7, 1))
        buf567 = empty_strided((8, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf566, buf567, 2560, 49, grid=grid(2560, 49), stream=stream0)
        buf568 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf569 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        buf570 = empty_strided((1, 320, 1, 1, 4), (1280, 1, 1280, 1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf567, buf568, buf569, buf570, 1280, 98, grid=grid(1280), stream=stream0)
        buf571 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf572 = empty_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda', dtype=torch.float32)
        buf574 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf568, buf569, buf570, primals_310, primals_311, buf571, buf572, buf574, primals_310, primals_311, 320, 4, grid=grid(320), stream=stream0)
        del primals_310
        del primals_311
        buf575 = reinterpret_tensor(buf566, (8, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf566  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_84.run(buf567, buf571, buf572, primals_101, primals_102, buf575, 125440, grid=grid(125440), stream=stream0)
        del buf572
        del primals_102
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 1280, 7, 7), (62720, 49, 7, 1))
        buf577 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf576, buf577, 10240, 49, grid=grid(10240, 49), stream=stream0)
        buf578 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf579 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        buf580 = empty_strided((1, 1280, 1, 1, 4), (5120, 1, 5120, 5120, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf577, buf578, buf579, buf580, 5120, 98, grid=grid(5120), stream=stream0)
        buf581 = reinterpret_tensor(buf570, (1, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf570  # reuse
        buf582 = reinterpret_tensor(buf569, (1, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf569  # reuse
        buf584 = reinterpret_tensor(buf568, (1280, ), (1, ), 0); del buf568  # reuse
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf578, buf579, buf580, primals_313, primals_314, buf581, buf582, buf584, primals_313, primals_314, 1280, 4, grid=grid(1280), stream=stream0)
        del buf578
        del buf579
        del buf580
        del primals_313
        del primals_314
        buf585 = reinterpret_tensor(buf576, (8, 1280, 7, 7), (62720, 1, 8960, 1280), 0); del buf576  # reuse
        buf589 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward]
        triton_poi_fused__native_batch_norm_legit_functional_hardtanh_backward_88.run(buf577, buf581, buf582, primals_103, primals_104, buf585, buf589, 501760, grid=grid(501760), stream=stream0)
        del buf582
        del primals_104
        buf586 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf587 = reinterpret_tensor(buf586, (8, 1280), (1280, 1), 0); del buf586  # reuse
        # Source Nodes: [x_288, x_289, x_291], Original ATen: [aten.hardtanh, aten.mean, aten.view]
        triton_per_fused_hardtanh_mean_view_89.run(buf587, buf585, 10240, 49, grid=grid(10240), stream=stream0)
        del buf585
        buf588 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf587, reinterpret_tensor(primals_157, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf588)
        del primals_158
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_159, primals_159, 1, grid=grid(1), stream=stream0)
        del primals_159
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_162, primals_162, 1, grid=grid(1), stream=stream0)
        del primals_162
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_165, primals_165, 1, grid=grid(1), stream=stream0)
        del primals_165
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_168, primals_168, 1, grid=grid(1), stream=stream0)
        del primals_168
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_171, primals_171, 1, grid=grid(1), stream=stream0)
        del primals_171
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_174, primals_174, 1, grid=grid(1), stream=stream0)
        del primals_174
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_177, primals_177, 1, grid=grid(1), stream=stream0)
        del primals_177
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_180, primals_180, 1, grid=grid(1), stream=stream0)
        del primals_180
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_183, primals_183, 1, grid=grid(1), stream=stream0)
        del primals_183
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_186, primals_186, 1, grid=grid(1), stream=stream0)
        del primals_186
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_189, primals_189, 1, grid=grid(1), stream=stream0)
        del primals_189
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_192, primals_192, 1, grid=grid(1), stream=stream0)
        del primals_192
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_195, primals_195, 1, grid=grid(1), stream=stream0)
        del primals_195
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_198, primals_198, 1, grid=grid(1), stream=stream0)
        del primals_198
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_201, primals_201, 1, grid=grid(1), stream=stream0)
        del primals_201
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_204, primals_204, 1, grid=grid(1), stream=stream0)
        del primals_204
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_207, primals_207, 1, grid=grid(1), stream=stream0)
        del primals_207
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_210, primals_210, 1, grid=grid(1), stream=stream0)
        del primals_210
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_213, primals_213, 1, grid=grid(1), stream=stream0)
        del primals_213
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_216, primals_216, 1, grid=grid(1), stream=stream0)
        del primals_216
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_219, primals_219, 1, grid=grid(1), stream=stream0)
        del primals_219
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_222, primals_222, 1, grid=grid(1), stream=stream0)
        del primals_222
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_90.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        return (buf588, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, buf0, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, buf1, buf3, buf13, buf15, buf17, buf27, buf29, buf31, buf41, buf42, buf44, buf54, buf56, buf58, buf68, buf70, buf72, buf82, buf83, buf85, buf95, buf97, buf99, buf109, buf111, buf113, buf123, buf124, buf126, buf136, buf138, buf140, buf147, buf149, buf151, buf158, buf159, buf161, buf168, buf170, buf172, buf179, buf181, buf183, buf190, buf191, buf193, buf200, buf202, buf204, buf211, buf213, buf215, buf222, buf223, buf225, buf232, buf234, buf236, buf243, buf245, buf247, buf254, buf255, buf257, buf264, buf266, buf268, buf275, buf277, buf279, buf286, buf287, buf289, buf296, buf298, buf300, buf307, buf309, buf311, buf318, buf319, buf321, buf328, buf330, buf332, buf339, buf341, buf343, buf350, buf351, buf353, buf360, buf362, buf364, buf371, buf373, buf375, buf382, buf383, buf385, buf392, buf394, buf396, buf403, buf405, buf407, buf414, buf415, buf417, buf424, buf426, buf428, buf435, buf437, buf439, buf446, buf447, buf449, buf456, buf458, buf460, buf467, buf469, buf471, buf478, buf479, buf481, buf488, buf490, buf492, buf499, buf501, buf503, buf510, buf511, buf513, buf520, buf522, buf524, buf531, buf533, buf535, buf542, buf543, buf545, buf552, buf554, buf556, buf563, buf565, buf567, buf574, buf575, buf577, buf584, buf587, reinterpret_tensor(primals_157, (1000, 1280), (1280, 1), 0), buf589, reinterpret_tensor(buf581, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf571, (1, 320, 1, 1), (320, 1, 1, 1), 0), buf590, reinterpret_tensor(buf560, (1, 960, 1, 1), (960, 1, 1, 1), 0), buf591, reinterpret_tensor(buf549, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf539, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf592, reinterpret_tensor(buf528, (1, 960, 1, 1), (960, 1, 1, 1), 0), buf593, reinterpret_tensor(buf517, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf507, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf594, reinterpret_tensor(buf496, (1, 960, 1, 1), (960, 1, 1, 1), 0), buf595, reinterpret_tensor(buf485, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf475, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf596, reinterpret_tensor(buf464, (1, 576, 1, 1), (576, 1, 1, 1), 0), buf597, reinterpret_tensor(buf453, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf443, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf598, reinterpret_tensor(buf432, (1, 576, 1, 1), (576, 1, 1, 1), 0), buf599, reinterpret_tensor(buf421, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf411, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf600, reinterpret_tensor(buf400, (1, 576, 1, 1), (576, 1, 1, 1), 0), buf601, reinterpret_tensor(buf389, (1, 576, 1, 1), (576, 1, 1, 1), 0), reinterpret_tensor(buf379, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf602, reinterpret_tensor(buf368, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf603, reinterpret_tensor(buf357, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf347, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf604, reinterpret_tensor(buf336, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf605, reinterpret_tensor(buf325, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf606, reinterpret_tensor(buf304, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf607, reinterpret_tensor(buf293, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf283, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf608, reinterpret_tensor(buf272, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf609, reinterpret_tensor(buf261, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf251, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf610, reinterpret_tensor(buf240, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf611, reinterpret_tensor(buf229, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf612, reinterpret_tensor(buf208, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf613, reinterpret_tensor(buf197, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf614, reinterpret_tensor(buf176, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf615, reinterpret_tensor(buf165, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf155, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf616, reinterpret_tensor(buf144, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf617, reinterpret_tensor(buf133, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf120, (1, 24, 1, 1), (24, 1, 1, 1), 0), buf618, reinterpret_tensor(buf106, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf619, reinterpret_tensor(buf92, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf79, (1, 24, 1, 1), (24, 1, 1, 1), 0), buf620, reinterpret_tensor(buf65, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf621, reinterpret_tensor(buf51, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf38, (1, 16, 1, 1), (16, 1, 1, 1), 0), buf622, reinterpret_tensor(buf24, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf623, reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


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
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_77 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((160, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((320, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_169 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_172 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_175 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_178 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_181 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_184 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_187 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_190 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_193 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_196 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_199 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_274 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    compiled_module_main('mobilenetv2_100', benchmark_compiled_module)
