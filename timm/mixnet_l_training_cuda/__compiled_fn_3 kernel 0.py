
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


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uqmiw35rvrh4phhpie6ndljzmn7bpawlfbd7sieefcjwofoj22.py
# Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_1 => add_15
# x_12 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
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
    tmp6 = 1e-05
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
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => var_mean_3
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
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => var_mean_3
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


# kernel path: /tmp/torchinductor_youkaichao/vg/cvg5v23godroqmaxrgxug63mifhr7fau6d3gkkokdp4iifnhe3mg.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => add_17, add_18, add_19, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
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


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5xolbcoubnwvpupnkhvxfpjfbhmcpfvgnfdal5vbjafmpjzr34.py
# Source Nodes: [x_19, x_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_19 => add_17, add_20, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_22 => relu_2
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


# kernel path: /tmp/torchinductor_youkaichao/jb/cjb3vmkns2kttuq535vlakbnoygtxkdjnfqgdov4blvov4kjrlco.py
# Source Nodes: [cat_80], Original ATen: [aten.cat]
# cat_80 => cat_1
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ep/cepxkabmrbuhc5yeuvexxy23pya44hvb3rcu2fna2t4rzzhwtujk.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/um/cum67h6l2ksviz5hjwi3wsxvadtbvnsb3z5se3sqtb6zts2xsiio.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/in/cinen2xi2dz3yf2y6bkemnghl54g6xbw6pb57dwkh4ngf2otvh6x.py
# Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
# x_25 => add_22, add_23, add_24, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/vd/cvddmzbignsqwralr4jzsxldfmf73iqafquabmw673n4gpf5xdcg.py
# Source Nodes: [x_25, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_25 => add_22, add_25, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_28 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7c4ytnt5q7374lu5lunbpkeeagede7wlnihhsprge223xwtbuo.py
# Source Nodes: [cat_79], Original ATen: [aten.cat]
# cat_79 => cat_2
triton_poi_fused_cat_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xw/cxw3wc2u76yxqlw4cgsw3a6rfpbkoe62q4rkl6hq4h575o743a5n.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3eu3w6frd52lpeoy43yhz66rv7lsnkgz6sbcqnh6ov4efeselg.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ki/ckigxs6hpzg5ytq6xm6l2y5paxya5mbqctrfarzjh6jgxwnd5gue.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => add_27, add_28, add_29, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/3o/c3oflf3owjtlvf2hljpgkloxdsh6ffob5u4xw2y4pcxjbey25drl.py
# Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
# x_32 => add_27, add_30, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csofcushvfkylwusprxwzwbvd7ztqyur5k3ntahqjiwhmilnhbeo.py
# Source Nodes: [cat_78], Original ATen: [aten.cat]
# cat_78 => cat_3
triton_poi_fused_cat_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjsm7appnhlpovkhl2csg5nobt3orr5kinbuoyv7zyhagfizca4.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwa27k35ii4qb37reitjn3kr3nlunjlgy6ppndqkuzfapoc2e3d.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sc/cscqrnosnnl6itusftd5l7ntkprmp3c2fz6bgbtdfmxosl55qnxa.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => add_32, add_33, add_34, mul_43, mul_44, mul_45, mul_46, mul_47, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/36/c36fn262l7zewoqf6f57j4vwabxsv3owngflpfcc5beaaedzbv4c.py
# Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_38 => add_32, add_35, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_41 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_29', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxldcif6kmuma4armbtjoyx6piggsx7f3hh3deom7zjjv6dsrk2.py
# Source Nodes: [x_42], Original ATen: [aten.convolution]
# x_42 => convolution_12
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciysz2jhlqp6ck7xz26g5ooth4g546hfglmlmicbnqrn3ioyev6n.py
# Source Nodes: [x_43, x_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_43 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# x_46 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yl/cylpfgays3z5qgepbismzv7ajy5fyvvosxl3cdxwrv2cepdodghe.py
# Source Nodes: [shortcut_3, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_46
# x_50 => add_42, add_45, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vm/cvmcxpbtv36xe3loq4g5iwz6gyzamyvhzh5nidlv3s7afblbbepz.py
# Source Nodes: [x_55], Original ATen: [aten.convolution]
# x_55 => convolution_15
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhsnrqxkpyouxsvyp3yzsokbkryceu66ean5mrkm2neqlivadgc.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
# x_56 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtaghkgisth62djbezq2yghwozxlurprsg4pjgsedarzdqtjks2.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
# x_56 => var_mean_9
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cy/ccypskahehmzynrwkdbicg2fcu3levwg32o5fuvjl3kvzzulvncp.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
# x_56 => add_48, add_49, add_50, mul_64, mul_65, mul_66, mul_67, mul_68, rsqrt_9, squeeze_28, var_mean_9
triton_per_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/zu/czuwm7hym2cfyynufakgsyci7fv5q3txiaydt4ovmuuf6ej6726l.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_56 => add_48, add_51, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_37', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zijd2csi3rnco6rojimximvx4xjxb7cx3qxotxkkxao4t777uk.py
# Source Nodes: [x_59], Original ATen: [aten.silu]
# x_59 => mul_70, sigmoid
triton_poi_fused_silu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_38', 'mutated_arg_names': []},
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
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (240*x2) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czy7idxxwt7v3jfthxn7zcwljgfn4ityfszx6afmdngiik363tp6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____0___conv_dw_0 => convolution_16
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curzvfwfgnnf3vqzggnh7le7t3tpkcogwqucxacmgbtrrtv5udpt.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____0___conv_dw_1 => convolution_17
triton_poi_fused_convolution_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (188160 + x2 + (3136*y0) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3wmnzftsvjyl7ekpbpuifaklhssq24rurqgw24mkvzc6ml2l7t.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____0___conv_dw_2 => convolution_18
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (376320 + x2 + (3136*y0) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmdbfvckgkc4zoplhfsbbwxzyrcle5zq2rmfyxktxkvylfx5jhe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____0___conv_dw_3 => convolution_19
triton_poi_fused_convolution_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (564480 + x2 + (3136*y0) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxknuy7udiqxbdraie3t76u4nj3q5teqpw7yvdq4kslogoe7wfs.py
# Source Nodes: [cat_76], Original ATen: [aten.cat]
# cat_76 => cat_5
triton_poi_fused_cat_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_43', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/c4/cc46bwb2ypbeujvas76guhwtye7mqkz4swii3d2a7ittda3exfo2.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbk2nxbmatga73jpj2prwglrl7k3gog3hckyfxal4rd4mzgzlla.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => add_53, add_54, add_55, mul_72, mul_73, mul_74, mul_75, mul_76, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kejta55jes6r3j2rsgvn3wnp2mwea2fvyanu7oalbn5jm7ywqc.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => add_53, add_56, mul_71, mul_77, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnyxapixdcfzuurg6ce2t6sr3hhhuaa5zmdsxnjy4tw2xxtede7.py
# Source Nodes: [x_65, x_se], Original ATen: [aten.mean, aten.silu]
# x_65 => mul_78, sigmoid_1
# x_se => mean
triton_red_fused_mean_silu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_47', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2fst5djg26jhetst2mfstnm5lh7plbnhczubxygteeq4emxakj.py
# Source Nodes: [x_65, x_se], Original ATen: [aten.mean, aten.silu]
# x_65 => mul_78, sigmoid_1
# x_se => mean
triton_per_fused_mean_silu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_48', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/ur/cur5ukcyrl7bbbf4tmkrd6c3w4qhrdrpc42th6rp3tyui4jz5j46.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
# x_se_1 => convolution_20
# x_se_2 => mul_79, sigmoid_2
triton_poi_fused_convolution_silu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_49', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7zggt4ljsjw72nu374xc2kksq6js7jwhawsk633hunqgfhvx5i.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_21
triton_poi_fused_convolution_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/iu/ciugokn5gvo5wbwe2fwx76ts2dfwkh5egqtky2ff4hmubcyukji7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_65, x_66], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
# x_65 => mul_78, sigmoid_1
# x_66 => mul_80
triton_poi_fused_mul_sigmoid_silu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_51', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ee/ceedne4wpq4tkcm45rbpdsdryjvr7jbx6twklge42czlqg2kvqsu.py
# Source Nodes: [x_67], Original ATen: [aten.convolution]
# x_67 => convolution_22
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4s/c4snxn5zizmibovozgtjyxcyc4mqf5oihrgg6vuvkqg4ib2ddtfc.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ai/caivk3btqicgf3zppjyusxtyv6zrny2wtxityzdkj2dmttfkvcme.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => add_58, add_59, add_60, mul_82, mul_83, mul_84, mul_85, mul_86, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ha/chaqazxqa2fdlccwvhsdww2rhiii536cwe75noyxfvbucvrpcgpu.py
# Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
# x_68 => add_58, add_61, mul_81, mul_87, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czw6dcghhvlsuidlterdwlh5bxlr5anelh7jsharsb5jriffkugw.py
# Source Nodes: [cat_75], Original ATen: [aten.cat]
# cat_75 => cat_6
triton_poi_fused_cat_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_56', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rq/crqoeu62qil5ee644p6lw357yxupsgjdfznikly5ymau5ydwpjkf.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
# x_74 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcme6lcbog7hysnbfcjiuzx2nuc7hhcl52ll4vjbn3ynkvb5v54.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
# x_74 => add_63, add_64, add_65, mul_89, mul_90, mul_91, mul_92, mul_93, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/nt/cntm45fwwusccgwaasef74cajoduobeukjq6an4lyqqtqughxpdi.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_74 => add_63, add_66, mul_88, mul_94, rsqrt_12, sub_12, var_mean_12
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_59', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/y3/cy3tjer72ilyvdllz4v32ge54drxxn7lttthu5jmcremy6dbrl3l.py
# Source Nodes: [x_77], Original ATen: [aten.silu]
# x_77 => mul_95, sigmoid_4
triton_poi_fused_silu_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_60', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zy/czynlfv34kadyruhpdkb3mk4o5t4baxmiueuzyp7ku4lfnao4ki4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_dw_0 => convolution_25
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


# kernel path: /tmp/torchinductor_youkaichao/ig/cigqw4v4jm5vxrohfjycmeyc2cimmrnxsoen5xqinirp2s5tsl5u.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____1___conv_dw_1 => convolution_26
triton_poi_fused_convolution_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_62', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5tvxmlacrahyhrradwhn72r3f2tpuufyu7wshuzp54u3ru7ls7.py
# Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
# x_80 => add_68, add_71, mul_102, mul_96, rsqrt_13, sub_13, var_mean_13
triton_poi_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyj5v6k24zgwmtojmovuj4nirroiuqktv2iopeqyghvdh4kpogij.py
# Source Nodes: [x_83, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_83 => mul_103, sigmoid_5
# x_se_4 => mean_1
triton_red_fused_mean_silu_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_64', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ly/clyztutx5rbjgcaqd72geatkb44rgzie3kppyxgsyiiuhyvhuw2k.py
# Source Nodes: [x_83, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_83 => mul_103, sigmoid_5
# x_se_4 => mean_1
triton_per_fused_mean_silu_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_65', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/6e/c6eypngrfmwhkwhplf3l7uxxytgy2k7lx67ra5u7woapdbvzzjlr.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
# x_se_5 => convolution_27
# x_se_6 => mul_104, sigmoid_6
triton_poi_fused_convolution_silu_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_66', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqbwontaree3hagkwozhytgvjif4msegqkumelh2yqb44dmosmc.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_28
triton_poi_fused_convolution_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_67', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/su/csu4ar7sgr6f3f5ikfoi27c4w6rzducob5dw6hbh64y3fngnfv72.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_83, x_84], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_7
# x_83 => mul_103, sigmoid_5
# x_84 => mul_105
triton_poi_fused_mul_sigmoid_silu_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_68', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4an76pmdkvihqpi4zqyhyzuqh2ex2ffk2zfqmnpk2didtyebir.py
# Source Nodes: [cat_73], Original ATen: [aten.cat]
# cat_73 => cat_8
triton_poi_fused_cat_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kqea3mn5cnncczzwjamdkt2eukgse2yhg3rjrreiyqfwzzumwf.py
# Source Nodes: [shortcut_5, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_5 => add_77
# x_87 => add_73, add_76, mul_106, mul_112, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_add_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_70', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cse46byw7xzfrkkrjxpjtuc6lzl47v4m4mxcicpeovmw2xumaou4.py
# Source Nodes: [x_132], Original ATen: [aten.convolution]
# x_132 => convolution_47
triton_poi_fused_convolution_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_71', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mo/cmoq5d2hamzknb2fe3fza4wcj5p7mg44d2kbe2dgnoyn4ltxdkli.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____0___conv_dw_0 => convolution_48
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y3/cy35lfjckeaaubfiajk6wwhusm54dxmyze7cmb6vhim253k6gq6t.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____0___conv_dw_1 => convolution_49
triton_poi_fused_convolution_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (87808 + x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7sezv2mwvucv37fjyvoo4nfn5fzwrstc6nkcxiphhoizs3g5lkn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____0___conv_dw_2 => convolution_50
triton_poi_fused_convolution_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (175616 + x2 + (784*y0) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (87808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgt57amadp5zmdfucxsfbv4uidav2fqwrlezpvueubueu26s44q.py
# Source Nodes: [cat_66], Original ATen: [aten.cat]
# cat_66 => cat_15
triton_poi_fused_cat_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_75', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/55/c55glkwtezpcukxkvuyzxdf77ue5xs4tykok54vg3von4br4x46l.py
# Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
# x_139 => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zk/czkfmgkmhpdoey6ghrfyplx7waix665asdxqcldwvkrjywgzed3q.py
# Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
# x_139 => add_116, add_117, add_118, mul_172, mul_173, mul_174, mul_175, mul_176, rsqrt_22, squeeze_67, var_mean_22
triton_per_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ys/cysphn2u2it4f5aqnqd2swttx2qgyhzkf4dr7tvz5tg5yg6sn5du.py
# Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
# x_139 => add_116, add_119, mul_171, mul_177, rsqrt_22, sub_22, var_mean_22
triton_poi_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xuspupvhacudxtemgficnfxwsaxcn2nhsuphttvnkhe4423b3h.py
# Source Nodes: [x_142, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_142 => mul_178, sigmoid_17
# x_se_16 => mean_4
triton_red_fused_mean_silu_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_79', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rp/crp6o7jtggcq2rwy3lzshveailn5vjhht4wbaajgfog4onfaopx6.py
# Source Nodes: [x_142, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_142 => mul_178, sigmoid_17
# x_se_16 => mean_4
triton_per_fused_mean_silu_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_80', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4d7cdcaiyhj3fhgovujv5wkhvfmkmhkxsy463gsfk5cmrbcwaz.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
# x_se_17 => convolution_51
# x_se_18 => mul_179, sigmoid_18
triton_poi_fused_convolution_silu_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_81', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/sh/cshhhurk2p64x6wol4qbzjgixgij3nk4f5ymrmt2fg47duusr2hr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_142, x_143], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
# x_142 => mul_178, sigmoid_17
# x_143 => mul_180
triton_poi_fused_mul_sigmoid_silu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_82', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jk/cjk2got5s7clujvsqrdchzxjtarqdh5w2zfpbkjuw4zym4vwpyo5.py
# Source Nodes: [x_144], Original ATen: [aten.convolution]
# x_144 => convolution_53
triton_poi_fused_convolution_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_83', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gv/cgv4aumgvr27ywtmvaybgjfncdqblbadrom2ny4ota24ybrlvrma.py
# Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
# x_145 => var_mean_23
triton_red_fused__native_batch_norm_legit_functional_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxyiqyhs4427eyazxmny5jv3oom6g7rvnszlchotote7pd3ajwo.py
# Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
# x_145 => add_121, add_122, add_123, mul_182, mul_183, mul_184, mul_185, mul_186, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_85', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/dj/cdj5jeo7vmdonfyxvpwc2o7xnqmbskkarfvzlpfa3g4ek2gw5yvl.py
# Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
# x_145 => add_121, add_124, mul_181, mul_187, rsqrt_23, sub_23, var_mean_23
triton_poi_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhbisozhhbhkgkh5m2t3milvm3ov2vqtqtac5jpf5ifx6dkogr7.py
# Source Nodes: [cat_65], Original ATen: [aten.cat]
# cat_65 => cat_16
triton_poi_fused_cat_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_87', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fa/cfaopggxxy7tzoj5myzzntgs53ks67dhllhxahp7x2abbhfvfgtu.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6b/c6btkxxciax2f6mkl6ejnj3tmwpidu2lagt4l6ueecyfrxruk7y5.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => add_126, add_127, add_128, mul_189, mul_190, mul_191, mul_192, mul_193, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_89', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzwufk3oo67iyl3v5yupczbkatbmh47yrg52ykkbbcsl73nqfj6.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_151 => add_126, add_129, mul_188, mul_194, rsqrt_24, sub_24, var_mean_24
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_90', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqr77txubvz2o3nnzumg2wann45so4mxozcco4fg6zizc6wrzjje.py
# Source Nodes: [x_154], Original ATen: [aten.silu]
# x_154 => mul_195, sigmoid_20
triton_poi_fused_silu_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_91', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ur/curghvumgjpbdvwgyxwruxvu4xnzsrcoohbxnogvvkxyohyvv5cs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_0 => convolution_56
triton_poi_fused_convolution_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_92', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czvedrsvpzb4jcdxz3ciiiy2a2segd6q5khit2ml2evottnqnuop.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_1 => convolution_57
triton_poi_fused_convolution_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_93', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3qloq3myzvdapzlxnklatje3wkdbw6u2j54ou6lij2gne7pg4k.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_2 => convolution_58
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
    tmp0 = tl.load(in_ptr0 + (61152 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csaznbrjg3ejmk42mjdtvn2n22st3gees2ynxjjrbx5yudd4hpeo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_dw_3 => convolution_59
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
    tmp0 = tl.load(in_ptr0 + (91728 + x2 + (196*y0) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (156*x2) + (30576*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceuzle7ts2f7h4m7xjzyheotsszhxs3vxjq7qsk7l2mhpvedcdlx.py
# Source Nodes: [cat_64], Original ATen: [aten.cat]
# cat_64 => cat_17
triton_poi_fused_cat_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_96', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mg/cmg3wfs6sshjrjd7rwpgqgqx7ez5jcs24l544ooeixwyv665n7jl.py
# Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
# x_157 => add_131, add_134, mul_196, mul_202, rsqrt_25, sub_25, var_mean_25
triton_poi_fused__native_batch_norm_legit_functional_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_97', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jx5vby5mshwbdjbymm4uut75ddzm4ri5vznqk2cirjocdoomtu.py
# Source Nodes: [x_160, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_160 => mul_203, sigmoid_21
# x_se_20 => mean_5
triton_red_fused_mean_silu_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlumjf5uw3jznyllmfwtsxrlkt366tbk2x22x2hgedl4jgnkol4.py
# Source Nodes: [x_160, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_160 => mul_203, sigmoid_21
# x_se_20 => mean_5
triton_per_fused_mean_silu_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_99', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/x6/cx643dx6y7dq4hdppukntalfg6ynju2cwlr4b3mkruksvcgb3dmp.py
# Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
# x_se_21 => convolution_60
# x_se_22 => mul_204, sigmoid_22
triton_poi_fused_convolution_silu_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_100', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvqp5a5q5pevs3yuarffvjvfkruubbcrfihb3mg5tdso3iai3f3.py
# Source Nodes: [x_se_23], Original ATen: [aten.convolution]
# x_se_23 => convolution_61
triton_poi_fused_convolution_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_101', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbfgydyzmgrgp2w7egukcejwvwguookmsh6ohqwnzv56p73vlgu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_160, x_161], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_23
# x_160 => mul_203, sigmoid_21
# x_161 => mul_205
triton_poi_fused_mul_sigmoid_silu_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_102', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkmvpbjdplfwrihny3qbksmx7o7if7p7ljruvmz53mtcvcrhn3k.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 => convolution_62
triton_poi_fused_convolution_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_103', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/lo/clo3unsszndfoex7nrejbaut6pmujwrcavlpxqpiv52icawlwmsx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1 => convolution_63
triton_poi_fused_convolution_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_104', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/g6/cg675iqffuljb6656ahmbjuznnrjm4iuqff7h2ll6fr2ztiznrre.py
# Source Nodes: [cat_63], Original ATen: [aten.cat]
# cat_63 => cat_18
triton_poi_fused_cat_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_105', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ab/cabj6vrfsbg4xjebrcdlzsb4y6sdo5q3axpwieyggkk5pm6ogput.py
# Source Nodes: [shortcut_9, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_9 => add_140
# x_164 => add_136, add_139, mul_206, mul_212, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_add_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_106', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxx5qypgw6g4ntwzggxzkvyve6sqe2srbstiyyg425vqlbuufr5.py
# Source Nodes: [x_209], Original ATen: [aten.convolution]
# x_209 => convolution_84
triton_poi_fused_convolution_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_107', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rf/crfxjma2tfmkreqbcpu5b3zy532dfny2o5rxvtxmssnb7kqqak52.py
# Source Nodes: [x_210, x_213], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_210 => add_174, add_177, mul_263, mul_269, rsqrt_33, sub_33, var_mean_33
# x_213 => mul_270, sigmoid_32
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_108', 'mutated_arg_names': []},
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
    tl.store(out_ptr1 + (x2), tmp15, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xeswhmeu6yvadzzfhclfbb2yd4ohh4ypg6jyh2cajx24mgf33w.py
# Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
# x_se_33 => convolution_86
# x_se_34 => mul_279, sigmoid_34
triton_poi_fused_convolution_silu_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_109', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgsabaot5fla5e2kktfi3hdbnhpnrpy52rpgjo5yqv4iketh3vo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_218, x_219], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
# x_218 => mul_278, sigmoid_33
# x_219 => mul_280
triton_poi_fused_mul_sigmoid_silu_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_110', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sd/csdwv3zqkj3sovzno45zc6pns456ip7ajan7w42j767zs2vcrqua.py
# Source Nodes: [x_220], Original ATen: [aten.convolution]
# x_220 => convolution_88
triton_poi_fused_convolution_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_111', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mubon6oeboysjywnm3touvmhvbqhzhtmmnay5ofp5nibt5npjs.py
# Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
# x_221 => var_mean_35
triton_red_fused__native_batch_norm_legit_functional_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_112', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkmmemvw6f4xic4z7j6zd2uwhpi7ayusdyf2upcqmow72x74xx3.py
# Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
# x_221 => add_184, add_185, add_186, mul_282, mul_283, mul_284, mul_285, mul_286, rsqrt_35, squeeze_106, var_mean_35
triton_per_fused__native_batch_norm_legit_functional_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_113', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/lw/clw5uw3dlgpdladfd7kux6yzo764dcj4leixhwase35o2x5tjy65.py
# Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
# x_221 => add_184, add_187, mul_281, mul_287, rsqrt_35, sub_35, var_mean_35
triton_poi_fused__native_batch_norm_legit_functional_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_114', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/ccizsf77zz4ln5iqxrel5kpnehocl6g6ctbqdrn56xhecssmtebw.py
# Source Nodes: [cat_56], Original ATen: [aten.cat]
# cat_56 => cat_25
triton_poi_fused_cat_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_115', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ur/curi2yiopy6c2fpr2rmx5smdh5orwpu5zxdy3nri4e7dpowoaja5.py
# Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
# x_227 => var_mean_36
triton_red_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwlcz5ya5lx3nbfp3q2btyobbo2myepgfp5h27mtu4ydgv36q5j.py
# Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
# x_227 => add_189, add_190, add_191, mul_289, mul_290, mul_291, mul_292, mul_293, rsqrt_36, squeeze_109, var_mean_36
triton_per_fused__native_batch_norm_legit_functional_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_117', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gmanpu27j7sbmp5kpbnmcnwfkywqfiotfgfd2b2nwevwjtmgy5.py
# Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_227 => add_189, add_192, mul_288, mul_294, rsqrt_36, sub_36, var_mean_36
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_118', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5dmx6v62mhfthobbcb64yapeoqn7tps7blstblhjtfcc4yqvwn.py
# Source Nodes: [x_230], Original ATen: [aten.silu]
# x_230 => mul_295, sigmoid_36
triton_poi_fused_silu_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_119', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xo/cxomy33vvn3ot6ma2u6b6pgl2c5mxbnijyzlgolizvz3p6ivzhff.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_0 => convolution_91
triton_poi_fused_convolution_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_120', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35qqyv5qi3pq2ylde3smshbuikijblmgu5pgfdnmemtlxcw5qjz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_1 => convolution_92
triton_poi_fused_convolution_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_121', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/my/cmydunmclelm2d5e6wiwlbq3iawtm5hk7rmeo3of3537gmsh5ugc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_2 => convolution_93
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
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvbdxo4sc75b7hwtssph7ppdxwkwqwrmhlqklvfoqokpuac6sbh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_dw_3 => convolution_94
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
    tmp0 = tl.load(in_ptr0 + (70560 + x2 + (196*y0) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxcy36vzjct2h3yo4sbk5nep6r4rlwsvdzplag4qp44exa4hyz6.py
# Source Nodes: [cat_55], Original ATen: [aten.cat]
# cat_55 => cat_26
triton_poi_fused_cat_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_124', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/55/c55lidy37x6vsynpmi4zfiuqtcee3sksaudkvnv2lng3yhlbcbp2.py
# Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
# x_233 => add_194, add_197, mul_296, mul_302, rsqrt_37, sub_37, var_mean_37
triton_poi_fused__native_batch_norm_legit_functional_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_125', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmodiuxydishxwjbjtazd2h2oxgteugjmlwwp5xhafn7gwlqhlye.py
# Source Nodes: [x_236, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_236 => mul_303, sigmoid_37
# x_se_36 => mean_9
triton_red_fused_mean_silu_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_126', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ea/ceanxzrx7rktzjf2wfb5d6ttnqvtnwr7r53gkmhuwi4ic55dddyd.py
# Source Nodes: [x_236, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_236 => mul_303, sigmoid_37
# x_se_36 => mean_9
triton_per_fused_mean_silu_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_127', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2hjwrz2xwxu2uznwsxwii72tsbnlz5jmgbschpywlsgle3va5k.py
# Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
# x_se_37 => convolution_95
# x_se_38 => mul_304, sigmoid_38
triton_poi_fused_convolution_silu_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_128', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/pm/cpmazg6isnpc3zkvtfcq3dhnrtjgkceltx2y22atkucr2yc2obzm.py
# Source Nodes: [x_se_39], Original ATen: [aten.convolution]
# x_se_39 => convolution_96
triton_poi_fused_convolution_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_129', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/25/c25q3kospmz65c5kz5nabcmucy35iq7ppqmsg7sterc66oie556h.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_236, x_237], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_236 => mul_303, sigmoid_37
# x_237 => mul_305
triton_poi_fused_mul_sigmoid_silu_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_130', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lgymejlbcfou6taduvu2y22uhleatbjoi3dtmfa4rx5u6etrhs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0 => convolution_97
triton_poi_fused_convolution_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_131', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/up/cupxbsxch5zeutoddsmzufu3cu33zgvw2um4dblqsvlb76ftjemo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1 => convolution_98
triton_poi_fused_convolution_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_132', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvzysfg6z4i7wwojnljltzsewifwsl4ohch4d3myv3lgc7ospja.py
# Source Nodes: [cat_54], Original ATen: [aten.cat]
# cat_54 => cat_27
triton_poi_fused_cat_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_133', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tg/ctggmgjambxptbuk4kbpfosbhy2jad74dnwi47h3w7fb4mbl776w.py
# Source Nodes: [shortcut_13, x_240], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_13 => add_203
# x_240 => add_199, add_202, mul_306, mul_312, rsqrt_38, sub_38, var_mean_38
triton_poi_fused__native_batch_norm_legit_functional_add_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_134', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hz/chzib6y2lageazodm2ltm3yw6mgibiacfrfrcv7z64w2dnimlj4x.py
# Source Nodes: [x_285], Original ATen: [aten.convolution]
# x_285 => convolution_119
triton_poi_fused_convolution_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_135', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yiaruv7slef5lvyb7bgnyni6qgr3erg2vyxrhmj54fzphsyzso.py
# Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
# x_286 => var_mean_45
triton_red_fused__native_batch_norm_legit_functional_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_136', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/th/cth3t5shcebqmu6kiysygeinnsevf64ntojmuyparzmt26mzog2a.py
# Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
# x_286 => add_237, add_238, add_239, mul_364, mul_365, mul_366, mul_367, mul_368, rsqrt_45, squeeze_136, var_mean_45
triton_per_fused__native_batch_norm_legit_functional_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_137', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/qr/cqr7tl3fgegxllw3elxor64wmao3jnc3sqb6cacw2b43uucpzs46.py
# Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_286 => add_237, add_240, mul_363, mul_369, rsqrt_45, sub_45, var_mean_45
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_138', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/lh/clhiz7pdalfqo6saoy4ak7os2jtlcvtzuevpsbkc6oemsirley6w.py
# Source Nodes: [x_289], Original ATen: [aten.silu]
# x_289 => mul_370, sigmoid_48
triton_poi_fused_silu_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_139', 'mutated_arg_names': []},
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
    y0 = yindex % 960
    y1 = (yindex // 960)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (960*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nijv2cyqwcizwumwarunyf6ceu4czihjr64udaju6xygxr4tlk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___conv_dw_0 => convolution_120
triton_poi_fused_convolution_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_140', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3li56xabfm2zyvkpwv63enq3fyga2oj5fvbw62xavzksu3d7opj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___conv_dw_1 => convolution_121
triton_poi_fused_convolution_141 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_141', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (47040 + x2 + (196*y0) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3aoa5762eqfoeldfw5gdqple2nhjmpt4lpjp6usanafjiqjstt.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___conv_dw_2 => convolution_122
triton_poi_fused_convolution_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_142', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (94080 + x2 + (196*y0) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chl5s3k7hymzchijdsyrunqlkgjd2jhrwjwtjabobofm6xxcilqy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___conv_dw_3 => convolution_123
triton_poi_fused_convolution_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_143', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (141120 + x2 + (196*y0) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceaioaehnfa4bsiuz5ayib7sn5ksdiwqyx777bpwxwpfdsxwi7ij.py
# Source Nodes: [cat_47], Original ATen: [aten.cat]
# cat_47 => cat_34
triton_poi_fused_cat_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_144', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmxahzfee5bg4adfgkzisutjzcmwllfutox7ktpw5uochog47op.py
# Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
# x_292 => var_mean_46
triton_red_fused__native_batch_norm_legit_functional_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_145', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yv/cyvysgvdkj7bl5bnfjj2oh3svym7c7by4k7bxl3gurfkq2rt6wwu.py
# Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
# x_292 => add_242, add_243, add_244, mul_372, mul_373, mul_374, mul_375, mul_376, rsqrt_46, squeeze_139, var_mean_46
triton_per_fused__native_batch_norm_legit_functional_146 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_146', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pbz2cxj5hrrs6mewbtb3h6us2md5i6szepxy2eya64prbts52w.py
# Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
# x_292 => add_242, add_245, mul_371, mul_377, rsqrt_46, sub_46, var_mean_46
triton_poi_fused__native_batch_norm_legit_functional_147 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_147', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy532rg7j22fdjlo25r2dmefhrpfbl4kab47vtrqcocac6oywwvf.py
# Source Nodes: [x_295, x_se_48], Original ATen: [aten.mean, aten.silu]
# x_295 => mul_378, sigmoid_49
# x_se_48 => mean_12
triton_per_fused_mean_silu_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_148', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5wq3kbtngu4ypuvey6uoykqc5wev74rwer7ehkrib2ammruuzq.py
# Source Nodes: [x_se_51], Original ATen: [aten.convolution]
# x_se_51 => convolution_125
triton_poi_fused_convolution_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_149', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhftb32rzmsabgcj66fha3q6zq4mecycilxlrhwd5dr2cukmoym.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_295, x_296], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
# x_295 => mul_378, sigmoid_49
# x_296 => mul_380
triton_poi_fused_mul_sigmoid_silu_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_150', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqmxiuuczomjceou2mobm5lz2p72kumtwxkgbpzhsis462te5wk.py
# Source Nodes: [x_297], Original ATen: [aten.convolution]
# x_297 => convolution_126
triton_poi_fused_convolution_151 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_151', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcn6i6jqjo3ldamq4zq5y2hndwop7vpucgmaq3ayoo3fxwfzglt.py
# Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
# x_298 => var_mean_47
triton_red_fused__native_batch_norm_legit_functional_152 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_152', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hu/chulgk3b3jcclepa6njnuf456b7nlysemebyqwcoho5lbuc5xhya.py
# Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
# x_298 => add_247, add_248, add_249, mul_382, mul_383, mul_384, mul_385, mul_386, rsqrt_47, squeeze_142, var_mean_47
triton_per_fused__native_batch_norm_legit_functional_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_153', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ai/caiz3vddxansa7o3yc2hbkx6tqoaigu7n3s6oe6kb5ycfoekekgi.py
# Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
# x_298 => add_247, add_250, mul_381, mul_387, rsqrt_47, sub_47, var_mean_47
triton_poi_fused__native_batch_norm_legit_functional_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_154', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctss7tfqyjvzuxoj4vvc3wz676abanqn2m5e5gizg32vpwe3ans7.py
# Source Nodes: [x_302], Original ATen: [aten.convolution]
# x_302 => convolution_127
triton_poi_fused_convolution_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_155', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hysbs74d4w2tvmmdlfp3e5urtzvkawn7mrydd2si7ol67snaa4.py
# Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
# x_303 => var_mean_48
triton_red_fused__native_batch_norm_legit_functional_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_156', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vqivbaqedij56aleb7cr6cb6vpox2sq3hnbno5q5uzwgx7prj6.py
# Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
# x_303 => add_252, add_253, add_254, mul_389, mul_390, mul_391, mul_392, mul_393, rsqrt_48, squeeze_145, var_mean_48
triton_per_fused__native_batch_norm_legit_functional_157 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_157', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ft/cftzrhvxg6nvrcmati4g7h3jdmq3zwiaqbivueqzxud6jvzf7bqk.py
# Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_303 => add_252, add_255, mul_388, mul_394, rsqrt_48, sub_48, var_mean_48
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_158 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_158', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkimlg4hwmgbha7nyme6i3rygngrljsofma6ldhlpajyiu3hrcf.py
# Source Nodes: [x_306], Original ATen: [aten.silu]
# x_306 => mul_395, sigmoid_52
triton_poi_fused_silu_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_159', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/37/c372favlavi6nmf5h4gb7nbtozxxrlmhlsjynj3ocs5iqwb6yyd5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_0 => convolution_128
triton_poi_fused_convolution_160 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_160', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vyilmaxelesdcppruowooxff235bj64vn4vgkjrw55smbiwotv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_1 => convolution_129
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
    tmp0 = tl.load(in_ptr0 + (19404 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxmo5bjqemciownj3ei5qite45kr7hqtrd5fr2zle56jv3nbvmf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_2 => convolution_130
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
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6jgbz6wslqjrjbff4vfzt5ce6knvptn7ufqomsnrcan4qppdgu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_dw_3 => convolution_131
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
    tmp0 = tl.load(in_ptr0 + (58212 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (396*x2) + (19404*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cdsdwon7k2k7d6ha6cxwivwczlh7lmkrohfqaj2vx45wkiv6tefn.py
# Source Nodes: [cat_46], Original ATen: [aten.cat]
# cat_46 => cat_35
triton_poi_fused_cat_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_164', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crhhqbzzzn327x23xyyezudz4pfimue3gkaz3dgssmgytzwc2qyw.py
# Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
# x_309 => add_257, add_260, mul_396, mul_402, rsqrt_49, sub_49, var_mean_49
triton_poi_fused__native_batch_norm_legit_functional_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_165', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/it/citjgrft7lnxswufsihpmvmgyvwvesb4e4a42ebuquhtqmizd6qq.py
# Source Nodes: [x_312, x_se_52], Original ATen: [aten.mean, aten.silu]
# x_312 => mul_403, sigmoid_53
# x_se_52 => mean_13
triton_per_fused_mean_silu_166 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_166', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/rt/crt3ohuxr4xzplqmqsk5eh5vyljomy2yysheuc4jjs3ocguwde6j.py
# Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
# x_se_53 => convolution_132
# x_se_54 => mul_404, sigmoid_54
triton_poi_fused_convolution_silu_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_167', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzmat6r7ddf4wzieky6khppn64y5ynpaskkxf6w6mfqmxw7q5hu.py
# Source Nodes: [x_se_55], Original ATen: [aten.convolution]
# x_se_55 => convolution_133
triton_poi_fused_convolution_168 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_168', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/yy/cyykta3bv3ecbjael2qfx2f6fpmupku4cic5fgjztcwdu7xsfqa4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_312, x_313], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_55
# x_312 => mul_403, sigmoid_53
# x_313 => mul_405
triton_poi_fused_mul_sigmoid_silu_169 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_169', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbv4rqmdiuckvwiiphz3zc5zycu376k43odwojqpnllorx3xts6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0 => convolution_134
triton_poi_fused_convolution_170 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_170', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dp/cdp3puam327psjdrqizdv7g6ueectwkfrqfdu6ns73zaylmsofmu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1 => convolution_135
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
    tmp0 = tl.load(in_ptr0 + (38808 + x2 + (49*y0) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (792*x2) + (38808*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45pzpb2rjxx4aamrcl2bkmnt6daahjp3d3di254qko564cmm3og.py
# Source Nodes: [cat_45], Original ATen: [aten.cat]
# cat_45 => cat_36
triton_poi_fused_cat_172 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_172', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/eg/cegsq3njkslhsmfh4shjdvuviqv6m6emfzxnkq432txnqqie3uwo.py
# Source Nodes: [shortcut_17, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_17 => add_266
# x_316 => add_262, add_265, mul_406, mul_412, rsqrt_50, sub_50, var_mean_50
triton_poi_fused__native_batch_norm_legit_functional_add_173 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_173', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5kigus4ojg6nboavtzjz2dp5r5qmebc4pxc6uifjb5vkwtcxd3.py
# Source Nodes: [x_360], Original ATen: [aten.convolution]
# x_360 => convolution_154
triton_poi_fused_convolution_174 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_174', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rbipv52sepofqeehg5wqacwj5ngixy6dzz2mbfdzrg3x26fker.py
# Source Nodes: [x_361], Original ATen: [aten._native_batch_norm_legit_functional]
# x_361 => var_mean_57
triton_red_fused__native_batch_norm_legit_functional_175 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_175', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xc/cxc4zubgkhlhksuofxqtnla3w4k7c6ouqmaui5yghjvidwwtmmvu.py
# Source Nodes: [x_361], Original ATen: [aten._native_batch_norm_legit_functional]
# x_361 => add_300, add_301, add_302, mul_464, mul_465, mul_466, mul_467, mul_468, rsqrt_57, squeeze_172, var_mean_57
triton_per_fused__native_batch_norm_legit_functional_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_176', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/3d/c3d3pbmp5wwdlc7y7e46vru4uf4md6u2l7d7ulcnnlgsxw35m5v5.py
# Source Nodes: [x_361, x_365], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_361 => add_300, add_303, mul_463, mul_469, rsqrt_57, sub_57, var_mean_57
# x_365 => relu_6
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_177 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_177', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bf/cbfv5oa3bdidksc4tsnkbyizeizkyhicdsa5jyf3wyxfcxyatks4.py
# Source Nodes: [x_366, x_368], Original ATen: [aten.mean, aten.view]
# x_366 => mean_16
# x_368 => view
triton_per_fused_mean_view_178 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_178', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrk6obqtikx5q4qkq2npi6wuaghrdmiqodek5l6hxittmhxlf5l.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_179 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_179', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (192, ), (1, ))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (40, ), (1, ))
    assert_size_stride(primals_12, (40, ), (1, ))
    assert_size_stride(primals_13, (120, ), (1, ))
    assert_size_stride(primals_14, (120, ), (1, ))
    assert_size_stride(primals_15, (120, ), (1, ))
    assert_size_stride(primals_16, (120, ), (1, ))
    assert_size_stride(primals_17, (40, ), (1, ))
    assert_size_stride(primals_18, (40, ), (1, ))
    assert_size_stride(primals_19, (240, ), (1, ))
    assert_size_stride(primals_20, (240, ), (1, ))
    assert_size_stride(primals_21, (240, ), (1, ))
    assert_size_stride(primals_22, (240, ), (1, ))
    assert_size_stride(primals_23, (56, ), (1, ))
    assert_size_stride(primals_24, (56, ), (1, ))
    assert_size_stride(primals_25, (336, ), (1, ))
    assert_size_stride(primals_26, (336, ), (1, ))
    assert_size_stride(primals_27, (336, ), (1, ))
    assert_size_stride(primals_28, (336, ), (1, ))
    assert_size_stride(primals_29, (56, ), (1, ))
    assert_size_stride(primals_30, (56, ), (1, ))
    assert_size_stride(primals_31, (336, ), (1, ))
    assert_size_stride(primals_32, (336, ), (1, ))
    assert_size_stride(primals_33, (336, ), (1, ))
    assert_size_stride(primals_34, (336, ), (1, ))
    assert_size_stride(primals_35, (56, ), (1, ))
    assert_size_stride(primals_36, (56, ), (1, ))
    assert_size_stride(primals_37, (336, ), (1, ))
    assert_size_stride(primals_38, (336, ), (1, ))
    assert_size_stride(primals_39, (336, ), (1, ))
    assert_size_stride(primals_40, (336, ), (1, ))
    assert_size_stride(primals_41, (56, ), (1, ))
    assert_size_stride(primals_42, (56, ), (1, ))
    assert_size_stride(primals_43, (336, ), (1, ))
    assert_size_stride(primals_44, (336, ), (1, ))
    assert_size_stride(primals_45, (336, ), (1, ))
    assert_size_stride(primals_46, (336, ), (1, ))
    assert_size_stride(primals_47, (104, ), (1, ))
    assert_size_stride(primals_48, (104, ), (1, ))
    assert_size_stride(primals_49, (624, ), (1, ))
    assert_size_stride(primals_50, (624, ), (1, ))
    assert_size_stride(primals_51, (624, ), (1, ))
    assert_size_stride(primals_52, (624, ), (1, ))
    assert_size_stride(primals_53, (104, ), (1, ))
    assert_size_stride(primals_54, (104, ), (1, ))
    assert_size_stride(primals_55, (624, ), (1, ))
    assert_size_stride(primals_56, (624, ), (1, ))
    assert_size_stride(primals_57, (624, ), (1, ))
    assert_size_stride(primals_58, (624, ), (1, ))
    assert_size_stride(primals_59, (104, ), (1, ))
    assert_size_stride(primals_60, (104, ), (1, ))
    assert_size_stride(primals_61, (624, ), (1, ))
    assert_size_stride(primals_62, (624, ), (1, ))
    assert_size_stride(primals_63, (624, ), (1, ))
    assert_size_stride(primals_64, (624, ), (1, ))
    assert_size_stride(primals_65, (104, ), (1, ))
    assert_size_stride(primals_66, (104, ), (1, ))
    assert_size_stride(primals_67, (624, ), (1, ))
    assert_size_stride(primals_68, (624, ), (1, ))
    assert_size_stride(primals_69, (624, ), (1, ))
    assert_size_stride(primals_70, (624, ), (1, ))
    assert_size_stride(primals_71, (160, ), (1, ))
    assert_size_stride(primals_72, (160, ), (1, ))
    assert_size_stride(primals_73, (480, ), (1, ))
    assert_size_stride(primals_74, (480, ), (1, ))
    assert_size_stride(primals_75, (480, ), (1, ))
    assert_size_stride(primals_76, (480, ), (1, ))
    assert_size_stride(primals_77, (160, ), (1, ))
    assert_size_stride(primals_78, (160, ), (1, ))
    assert_size_stride(primals_79, (480, ), (1, ))
    assert_size_stride(primals_80, (480, ), (1, ))
    assert_size_stride(primals_81, (480, ), (1, ))
    assert_size_stride(primals_82, (480, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_84, (160, ), (1, ))
    assert_size_stride(primals_85, (480, ), (1, ))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_87, (480, ), (1, ))
    assert_size_stride(primals_88, (480, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_90, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_92, (960, ), (1, ))
    assert_size_stride(primals_93, (960, ), (1, ))
    assert_size_stride(primals_94, (960, ), (1, ))
    assert_size_stride(primals_95, (264, ), (1, ))
    assert_size_stride(primals_96, (264, ), (1, ))
    assert_size_stride(primals_97, (1584, ), (1, ))
    assert_size_stride(primals_98, (1584, ), (1, ))
    assert_size_stride(primals_99, (1584, ), (1, ))
    assert_size_stride(primals_100, (1584, ), (1, ))
    assert_size_stride(primals_101, (264, ), (1, ))
    assert_size_stride(primals_102, (264, ), (1, ))
    assert_size_stride(primals_103, (1584, ), (1, ))
    assert_size_stride(primals_104, (1584, ), (1, ))
    assert_size_stride(primals_105, (1584, ), (1, ))
    assert_size_stride(primals_106, (1584, ), (1, ))
    assert_size_stride(primals_107, (264, ), (1, ))
    assert_size_stride(primals_108, (264, ), (1, ))
    assert_size_stride(primals_109, (1584, ), (1, ))
    assert_size_stride(primals_110, (1584, ), (1, ))
    assert_size_stride(primals_111, (1584, ), (1, ))
    assert_size_stride(primals_112, (1584, ), (1, ))
    assert_size_stride(primals_113, (264, ), (1, ))
    assert_size_stride(primals_114, (264, ), (1, ))
    assert_size_stride(primals_115, (1536, ), (1, ))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_117, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_118, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_120, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_121, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_122, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_123, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_124, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_125, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_126, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_127, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_128, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_129, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_131, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_132, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_133, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_135, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_136, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_137, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_138, (20, ), (1, ))
    assert_size_stride(primals_139, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_140, (240, ), (1, ))
    assert_size_stride(primals_141, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_142, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_143, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_144, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_147, (28, ), (1, ))
    assert_size_stride(primals_148, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_149, (336, ), (1, ))
    assert_size_stride(primals_150, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_151, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_152, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_153, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_154, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_156, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_157, (28, ), (1, ))
    assert_size_stride(primals_158, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_159, (336, ), (1, ))
    assert_size_stride(primals_160, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_161, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_162, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_163, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_164, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_165, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_166, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_167, (28, ), (1, ))
    assert_size_stride(primals_168, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_169, (336, ), (1, ))
    assert_size_stride(primals_170, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_171, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_172, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_173, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_174, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_175, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_176, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_177, (14, ), (1, ))
    assert_size_stride(primals_178, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_179, (336, ), (1, ))
    assert_size_stride(primals_180, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_181, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_182, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_183, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_184, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_186, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_187, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_188, (26, ), (1, ))
    assert_size_stride(primals_189, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_190, (624, ), (1, ))
    assert_size_stride(primals_191, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_192, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_193, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_194, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_195, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_196, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_197, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_198, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_199, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_200, (26, ), (1, ))
    assert_size_stride(primals_201, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_202, (624, ), (1, ))
    assert_size_stride(primals_203, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_204, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_205, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_206, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_207, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_208, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_209, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_210, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_211, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_212, (26, ), (1, ))
    assert_size_stride(primals_213, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_214, (624, ), (1, ))
    assert_size_stride(primals_215, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_216, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_217, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_218, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_219, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_220, (52, ), (1, ))
    assert_size_stride(primals_221, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_222, (624, ), (1, ))
    assert_size_stride(primals_223, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_224, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_225, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_226, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_228, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_229, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_230, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_231, (80, ), (1, ))
    assert_size_stride(primals_232, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_233, (480, ), (1, ))
    assert_size_stride(primals_234, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_235, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_236, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_237, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_238, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_239, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_240, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_241, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_242, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_243, (80, ), (1, ))
    assert_size_stride(primals_244, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_245, (480, ), (1, ))
    assert_size_stride(primals_246, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_247, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_248, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_249, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_250, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_251, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_252, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_253, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_254, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_255, (80, ), (1, ))
    assert_size_stride(primals_256, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_257, (480, ), (1, ))
    assert_size_stride(primals_258, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_259, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_260, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_261, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_262, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_263, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_264, (240, 1, 9, 9), (81, 81, 9, 1))
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
        triton_poi_fused_0.run(primals_117, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_117
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_480, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_480
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
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_307, primals_308, buf10, buf11, buf13, primals_307, primals_308, 32, 7, grid=grid(32), stream=stream0)
        del primals_307
        del primals_308
        buf14 = reinterpret_tensor(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf3, buf10, buf11, primals_1, primals_2, buf14, 3211264, grid=grid(3211264), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
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
        buf23 = buf11; del buf11  # reuse
        buf24 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf26 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf20, buf21, buf22, primals_310, primals_311, buf23, buf24, buf26, primals_310, primals_311, 32, 7, grid=grid(32), stream=stream0)
        del primals_310
        del primals_311
        buf27 = reinterpret_tensor(buf15, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf15  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf16, buf23, buf24, primals_3, primals_4, buf27, 3211264, grid=grid(3211264), stream=stream0)
        del primals_4
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf29 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf28, buf29, 256, 12544, grid=grid(256, 12544), stream=stream0)
        buf30 = buf19; del buf19  # reuse
        buf31 = buf18; del buf18  # reuse
        buf32 = buf17; del buf17  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf29, buf30, buf31, buf32, 25088, 128, grid=grid(25088), stream=stream0)
        buf33 = buf22; del buf22  # reuse
        buf34 = buf21; del buf21  # reuse
        buf35 = buf20; del buf20  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf30, buf31, buf32, buf33, buf34, buf35, 224, 112, grid=grid(224), stream=stream0)
        del buf30
        del buf31
        del buf32
        buf36 = buf24; del buf24  # reuse
        buf37 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf39 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf33, buf34, buf35, primals_313, primals_314, buf36, buf37, buf39, primals_313, primals_314, 32, 7, grid=grid(32), stream=stream0)
        del primals_313
        del primals_314
        buf40 = buf28; del buf28  # reuse
        # Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_7.run(buf29, buf36, buf37, primals_5, primals_6, buf14, buf40, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del buf37
        del primals_6
        buf41 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf40, buf41, 128, 12544, grid=grid(128, 12544), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        buf43 = buf41; del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf40, buf43, 128, 12544, grid=grid(128, 12544), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
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
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf45, buf46, buf47, buf48, 98304, 196, grid=grid(98304), stream=stream0)
        buf49 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 192, 1, 1, 4), (768, 1, 768, 768, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_12.run(buf46, buf47, buf48, buf49, buf50, buf51, 768, 128, grid=grid(768), stream=stream0)
        del buf46
        del buf47
        del buf48
        buf52 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf55 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_13.run(buf49, buf50, buf51, primals_316, primals_317, buf52, buf53, buf55, primals_316, primals_317, 192, 4, grid=grid(192), stream=stream0)
        del buf49
        del buf50
        del buf51
        del primals_316
        del primals_317
        buf56 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        buf934 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_19, x_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_14.run(buf45, buf52, buf53, primals_7, primals_8, buf56, buf934, 19267584, grid=grid(19267584), stream=stream0)
        del primals_8
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_dw_0], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(reinterpret_tensor(buf56, (8, 64, 112, 112), (2408448, 1, 21504, 192), 0), primals_122, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf57, (8, 64, 56, 56), (200704, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_dw_1], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(reinterpret_tensor(buf56, (8, 64, 112, 112), (2408448, 1, 21504, 192), 64), primals_123, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf58, (8, 64, 56, 56), (200704, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_dw_2], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(reinterpret_tensor(buf56, (8, 64, 112, 112), (2408448, 1, 21504, 192), 128), primals_124, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf59, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf60 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf57, buf58, buf59, buf60, 1536, 3136, grid=grid(1536, 3136), stream=stream0)
        del buf57
        del buf58
        del buf59
        buf61 = empty_strided((1, 192, 1, 1, 196), (37632, 1, 37632, 37632, 192), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((1, 192, 1, 1, 196), (37632, 1, 37632, 37632, 192), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((1, 192, 1, 1, 196), (37632, 1, 37632, 37632, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf60, buf61, buf62, buf63, 37632, 128, grid=grid(37632), stream=stream0)
        buf64 = empty_strided((1, 192, 1, 1, 2), (384, 1, 384, 384, 192), device='cuda', dtype=torch.float32)
        buf65 = empty_strided((1, 192, 1, 1, 2), (384, 1, 384, 384, 192), device='cuda', dtype=torch.float32)
        buf66 = empty_strided((1, 192, 1, 1, 2), (384, 1, 384, 384, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf61, buf62, buf63, buf64, buf65, buf66, 384, 98, grid=grid(384), stream=stream0)
        del buf61
        del buf62
        del buf63
        buf67 = buf53; del buf53  # reuse
        buf68 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf70 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_18.run(buf64, buf65, buf66, primals_319, primals_320, buf67, buf68, buf70, primals_319, primals_320, 192, 2, grid=grid(192), stream=stream0)
        del buf64
        del buf65
        del buf66
        del primals_319
        del primals_320
        buf71 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda', dtype=torch.float32)
        buf933 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_25, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf60, buf67, buf68, primals_9, primals_10, buf71, buf933, 4816896, grid=grid(4816896), stream=stream0)
        del buf68
        del primals_10
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(reinterpret_tensor(buf71, (8, 96, 56, 56), (602112, 1, 10752, 192), 0), primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 20, 56, 56), (62720, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(reinterpret_tensor(buf71, (8, 96, 56, 56), (602112, 1, 10752, 192), 96), primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 20, 56, 56), (62720, 3136, 56, 1))
        buf74 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_20.run(buf72, buf73, buf74, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del buf72
        del buf73
        buf75 = empty_strided((1, 40, 1, 1, 196), (7840, 1, 7840, 7840, 40), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((1, 40, 1, 1, 196), (7840, 1, 7840, 7840, 40), device='cuda', dtype=torch.float32)
        buf77 = empty_strided((1, 40, 1, 1, 196), (7840, 1, 7840, 7840, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf74, buf75, buf76, buf77, 7840, 128, grid=grid(7840), stream=stream0)
        buf78 = empty_strided((1, 40, 1, 1, 2), (80, 1, 80, 80, 40), device='cuda', dtype=torch.float32)
        buf79 = empty_strided((1, 40, 1, 1, 2), (80, 1, 80, 80, 40), device='cuda', dtype=torch.float32)
        buf80 = empty_strided((1, 40, 1, 1, 2), (80, 1, 80, 80, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_22.run(buf75, buf76, buf77, buf78, buf79, buf80, 80, 98, grid=grid(80), stream=stream0)
        buf81 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf82 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf84 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_23.run(buf78, buf79, buf80, primals_322, primals_323, buf81, buf82, buf84, primals_322, primals_323, 40, 2, grid=grid(40), stream=stream0)
        del primals_322
        del primals_323
        buf85 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_24.run(buf74, buf81, buf82, primals_11, primals_12, buf85, 1003520, grid=grid(1003520), stream=stream0)
        del primals_12
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(reinterpret_tensor(buf85, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 60, 56, 56), (188160, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(reinterpret_tensor(buf85, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 60, 56, 56), (188160, 3136, 56, 1))
        buf88 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_78], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf86, buf87, buf88, 960, 3136, grid=grid(960, 3136), stream=stream0)
        buf89 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf88, buf89, buf90, buf91, 23520, 128, grid=grid(23520), stream=stream0)
        buf92 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        buf93 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        buf94 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf89, buf90, buf91, buf92, buf93, buf94, 240, 98, grid=grid(240), stream=stream0)
        buf95 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf96 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf98 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_28.run(buf92, buf93, buf94, primals_325, primals_326, buf95, buf96, buf98, primals_325, primals_326, 120, 2, grid=grid(120), stream=stream0)
        del primals_325
        del primals_326
        buf99 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_29.run(buf88, buf95, buf96, primals_13, primals_14, buf99, 3010560, grid=grid(3010560), stream=stream0)
        del primals_14
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf100, (8, 120, 56, 56), (376320, 3136, 56, 1))
        buf101 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf100, buf101, 960, 3136, grid=grid(960, 3136), stream=stream0)
        buf102 = buf91; del buf91  # reuse
        buf103 = buf90; del buf90  # reuse
        buf104 = buf89; del buf89  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf101, buf102, buf103, buf104, 23520, 128, grid=grid(23520), stream=stream0)
        buf105 = buf94; del buf94  # reuse
        buf106 = buf93; del buf93  # reuse
        buf107 = buf92; del buf92  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf102, buf103, buf104, buf105, buf106, buf107, 240, 98, grid=grid(240), stream=stream0)
        del buf102
        del buf103
        del buf104
        buf108 = buf96; del buf96  # reuse
        buf109 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf111 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_28.run(buf105, buf106, buf107, primals_328, primals_329, buf108, buf109, buf111, primals_328, primals_329, 120, 2, grid=grid(120), stream=stream0)
        del primals_328
        del primals_329
        buf112 = reinterpret_tensor(buf100, (8, 120, 56, 56), (376320, 1, 6720, 120), 0); del buf100  # reuse
        buf932 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_43, x_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf101, buf108, buf109, primals_15, primals_16, buf112, buf932, 3010560, grid=grid(3010560), stream=stream0)
        del buf109
        del primals_16
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(reinterpret_tensor(buf112, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 20, 56, 56), (62720, 3136, 56, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(reinterpret_tensor(buf112, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 20, 56, 56), (62720, 3136, 56, 1))
        buf115 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_77], Original ATen: [aten.cat]
        triton_poi_fused_cat_20.run(buf113, buf114, buf115, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del buf113
        del buf114
        buf116 = buf77; del buf77  # reuse
        buf117 = buf76; del buf76  # reuse
        buf118 = buf75; del buf75  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf115, buf116, buf117, buf118, 7840, 128, grid=grid(7840), stream=stream0)
        buf119 = buf80; del buf80  # reuse
        buf120 = buf79; del buf79  # reuse
        buf121 = buf78; del buf78  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_22.run(buf116, buf117, buf118, buf119, buf120, buf121, 80, 98, grid=grid(80), stream=stream0)
        del buf116
        del buf117
        del buf118
        buf122 = buf82; del buf82  # reuse
        buf123 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf125 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_23.run(buf119, buf120, buf121, primals_331, primals_332, buf122, buf123, buf125, primals_331, primals_332, 40, 2, grid=grid(40), stream=stream0)
        del buf119
        del buf120
        del buf121
        del primals_331
        del primals_332
        buf126 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_32.run(buf115, buf122, buf123, primals_17, primals_18, buf85, buf126, 1003520, grid=grid(1003520), stream=stream0)
        del buf123
        del primals_18
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 240, 56, 56), (752640, 3136, 56, 1))
        buf128 = empty_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf127, buf128, 1920, 3136, grid=grid(1920, 3136), stream=stream0)
        buf129 = empty_strided((1, 240, 1, 1, 196), (47040, 1, 47040, 47040, 240), device='cuda', dtype=torch.float32)
        buf130 = empty_strided((1, 240, 1, 1, 196), (47040, 1, 47040, 47040, 240), device='cuda', dtype=torch.float32)
        buf131 = empty_strided((1, 240, 1, 1, 196), (47040, 1, 47040, 47040, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf128, buf129, buf130, buf131, 47040, 128, grid=grid(47040), stream=stream0)
        buf132 = empty_strided((1, 240, 1, 1, 2), (480, 1, 480, 480, 240), device='cuda', dtype=torch.float32)
        buf133 = empty_strided((1, 240, 1, 1, 2), (480, 1, 480, 480, 240), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((1, 240, 1, 1, 2), (480, 1, 480, 480, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf129, buf130, buf131, buf132, buf133, buf134, 480, 98, grid=grid(480), stream=stream0)
        del buf129
        del buf130
        del buf131
        buf135 = reinterpret_tensor(buf107, (1, 240, 1, 1), (240, 1, 240, 240), 0); del buf107  # reuse
        buf136 = reinterpret_tensor(buf106, (1, 240, 1, 1), (240, 1, 240, 240), 0); del buf106  # reuse
        buf138 = reinterpret_tensor(buf105, (240, ), (1, ), 0); del buf105  # reuse
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf132, buf133, buf134, primals_334, primals_335, buf135, buf136, buf138, primals_334, primals_335, 240, 2, grid=grid(240), stream=stream0)
        del primals_334
        del primals_335
        buf139 = reinterpret_tensor(buf127, (8, 240, 56, 56), (752640, 1, 13440, 240), 0); del buf127  # reuse
        buf931 = empty_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_37.run(buf128, buf135, buf136, primals_19, primals_20, buf139, buf931, 6021120, grid=grid(6021120), stream=stream0)
        del primals_20
        buf140 = empty((8, 240, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten.silu]
        triton_poi_fused_silu_38.run(buf139, buf140, 1920, 3136, grid=grid(1920, 3136), stream=stream0)
        del buf139
        buf141 = reinterpret_tensor(buf87, (8, 60, 56, 56), (188160, 1, 3360, 60), 0); del buf87  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf140, buf141, 480, 3136, grid=grid(480, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_0], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf142, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf143 = buf141; del buf141  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_40.run(buf140, buf143, 480, 3136, grid=grid(480, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_1], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_134, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf144, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf145 = buf143; del buf143  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf140, buf145, 480, 3136, grid=grid(480, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_2], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_135, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf146, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf147 = buf145; del buf145  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_42.run(buf140, buf147, 480, 3136, grid=grid(480, 3136), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_3], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_136, stride=(2, 2), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf148, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf149 = reinterpret_tensor(buf147, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf147  # reuse
        # Source Nodes: [cat_76], Original ATen: [aten.cat]
        triton_poi_fused_cat_43.run(buf142, buf144, buf146, buf148, buf149, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del buf142
        del buf144
        del buf146
        del buf148
        buf150 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf151 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf152 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf149, buf150, buf151, buf152, 11760, 128, grid=grid(11760), stream=stream0)
        buf153 = buf136; del buf136  # reuse
        buf154 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf156 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf150, buf151, buf152, primals_337, primals_338, buf153, buf154, buf156, primals_337, primals_338, 240, 49, grid=grid(240), stream=stream0)
        del buf150
        del buf151
        del buf152
        del primals_337
        del primals_338
        buf157 = reinterpret_tensor(buf86, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf86  # reuse
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_46.run(buf149, buf153, buf154, primals_21, primals_22, buf157, 1505280, grid=grid(1505280), stream=stream0)
        del buf154
        del primals_22
        buf158 = empty_strided((8, 240, 1, 1, 7), (1680, 1, 13440, 13440, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_47.run(buf157, buf158, 13440, 112, grid=grid(13440), stream=stream0)
        buf159 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf160 = reinterpret_tensor(buf159, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf159  # reuse
        # Source Nodes: [x_65, x_se], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_48.run(buf160, buf158, 1920, 7, grid=grid(1920), stream=stream0)
        del buf158
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 20, 1, 1), (20, 1, 1, 1))
        buf162 = reinterpret_tensor(buf161, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf161  # reuse
        buf163 = empty_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_49.run(buf162, primals_138, buf163, 160, grid=grid(160), stream=stream0)
        del primals_138
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 240, 1, 1), (240, 1, 1, 1))
        buf165 = reinterpret_tensor(buf164, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf164  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(buf165, primals_140, 1920, grid=grid(1920), stream=stream0)
        del primals_140
        buf166 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_65, x_66], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_51.run(buf157, buf165, buf166, 1505280, grid=grid(1505280), stream=stream0)
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 56, 28, 28), (43904, 784, 28, 1))
        buf168 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf167, buf168, 448, 784, grid=grid(448, 784), stream=stream0)
        buf169 = empty_strided((1, 56, 1, 1, 49), (2744, 1, 2744, 2744, 56), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((1, 56, 1, 1, 49), (2744, 1, 2744, 2744, 56), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((1, 56, 1, 1, 49), (2744, 1, 2744, 2744, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf168, buf169, buf170, buf171, 2744, 128, grid=grid(2744), stream=stream0)
        buf172 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf173 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf175 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf169, buf170, buf171, primals_340, primals_341, buf172, buf173, buf175, primals_340, primals_341, 56, 49, grid=grid(56), stream=stream0)
        del primals_340
        del primals_341
        buf176 = reinterpret_tensor(buf167, (8, 56, 28, 28), (43904, 1, 1568, 56), 0); del buf167  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_55.run(buf168, buf172, buf173, primals_23, primals_24, buf176, 351232, grid=grid(351232), stream=stream0)
        del primals_24
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(reinterpret_tensor(buf176, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 168, 28, 28), (131712, 784, 28, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(reinterpret_tensor(buf176, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf179 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_75], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf177, buf178, buf179, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf177
        buf180 = empty_strided((1, 336, 1, 1, 49), (16464, 1, 16464, 16464, 336), device='cuda', dtype=torch.float32)
        buf181 = empty_strided((1, 336, 1, 1, 49), (16464, 1, 16464, 16464, 336), device='cuda', dtype=torch.float32)
        buf182 = empty_strided((1, 336, 1, 1, 49), (16464, 1, 16464, 16464, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf179, buf180, buf181, buf182, 16464, 128, grid=grid(16464), stream=stream0)
        buf183 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf184 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf186 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf180, buf181, buf182, primals_343, primals_344, buf183, buf184, buf186, primals_343, primals_344, 336, 49, grid=grid(336), stream=stream0)
        del primals_343
        del primals_344
        buf187 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        buf930 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_59.run(buf179, buf183, buf184, primals_25, primals_26, buf187, buf930, 2107392, grid=grid(2107392), stream=stream0)
        del primals_26
        buf188 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.silu]
        triton_poi_fused_silu_60.run(buf187, buf188, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf189 = reinterpret_tensor(buf178, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf178  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf188, buf189, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf190, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf191 = buf189; del buf189  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf188, buf191, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_145, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf192, (8, 168, 28, 28), (131712, 784, 28, 1))
        del buf191
        buf193 = buf187; del buf187  # reuse
        # Source Nodes: [cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf190, buf192, buf193, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf190
        buf194 = buf182; del buf182  # reuse
        buf195 = buf181; del buf181  # reuse
        buf196 = buf180; del buf180  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf193, buf194, buf195, buf196, 16464, 128, grid=grid(16464), stream=stream0)
        buf197 = buf184; del buf184  # reuse
        buf198 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf200 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf194, buf195, buf196, primals_346, primals_347, buf197, buf198, buf200, primals_346, primals_347, 336, 49, grid=grid(336), stream=stream0)
        del primals_346
        del primals_347
        buf201 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_63.run(buf193, buf197, buf198, primals_27, primals_28, buf201, 2107392, grid=grid(2107392), stream=stream0)
        del primals_28
        buf202 = empty_strided((8, 336, 1, 1, 7), (2352, 1, 18816, 18816, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_64.run(buf201, buf202, 18816, 112, grid=grid(18816), stream=stream0)
        buf203 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf204 = reinterpret_tensor(buf203, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf203  # reuse
        # Source Nodes: [x_83, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_65.run(buf204, buf202, 2688, 7, grid=grid(2688), stream=stream0)
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 28, 1, 1), (28, 1, 1, 1))
        buf206 = reinterpret_tensor(buf205, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf205  # reuse
        buf207 = reinterpret_tensor(buf35, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf35  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_66.run(buf206, primals_147, buf207, 224, grid=grid(224), stream=stream0)
        del primals_147
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 336, 1, 1), (336, 1, 1, 1))
        buf209 = reinterpret_tensor(buf208, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf208  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf209, primals_149, 2688, grid=grid(2688), stream=stream0)
        del primals_149
        buf210 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_83, x_84], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_68.run(buf201, buf209, buf210, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf211 = reinterpret_tensor(buf192, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf192  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf210, buf211, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf213 = buf211; del buf211  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf210, buf213, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf213
        buf215 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_73], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf212, buf214, buf215, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf212
        del buf214
        buf216 = buf171; del buf171  # reuse
        buf217 = buf170; del buf170  # reuse
        buf218 = buf169; del buf169  # reuse
        # Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf215, buf216, buf217, buf218, 2744, 128, grid=grid(2744), stream=stream0)
        buf219 = buf173; del buf173  # reuse
        buf220 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf222 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf216, buf217, buf218, primals_349, primals_350, buf219, buf220, buf222, primals_349, primals_350, 56, 49, grid=grid(56), stream=stream0)
        del primals_349
        del primals_350
        buf223 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_70.run(buf215, buf219, buf220, primals_29, primals_30, buf176, buf223, 351232, grid=grid(351232), stream=stream0)
        del primals_30
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 168, 28, 28), (131712, 784, 28, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf226 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_72], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf224, buf225, buf226, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf224
        buf227 = buf196; del buf196  # reuse
        buf228 = buf195; del buf195  # reuse
        buf229 = buf194; del buf194  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf226, buf227, buf228, buf229, 16464, 128, grid=grid(16464), stream=stream0)
        buf230 = buf198; del buf198  # reuse
        buf231 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf233 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf227, buf228, buf229, primals_352, primals_353, buf230, buf231, buf233, primals_352, primals_353, 336, 49, grid=grid(336), stream=stream0)
        del primals_352
        del primals_353
        buf234 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        buf929 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_59.run(buf226, buf230, buf231, primals_31, primals_32, buf234, buf929, 2107392, grid=grid(2107392), stream=stream0)
        del primals_32
        buf235 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten.silu]
        triton_poi_fused_silu_60.run(buf234, buf235, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf236 = reinterpret_tensor(buf225, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf225  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf235, buf236, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf237, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf238 = buf236; del buf236  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf235, buf238, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_155, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf239, (8, 168, 28, 28), (131712, 784, 28, 1))
        del buf238
        buf240 = buf234; del buf234  # reuse
        # Source Nodes: [cat_71], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf237, buf239, buf240, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf237
        buf241 = buf229; del buf229  # reuse
        buf242 = buf228; del buf228  # reuse
        buf243 = buf227; del buf227  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf240, buf241, buf242, buf243, 16464, 128, grid=grid(16464), stream=stream0)
        buf244 = buf231; del buf231  # reuse
        buf245 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf247 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf241, buf242, buf243, primals_355, primals_356, buf244, buf245, buf247, primals_355, primals_356, 336, 49, grid=grid(336), stream=stream0)
        del primals_355
        del primals_356
        buf248 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_63.run(buf240, buf244, buf245, primals_33, primals_34, buf248, 2107392, grid=grid(2107392), stream=stream0)
        del primals_34
        buf249 = buf202; del buf202  # reuse
        # Source Nodes: [x_103, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_64.run(buf248, buf249, 18816, 112, grid=grid(18816), stream=stream0)
        buf250 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf251 = reinterpret_tensor(buf250, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf250  # reuse
        # Source Nodes: [x_103, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_65.run(buf251, buf249, 2688, 7, grid=grid(2688), stream=stream0)
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 28, 1, 1), (28, 1, 1, 1))
        buf253 = reinterpret_tensor(buf252, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf252  # reuse
        buf254 = reinterpret_tensor(buf34, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf34  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_66.run(buf253, primals_157, buf254, 224, grid=grid(224), stream=stream0)
        del primals_157
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (8, 336, 1, 1), (336, 1, 1, 1))
        buf256 = reinterpret_tensor(buf255, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf255  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf256, primals_159, 2688, grid=grid(2688), stream=stream0)
        del primals_159
        buf257 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_103, x_104], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_68.run(buf248, buf256, buf257, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf258 = reinterpret_tensor(buf239, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf239  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf257, buf258, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf260 = buf258; del buf258  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf257, buf260, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf260
        buf262 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_70], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf259, buf261, buf262, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf259
        del buf261
        buf263 = buf218; del buf218  # reuse
        buf264 = buf217; del buf217  # reuse
        buf265 = buf216; del buf216  # reuse
        # Source Nodes: [x_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf262, buf263, buf264, buf265, 2744, 128, grid=grid(2744), stream=stream0)
        buf266 = buf220; del buf220  # reuse
        buf267 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf269 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf263, buf264, buf265, primals_358, primals_359, buf266, buf267, buf269, primals_358, primals_359, 56, 49, grid=grid(56), stream=stream0)
        del primals_358
        del primals_359
        buf270 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6, x_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_70.run(buf262, buf266, buf267, primals_35, primals_36, buf223, buf270, 351232, grid=grid(351232), stream=stream0)
        del primals_36
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(reinterpret_tensor(buf270, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (8, 168, 28, 28), (131712, 784, 28, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(reinterpret_tensor(buf270, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf273 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_69], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf271, buf272, buf273, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf271
        buf274 = buf243; del buf243  # reuse
        buf275 = buf242; del buf242  # reuse
        buf276 = buf241; del buf241  # reuse
        # Source Nodes: [x_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf273, buf274, buf275, buf276, 16464, 128, grid=grid(16464), stream=stream0)
        buf277 = buf245; del buf245  # reuse
        buf278 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf280 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf274, buf275, buf276, primals_361, primals_362, buf277, buf278, buf280, primals_361, primals_362, 336, 49, grid=grid(336), stream=stream0)
        del primals_361
        del primals_362
        buf281 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        buf928 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_59.run(buf273, buf277, buf278, primals_37, primals_38, buf281, buf928, 2107392, grid=grid(2107392), stream=stream0)
        del primals_38
        buf282 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.silu]
        triton_poi_fused_silu_60.run(buf281, buf282, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf283 = reinterpret_tensor(buf272, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf272  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf282, buf283, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf284, (8, 168, 28, 28), (131712, 784, 28, 1))
        buf285 = buf283; del buf283  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf282, buf285, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_165, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
        assert_size_stride(buf286, (8, 168, 28, 28), (131712, 784, 28, 1))
        del buf285
        buf287 = buf281; del buf281  # reuse
        # Source Nodes: [cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf284, buf286, buf287, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf284
        buf288 = buf276; del buf276  # reuse
        buf289 = buf275; del buf275  # reuse
        buf290 = buf274; del buf274  # reuse
        # Source Nodes: [x_120], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf287, buf288, buf289, buf290, 16464, 128, grid=grid(16464), stream=stream0)
        buf291 = buf278; del buf278  # reuse
        buf292 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf294 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf288, buf289, buf290, primals_364, primals_365, buf291, buf292, buf294, primals_364, primals_365, 336, 49, grid=grid(336), stream=stream0)
        del primals_364
        del primals_365
        buf295 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_63.run(buf287, buf291, buf292, primals_39, primals_40, buf295, 2107392, grid=grid(2107392), stream=stream0)
        del primals_40
        buf296 = buf249; del buf249  # reuse
        # Source Nodes: [x_123, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_64.run(buf295, buf296, 18816, 112, grid=grid(18816), stream=stream0)
        buf297 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf298 = reinterpret_tensor(buf297, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf297  # reuse
        # Source Nodes: [x_123, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_65.run(buf298, buf296, 2688, 7, grid=grid(2688), stream=stream0)
        del buf296
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 28, 1, 1), (28, 1, 1, 1))
        buf300 = reinterpret_tensor(buf299, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf299  # reuse
        buf301 = reinterpret_tensor(buf33, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf33  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_66.run(buf300, primals_167, buf301, 224, grid=grid(224), stream=stream0)
        del primals_167
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 336, 1, 1), (336, 1, 1, 1))
        buf303 = reinterpret_tensor(buf302, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf302  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf303, primals_169, 2688, grid=grid(2688), stream=stream0)
        del primals_169
        buf304 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_123, x_124], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_68.run(buf295, buf303, buf304, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf305 = reinterpret_tensor(buf286, (8, 168, 28, 28), (131712, 1, 4704, 168), 0); del buf286  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf304, buf305, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 28, 28, 28), (21952, 784, 28, 1))
        buf307 = buf305; del buf305  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_62.run(buf304, buf307, 1344, 784, grid=grid(1344, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (8, 28, 28, 28), (21952, 784, 28, 1))
        del buf307
        buf309 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_67], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf306, buf308, buf309, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf306
        del buf308
        buf310 = buf265; del buf265  # reuse
        buf311 = buf264; del buf264  # reuse
        buf312 = buf263; del buf263  # reuse
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf309, buf310, buf311, buf312, 2744, 128, grid=grid(2744), stream=stream0)
        buf313 = buf267; del buf267  # reuse
        buf314 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf316 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf310, buf311, buf312, primals_367, primals_368, buf313, buf314, buf316, primals_367, primals_368, 56, 49, grid=grid(56), stream=stream0)
        del buf310
        del buf311
        del buf312
        del primals_367
        del primals_368
        buf317 = empty_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_7, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_70.run(buf309, buf313, buf314, primals_41, primals_42, buf270, buf317, 351232, grid=grid(351232), stream=stream0)
        del buf314
        del primals_42
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (8, 336, 28, 28), (263424, 784, 28, 1))
        buf319 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_71.run(buf318, buf319, 2688, 784, grid=grid(2688, 784), stream=stream0)
        buf320 = buf290; del buf290  # reuse
        buf321 = buf289; del buf289  # reuse
        buf322 = buf288; del buf288  # reuse
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf319, buf320, buf321, buf322, 16464, 128, grid=grid(16464), stream=stream0)
        buf323 = buf292; del buf292  # reuse
        buf324 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf326 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf320, buf321, buf322, primals_370, primals_371, buf323, buf324, buf326, primals_370, primals_371, 336, 49, grid=grid(336), stream=stream0)
        del buf320
        del buf321
        del buf322
        del primals_370
        del primals_371
        buf327 = reinterpret_tensor(buf318, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf318  # reuse
        buf927 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_59.run(buf319, buf323, buf324, primals_43, primals_44, buf327, buf927, 2107392, grid=grid(2107392), stream=stream0)
        del primals_44
        buf328 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_136], Original ATen: [aten.silu]
        triton_poi_fused_silu_60.run(buf327, buf328, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf327
        buf329 = empty_strided((8, 112, 28, 28), (87808, 1, 3136, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf328, buf329, 896, 784, grid=grid(896, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_0], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_173, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf330, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf331 = buf329; del buf329  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf328, buf331, 896, 784, grid=grid(896, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_1], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_174, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf332, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf333 = buf331; del buf331  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf328, buf333, 896, 784, grid=grid(896, 784), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_2], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_175, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf334, (8, 112, 14, 14), (21952, 196, 14, 1))
        del buf333
        buf335 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf330, buf332, buf334, buf335, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del buf330
        del buf332
        del buf334
        buf336 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf337 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf338 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf335, buf336, buf337, buf338, 4368, 121, grid=grid(4368), stream=stream0)
        buf339 = buf324; del buf324  # reuse
        buf340 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf342 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf336, buf337, buf338, primals_373, primals_374, buf339, buf340, buf342, primals_373, primals_374, 336, 13, grid=grid(336), stream=stream0)
        del buf336
        del buf337
        del buf338
        del primals_373
        del primals_374
        buf343 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_78.run(buf335, buf339, buf340, primals_45, primals_46, buf343, 526848, grid=grid(526848), stream=stream0)
        del buf340
        del primals_46
        buf344 = empty_strided((8, 336, 1, 1, 2), (672, 1, 5376, 5376, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_79.run(buf343, buf344, 5376, 98, grid=grid(5376), stream=stream0)
        buf345 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf346 = reinterpret_tensor(buf345, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf345  # reuse
        # Source Nodes: [x_142, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_80.run(buf346, buf344, 2688, 2, grid=grid(2688), stream=stream0)
        del buf344
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 14, 1, 1), (14, 1, 1, 1))
        buf348 = reinterpret_tensor(buf347, (8, 14, 1, 1), (14, 1, 14, 14), 0); del buf347  # reuse
        buf349 = empty_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_81.run(buf348, primals_177, buf349, 112, grid=grid(112), stream=stream0)
        del primals_177
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 336, 1, 1), (336, 1, 1, 1))
        buf351 = reinterpret_tensor(buf350, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf350  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_67.run(buf351, primals_179, 2688, grid=grid(2688), stream=stream0)
        del primals_179
        buf352 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_142, x_143], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_82.run(buf343, buf351, buf352, 526848, grid=grid(526848), stream=stream0)
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 104, 14, 14), (20384, 196, 14, 1))
        buf354 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf353, buf354, 832, 196, grid=grid(832, 196), stream=stream0)
        buf355 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        buf356 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        buf357 = empty_strided((1, 104, 1, 1, 13), (1352, 1, 1352, 1352, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf354, buf355, buf356, buf357, 1352, 121, grid=grid(1352), stream=stream0)
        buf358 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf359 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf361 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf355, buf356, buf357, primals_376, primals_377, buf358, buf359, buf361, primals_376, primals_377, 104, 13, grid=grid(104), stream=stream0)
        del primals_376
        del primals_377
        buf362 = reinterpret_tensor(buf353, (8, 104, 14, 14), (20384, 1, 1456, 104), 0); del buf353  # reuse
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_86.run(buf354, buf358, buf359, primals_47, primals_48, buf362, 163072, grid=grid(163072), stream=stream0)
        del primals_48
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(reinterpret_tensor(buf362, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 312, 14, 14), (61152, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(reinterpret_tensor(buf362, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 312, 14, 14), (61152, 196, 14, 1))
        buf365 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf363, buf364, buf365, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf363
        buf366 = empty_strided((1, 624, 1, 1, 13), (8112, 1, 8112, 8112, 624), device='cuda', dtype=torch.float32)
        buf367 = empty_strided((1, 624, 1, 1, 13), (8112, 1, 8112, 8112, 624), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((1, 624, 1, 1, 13), (8112, 1, 8112, 8112, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf365, buf366, buf367, buf368, 8112, 121, grid=grid(8112), stream=stream0)
        buf369 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf370 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf372 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf366, buf367, buf368, primals_379, primals_380, buf369, buf370, buf372, primals_379, primals_380, 624, 13, grid=grid(624), stream=stream0)
        del primals_379
        del primals_380
        buf373 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        buf926 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_90.run(buf365, buf369, buf370, primals_49, primals_50, buf373, buf926, 978432, grid=grid(978432), stream=stream0)
        del primals_50
        buf374 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.silu]
        triton_poi_fused_silu_91.run(buf373, buf374, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf375 = empty_strided((8, 156, 14, 14), (30576, 1, 2184, 156), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf374, buf375, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf376, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf377 = buf375; del buf375  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf374, buf377, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_184, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf378, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf379 = buf377; del buf377  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf374, buf379, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_185, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf380, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf381 = buf379; del buf379  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf374, buf381, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_186, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf382, (8, 156, 14, 14), (30576, 196, 14, 1))
        del buf381
        buf383 = buf373; del buf373  # reuse
        # Source Nodes: [cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf376, buf378, buf380, buf382, buf383, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf376
        del buf378
        del buf380
        buf384 = buf368; del buf368  # reuse
        buf385 = buf367; del buf367  # reuse
        buf386 = buf366; del buf366  # reuse
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf383, buf384, buf385, buf386, 8112, 121, grid=grid(8112), stream=stream0)
        buf387 = buf370; del buf370  # reuse
        buf388 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf390 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf384, buf385, buf386, primals_382, primals_383, buf387, buf388, buf390, primals_382, primals_383, 624, 13, grid=grid(624), stream=stream0)
        del primals_382
        del primals_383
        buf391 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_97.run(buf383, buf387, buf388, primals_51, primals_52, buf391, 978432, grid=grid(978432), stream=stream0)
        del primals_52
        buf392 = empty_strided((8, 624, 1, 1, 2), (1248, 1, 9984, 9984, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_98.run(buf391, buf392, 9984, 98, grid=grid(9984), stream=stream0)
        buf393 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf394 = reinterpret_tensor(buf393, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf393  # reuse
        # Source Nodes: [x_160, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_99.run(buf394, buf392, 4992, 2, grid=grid(4992), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (8, 26, 1, 1), (26, 1, 1, 1))
        buf396 = reinterpret_tensor(buf395, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf395  # reuse
        buf397 = empty_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_100.run(buf396, primals_188, buf397, 208, grid=grid(208), stream=stream0)
        del primals_188
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (8, 624, 1, 1), (624, 1, 1, 1))
        buf399 = reinterpret_tensor(buf398, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf398  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_101.run(buf399, primals_190, 4992, grid=grid(4992), stream=stream0)
        del primals_190
        buf400 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_160, x_161], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_102.run(buf391, buf399, buf400, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf401 = reinterpret_tensor(buf364, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf364  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf400, buf401, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 52, 14, 14), (10192, 196, 14, 1))
        buf403 = buf401; del buf401  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_104.run(buf400, buf403, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 52, 14, 14), (10192, 196, 14, 1))
        del buf403
        buf405 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf402, buf404, buf405, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf402
        del buf404
        buf406 = buf357; del buf357  # reuse
        buf407 = buf356; del buf356  # reuse
        buf408 = buf355; del buf355  # reuse
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf405, buf406, buf407, buf408, 1352, 121, grid=grid(1352), stream=stream0)
        buf409 = buf359; del buf359  # reuse
        buf410 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf412 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf406, buf407, buf408, primals_385, primals_386, buf409, buf410, buf412, primals_385, primals_386, 104, 13, grid=grid(104), stream=stream0)
        del primals_385
        del primals_386
        buf413 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_106.run(buf405, buf409, buf410, primals_53, primals_54, buf362, buf413, 163072, grid=grid(163072), stream=stream0)
        del primals_54
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(reinterpret_tensor(buf413, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 312, 14, 14), (61152, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(reinterpret_tensor(buf413, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 312, 14, 14), (61152, 196, 14, 1))
        buf416 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf414, buf415, buf416, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf414
        buf417 = buf386; del buf386  # reuse
        buf418 = buf385; del buf385  # reuse
        buf419 = buf384; del buf384  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf416, buf417, buf418, buf419, 8112, 121, grid=grid(8112), stream=stream0)
        buf420 = buf388; del buf388  # reuse
        buf421 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf423 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf417, buf418, buf419, primals_388, primals_389, buf420, buf421, buf423, primals_388, primals_389, 624, 13, grid=grid(624), stream=stream0)
        del primals_388
        del primals_389
        buf424 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        buf925 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_90.run(buf416, buf420, buf421, primals_55, primals_56, buf424, buf925, 978432, grid=grid(978432), stream=stream0)
        del primals_56
        buf425 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174], Original ATen: [aten.silu]
        triton_poi_fused_silu_91.run(buf424, buf425, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf426 = reinterpret_tensor(buf382, (8, 156, 14, 14), (30576, 1, 2184, 156), 0); del buf382  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf425, buf426, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf427, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf428 = buf426; del buf426  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf425, buf428, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, primals_196, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf429, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf430 = buf428; del buf428  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf425, buf430, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_197, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf431, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf432 = buf430; del buf430  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf425, buf432, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, primals_198, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf433, (8, 156, 14, 14), (30576, 196, 14, 1))
        del buf432
        buf434 = buf424; del buf424  # reuse
        # Source Nodes: [cat_61], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf427, buf429, buf431, buf433, buf434, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf427
        del buf429
        del buf431
        buf435 = buf419; del buf419  # reuse
        buf436 = buf418; del buf418  # reuse
        buf437 = buf417; del buf417  # reuse
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf434, buf435, buf436, buf437, 8112, 121, grid=grid(8112), stream=stream0)
        buf438 = buf421; del buf421  # reuse
        buf439 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf441 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf435, buf436, buf437, primals_391, primals_392, buf438, buf439, buf441, primals_391, primals_392, 624, 13, grid=grid(624), stream=stream0)
        del primals_391
        del primals_392
        buf442 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_97.run(buf434, buf438, buf439, primals_57, primals_58, buf442, 978432, grid=grid(978432), stream=stream0)
        del primals_58
        buf443 = buf392; del buf392  # reuse
        # Source Nodes: [x_180, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_98.run(buf442, buf443, 9984, 98, grid=grid(9984), stream=stream0)
        buf444 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf445 = reinterpret_tensor(buf444, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf444  # reuse
        # Source Nodes: [x_180, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_99.run(buf445, buf443, 4992, 2, grid=grid(4992), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 26, 1, 1), (26, 1, 1, 1))
        buf447 = reinterpret_tensor(buf446, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf446  # reuse
        buf448 = empty_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_100.run(buf447, primals_200, buf448, 208, grid=grid(208), stream=stream0)
        del primals_200
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (8, 624, 1, 1), (624, 1, 1, 1))
        buf450 = reinterpret_tensor(buf449, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf449  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_101.run(buf450, primals_202, 4992, grid=grid(4992), stream=stream0)
        del primals_202
        buf451 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_180, x_181], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_102.run(buf442, buf450, buf451, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf452 = reinterpret_tensor(buf415, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf415  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf451, buf452, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (8, 52, 14, 14), (10192, 196, 14, 1))
        buf454 = buf452; del buf452  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_104.run(buf451, buf454, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf454, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (8, 52, 14, 14), (10192, 196, 14, 1))
        del buf454
        buf456 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf453, buf455, buf456, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf453
        del buf455
        buf457 = buf408; del buf408  # reuse
        buf458 = buf407; del buf407  # reuse
        buf459 = buf406; del buf406  # reuse
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf456, buf457, buf458, buf459, 1352, 121, grid=grid(1352), stream=stream0)
        buf460 = buf410; del buf410  # reuse
        buf461 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf463 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf457, buf458, buf459, primals_394, primals_395, buf460, buf461, buf463, primals_394, primals_395, 104, 13, grid=grid(104), stream=stream0)
        del primals_394
        del primals_395
        buf464 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_106.run(buf456, buf460, buf461, primals_59, primals_60, buf413, buf464, 163072, grid=grid(163072), stream=stream0)
        del primals_60
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(reinterpret_tensor(buf464, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (8, 312, 14, 14), (61152, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(reinterpret_tensor(buf464, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 312, 14, 14), (61152, 196, 14, 1))
        buf467 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_59], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf465, buf466, buf467, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf465
        buf468 = buf437; del buf437  # reuse
        buf469 = buf436; del buf436  # reuse
        buf470 = buf435; del buf435  # reuse
        # Source Nodes: [x_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf467, buf468, buf469, buf470, 8112, 121, grid=grid(8112), stream=stream0)
        buf471 = buf439; del buf439  # reuse
        buf472 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf474 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf468, buf469, buf470, primals_397, primals_398, buf471, buf472, buf474, primals_397, primals_398, 624, 13, grid=grid(624), stream=stream0)
        del primals_397
        del primals_398
        buf475 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        buf924 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_90.run(buf467, buf471, buf472, primals_61, primals_62, buf475, buf924, 978432, grid=grid(978432), stream=stream0)
        del primals_62
        buf476 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten.silu]
        triton_poi_fused_silu_91.run(buf475, buf476, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf477 = reinterpret_tensor(buf433, (8, 156, 14, 14), (30576, 1, 2184, 156), 0); del buf433  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf476, buf477, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf477, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf478, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf479 = buf477; del buf477  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf476, buf479, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_208, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf480, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf481 = buf479; del buf479  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_94.run(buf476, buf481, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, primals_209, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf482, (8, 156, 14, 14), (30576, 196, 14, 1))
        buf483 = buf481; del buf481  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_95.run(buf476, buf483, 1248, 196, grid=grid(1248, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_210, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
        assert_size_stride(buf484, (8, 156, 14, 14), (30576, 196, 14, 1))
        del buf483
        buf485 = buf475; del buf475  # reuse
        # Source Nodes: [cat_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf478, buf480, buf482, buf484, buf485, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf478
        del buf480
        del buf482
        del buf484
        buf486 = buf470; del buf470  # reuse
        buf487 = buf469; del buf469  # reuse
        buf488 = buf468; del buf468  # reuse
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf485, buf486, buf487, buf488, 8112, 121, grid=grid(8112), stream=stream0)
        buf489 = buf472; del buf472  # reuse
        buf490 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf492 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf486, buf487, buf488, primals_400, primals_401, buf489, buf490, buf492, primals_400, primals_401, 624, 13, grid=grid(624), stream=stream0)
        del primals_400
        del primals_401
        buf493 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_97.run(buf485, buf489, buf490, primals_63, primals_64, buf493, 978432, grid=grid(978432), stream=stream0)
        del primals_64
        buf494 = buf443; del buf443  # reuse
        # Source Nodes: [x_200, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_98.run(buf493, buf494, 9984, 98, grid=grid(9984), stream=stream0)
        buf495 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf496 = reinterpret_tensor(buf495, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf495  # reuse
        # Source Nodes: [x_200, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_99.run(buf496, buf494, 4992, 2, grid=grid(4992), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 26, 1, 1), (26, 1, 1, 1))
        buf498 = reinterpret_tensor(buf497, (8, 26, 1, 1), (26, 1, 26, 26), 0); del buf497  # reuse
        buf499 = empty_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_100.run(buf498, primals_212, buf499, 208, grid=grid(208), stream=stream0)
        del primals_212
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (8, 624, 1, 1), (624, 1, 1, 1))
        buf501 = reinterpret_tensor(buf500, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf500  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_101.run(buf501, primals_214, 4992, grid=grid(4992), stream=stream0)
        del primals_214
        buf502 = empty((8, 624, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_200, x_201], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_102.run(buf493, buf501, buf502, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf503 = reinterpret_tensor(buf466, (8, 312, 14, 14), (61152, 1, 4368, 312), 0); del buf466  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf502, buf503, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (8, 52, 14, 14), (10192, 196, 14, 1))
        buf505 = buf503; del buf503  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_104.run(buf502, buf505, 2496, 196, grid=grid(2496, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (8, 52, 14, 14), (10192, 196, 14, 1))
        del buf505
        buf507 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_57], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf504, buf506, buf507, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf504
        del buf506
        buf508 = buf459; del buf459  # reuse
        buf509 = buf458; del buf458  # reuse
        buf510 = buf457; del buf457  # reuse
        # Source Nodes: [x_204], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf507, buf508, buf509, buf510, 1352, 121, grid=grid(1352), stream=stream0)
        buf511 = buf461; del buf461  # reuse
        buf512 = empty_strided((1, 104, 1, 1), (104, 1, 104, 104), device='cuda', dtype=torch.float32)
        buf514 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf508, buf509, buf510, primals_403, primals_404, buf511, buf512, buf514, primals_403, primals_404, 104, 13, grid=grid(104), stream=stream0)
        del buf508
        del buf509
        del buf510
        del primals_403
        del primals_404
        buf515 = empty_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_11, x_204], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_106.run(buf507, buf511, buf512, primals_65, primals_66, buf464, buf515, 163072, grid=grid(163072), stream=stream0)
        del buf512
        del primals_66
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 624, 14, 14), (122304, 196, 14, 1))
        buf517 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf516, buf517, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf518 = buf488; del buf488  # reuse
        buf519 = buf487; del buf487  # reuse
        buf520 = buf486; del buf486  # reuse
        # Source Nodes: [x_210], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf517, buf518, buf519, buf520, 8112, 121, grid=grid(8112), stream=stream0)
        buf521 = buf490; del buf490  # reuse
        buf522 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf524 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf518, buf519, buf520, primals_406, primals_407, buf521, buf522, buf524, primals_406, primals_407, 624, 13, grid=grid(624), stream=stream0)
        del primals_406
        del primals_407
        buf526 = reinterpret_tensor(buf516, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf516  # reuse
        buf923 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210, x_213], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_108.run(buf517, buf521, buf522, primals_67, primals_68, buf526, buf923, 978432, grid=grid(978432), stream=stream0)
        del primals_68
        # Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=624, bias=None)
        assert_size_stride(buf527, (8, 624, 14, 14), (122304, 196, 14, 1))
        buf528 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf527, buf528, 4992, 196, grid=grid(4992, 196), stream=stream0)
        buf529 = buf520; del buf520  # reuse
        buf530 = buf519; del buf519  # reuse
        buf531 = buf518; del buf518  # reuse
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf528, buf529, buf530, buf531, 8112, 121, grid=grid(8112), stream=stream0)
        buf532 = buf522; del buf522  # reuse
        buf533 = empty_strided((1, 624, 1, 1), (624, 1, 624, 624), device='cuda', dtype=torch.float32)
        buf535 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf529, buf530, buf531, primals_409, primals_410, buf532, buf533, buf535, primals_409, primals_410, 624, 13, grid=grid(624), stream=stream0)
        del buf529
        del buf530
        del buf531
        del primals_409
        del primals_410
        buf536 = reinterpret_tensor(buf527, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf527  # reuse
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_97.run(buf528, buf532, buf533, primals_69, primals_70, buf536, 978432, grid=grid(978432), stream=stream0)
        del buf533
        del primals_70
        buf537 = buf494; del buf494  # reuse
        # Source Nodes: [x_218, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_98.run(buf536, buf537, 9984, 98, grid=grid(9984), stream=stream0)
        buf538 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf539 = reinterpret_tensor(buf538, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf538  # reuse
        # Source Nodes: [x_218, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_99.run(buf539, buf537, 4992, 2, grid=grid(4992), stream=stream0)
        del buf537
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf540 = extern_kernels.convolution(buf539, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf540, (8, 52, 1, 1), (52, 1, 1, 1))
        buf541 = reinterpret_tensor(buf540, (8, 52, 1, 1), (52, 1, 52, 52), 0); del buf540  # reuse
        buf542 = empty_strided((8, 52, 1, 1), (52, 1, 52, 52), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_109.run(buf541, primals_220, buf542, 416, grid=grid(416), stream=stream0)
        del primals_220
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (8, 624, 1, 1), (624, 1, 1, 1))
        buf544 = reinterpret_tensor(buf543, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf543  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_101.run(buf544, primals_222, 4992, grid=grid(4992), stream=stream0)
        del primals_222
        buf545 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_218, x_219], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_110.run(buf536, buf544, buf545, 978432, grid=grid(978432), stream=stream0)
        # Source Nodes: [x_220], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 160, 14, 14), (31360, 196, 14, 1))
        buf547 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_111.run(buf546, buf547, 1280, 196, grid=grid(1280, 196), stream=stream0)
        buf548 = empty_strided((1, 160, 1, 1, 13), (2080, 1, 2080, 2080, 160), device='cuda', dtype=torch.float32)
        buf549 = empty_strided((1, 160, 1, 1, 13), (2080, 1, 2080, 2080, 160), device='cuda', dtype=torch.float32)
        buf550 = empty_strided((1, 160, 1, 1, 13), (2080, 1, 2080, 2080, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_112.run(buf547, buf548, buf549, buf550, 2080, 121, grid=grid(2080), stream=stream0)
        buf551 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf552 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf554 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_113.run(buf548, buf549, buf550, primals_412, primals_413, buf551, buf552, buf554, primals_412, primals_413, 160, 13, grid=grid(160), stream=stream0)
        del primals_412
        del primals_413
        buf555 = reinterpret_tensor(buf546, (8, 160, 14, 14), (31360, 1, 2240, 160), 0); del buf546  # reuse
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_114.run(buf547, buf551, buf552, primals_71, primals_72, buf555, 250880, grid=grid(250880), stream=stream0)
        del primals_72
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
        buf556 = extern_kernels.convolution(reinterpret_tensor(buf555, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf556, (8, 240, 14, 14), (47040, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(reinterpret_tensor(buf555, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf557, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf558 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_115.run(buf556, buf557, buf558, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf556
        buf559 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf560 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf561 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf558, buf559, buf560, buf561, 6240, 121, grid=grid(6240), stream=stream0)
        buf562 = reinterpret_tensor(buf134, (1, 480, 1, 1), (480, 1, 480, 480), 0); del buf134  # reuse
        buf563 = reinterpret_tensor(buf133, (1, 480, 1, 1), (480, 1, 480, 480), 0); del buf133  # reuse
        buf565 = reinterpret_tensor(buf132, (480, ), (1, ), 0); del buf132  # reuse
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf559, buf560, buf561, primals_415, primals_416, buf562, buf563, buf565, primals_415, primals_416, 480, 13, grid=grid(480), stream=stream0)
        del primals_415
        del primals_416
        buf566 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf922 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_118.run(buf558, buf562, buf563, primals_73, primals_74, buf566, buf922, 752640, grid=grid(752640), stream=stream0)
        del primals_74
        buf567 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten.silu]
        triton_poi_fused_silu_119.run(buf566, buf567, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf568 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_120.run(buf567, buf568, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf569 = extern_kernels.convolution(buf568, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf569, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf570 = buf568; del buf568  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_121.run(buf567, buf570, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, primals_227, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf571, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf572 = buf570; del buf570  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf567, buf572, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf573 = extern_kernels.convolution(buf572, primals_228, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf573, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf574 = buf572; del buf572  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_123.run(buf567, buf574, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, primals_229, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf575, (8, 120, 14, 14), (23520, 196, 14, 1))
        del buf574
        buf576 = buf566; del buf566  # reuse
        # Source Nodes: [cat_55], Original ATen: [aten.cat]
        triton_poi_fused_cat_124.run(buf569, buf571, buf573, buf575, buf576, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf569
        del buf571
        del buf573
        buf577 = buf561; del buf561  # reuse
        buf578 = buf560; del buf560  # reuse
        buf579 = buf559; del buf559  # reuse
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf576, buf577, buf578, buf579, 6240, 121, grid=grid(6240), stream=stream0)
        buf580 = buf563; del buf563  # reuse
        buf581 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf583 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf577, buf578, buf579, primals_418, primals_419, buf580, buf581, buf583, primals_418, primals_419, 480, 13, grid=grid(480), stream=stream0)
        del primals_418
        del primals_419
        buf584 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_125.run(buf576, buf580, buf581, primals_75, primals_76, buf584, 752640, grid=grid(752640), stream=stream0)
        del primals_76
        buf585 = empty_strided((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_126.run(buf584, buf585, 7680, 98, grid=grid(7680), stream=stream0)
        buf586 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf587 = reinterpret_tensor(buf586, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf586  # reuse
        # Source Nodes: [x_236, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_127.run(buf587, buf585, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf587, primals_230, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (8, 80, 1, 1), (80, 1, 1, 1))
        buf589 = reinterpret_tensor(buf588, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf588  # reuse
        buf590 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_128.run(buf589, primals_231, buf590, 640, grid=grid(640), stream=stream0)
        del primals_231
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf590, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (8, 480, 1, 1), (480, 1, 1, 1))
        buf592 = reinterpret_tensor(buf591, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf591  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_129.run(buf592, primals_233, 3840, grid=grid(3840), stream=stream0)
        del primals_233
        buf593 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_236, x_237], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_130.run(buf584, buf592, buf593, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf594 = reinterpret_tensor(buf557, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf557  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_131.run(buf593, buf594, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf595 = extern_kernels.convolution(buf594, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf595, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf596 = buf594; del buf594  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_132.run(buf593, buf596, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf597 = extern_kernels.convolution(buf596, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (8, 80, 14, 14), (15680, 196, 14, 1))
        del buf596
        buf598 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_133.run(buf595, buf597, buf598, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf595
        del buf597
        buf599 = buf550; del buf550  # reuse
        buf600 = buf549; del buf549  # reuse
        buf601 = buf548; del buf548  # reuse
        # Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_112.run(buf598, buf599, buf600, buf601, 2080, 121, grid=grid(2080), stream=stream0)
        buf602 = buf552; del buf552  # reuse
        buf603 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf605 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_113.run(buf599, buf600, buf601, primals_421, primals_422, buf602, buf603, buf605, primals_421, primals_422, 160, 13, grid=grid(160), stream=stream0)
        del primals_421
        del primals_422
        buf606 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_13, x_240], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_134.run(buf598, buf602, buf603, primals_77, primals_78, buf555, buf606, 250880, grid=grid(250880), stream=stream0)
        del primals_78
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_0], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(reinterpret_tensor(buf606, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (8, 240, 14, 14), (47040, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(reinterpret_tensor(buf606, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf609 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_115.run(buf607, buf608, buf609, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf607
        buf610 = buf579; del buf579  # reuse
        buf611 = buf578; del buf578  # reuse
        buf612 = buf577; del buf577  # reuse
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf609, buf610, buf611, buf612, 6240, 121, grid=grid(6240), stream=stream0)
        buf613 = buf581; del buf581  # reuse
        buf614 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf616 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf610, buf611, buf612, primals_424, primals_425, buf613, buf614, buf616, primals_424, primals_425, 480, 13, grid=grid(480), stream=stream0)
        del primals_424
        del primals_425
        buf617 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf921 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_118.run(buf609, buf613, buf614, primals_79, primals_80, buf617, buf921, 752640, grid=grid(752640), stream=stream0)
        del primals_80
        buf618 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten.silu]
        triton_poi_fused_silu_119.run(buf617, buf618, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf619 = reinterpret_tensor(buf575, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf575  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_120.run(buf618, buf619, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf620, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf621 = buf619; del buf619  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_121.run(buf618, buf621, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_239, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf622, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf623 = buf621; del buf621  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf618, buf623, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf623, primals_240, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf624, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf625 = buf623; del buf623  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_123.run(buf618, buf625, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_241, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf626, (8, 120, 14, 14), (23520, 196, 14, 1))
        del buf625
        buf627 = buf617; del buf617  # reuse
        # Source Nodes: [cat_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_124.run(buf620, buf622, buf624, buf626, buf627, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf620
        del buf622
        del buf624
        buf628 = buf612; del buf612  # reuse
        buf629 = buf611; del buf611  # reuse
        buf630 = buf610; del buf610  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf627, buf628, buf629, buf630, 6240, 121, grid=grid(6240), stream=stream0)
        buf631 = buf614; del buf614  # reuse
        buf632 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf634 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf628, buf629, buf630, primals_427, primals_428, buf631, buf632, buf634, primals_427, primals_428, 480, 13, grid=grid(480), stream=stream0)
        del primals_427
        del primals_428
        buf635 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_125.run(buf627, buf631, buf632, primals_81, primals_82, buf635, 752640, grid=grid(752640), stream=stream0)
        del primals_82
        buf636 = buf585; del buf585  # reuse
        # Source Nodes: [x_256, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_126.run(buf635, buf636, 7680, 98, grid=grid(7680), stream=stream0)
        buf637 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf638 = reinterpret_tensor(buf637, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf637  # reuse
        # Source Nodes: [x_256, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_127.run(buf638, buf636, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (8, 80, 1, 1), (80, 1, 1, 1))
        buf640 = reinterpret_tensor(buf639, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf639  # reuse
        buf641 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_128.run(buf640, primals_243, buf641, 640, grid=grid(640), stream=stream0)
        del primals_243
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (8, 480, 1, 1), (480, 1, 1, 1))
        buf643 = reinterpret_tensor(buf642, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf642  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_129.run(buf643, primals_245, 3840, grid=grid(3840), stream=stream0)
        del primals_245
        buf644 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_256, x_257], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_130.run(buf635, buf643, buf644, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf645 = reinterpret_tensor(buf608, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf608  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_131.run(buf644, buf645, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf645, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf647 = buf645; del buf645  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_132.run(buf644, buf647, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf648 = extern_kernels.convolution(buf647, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf648, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf649 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_51], Original ATen: [aten.cat]
        triton_poi_fused_cat_133.run(buf646, buf648, buf649, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf646
        del buf648
        buf650 = buf601; del buf601  # reuse
        buf651 = buf600; del buf600  # reuse
        buf652 = buf599; del buf599  # reuse
        # Source Nodes: [x_260], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_112.run(buf649, buf650, buf651, buf652, 2080, 121, grid=grid(2080), stream=stream0)
        buf653 = buf603; del buf603  # reuse
        buf654 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf656 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_113.run(buf650, buf651, buf652, primals_430, primals_431, buf653, buf654, buf656, primals_430, primals_431, 160, 13, grid=grid(160), stream=stream0)
        del primals_430
        del primals_431
        buf657 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_14, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_134.run(buf649, buf653, buf654, primals_83, primals_84, buf606, buf657, 250880, grid=grid(250880), stream=stream0)
        del primals_84
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_0], Original ATen: [aten.convolution]
        buf658 = extern_kernels.convolution(reinterpret_tensor(buf657, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf658, (8, 240, 14, 14), (47040, 196, 14, 1))
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(reinterpret_tensor(buf657, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf660 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_50], Original ATen: [aten.cat]
        triton_poi_fused_cat_115.run(buf658, buf659, buf660, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf661 = buf630; del buf630  # reuse
        buf662 = buf629; del buf629  # reuse
        buf663 = buf628; del buf628  # reuse
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf660, buf661, buf662, buf663, 6240, 121, grid=grid(6240), stream=stream0)
        buf664 = buf632; del buf632  # reuse
        buf665 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf667 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf661, buf662, buf663, primals_433, primals_434, buf664, buf665, buf667, primals_433, primals_434, 480, 13, grid=grid(480), stream=stream0)
        del primals_433
        del primals_434
        buf668 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        buf920 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_118.run(buf660, buf664, buf665, primals_85, primals_86, buf668, buf920, 752640, grid=grid(752640), stream=stream0)
        del primals_86
        buf669 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten.silu]
        triton_poi_fused_silu_119.run(buf668, buf669, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf670 = reinterpret_tensor(buf626, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf626  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_120.run(buf669, buf670, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf671 = extern_kernels.convolution(buf670, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf671, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf672 = buf670; del buf670  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_121.run(buf669, buf672, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, primals_251, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf673, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf674 = buf672; del buf672  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_122.run(buf669, buf674, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf675 = extern_kernels.convolution(buf674, primals_252, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf675, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf676 = buf674; del buf674  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_123.run(buf669, buf676, 960, 196, grid=grid(960, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf677 = extern_kernels.convolution(buf676, primals_253, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf677, (8, 120, 14, 14), (23520, 196, 14, 1))
        del buf676
        buf678 = buf668; del buf668  # reuse
        # Source Nodes: [cat_49], Original ATen: [aten.cat]
        triton_poi_fused_cat_124.run(buf671, buf673, buf675, buf677, buf678, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf671
        del buf673
        del buf675
        del buf677
        buf679 = buf663; del buf663  # reuse
        buf680 = buf662; del buf662  # reuse
        buf681 = buf661; del buf661  # reuse
        # Source Nodes: [x_273], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf678, buf679, buf680, buf681, 6240, 121, grid=grid(6240), stream=stream0)
        buf682 = buf665; del buf665  # reuse
        buf683 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf685 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf679, buf680, buf681, primals_436, primals_437, buf682, buf683, buf685, primals_436, primals_437, 480, 13, grid=grid(480), stream=stream0)
        del buf679
        del buf680
        del buf681
        del primals_436
        del primals_437
        buf686 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_125.run(buf678, buf682, buf683, primals_87, primals_88, buf686, 752640, grid=grid(752640), stream=stream0)
        del buf683
        del primals_88
        buf687 = buf636; del buf636  # reuse
        # Source Nodes: [x_276, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_126.run(buf686, buf687, 7680, 98, grid=grid(7680), stream=stream0)
        buf688 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf689 = reinterpret_tensor(buf688, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf688  # reuse
        # Source Nodes: [x_276, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_127.run(buf689, buf687, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf690 = extern_kernels.convolution(buf689, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf690, (8, 80, 1, 1), (80, 1, 1, 1))
        buf691 = reinterpret_tensor(buf690, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf690  # reuse
        buf692 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_128.run(buf691, primals_255, buf692, 640, grid=grid(640), stream=stream0)
        del primals_255
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf693 = extern_kernels.convolution(buf692, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (8, 480, 1, 1), (480, 1, 1, 1))
        buf694 = reinterpret_tensor(buf693, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf693  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_129.run(buf694, primals_257, 3840, grid=grid(3840), stream=stream0)
        del primals_257
        buf695 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_276, x_277], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_130.run(buf686, buf694, buf695, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf696 = reinterpret_tensor(buf659, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf659  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_131.run(buf695, buf696, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf697 = extern_kernels.convolution(buf696, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf697, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf698 = buf696; del buf696  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_132.run(buf695, buf698, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf699 = extern_kernels.convolution(buf698, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf699, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf700 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_133.run(buf697, buf699, buf700, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf697
        del buf699
        buf701 = buf652; del buf652  # reuse
        buf702 = buf651; del buf651  # reuse
        buf703 = buf650; del buf650  # reuse
        # Source Nodes: [x_280], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_112.run(buf700, buf701, buf702, buf703, 2080, 121, grid=grid(2080), stream=stream0)
        buf704 = buf654; del buf654  # reuse
        buf705 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf707 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_113.run(buf701, buf702, buf703, primals_439, primals_440, buf704, buf705, buf707, primals_439, primals_440, 160, 13, grid=grid(160), stream=stream0)
        del buf701
        del buf702
        del buf703
        del primals_439
        del primals_440
        buf708 = empty_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_15, x_280], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_134.run(buf700, buf704, buf705, primals_89, primals_90, buf657, buf708, 250880, grid=grid(250880), stream=stream0)
        del buf705
        del primals_90
        # Source Nodes: [x_285], Original ATen: [aten.convolution]
        buf709 = extern_kernels.convolution(buf708, primals_260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf709, (8, 960, 14, 14), (188160, 196, 14, 1))
        buf710 = empty_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_135.run(buf709, buf710, 7680, 196, grid=grid(7680, 196), stream=stream0)
        buf711 = empty_strided((1, 960, 1, 1, 13), (12480, 1, 12480, 12480, 960), device='cuda', dtype=torch.float32)
        buf712 = empty_strided((1, 960, 1, 1, 13), (12480, 1, 12480, 12480, 960), device='cuda', dtype=torch.float32)
        buf713 = empty_strided((1, 960, 1, 1, 13), (12480, 1, 12480, 12480, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_136.run(buf710, buf711, buf712, buf713, 12480, 121, grid=grid(12480), stream=stream0)
        buf714 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf715 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf717 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_137.run(buf711, buf712, buf713, primals_442, primals_443, buf714, buf715, buf717, primals_442, primals_443, 960, 13, grid=grid(960), stream=stream0)
        del buf711
        del buf712
        del buf713
        del primals_442
        del primals_443
        buf718 = reinterpret_tensor(buf709, (8, 960, 14, 14), (188160, 1, 13440, 960), 0); del buf709  # reuse
        buf919 = empty_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_138.run(buf710, buf714, buf715, primals_91, primals_92, buf718, buf919, 1505280, grid=grid(1505280), stream=stream0)
        del primals_92
        buf719 = empty((8, 960, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten.silu]
        triton_poi_fused_silu_139.run(buf718, buf719, 7680, 196, grid=grid(7680, 196), stream=stream0)
        del buf718
        buf720 = buf698; del buf698  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_140.run(buf719, buf720, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_0], Original ATen: [aten.convolution]
        buf721 = extern_kernels.convolution(buf720, primals_261, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf721, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf722 = buf720; del buf720  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf719, buf722, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_1], Original ATen: [aten.convolution]
        buf723 = extern_kernels.convolution(buf722, primals_262, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf723, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf724 = buf722; del buf722  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_142.run(buf719, buf724, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_2], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, primals_263, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf725, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf726 = buf724; del buf724  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_143.run(buf719, buf726, 1920, 196, grid=grid(1920, 196), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_3], Original ATen: [aten.convolution]
        buf727 = extern_kernels.convolution(buf726, primals_264, stride=(2, 2), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf727, (8, 240, 7, 7), (11760, 49, 7, 1))
        buf728 = reinterpret_tensor(buf726, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf726  # reuse
        # Source Nodes: [cat_47], Original ATen: [aten.cat]
        triton_poi_fused_cat_144.run(buf721, buf723, buf725, buf727, buf728, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del buf721
        del buf723
        del buf725
        del buf727
        buf729 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf730 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf731 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_145.run(buf728, buf729, buf730, buf731, 3840, 98, grid=grid(3840), stream=stream0)
        buf732 = buf715; del buf715  # reuse
        buf733 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf735 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_146.run(buf729, buf730, buf731, primals_445, primals_446, buf732, buf733, buf735, primals_445, primals_446, 960, 4, grid=grid(960), stream=stream0)
        del buf729
        del buf730
        del buf731
        del primals_445
        del primals_446
        buf736 = reinterpret_tensor(buf658, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf658  # reuse
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_147.run(buf728, buf732, buf733, primals_93, primals_94, buf736, 376320, grid=grid(376320), stream=stream0)
        del buf733
        del primals_94
        buf737 = reinterpret_tensor(buf687, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf687  # reuse
        buf738 = reinterpret_tensor(buf737, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf737  # reuse
        # Source Nodes: [x_295, x_se_48], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_148.run(buf738, buf736, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf739 = extern_kernels.convolution(buf738, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf739, (8, 80, 1, 1), (80, 1, 1, 1))
        buf740 = reinterpret_tensor(buf739, (8, 80, 1, 1), (80, 1, 80, 80), 0); del buf739  # reuse
        buf741 = empty_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_128.run(buf740, primals_266, buf741, 640, grid=grid(640), stream=stream0)
        del primals_266
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf742 = extern_kernels.convolution(buf741, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf742, (8, 960, 1, 1), (960, 1, 1, 1))
        buf743 = reinterpret_tensor(buf742, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf742  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_149.run(buf743, primals_268, 7680, grid=grid(7680), stream=stream0)
        del primals_268
        buf744 = reinterpret_tensor(buf647, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf647  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_295, x_296], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_150.run(buf736, buf743, buf744, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf745 = extern_kernels.convolution(buf744, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf745, (8, 264, 7, 7), (12936, 49, 7, 1))
        buf746 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_151.run(buf745, buf746, 2112, 49, grid=grid(2112, 49), stream=stream0)
        buf747 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        buf748 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        buf749 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_152.run(buf746, buf747, buf748, buf749, 1056, 98, grid=grid(1056), stream=stream0)
        buf750 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf751 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf753 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_153.run(buf747, buf748, buf749, primals_448, primals_449, buf750, buf751, buf753, primals_448, primals_449, 264, 4, grid=grid(264), stream=stream0)
        del primals_448
        del primals_449
        buf754 = reinterpret_tensor(buf745, (8, 264, 7, 7), (12936, 1, 1848, 264), 0); del buf745  # reuse
        # Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_154.run(buf746, buf750, buf751, primals_95, primals_96, buf754, 103488, grid=grid(103488), stream=stream0)
        del primals_96
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        buf755 = extern_kernels.convolution(buf754, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf755, (8, 1584, 7, 7), (77616, 49, 7, 1))
        buf756 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_155.run(buf755, buf756, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf757 = empty_strided((1, 1584, 1, 1, 4), (6336, 1, 6336, 6336, 1584), device='cuda', dtype=torch.float32)
        buf758 = empty_strided((1, 1584, 1, 1, 4), (6336, 1, 6336, 6336, 1584), device='cuda', dtype=torch.float32)
        buf759 = empty_strided((1, 1584, 1, 1, 4), (6336, 1, 6336, 6336, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_156.run(buf756, buf757, buf758, buf759, 6336, 98, grid=grid(6336), stream=stream0)
        buf760 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf761 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf763 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_157.run(buf757, buf758, buf759, primals_451, primals_452, buf760, buf761, buf763, primals_451, primals_452, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_451
        del primals_452
        buf764 = reinterpret_tensor(buf755, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf755  # reuse
        buf918 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_158.run(buf756, buf760, buf761, primals_97, primals_98, buf764, buf918, 620928, grid=grid(620928), stream=stream0)
        del primals_98
        buf765 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_306], Original ATen: [aten.silu]
        triton_poi_fused_silu_159.run(buf764, buf765, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf766 = empty_strided((8, 396, 7, 7), (19404, 1, 2772, 396), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_160.run(buf765, buf766, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
        buf767 = extern_kernels.convolution(buf766, primals_271, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf767, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf768 = buf766; del buf766  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_161.run(buf765, buf768, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
        buf769 = extern_kernels.convolution(buf768, primals_272, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf769, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf770 = buf768; del buf768  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_162.run(buf765, buf770, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
        buf771 = extern_kernels.convolution(buf770, primals_273, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf771, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf772 = buf770; del buf770  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_163.run(buf765, buf772, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
        buf773 = extern_kernels.convolution(buf772, primals_274, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf773, (8, 396, 7, 7), (19404, 49, 7, 1))
        del buf772
        buf774 = buf764; del buf764  # reuse
        # Source Nodes: [cat_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_164.run(buf767, buf769, buf771, buf773, buf774, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf767
        del buf769
        del buf771
        buf775 = buf759; del buf759  # reuse
        buf776 = buf758; del buf758  # reuse
        buf777 = buf757; del buf757  # reuse
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_156.run(buf774, buf775, buf776, buf777, 6336, 98, grid=grid(6336), stream=stream0)
        buf778 = buf761; del buf761  # reuse
        buf779 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf781 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_157.run(buf775, buf776, buf777, primals_454, primals_455, buf778, buf779, buf781, primals_454, primals_455, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_454
        del primals_455
        buf782 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_165.run(buf774, buf778, buf779, primals_99, primals_100, buf782, 620928, grid=grid(620928), stream=stream0)
        del primals_100
        buf783 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf784 = reinterpret_tensor(buf783, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf783  # reuse
        # Source Nodes: [x_312, x_se_52], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_166.run(buf784, buf782, 12672, 49, grid=grid(12672), stream=stream0)
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf785 = extern_kernels.convolution(buf784, primals_275, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf785, (8, 132, 1, 1), (132, 1, 1, 1))
        buf786 = reinterpret_tensor(buf785, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf785  # reuse
        buf787 = reinterpret_tensor(buf749, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf749  # reuse
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_167.run(buf786, primals_276, buf787, 1056, grid=grid(1056), stream=stream0)
        del primals_276
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf788 = extern_kernels.convolution(buf787, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf788, (8, 1584, 1, 1), (1584, 1, 1, 1))
        buf789 = reinterpret_tensor(buf788, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf788  # reuse
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_168.run(buf789, primals_278, 12672, grid=grid(12672), stream=stream0)
        del primals_278
        buf790 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_312, x_313], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_169.run(buf782, buf789, buf790, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf791 = empty_strided((8, 792, 7, 7), (38808, 1, 5544, 792), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_170.run(buf790, buf791, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
        buf792 = extern_kernels.convolution(buf791, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf792, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf793 = buf791; del buf791  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_171.run(buf790, buf793, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
        buf794 = extern_kernels.convolution(buf793, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf794, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf795 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_45], Original ATen: [aten.cat]
        triton_poi_fused_cat_172.run(buf792, buf794, buf795, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf792
        del buf794
        buf796 = buf748; del buf748  # reuse
        buf797 = buf747; del buf747  # reuse
        buf798 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_152.run(buf795, buf796, buf797, buf798, 1056, 98, grid=grid(1056), stream=stream0)
        buf799 = buf751; del buf751  # reuse
        buf800 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf802 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_316], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_153.run(buf796, buf797, buf798, primals_457, primals_458, buf799, buf800, buf802, primals_457, primals_458, 264, 4, grid=grid(264), stream=stream0)
        del primals_457
        del primals_458
        buf803 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_173.run(buf795, buf799, buf800, primals_101, primals_102, buf754, buf803, 103488, grid=grid(103488), stream=stream0)
        del primals_102
        # Source Nodes: [x_321], Original ATen: [aten.convolution]
        buf804 = extern_kernels.convolution(buf803, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf804, (8, 1584, 7, 7), (77616, 49, 7, 1))
        buf805 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_321], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_155.run(buf804, buf805, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf806 = buf777; del buf777  # reuse
        buf807 = buf776; del buf776  # reuse
        buf808 = buf775; del buf775  # reuse
        # Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_156.run(buf805, buf806, buf807, buf808, 6336, 98, grid=grid(6336), stream=stream0)
        buf809 = buf779; del buf779  # reuse
        buf810 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf812 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_157.run(buf806, buf807, buf808, primals_460, primals_461, buf809, buf810, buf812, primals_460, primals_461, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_460
        del primals_461
        buf813 = reinterpret_tensor(buf804, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf804  # reuse
        buf917 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_158.run(buf805, buf809, buf810, primals_103, primals_104, buf813, buf917, 620928, grid=grid(620928), stream=stream0)
        del primals_104
        buf814 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325], Original ATen: [aten.silu]
        triton_poi_fused_silu_159.run(buf813, buf814, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf815 = reinterpret_tensor(buf773, (8, 396, 7, 7), (19404, 1, 2772, 396), 0); del buf773  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_160.run(buf814, buf815, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
        buf816 = extern_kernels.convolution(buf815, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf816, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf817 = buf815; del buf815  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_161.run(buf814, buf817, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
        buf818 = extern_kernels.convolution(buf817, primals_283, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf818, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf819 = buf817; del buf817  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_162.run(buf814, buf819, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
        buf820 = extern_kernels.convolution(buf819, primals_284, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf820, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf821 = buf819; del buf819  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_163.run(buf814, buf821, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
        buf822 = extern_kernels.convolution(buf821, primals_285, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf822, (8, 396, 7, 7), (19404, 49, 7, 1))
        del buf821
        buf823 = buf813; del buf813  # reuse
        # Source Nodes: [cat_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_164.run(buf816, buf818, buf820, buf822, buf823, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf816
        del buf818
        del buf820
        buf824 = buf808; del buf808  # reuse
        buf825 = buf807; del buf807  # reuse
        buf826 = buf806; del buf806  # reuse
        # Source Nodes: [x_328], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_156.run(buf823, buf824, buf825, buf826, 6336, 98, grid=grid(6336), stream=stream0)
        buf827 = buf810; del buf810  # reuse
        buf828 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf830 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_328], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_157.run(buf824, buf825, buf826, primals_463, primals_464, buf827, buf828, buf830, primals_463, primals_464, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_463
        del primals_464
        buf831 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_328], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_165.run(buf823, buf827, buf828, primals_105, primals_106, buf831, 620928, grid=grid(620928), stream=stream0)
        del primals_106
        buf832 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf833 = reinterpret_tensor(buf832, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf832  # reuse
        # Source Nodes: [x_331, x_se_56], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_166.run(buf833, buf831, 12672, 49, grid=grid(12672), stream=stream0)
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf834 = extern_kernels.convolution(buf833, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf834, (8, 132, 1, 1), (132, 1, 1, 1))
        buf835 = reinterpret_tensor(buf834, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf834  # reuse
        buf836 = reinterpret_tensor(buf798, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf798  # reuse
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_167.run(buf835, primals_287, buf836, 1056, grid=grid(1056), stream=stream0)
        del primals_287
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf837 = extern_kernels.convolution(buf836, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf837, (8, 1584, 1, 1), (1584, 1, 1, 1))
        buf838 = reinterpret_tensor(buf837, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf837  # reuse
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_168.run(buf838, primals_289, 12672, grid=grid(12672), stream=stream0)
        del primals_289
        buf839 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_331, x_332], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_169.run(buf831, buf838, buf839, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf840 = buf793; del buf793  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_170.run(buf839, buf840, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
        buf841 = extern_kernels.convolution(buf840, primals_290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf841, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf842 = buf840; del buf840  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_171.run(buf839, buf842, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
        buf843 = extern_kernels.convolution(buf842, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf843, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf844 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_172.run(buf841, buf843, buf844, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf841
        del buf843
        buf845 = buf797; del buf797  # reuse
        buf846 = buf796; del buf796  # reuse
        buf847 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_152.run(buf844, buf845, buf846, buf847, 1056, 98, grid=grid(1056), stream=stream0)
        buf848 = buf800; del buf800  # reuse
        buf849 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf851 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_153.run(buf845, buf846, buf847, primals_466, primals_467, buf848, buf849, buf851, primals_466, primals_467, 264, 4, grid=grid(264), stream=stream0)
        del primals_466
        del primals_467
        buf852 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_18, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_173.run(buf844, buf848, buf849, primals_107, primals_108, buf803, buf852, 103488, grid=grid(103488), stream=stream0)
        del primals_108
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        buf853 = extern_kernels.convolution(buf852, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf853, (8, 1584, 7, 7), (77616, 49, 7, 1))
        buf854 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_155.run(buf853, buf854, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf855 = buf826; del buf826  # reuse
        buf856 = buf825; del buf825  # reuse
        buf857 = buf824; del buf824  # reuse
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_156.run(buf854, buf855, buf856, buf857, 6336, 98, grid=grid(6336), stream=stream0)
        buf858 = buf828; del buf828  # reuse
        buf859 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf861 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_157.run(buf855, buf856, buf857, primals_469, primals_470, buf858, buf859, buf861, primals_469, primals_470, 1584, 4, grid=grid(1584), stream=stream0)
        del primals_469
        del primals_470
        buf862 = reinterpret_tensor(buf853, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf853  # reuse
        buf916 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_158.run(buf854, buf858, buf859, primals_109, primals_110, buf862, buf916, 620928, grid=grid(620928), stream=stream0)
        del primals_110
        buf863 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_344], Original ATen: [aten.silu]
        triton_poi_fused_silu_159.run(buf862, buf863, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf864 = reinterpret_tensor(buf822, (8, 396, 7, 7), (19404, 1, 2772, 396), 0); del buf822  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_160.run(buf863, buf864, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
        buf865 = extern_kernels.convolution(buf864, primals_293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf865, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf866 = buf864; del buf864  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_161.run(buf863, buf866, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
        buf867 = extern_kernels.convolution(buf866, primals_294, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf867, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf868 = buf866; del buf866  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_162.run(buf863, buf868, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
        buf869 = extern_kernels.convolution(buf868, primals_295, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf869, (8, 396, 7, 7), (19404, 49, 7, 1))
        buf870 = buf868; del buf868  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_163.run(buf863, buf870, 3168, 49, grid=grid(3168, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
        buf871 = extern_kernels.convolution(buf870, primals_296, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
        assert_size_stride(buf871, (8, 396, 7, 7), (19404, 49, 7, 1))
        del buf870
        buf872 = buf862; del buf862  # reuse
        # Source Nodes: [cat_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_164.run(buf865, buf867, buf869, buf871, buf872, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf865
        del buf867
        del buf869
        del buf871
        buf873 = buf857; del buf857  # reuse
        buf874 = buf856; del buf856  # reuse
        buf875 = buf855; del buf855  # reuse
        # Source Nodes: [x_347], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_156.run(buf872, buf873, buf874, buf875, 6336, 98, grid=grid(6336), stream=stream0)
        buf876 = buf859; del buf859  # reuse
        buf877 = empty_strided((1, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda', dtype=torch.float32)
        buf879 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_347], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_157.run(buf873, buf874, buf875, primals_472, primals_473, buf876, buf877, buf879, primals_472, primals_473, 1584, 4, grid=grid(1584), stream=stream0)
        del buf873
        del buf874
        del buf875
        del primals_472
        del primals_473
        buf880 = empty_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_347], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_165.run(buf872, buf876, buf877, primals_111, primals_112, buf880, 620928, grid=grid(620928), stream=stream0)
        del buf877
        del primals_112
        buf881 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf882 = reinterpret_tensor(buf881, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf881  # reuse
        # Source Nodes: [x_350, x_se_60], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_166.run(buf882, buf880, 12672, 49, grid=grid(12672), stream=stream0)
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (8, 132, 1, 1), (132, 1, 1, 1))
        buf884 = reinterpret_tensor(buf883, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf883  # reuse
        buf885 = reinterpret_tensor(buf847, (8, 132, 1, 1), (132, 1, 132, 132), 0); del buf847  # reuse
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_167.run(buf884, primals_298, buf885, 1056, grid=grid(1056), stream=stream0)
        del primals_298
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf886 = extern_kernels.convolution(buf885, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf886, (8, 1584, 1, 1), (1584, 1, 1, 1))
        buf887 = reinterpret_tensor(buf886, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf886  # reuse
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_168.run(buf887, primals_300, 12672, grid=grid(12672), stream=stream0)
        del primals_300
        buf888 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_350, x_351], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_169.run(buf880, buf887, buf888, 12672, 49, grid=grid(12672, 49), stream=stream0)
        buf889 = buf842; del buf842  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_170.run(buf888, buf889, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
        buf890 = extern_kernels.convolution(buf889, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf890, (8, 132, 7, 7), (6468, 49, 7, 1))
        buf891 = buf889; del buf889  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_171.run(buf888, buf891, 6336, 49, grid=grid(6336, 49), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
        buf892 = extern_kernels.convolution(buf891, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf892, (8, 132, 7, 7), (6468, 49, 7, 1))
        del buf891
        buf893 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_41], Original ATen: [aten.cat]
        triton_poi_fused_cat_172.run(buf890, buf892, buf893, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf890
        del buf892
        buf894 = buf846; del buf846  # reuse
        buf895 = buf845; del buf845  # reuse
        buf896 = empty_strided((1, 264, 1, 1, 4), (1056, 1, 1056, 1056, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_152.run(buf893, buf894, buf895, buf896, 1056, 98, grid=grid(1056), stream=stream0)
        buf897 = buf849; del buf849  # reuse
        buf898 = empty_strided((1, 264, 1, 1), (264, 1, 264, 264), device='cuda', dtype=torch.float32)
        buf900 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_153.run(buf894, buf895, buf896, primals_475, primals_476, buf897, buf898, buf900, primals_475, primals_476, 264, 4, grid=grid(264), stream=stream0)
        del buf894
        del buf895
        del buf896
        del primals_475
        del primals_476
        buf901 = empty_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354, x_359], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_173.run(buf893, buf897, buf898, primals_113, primals_114, buf852, buf901, 103488, grid=grid(103488), stream=stream0)
        del buf898
        del primals_114
        # Source Nodes: [x_360], Original ATen: [aten.convolution]
        buf902 = extern_kernels.convolution(buf901, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf902, (8, 1536, 7, 7), (75264, 49, 7, 1))
        buf903 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_360], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_174.run(buf902, buf903, 12288, 49, grid=grid(12288, 49), stream=stream0)
        buf904 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        buf905 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        buf906 = empty_strided((1, 1536, 1, 1, 4), (6144, 1, 6144, 6144, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_361], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_175.run(buf903, buf904, buf905, buf906, 6144, 98, grid=grid(6144), stream=stream0)
        buf907 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf908 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf910 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_361], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_176.run(buf904, buf905, buf906, primals_478, primals_479, buf907, buf908, buf910, primals_478, primals_479, 1536, 4, grid=grid(1536), stream=stream0)
        del buf904
        del buf905
        del buf906
        del primals_478
        del primals_479
        buf911 = reinterpret_tensor(buf902, (8, 1536, 7, 7), (75264, 1, 10752, 1536), 0); del buf902  # reuse
        buf915 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_361, x_365], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_177.run(buf903, buf907, buf908, primals_115, primals_116, buf911, buf915, 602112, grid=grid(602112), stream=stream0)
        del buf908
        del primals_116
        buf912 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf913 = reinterpret_tensor(buf912, (8, 1536), (1536, 1), 0); del buf912  # reuse
        # Source Nodes: [x_366, x_368], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_178.run(buf913, buf911, 12288, 49, grid=grid(12288), stream=stream0)
        del buf911
        buf914 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_305, buf913, reinterpret_tensor(primals_304, (1536, 1000), (1, 1536), 0), alpha=1, beta=1, out=buf914)
        del primals_305
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_345, primals_345, 1, grid=grid(1), stream=stream0)
        del primals_345
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_348, primals_348, 1, grid=grid(1), stream=stream0)
        del primals_348
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_351, primals_351, 1, grid=grid(1), stream=stream0)
        del primals_351
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_354, primals_354, 1, grid=grid(1), stream=stream0)
        del primals_354
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_357, primals_357, 1, grid=grid(1), stream=stream0)
        del primals_357
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_360, primals_360, 1, grid=grid(1), stream=stream0)
        del primals_360
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_363, primals_363, 1, grid=grid(1), stream=stream0)
        del primals_363
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_366, primals_366, 1, grid=grid(1), stream=stream0)
        del primals_366
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_369, primals_369, 1, grid=grid(1), stream=stream0)
        del primals_369
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_372, primals_372, 1, grid=grid(1), stream=stream0)
        del primals_372
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_375, primals_375, 1, grid=grid(1), stream=stream0)
        del primals_375
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_378, primals_378, 1, grid=grid(1), stream=stream0)
        del primals_378
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_381, primals_381, 1, grid=grid(1), stream=stream0)
        del primals_381
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_384, primals_384, 1, grid=grid(1), stream=stream0)
        del primals_384
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_387, primals_387, 1, grid=grid(1), stream=stream0)
        del primals_387
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_390, primals_390, 1, grid=grid(1), stream=stream0)
        del primals_390
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_393, primals_393, 1, grid=grid(1), stream=stream0)
        del primals_393
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_396, primals_396, 1, grid=grid(1), stream=stream0)
        del primals_396
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_399, primals_399, 1, grid=grid(1), stream=stream0)
        del primals_399
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_402, primals_402, 1, grid=grid(1), stream=stream0)
        del primals_402
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_405, primals_405, 1, grid=grid(1), stream=stream0)
        del primals_405
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_408, primals_408, 1, grid=grid(1), stream=stream0)
        del primals_408
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_411, primals_411, 1, grid=grid(1), stream=stream0)
        del primals_411
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_414, primals_414, 1, grid=grid(1), stream=stream0)
        del primals_414
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_417, primals_417, 1, grid=grid(1), stream=stream0)
        del primals_417
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_420, primals_420, 1, grid=grid(1), stream=stream0)
        del primals_420
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_423, primals_423, 1, grid=grid(1), stream=stream0)
        del primals_423
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_426, primals_426, 1, grid=grid(1), stream=stream0)
        del primals_426
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_429, primals_429, 1, grid=grid(1), stream=stream0)
        del primals_429
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_432, primals_432, 1, grid=grid(1), stream=stream0)
        del primals_432
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_435, primals_435, 1, grid=grid(1), stream=stream0)
        del primals_435
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_438, primals_438, 1, grid=grid(1), stream=stream0)
        del primals_438
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_441, primals_441, 1, grid=grid(1), stream=stream0)
        del primals_441
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_444, primals_444, 1, grid=grid(1), stream=stream0)
        del primals_444
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_447, primals_447, 1, grid=grid(1), stream=stream0)
        del primals_447
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_450, primals_450, 1, grid=grid(1), stream=stream0)
        del primals_450
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_453, primals_453, 1, grid=grid(1), stream=stream0)
        del primals_453
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_456, primals_456, 1, grid=grid(1), stream=stream0)
        del primals_456
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_459, primals_459, 1, grid=grid(1), stream=stream0)
        del primals_459
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_462, primals_462, 1, grid=grid(1), stream=stream0)
        del primals_462
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_465, primals_465, 1, grid=grid(1), stream=stream0)
        del primals_465
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_468, primals_468, 1, grid=grid(1), stream=stream0)
        del primals_468
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_471, primals_471, 1, grid=grid(1), stream=stream0)
        del primals_471
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_474, primals_474, 1, grid=grid(1), stream=stream0)
        del primals_474
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_179.run(primals_477, primals_477, 1, grid=grid(1), stream=stream0)
        del primals_477
        return (buf914, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, buf0, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_158, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_178, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_232, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_244, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_256, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, buf1, buf3, buf13, buf14, buf16, buf26, buf27, buf29, buf39, reinterpret_tensor(buf40, (8, 16, 112, 112), (401408, 12544, 112, 1), 0), reinterpret_tensor(buf40, (8, 16, 112, 112), (401408, 12544, 112, 1), 200704), buf45, buf55, reinterpret_tensor(buf56, (8, 64, 112, 112), (2408448, 1, 21504, 192), 0), reinterpret_tensor(buf56, (8, 64, 112, 112), (2408448, 1, 21504, 192), 64), reinterpret_tensor(buf56, (8, 64, 112, 112), (2408448, 1, 21504, 192), 128), buf60, buf70, reinterpret_tensor(buf71, (8, 96, 56, 56), (602112, 1, 10752, 192), 0), reinterpret_tensor(buf71, (8, 96, 56, 56), (602112, 1, 10752, 192), 96), buf74, buf84, reinterpret_tensor(buf85, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), reinterpret_tensor(buf85, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), buf88, buf98, buf99, buf101, buf111, reinterpret_tensor(buf112, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), reinterpret_tensor(buf112, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), buf115, buf125, buf126, buf128, buf138, reinterpret_tensor(buf140, (8, 60, 56, 56), (752640, 3136, 56, 1), 0), reinterpret_tensor(buf140, (8, 60, 56, 56), (752640, 3136, 56, 1), 188160), reinterpret_tensor(buf140, (8, 60, 56, 56), (752640, 3136, 56, 1), 376320), reinterpret_tensor(buf140, (8, 60, 56, 56), (752640, 3136, 56, 1), 564480), buf149, buf156, buf157, buf160, buf162, buf163, buf165, buf166, buf168, buf175, reinterpret_tensor(buf176, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), reinterpret_tensor(buf176, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), buf179, buf186, reinterpret_tensor(buf188, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf188, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf193, buf200, buf201, buf204, buf206, buf207, buf209, reinterpret_tensor(buf210, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf210, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf215, buf222, reinterpret_tensor(buf223, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), reinterpret_tensor(buf223, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), buf226, buf233, reinterpret_tensor(buf235, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf235, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf240, buf247, buf248, buf251, buf253, buf254, buf256, reinterpret_tensor(buf257, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf257, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf262, buf269, reinterpret_tensor(buf270, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), reinterpret_tensor(buf270, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), buf273, buf280, reinterpret_tensor(buf282, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf282, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf287, buf294, buf295, buf298, buf300, buf301, buf303, reinterpret_tensor(buf304, (8, 168, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf304, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), buf309, buf316, buf317, buf319, buf326, reinterpret_tensor(buf328, (8, 112, 28, 28), (263424, 784, 28, 1), 0), reinterpret_tensor(buf328, (8, 112, 28, 28), (263424, 784, 28, 1), 87808), reinterpret_tensor(buf328, (8, 112, 28, 28), (263424, 784, 28, 1), 175616), buf335, buf342, buf343, buf346, buf348, buf349, buf351, buf352, buf354, buf361, reinterpret_tensor(buf362, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), reinterpret_tensor(buf362, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), buf365, buf372, reinterpret_tensor(buf374, (8, 156, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf374, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), reinterpret_tensor(buf374, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), reinterpret_tensor(buf374, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), buf383, buf390, buf391, buf394, buf396, buf397, buf399, reinterpret_tensor(buf400, (8, 312, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf400, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), buf405, buf412, reinterpret_tensor(buf413, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), reinterpret_tensor(buf413, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), buf416, buf423, reinterpret_tensor(buf425, (8, 156, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf425, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), reinterpret_tensor(buf425, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), reinterpret_tensor(buf425, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), buf434, buf441, buf442, buf445, buf447, buf448, buf450, reinterpret_tensor(buf451, (8, 312, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf451, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), buf456, buf463, reinterpret_tensor(buf464, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), reinterpret_tensor(buf464, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), buf467, buf474, reinterpret_tensor(buf476, (8, 156, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf476, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), reinterpret_tensor(buf476, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), reinterpret_tensor(buf476, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), buf485, buf492, buf493, buf496, buf498, buf499, buf501, reinterpret_tensor(buf502, (8, 312, 14, 14), (122304, 196, 14, 1), 0), reinterpret_tensor(buf502, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), buf507, buf514, buf515, buf517, buf524, buf526, buf528, buf535, buf536, buf539, buf541, buf542, buf544, buf545, buf547, buf554, reinterpret_tensor(buf555, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), reinterpret_tensor(buf555, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), buf558, buf565, reinterpret_tensor(buf567, (8, 120, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf567, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), reinterpret_tensor(buf567, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), reinterpret_tensor(buf567, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), buf576, buf583, buf584, buf587, buf589, buf590, buf592, reinterpret_tensor(buf593, (8, 240, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf593, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), buf598, buf605, reinterpret_tensor(buf606, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), reinterpret_tensor(buf606, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), buf609, buf616, reinterpret_tensor(buf618, (8, 120, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf618, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), reinterpret_tensor(buf618, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), reinterpret_tensor(buf618, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), buf627, buf634, buf635, buf638, buf640, buf641, buf643, reinterpret_tensor(buf644, (8, 240, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf644, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), buf649, buf656, reinterpret_tensor(buf657, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), reinterpret_tensor(buf657, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), buf660, buf667, reinterpret_tensor(buf669, (8, 120, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf669, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), reinterpret_tensor(buf669, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), reinterpret_tensor(buf669, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), buf678, buf685, buf686, buf689, buf691, buf692, buf694, reinterpret_tensor(buf695, (8, 240, 14, 14), (94080, 196, 14, 1), 0), reinterpret_tensor(buf695, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), buf700, buf707, buf708, buf710, buf717, reinterpret_tensor(buf719, (8, 240, 14, 14), (188160, 196, 14, 1), 0), reinterpret_tensor(buf719, (8, 240, 14, 14), (188160, 196, 14, 1), 47040), reinterpret_tensor(buf719, (8, 240, 14, 14), (188160, 196, 14, 1), 94080), reinterpret_tensor(buf719, (8, 240, 14, 14), (188160, 196, 14, 1), 141120), buf728, buf735, buf736, buf738, buf740, buf741, buf743, buf744, buf746, buf753, buf754, buf756, buf763, reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), reinterpret_tensor(buf765, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), buf774, buf781, buf782, buf784, buf786, buf787, buf789, reinterpret_tensor(buf790, (8, 792, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf790, (8, 792, 7, 7), (77616, 49, 7, 1), 38808), buf795, buf802, buf803, buf805, buf812, reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), reinterpret_tensor(buf814, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), buf823, buf830, buf831, buf833, buf835, buf836, buf838, reinterpret_tensor(buf839, (8, 792, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf839, (8, 792, 7, 7), (77616, 49, 7, 1), 38808), buf844, buf851, buf852, buf854, buf861, reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), reinterpret_tensor(buf863, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), buf872, buf879, buf880, buf882, buf884, buf885, buf887, reinterpret_tensor(buf888, (8, 792, 7, 7), (77616, 49, 7, 1), 0), reinterpret_tensor(buf888, (8, 792, 7, 7), (77616, 49, 7, 1), 38808), buf893, buf900, buf901, buf903, buf910, buf913, reinterpret_tensor(primals_304, (1000, 1536), (1536, 1), 0), buf915, reinterpret_tensor(buf907, (1, 1536, 1, 1), (1536, 1, 1, 1), 0), reinterpret_tensor(buf897, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf876, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), buf916, reinterpret_tensor(buf858, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), reinterpret_tensor(buf848, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf827, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), buf917, reinterpret_tensor(buf809, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), reinterpret_tensor(buf799, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf778, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), buf918, reinterpret_tensor(buf760, (1, 1584, 1, 1), (1584, 1, 1, 1), 0), reinterpret_tensor(buf750, (1, 264, 1, 1), (264, 1, 1, 1), 0), reinterpret_tensor(buf732, (1, 960, 1, 1), (960, 1, 1, 1), 0), buf919, reinterpret_tensor(buf714, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf704, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf682, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf920, reinterpret_tensor(buf664, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf653, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf631, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf921, reinterpret_tensor(buf613, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf602, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf580, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf922, reinterpret_tensor(buf562, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf551, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf532, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf923, reinterpret_tensor(buf521, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf511, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf489, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf924, reinterpret_tensor(buf471, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf460, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf438, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf925, reinterpret_tensor(buf420, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf409, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 624, 1, 1), (624, 1, 1, 1), 0), buf926, reinterpret_tensor(buf369, (1, 624, 1, 1), (624, 1, 1, 1), 0), reinterpret_tensor(buf358, (1, 104, 1, 1), (104, 1, 1, 1), 0), reinterpret_tensor(buf339, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf927, reinterpret_tensor(buf323, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf313, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf291, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf928, reinterpret_tensor(buf277, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf266, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf244, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf929, reinterpret_tensor(buf230, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf197, (1, 336, 1, 1), (336, 1, 1, 1), 0), buf930, reinterpret_tensor(buf183, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf172, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf931, reinterpret_tensor(buf135, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf122, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf932, reinterpret_tensor(buf108, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf95, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf933, reinterpret_tensor(buf67, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf934, reinterpret_tensor(buf52, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf36, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf23, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
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
    compiled_module_main('mixnet_l', benchmark_compiled_module)
