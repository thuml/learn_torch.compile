
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtjh24prmexvvkuuwa353z2t4zcrhpxepjdtjmzrod7rpi6wxxd.py
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
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxfvjtaoo3x3r2xvx3jfggbvzxqjc3nhiwe52brww3bcipojlip.py
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
    size_hints=[128, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqi3bcospdshhy77arzdzftlmesnjpnyigu3ffsb6wq2goc7yspj.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5a/c5avmt44lal22e7sgkmi7othfki4qesr5amwmtrkkeswvbcp5vw2.py
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
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rq/crqu452wa4jiw5vzg56vgktejk4ke3hd66i5k236dy6skxihgn5y.py
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
    size_hints=[16, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/vu/cvu7hsq5w5j7rfe26kjvgxl3l3lofkirulcqnzsj4kka6lzesy3g.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_6', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvx4yehfgzk6ml7alw733sscgktwq57q6tq7q7iruau56e5wnv5q.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0 => convolution_1
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (8*x2) + (100352*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chcoifn65zkso7l5ie5berqdugurrqlym72gmmzkh5h7xeadnlag.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (8*r2) + (1024*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kb/ckbcfdwfxbkaurk5kj7azlm4wswvnqdi2ogbcbu2tnc3rky4w7d5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56
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
        tmp0 = tl.load(in_ptr0 + (x1 + (8*r2) + (896*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (8*r2) + (896*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (8*r2) + (896*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (8*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (8*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (8*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5r2rr6pmovfoneebkpjcmwsyvkeifo7s5m2s7lmgsxnhjr3nlhg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (8*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (8*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2j/c2j36scl7ici3ln5omh2a3d7svrnv2bike6z7p2ebbnktaghasig.py
# Source Nodes: [cat_63, getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1, x1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_63 => cat
# getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# x1 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    x1 = (xindex // 8)
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
    tl.store(out_ptr1 + (x0 + (16*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwrpwbxpoh3bowbb4yzhxjccjexnuudljk2lzqberz4ydpuczks.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1, x2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    x1 = (xindex // 8)
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
    tl.store(out_ptr0 + (x0 + (16*x1)), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czoouzypymm5kktyvyqby53s6vimvvuo5y5aid7pwxsbjgq7wykv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
triton_poi_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
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


# kernel path: /tmp/torchinductor_youkaichao/or/corywvyqs43e3idojn4shlct2gkd4z2gwwiul7blbhlmicnifkmf.py
# Source Nodes: [cat_62, shortcut_1], Original ATen: [aten.add, aten.cat]
# cat_62 => cat_1
# shortcut_1 => add_25
triton_poi_fused_add_cat_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), None)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (8*x1)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-8) + x0 + (8*x1)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-8) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-8) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp15 = 100352.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-8) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-8) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coevlszuppapsyo5hcmkxab3tz4mn3z3m27j2x453sx6z4amzved.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0 => convolution_5
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7t7d7uwsbtryqcf4wl3qahqvydejyq5ikiznbkimryprmjylrb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
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


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4qwdzedeohw2dqrq332zfkbwu2qgcl3otyfovscnm5hc2mbdpo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 168
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
        tmp0 = tl.load(in_ptr0 + (x1 + (24*r2) + (2688*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (2688*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (24*r2) + (2688*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ju/cju4nnyiyn3hn2xfyinfthn62t72hx5rbglsabw36ekfmgw6iz6h.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 => add_27, add_28, add_29, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3elhsuvzncbzj7t54r57iechmbibfim6in4e2hqdncwo4r5ip5.py
# Source Nodes: [cat_61, getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1, x1_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_61 => cat_2
# getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 => add_27, add_30, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x1_2 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    x1 = (xindex // 24)
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
    tl.store(out_ptr1 + (x0 + (48*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ynpmnrlnfhbtgvms23px4r5csgyiz4jk5eetj5eo6rc57qr6r4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1, x2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1 => add_32, add_35, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x2_2 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    x1 = (xindex // 24)
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
    tl.store(out_ptr0 + (x0 + (48*x1)), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cse46ryz7y7ksu2yhv56gnamld5w2ejc6owdub4owhhaterkwajw.py
# Source Nodes: [x_7], Original ATen: [aten.convolution]
# x_7 => convolution_7
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3p/c3py2w3gxbckifahsgiufch3osqrh2laxpkjpqrdfe2nxp7bovly.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4d/c4d4k7gi6npofklmb5h3ukzk3alb5xzantlx47yhgsccxj3dvkog.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/56/c56yckkcq5byrshjdzg5krhg5aovwbfk4ibw3f7cm6nauzg5tuej.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => add_37, add_38, add_39, mul_50, mul_51, mul_52, mul_53, mul_54, rsqrt_7, squeeze_22, var_mean_7
triton_per_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3pcgbmgecdanttqcn44t3tu6frihoo3regzqp7v3whoj4ih5aa.py
# Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_8 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
triton_poi_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5aemt67v4ee3f7hiema2tixhar4ahqpu2yk6vz3bk34snmywgg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0 => convolution_8
triton_poi_fused_convolution_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12
    y1 = (yindex // 12)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (12*x2) + (37632*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2knqax3lscszqi2gxod3e5iwr6jbfs44ffjbk5xv23h5vb2yfy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 => var_mean_8
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 12
    x1 = (xindex // 12)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (12*r2) + (1536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5rb735ukujjr7y7en3akrc3atsxwty27l2ajcp5wpww32cgs25.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 => var_mean_8
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
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
        tmp0 = tl.load(in_ptr0 + (x1 + (12*r2) + (1176*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (12*r2) + (1176*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (12*r2) + (1176*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (12*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (12*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (12*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmby77427i7signvzlez2x4ypfviu4p3iinclsuv3iad6ka2bka.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 => add_42, add_43, add_44, mul_57, mul_58, mul_59, mul_60, mul_61, rsqrt_8, squeeze_25, var_mean_8
triton_per_fused__native_batch_norm_legit_functional_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_29', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (12*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (12*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (12*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6u5bon724lbyelddakqf43npluk6zbznzh2o2cpzxdpyy5akxf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 => add_42, add_45, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 12
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


# kernel path: /tmp/torchinductor_youkaichao/da/cdam2qgwlni2d7azu23qi6sasmv6oxrasufx2ww5tdg2jwu7imor.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_0 => convolution_10
triton_poi_fused_convolution_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (50176*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyffxlht2y5nh6uhdhshrtnvcw53m5igu7kqomhgxip2hbr4vfzn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_1 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
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


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzjfphofbpqbvluchrcxk5537jhqpufayi3tu2mb67oejhcpiwe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_1 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
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
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (1568*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (16*r2) + (1568*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (16*r2) + (1568*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ky/ckyixwgaojmw3rbzm7p4ph3nfoxhyrxh4mgl6cj63d5kcm76liqi.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_1 => add_52, add_53, add_54, mul_71, mul_72, mul_73, mul_74, mul_75, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/c5/cc57piiw37nwuqxtsr5offdjpr2cp3jlcvle4kp2cieizlauc3w3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_1 => add_52, add_55, mul_70, mul_76, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_youkaichao/vo/cvomkxbrou2sefxo6junl3o6m6hk4ku6aircw3yx5cg33b3awymr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_2 => convolution_11
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crh64kismos3pbevnfksvb56f56czl25cjdbfwy22ztx7ngfvymc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_3 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lcy5xrny2g7jtcevpmrbsaby7ehn5yeotlntm5jdohcwzdesbe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_3 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tayoqjmlhfpvy5ndqk4b2uxegd3e5ortkxcrlzpafyyts3b7nc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___1_____0___shortcut_3 => add_57, add_58, add_59, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/li/cli3dulalevt57xesd5an3g7eyavpv7q2jxyptgkdt46t27pzlj4.py
# Source Nodes: [cat_60, getattr_getattr_l__mod___blocks___1_____0___shortcut_3, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
# cat_60 => cat_3
# getattr_getattr_l__mod___blocks___1_____0___shortcut_3 => add_57, add_60, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# shortcut_2 => add_61
triton_poi_fused__native_batch_norm_legit_functional_add_cat_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), None)
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (12*x1)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 24, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-12) + x0 + (12*x1)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp15 = 25088.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 / tmp15
    tmp33 = tmp32 + tmp17
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp30 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp27 + tmp39
    tl.store(out_ptr0 + (x2), tmp40, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqk6atnv7eortxhr764jv5dubufd2dzzucah6svu32ytvhalhydo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0 => convolution_12
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
    ynumel = 288
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (36*x2) + (112896*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crj6oofwa5wbbd2mgfextvv7zvfw2mwnulohrij7obuwciiwnzhx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 => var_mean_12
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 36
    x1 = (xindex // 36)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (36*r2) + (4608*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/oo/cooz342qvi5lfke72nfxbs3a22n3hnvwj4gsukprwbot7bkrsscq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (x1 + (36*r2) + (3528*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (36*r2) + (3528*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (36*r2) + (3528*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (36*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (36*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (36*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvr7klfindnuw76qmj7ejxb4temg2yktu5xnmae7jo75sb5zzvye.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 => add_63, add_64, add_65, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (36*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (36*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (36*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck346x52pwhdx75rq3epxdqhp3hqnoknsntqc3bhns6uncvqwbnx.py
# Source Nodes: [cat_59, getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1, x1_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_59 => cat_4
# getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 => add_63, add_66, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# x1_4 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    x1 = (xindex // 36)
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
    tl.store(out_ptr1 + (x0 + (72*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/66/c66ng3ce6wm7ospqac4qcg3p2zqujbprdonvarj636i3kh3tomra.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1, x2_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1 => add_68, add_71, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# x2_4 => relu_6
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    x1 = (xindex // 36)
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
    tl.store(out_ptr0 + (x0 + (72*x1)), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qp/cqppv6jcytdgcn7brxz7pt6rarijoi3o3nurvscic75ihy6zaopb.py
# Source Nodes: [cat_58, shortcut_3], Original ATen: [aten.add, aten.cat]
# cat_58 => cat_5
# shortcut_3 => add_82
triton_poi_fused_add_cat_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), None)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (12*x1)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 24, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-12) + x0 + (12*x1)), tmp8, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp15 = 25088.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-12) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzocz6mwcvcjw3oz5fllh3ejrzgvm4aaw4vnbtilavmydxhq4ce.py
# Source Nodes: [x_15], Original ATen: [aten.convolution]
# x_15 => convolution_18
triton_poi_fused_convolution_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctctwlaba7v6nitqg4y7p645cncyqe7be22di6ezy6635o4o2wtn.py
# Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_functional]
# x_16 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_49', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwrcnzmxalnlypf4txgadwczxjgcdn54cdd2aykh4xx5gjcnzui.py
# Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_functional]
# x_16 => add_94, add_95, add_96, mul_127, mul_128, mul_129, mul_130, mul_131, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/bi/cbi6evocp2wfpam3bhmgjpomxs5ncvrz5v6ueclrmr32hpywzslv.py
# Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_functional]
# x_16 => add_94, add_97, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
triton_poi_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33m5wugmoaaf65hfm4ychgonk5zlnwopihi4key72k2tzkyewnc.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_red_fused_mean_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (8064*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cuemydliccb73otfa55ofq6xqum35wf5hqffzu6o63d3nirxzvit.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_per_fused_mean_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_53', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 72
    x1 = (xindex // 72)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (504*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxwympqoxaw2k2kbvct7wiawotmrhnz6krwc6d4odqe7scwbxv3.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_19
# x_se_2 => relu_9
triton_poi_fused_convolution_relu_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqabons2ortlck6wjfrdraxkto3zn4yualastikvqgt7ihtggzvs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => add_98, clamp_max, clamp_min, div
# x_se_3 => convolution_20
triton_poi_fused_convolution_hardsigmoid_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblteeu22m3coxrxhncizpkkpgauqvudhtm6o6kge3yr4wapta4h.py
# Source Nodes: [x_17], Original ATen: [aten.mul]
# x_17 => mul_133
triton_poi_fused_mul_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 72
    x2 = (xindex // 56448)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (72*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmqkzlvodthucu5hshyzcfw5okcp3x6t4j75z3smhmybwww2zak.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0 => convolution_21
triton_poi_fused_convolution_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 20
    y1 = (yindex // 20)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (20*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cupgennpolb7gjfma7ji346sbnai6lfnt5pez2lmmklqnnq6n4gb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1 => var_mean_19
triton_red_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 980
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 20
    x1 = (xindex // 20)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (20*r2) + (2560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3rfbeglhrjghnhqy4ctb6qpcpbmbu3iifcjyp6qmadpnn54lxe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1 => add_100, add_101, add_102, mul_135, mul_136, mul_137, mul_138, mul_139, rsqrt_19, squeeze_58, var_mean_19
triton_per_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (20*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (20*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (20*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/je/cjenxpehb44t5e6boy5ksiyt5foiq5mdyikwd4tfazak5sdhjltw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1 => add_100, add_103, mul_134, mul_140, rsqrt_19, sub_19, var_mean_19
triton_poi_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
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


# kernel path: /tmp/torchinductor_youkaichao/ue/cuejsnvxppzlmckeq4vgnljz44nd26f3sc4vu4wtqnjtjevqiodo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_0 => convolution_23
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (18816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cunqq72ktip6zlfsoevjudotm3ehrx73eu47ohjylxsk3jawfn3h.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_1 => var_mean_21
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1176
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


# kernel path: /tmp/torchinductor_youkaichao/5w/c5widrvij324tq76ts67exkrxoq4y5sh5r7k53lmoativnywssb4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_1 => add_110, add_111, add_112, mul_149, mul_150, mul_151, mul_152, mul_153, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/cl/ccl3xcbq2t73jmrcdze3llvzt7pxrzadslfzpcskecdnsw37o5vz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_1 => add_110, add_113, mul_148, mul_154, rsqrt_21, sub_21, var_mean_21
triton_poi_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
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


# kernel path: /tmp/torchinductor_youkaichao/zh/czhu742wacwk2icmeu3uplook4kbkeulhhqlxgz7oaj3fa45a2lv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_2 => convolution_24
triton_poi_fused_convolution_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_65', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbl6xwzpcoexvcdybvubhpyyyzuacd7fwffzfnrxfsomdu3ymya.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_3 => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vn/cvndy75eeraf27fkr2i4hqb7bdtjqtubb7wtl3pxmbdt2np4cjze.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___3_____0___shortcut_3 => add_115, add_116, add_117, mul_156, mul_157, mul_158, mul_159, mul_160, rsqrt_22, squeeze_67, var_mean_22
triton_per_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_67', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2jwauqlda5rptnmvqpj3zp2kgnqza6bohfjx5kewrjdzubs7jv.py
# Source Nodes: [cat_56, getattr_getattr_l__mod___blocks___3_____0___shortcut_3, shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
# cat_56 => cat_7
# getattr_getattr_l__mod___blocks___3_____0___shortcut_3 => add_115, add_118, mul_155, mul_161, rsqrt_22, sub_22, var_mean_22
# shortcut_4 => add_119
triton_poi_fused__native_batch_norm_legit_functional_add_cat_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (20*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-20) + x0 + (20*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 6272.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 / tmp15
    tmp33 = tmp32 + tmp17
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp30 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp27 + tmp39
    tl.store(out_ptr0 + (x2), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5a5uxb6llhtyohftaoqtuoi6zsw74aucwce5qila7xn3qlghm5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0 => convolution_25
triton_poi_fused_convolution_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (60*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpx57kbr3pehkezlmc5sp46bdyqygrootch65tazuvnwi4mmc53h.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1 => var_mean_23
triton_red_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2940
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 60
    x1 = (xindex // 60)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (60*r2) + (7680*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2tokol2i6r3hiye5xmtl6daecc6p46g6565xf335e7bvhbisir.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1 => add_121, add_122, add_123, mul_163, mul_164, mul_165, mul_166, mul_167, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 60
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (60*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (60*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (60*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5obibjbypcvnm25gcorl25ay3jlz2otgu3bw743vzmr7shv5io.py
# Source Nodes: [cat_55, getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1, x1_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_55 => cat_8
# getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1 => add_121, add_124, mul_162, mul_168, rsqrt_23, sub_23, var_mean_23
# x1_8 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 60
    x1 = (xindex // 60)
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
    tl.store(out_ptr1 + (x0 + (120*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvicealqon2midnwl7vhhn32d6v3dufvfi5ee2cd57b2wuhaowfx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1, x2_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1 => add_126, add_129, mul_169, mul_175, rsqrt_24, sub_24, var_mean_24
# x2_8 => relu_11
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 60
    x1 = (xindex // 60)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (120*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cnaolwdd2p5yepcjuo43bzr6hhyandy5ejsvdqe4eexkjpv5xb6d.py
# Source Nodes: [x_se_4], Original ATen: [aten.mean]
# x_se_4 => mean_1
triton_red_fused_mean_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (13440*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwepdrpezze6pthdytxg3jyhu7jduo2hueqtndmb73jgqzrura4.py
# Source Nodes: [x_se_4], Original ATen: [aten.mean]
# x_se_4 => mean_1
triton_per_fused_mean_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_75', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 120
    x1 = (xindex // 120)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (840*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/cayim6ra74sx2l5gfjvvybpnl35bf36wxivdqrq7j7qzsmonevff.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
# x_se_5 => convolution_27
# x_se_6 => relu_12
triton_poi_fused_convolution_relu_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_76', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcubqbunbrce6dh4llyjurqvydkgitkl5dj2yp2j4uyul2f5vza.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => add_130, clamp_max_1, clamp_min_1, div_1
# x_se_7 => convolution_28
triton_poi_fused_convolution_hardsigmoid_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqrlfo5eni2ic7yxa6g6vltmnwoaer76lpwjk55ozhimqjddyqv.py
# Source Nodes: [x_21], Original ATen: [aten.mul]
# x_21 => mul_176
triton_poi_fused_mul_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 120
    x2 = (xindex // 94080)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (120*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ft/cftn467g3zjq2gvkcipmo77mkjlv33avyypxvhhwxfdd7wg3kk6h.py
# Source Nodes: [cat_54, shortcut_5], Original ATen: [aten.add, aten.cat]
# cat_54 => cat_9
# shortcut_5 => add_141
triton_poi_fused_add_cat_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 20, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (20*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 40, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-20) + x0 + (20*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 6272.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-20) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmo6o6xsa7gi3ce5srifsur72gi7setbhzjgdvycjfqwtnqn7npu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0 => convolution_31
triton_poi_fused_convolution_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_80', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/si/csi7grafqmbj6f6a54cs3qhoaxiymrqihamy2rmggnb53t5vquyl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1 => var_mean_27
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bz/cbzcn3a4rey3b4t5km5ykxsx3ahnwowgs5p2jifzo7wr4sclcwfl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1 => add_143, add_144, add_145, mul_192, mul_193, mul_194, mul_195, mul_196, rsqrt_27, squeeze_82, var_mean_27
triton_per_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/qh/cqh4b66ivtltfttqofahyszbmhr6kb43cur2j2b7dcx52rkw6jm6.py
# Source Nodes: [cat_53, getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1, x1_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_53 => cat_10
# getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1 => add_143, add_146, mul_191, mul_197, rsqrt_27, sub_27, var_mean_27
# x1_10 => relu_13
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    x1 = (xindex // 120)
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
    tl.store(out_ptr1 + (x0 + (240*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/cizwj72f7775vzo7k37yvj333fq6i7vns3bnmthp23clroc7u2gg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1, x2_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1 => add_148, add_151, mul_198, mul_204, rsqrt_28, sub_28, var_mean_28
# x2_10 => relu_14
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    x1 = (xindex // 120)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (240*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6naxjvwnmehcn6gwvb5m27gr432zntn7cnmdtxl5z4uz45uwhyz.py
# Source Nodes: [x_25], Original ATen: [aten.convolution]
# x_25 => convolution_33
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmfwqip7qf6t53orvxnc4nrjypahqwvgooda2axi3p3ij27aj6v.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => var_mean_29
triton_red_fused__native_batch_norm_legit_functional_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zw/czwficl745ud7e7bu4rvmltvqtjaxmovq5nxuex7yncick4enu3x.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => add_153, add_154, add_155, mul_206, mul_207, mul_208, mul_209, mul_210, rsqrt_29, squeeze_88, var_mean_29
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/og/cogru373i7cc6wkwnv3fvld2bgb3vhkc7mru6e4vcsgg434ieqrv.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => add_153, add_156, mul_205, mul_211, rsqrt_29, sub_29, var_mean_29
triton_poi_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/czt4wbc3dn3l2v7iskvzkvougytivee4ig5yve4e5wz4m2hjmm7w.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0 => convolution_34
triton_poi_fused_convolution_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (40*x2) + (7840*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xcanc64fhletkoe3clio2lx465k2kaey54b2tdti54q4ebn6se.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 520
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 40)
    x0 = xindex % 40
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
        tmp3 = tl.load(in_ptr0 + (x0 + (40*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6sdheiywmz2poj377tuhmmvnseswjyq2mef5xhndevz6bqijrv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1 => add_158, add_159, add_160, mul_213, mul_214, mul_215, mul_216, mul_217, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_91', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxm6potcbuqr6ij3ojyvhansihs2k73g7duk7c6d7vmccn22vnp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1 => add_158, add_161, mul_212, mul_218, rsqrt_30, sub_30, var_mean_30
triton_poi_fused__native_batch_norm_legit_functional_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
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


# kernel path: /tmp/torchinductor_youkaichao/jr/cjriduhwmcauzcmfc7ifyo7n5p2albdh6t2bit4mgbptjbx5ugck.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___5_____0___shortcut_2 => convolution_37
triton_poi_fused_convolution_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_93', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6c6lxdbpmishk7gvzl6tqafq53h7gqjdmrpqqy3cb2zcp46lc3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___shortcut_3 => var_mean_33
triton_red_fused__native_batch_norm_legit_functional_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_94', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/uj/cujibg4pbhys5dsdzoensbkdio56ojzggeqepjoyih7bp3ldmotl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___5_____0___shortcut_3 => add_173, add_174, add_175, mul_234, mul_235, mul_236, mul_237, mul_238, rsqrt_33, squeeze_100, var_mean_33
triton_per_fused__native_batch_norm_legit_functional_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_95', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/rj/crjtlzbg7yo7b24aykpd27nti5nfqfozziywtcubtmkvj7einwmj.py
# Source Nodes: [cat_52, getattr_getattr_l__mod___blocks___5_____0___shortcut_3, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
# cat_52 => cat_11
# getattr_getattr_l__mod___blocks___5_____0___shortcut_3 => add_173, add_176, mul_233, mul_239, rsqrt_33, sub_33, var_mean_33
# shortcut_6 => add_177
triton_poi_fused__native_batch_norm_legit_functional_add_cat_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 40, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (40*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-40) + x0 + (40*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1568.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 / tmp15
    tmp33 = tmp32 + tmp17
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp30 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp27 + tmp39
    tl.store(out_ptr0 + (x2), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dv4hrlnj2ohhpbsu24gtsjmctde5r2zgfw4oozeieiunqzamvf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0 => convolution_38
triton_poi_fused_convolution_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 800
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 100
    y1 = (yindex // 100)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (100*x2) + (19600*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/csh6xwskzcipqfwye47sckszo3btsclzz5eqymjml7dzksrd4j3t.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1 => var_mean_34
triton_red_fused__native_batch_norm_legit_functional_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1300
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 100)
    x0 = xindex % 100
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
        tmp3 = tl.load(in_ptr0 + (x0 + (100*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lo/cloollytbmr7xbsrokz4hejjawjizbcmcpom2lqugwsoowyslxzx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1 => add_179, add_180, add_181, mul_241, mul_242, mul_243, mul_244, mul_245, rsqrt_34, squeeze_103, var_mean_34
triton_per_fused__native_batch_norm_legit_functional_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_99', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (100*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (100*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (100*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/is/cisjvjlpoyoask2kyxivln6qzusmmxvbx6xlorf7bb7ep6wcc2vb.py
# Source Nodes: [cat_51, getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1, x1_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_51 => cat_12
# getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1 => add_179, add_182, mul_240, mul_246, rsqrt_34, sub_34, var_mean_34
# x1_12 => relu_15
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100
    x1 = (xindex // 100)
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
    tl.store(out_ptr1 + (x0 + (200*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pzjs5yzhyv3weimlwg4dsutawhks5yb4lpivxwrpz2hb7htkg3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1, x2_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1 => add_184, add_187, mul_247, mul_253, rsqrt_35, sub_35, var_mean_35
# x2_12 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100
    x1 = (xindex // 100)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (200*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwi4d65fsr2ihznyfqayvn2efobdhc5l6sts2aluzpmcojimzwnw.py
# Source Nodes: [cat_50, shortcut_7], Original ATen: [aten.add, aten.cat]
# cat_50 => cat_13
# shortcut_7 => add_198
triton_poi_fused_add_cat_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 40, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (40*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-40) + x0 + (40*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1568.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-40) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbw326c4zwxuep4bi5nvxu3xb6j3jp5g2g5esjjnge7ey6ldh3zo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0 => convolution_42
triton_poi_fused_convolution_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 736
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 92
    y1 = (yindex // 92)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (92*x2) + (18032*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clafhaeecvqf7pdixydegsouqdkwjmfin4x23tpeggkddk5wajzz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1 => var_mean_38
triton_red_fused__native_batch_norm_legit_functional_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1196
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 92)
    x0 = xindex % 92
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
        tmp3 = tl.load(in_ptr0 + (x0 + (92*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5v4egbzbm73csdzgmedodudk3mrgsmknpgcyqpgonxikza5owv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1 => add_200, add_201, add_202, mul_269, mul_270, mul_271, mul_272, mul_273, rsqrt_38, squeeze_115, var_mean_38
triton_per_fused__native_batch_norm_legit_functional_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_105', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 92
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (92*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (92*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (92*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eb/cebzxm447wwccqkho2tvl2xffzzoi2mvkv43zfzqx22diie3hlbv.py
# Source Nodes: [cat_49, getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1, x1_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_49 => cat_14
# getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1 => add_200, add_203, mul_268, mul_274, rsqrt_38, sub_38, var_mean_38
# x1_14 => relu_17
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 92
    x1 = (xindex // 92)
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
    tl.store(out_ptr1 + (x0 + (184*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qs/cqstpboa2iso6panjnx6c7jflmoops6c7zldfrfeyufqls6ukmyc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1, x2_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1 => add_205, add_208, mul_275, mul_281, rsqrt_39, sub_39, var_mean_39
# x2_14 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 92
    x1 = (xindex // 92)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (184*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wlutqjowvp3bvbxvcauhpftxie25t5evbaidaodca5mjjsvple.py
# Source Nodes: [cat_45, getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1, x1_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_45 => cat_18
# getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1 => add_242, add_245, mul_324, mul_330, rsqrt_46, sub_46, var_mean_46
# x1_18 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    x1 = (xindex // 240)
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
    tl.store(out_ptr1 + (x0 + (480*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfwx4ygph6mg2sggdgowfosnfwnogwfvca5q6wat6j4soawvwxk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1, x2_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1 => add_247, add_250, mul_331, mul_337, rsqrt_47, sub_47, var_mean_47
# x2_18 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    x1 = (xindex // 240)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (480*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegvmqqwtzcvahcy7rfkhcu4ul3kamqbkijffi552vbrnlzkyxuy.py
# Source Nodes: [x_se_8], Original ATen: [aten.mean]
# x_se_8 => mean_2
triton_red_fused_mean_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_110', 'mutated_arg_names': []}
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
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck72sy3xrzcfo7igkww5rgklbhfijbo6fcyvax2mwophby35ybpm.py
# Source Nodes: [x_se_8], Original ATen: [aten.mean]
# x_se_8 => mean_2
triton_per_fused_mean_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_111', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/fq/cfq2wsfzejdhfrxfgbfhsiiziwrxmpbg26cetaqvydofgknkfpmn.py
# Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
# x_se_10 => relu_23
# x_se_9 => convolution_52
triton_poi_fused_convolution_relu_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csomrz3fwhetumlrrenvwoeozkpndsp65o226wp6o6rbjfib3zqu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___se_gate, x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___6_____3___se_gate => add_251, clamp_max_2, clamp_min_2, div_2
# x_se_11 => convolution_53
triton_poi_fused_convolution_hardsigmoid_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cryg6gv6jhtrqrpgxlkcww6uuq5rygy4w2h3kcclqhc3tfirgfpi.py
# Source Nodes: [x_39], Original ATen: [aten.mul]
# x_39 => mul_338
triton_poi_fused_mul_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_114', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x0 + (480*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjclrly2odrwwqprrczadmbyogwzvlh7rtu3znxr5qgcb57gupc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0 => convolution_54
triton_poi_fused_convolution_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (10976*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgwatubmkmoqofa4txs447tuvig23jrultdvo4m4zgkdvoal7rm.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1 => var_mean_48
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 728
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 56)
    x0 = xindex % 56
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
        tmp3 = tl.load(in_ptr0 + (x0 + (56*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2st4vkrwnp7ngglcqa2i3d55jcqpswg6zaufyrp4uipm2bmkw5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1 => add_253, add_254, add_255, mul_340, mul_341, mul_342, mul_343, mul_344, rsqrt_48, squeeze_145, var_mean_48
triton_per_fused__native_batch_norm_legit_functional_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_117', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/ge/cgepzpslb4goumqqzublzullxp23rmpjp4cohd2nq4rl7qv5tx55.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1 => add_253, add_256, mul_339, mul_345, rsqrt_48, sub_48, var_mean_48
triton_poi_fused__native_batch_norm_legit_functional_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dxw2pjcvlgsaib2lkfeqoege6cjg3fgmsotxwfjvi26gm5or7f.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____3___shortcut_1 => add_263, add_266, mul_353, mul_359, rsqrt_50, sub_50, var_mean_50
triton_poi_fused__native_batch_norm_legit_functional_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_119', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rs/crspdq55xpv2bfbair3cjeiiagcousxaxgxlrvldix3n5sf57tca.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___6_____3___shortcut_2 => convolution_57
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


# kernel path: /tmp/torchinductor_youkaichao/66/c66xw6i4ll3rz6n2ljxlwwwjs3uahh4h4j4su3okvk5x4bshpgk6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____3___shortcut_3 => var_mean_51
triton_red_fused__native_batch_norm_legit_functional_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_121', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3nlzjcfx2vhda5ngwjkbymb4rstvomguij5j5z2uu6zjbrwurj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____3___shortcut_3 => add_268, add_269, add_270, mul_361, mul_362, mul_363, mul_364, mul_365, rsqrt_51, squeeze_154, var_mean_51
triton_per_fused__native_batch_norm_legit_functional_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_122', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ns/cns5hvaibwprsgydnctetyv3d5i3ecrtkdi2e52nmvntoczfc6ce.py
# Source Nodes: [cat_44, getattr_getattr_l__mod___blocks___6_____3___shortcut_3, shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
# cat_44 => cat_19
# getattr_getattr_l__mod___blocks___6_____3___shortcut_3 => add_268, add_271, mul_360, mul_366, rsqrt_51, sub_51, var_mean_51
# shortcut_10 => add_272
triton_poi_fused__native_batch_norm_legit_functional_add_cat_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (56*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 112, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-56) + x0 + (56*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1568.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 / tmp15
    tmp33 = tmp32 + tmp17
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp30 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp27 + tmp39
    tl.store(out_ptr0 + (x2), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjop5yrfkqgow7gevo5o6u2iw5jgm5bmaifivd3lvkmdkwtovtuw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0 => convolution_58
triton_poi_fused_convolution_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (336*x2) + (65856*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casxpkzpw7l3fzueb3g7ucoh6ayf3fbgohppb4yup4w3ae3the6m.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1 => var_mean_52
triton_red_fused__native_batch_norm_legit_functional_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_125', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ii/cii22px56lsi7m5qluc6etn5m5s6z3hty4cgsk4hhjozycc7thku.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1 => add_274, add_275, add_276, mul_368, mul_369, mul_370, mul_371, mul_372, rsqrt_52, squeeze_157, var_mean_52
triton_per_fused__native_batch_norm_legit_functional_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_126', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/si/csivwt6ywhehjdpwrhp7qmwscqvfltba2ilq2vyhufjubbj46nh2.py
# Source Nodes: [cat_43, getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1, x1_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_43 => cat_20
# getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1 => add_274, add_277, mul_367, mul_373, rsqrt_52, sub_52, var_mean_52
# x1_20 => relu_24
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_127', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
    x1 = (xindex // 336)
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
    tl.store(out_ptr1 + (x0 + (672*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jiphcnhbkuhiy44vjr6vwqncy3izskey2zp47p4syyxzhyvn57.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1, x2_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1 => add_279, add_282, mul_374, mul_380, rsqrt_53, sub_53, var_mean_53
# x2_20 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_128', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 336
    x1 = (xindex // 336)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (672*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cambzmjlqghwvk24rdhsdl6dt7pwbip6pjb2mz5sn7jn35wvpq5d.py
# Source Nodes: [x_se_12], Original ATen: [aten.mean]
# x_se_12 => mean_3
triton_red_fused_mean_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_129', 'mutated_arg_names': []}
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
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwhdknq7if7z6kmp4rb3bwdkclc56s3yao6kcurxm3ox4aekuow.py
# Source Nodes: [x_se_12], Original ATen: [aten.mean]
# x_se_12 => mean_3
triton_per_fused_mean_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_130', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bo7zbggmtkym3xbx2uar242qscybxxlhfmkk5whprwdknmdrtq.py
# Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
# x_se_13 => convolution_60
# x_se_14 => relu_26
triton_poi_fused_convolution_relu_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_131', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 168
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqy3aaklbp6yecglkd6fflmol6cebb22rdu34jayhvmhsxee4ku.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___se_gate, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___6_____4___se_gate => add_283, clamp_max_3, clamp_min_3, div_3
# x_se_15 => convolution_61
triton_poi_fused_convolution_hardsigmoid_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77h5gspyg7ghorhxasspcz3ji6tpzdorcb5his7tvjtefij35sq.py
# Source Nodes: [x_43], Original ATen: [aten.mul]
# x_43 => mul_381
triton_poi_fused_mul_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_133', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvapskyn3xcxczdpt2f2uynhz7tied5s35vttu4llufsmlctgzg7.py
# Source Nodes: [cat_42, shortcut_11], Original ATen: [aten.add, aten.cat]
# cat_42 => cat_21
# shortcut_11 => add_294
triton_poi_fused_add_cat_134 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_134', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 112
    x1 = (xindex // 112)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (56*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 112, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-56) + x0 + (56*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1568.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-56) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3u/c3utm2yonibf5uej2zovrd6rlip5wxuxn5i7xl4sxj6x6fjm27vf.py
# Source Nodes: [x_47], Original ATen: [aten.convolution]
# x_47 => convolution_66
triton_poi_fused_convolution_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_135', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vi/cvipdtapefctqsudcspg6bzur57stzajdl57i7gletef4ypfevar.py
# Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
# x_48 => var_mean_58
triton_red_fused__native_batch_norm_legit_functional_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_136', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yu/cyucg2mevbkxpqiw7perwh3ufmoglibvjbovswmyfqqidak4omst.py
# Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
# x_48 => add_306, add_307, add_308, mul_411, mul_412, mul_413, mul_414, mul_415, rsqrt_58, squeeze_175, var_mean_58
triton_per_fused__native_batch_norm_legit_functional_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_137', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/5h/c5ht2de4gpc4lm7k273lnedwqhynnonp2phwmejoczffp3zbbuq3.py
# Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
# x_48 => add_306, add_309, mul_410, mul_416, rsqrt_58, sub_58, var_mean_58
triton_poi_fused__native_batch_norm_legit_functional_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_138', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gyjd6kk6sjpx7xzojygrkp5cmyubdxsufhpusasmidmvqofnvk.py
# Source Nodes: [x_se_16], Original ATen: [aten.mean]
# x_se_16 => mean_4
triton_per_fused_mean_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_139', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5ij67nt22qenyi2sp2yythzntoe7b4o2whqitrmxw72sh54fbwp.py
# Source Nodes: [x_49], Original ATen: [aten.mul]
# x_49 => mul_417
triton_poi_fused_mul_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_140', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhie2ff2imuz7ys5fitdwfrmjc2mgvlotpotp5sabas64hgqmpf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0 => convolution_69
triton_poi_fused_convolution_141 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_141', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (3920*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ma/cmaolvhk7pnomuca4ilz6yg5eenacodvoexogvnobqd7ylwo4ewf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1 => var_mean_59
triton_red_fused__native_batch_norm_legit_functional_142 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_142', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (80*r2) + (7840*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmptyjxzht7tug3idjnp62v36kini72pbco62d6gycutwed3rek.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1 => add_312, add_313, add_314, mul_419, mul_420, mul_421, mul_422, mul_423, rsqrt_59, squeeze_178, var_mean_59
triton_per_fused__native_batch_norm_legit_functional_143 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_143', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/or/corbp5y6rcdsyx7ztnenmqxniqpbvkoiqufa6ghcg2a2v6o2mzmk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1 => add_312, add_315, mul_418, mul_424, rsqrt_59, sub_59, var_mean_59
triton_poi_fused__native_batch_norm_legit_functional_144 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_144', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
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


# kernel path: /tmp/torchinductor_youkaichao/um/cumjl65cvesbxzloxxlq7raiusfapfh2q3yumep73fh253nd4ehx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_0 => convolution_71
triton_poi_fused_convolution_145 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_145', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (5488*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eefncjwxvjhohhgn6bn7xsb53ni7cxsfyktuol6me2xfq3qw3d.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_1 => var_mean_61
triton_red_fused__native_batch_norm_legit_functional_146 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (112*r2) + (10976*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxkmnyujg4mvro35zwlp5fh3y6a4tasqkvtxqnwqtighl3meafd.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_1 => add_322, add_323, add_324, mul_433, mul_434, mul_435, mul_436, mul_437, rsqrt_61, squeeze_184, var_mean_61
triton_per_fused__native_batch_norm_legit_functional_147 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_147', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/6j/c6jdf3s7e7vmnkcors5fc7hswkj5xlnpobci4plx6wh2f3zhivix.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_1 => add_322, add_325, mul_432, mul_438, rsqrt_61, sub_61, var_mean_61
triton_poi_fused__native_batch_norm_legit_functional_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_148', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7wc43hrp7hrtvanvfb7hpnqaoz5xqm6nekmjb2hxufw6jwhkbg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_2], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_2 => convolution_72
triton_poi_fused_convolution_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_149', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/34/c34az43wh62x23bxvlcy3abe7k5bz2wlizzyijcwcnmowevn7gdm.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_3 => var_mean_62
triton_red_fused__native_batch_norm_legit_functional_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_150', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqzve5pnp6blrprzhrmzgdhabpo5pbzwzjclm2gw67z2mvcbcen.py
# Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___7_____0___shortcut_3 => add_327, add_328, add_329, mul_440, mul_441, mul_442, mul_443, mul_444, rsqrt_62, squeeze_187, var_mean_62
triton_per_fused__native_batch_norm_legit_functional_151 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_151', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdvsfvbxjwjgw2oezic4p6jygixjsyr4mux77g4nrwn3l76scth.py
# Source Nodes: [cat_40, getattr_getattr_l__mod___blocks___7_____0___shortcut_3, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
# cat_40 => cat_23
# getattr_getattr_l__mod___blocks___7_____0___shortcut_3 => add_327, add_330, mul_439, mul_445, rsqrt_62, sub_62, var_mean_62
# shortcut_12 => add_331
triton_poi_fused__native_batch_norm_legit_functional_add_cat_152 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_cat_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 160
    x1 = (xindex // 160)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (80*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-80) + x0 + (80*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 392.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 / tmp15
    tmp33 = tmp32 + tmp17
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp30 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp27 + tmp39
    tl.store(out_ptr0 + (x2), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijo7n3r5kukni7m76tac2j7xekl5tuyd4nabqn6koilvqsxjllo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0 => convolution_73
triton_poi_fused_convolution_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_153', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (480*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kkd5igcadtwjtnh5xdfxekzsl4axh4tbnj4o3al3ejj34pn3gy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1 => var_mean_63
triton_red_fused__native_batch_norm_legit_functional_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/py/cpyowvgpu24l6hcwtycyqha3whwqookb3tekxcfp2da24xigx5vf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1 => add_333, add_334, add_335, mul_447, mul_448, mul_449, mul_450, mul_451, rsqrt_63, squeeze_190, var_mean_63
triton_per_fused__native_batch_norm_legit_functional_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_155', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx53neswyavx7izlwjahrpdmkqz2e26gc36psv7fpadq3gcyv7tf.py
# Source Nodes: [cat_39, getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1, x1_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_39 => cat_24
# getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1 => add_333, add_336, mul_446, mul_452, rsqrt_63, sub_63, var_mean_63
# x1_24 => relu_30
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_156', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    x1 = (xindex // 480)
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
    tl.store(out_ptr1 + (x0 + (960*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72wgt2jb45wdhs5c6uqiif7lqib2qx2y54f66kyznq4aectykwb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1, x2_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1 => add_338, add_341, mul_453, mul_459, rsqrt_64, sub_64, var_mean_64
# x2_24 => relu_31
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_157 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_157', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    x1 = (xindex // 480)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (960*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6ndop5lrbcmxrki5xc7vcwtcbn75ftl3ogaadnkj57qosysw5h.py
# Source Nodes: [cat_38, shortcut_13], Original ATen: [aten.add, aten.cat]
# cat_38 => cat_25
# shortcut_13 => add_352
triton_poi_fused_add_cat_158 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_158', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 160
    x1 = (xindex // 160)
    x2 = xindex
    tmp28 = tl.load(in_ptr6 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 80, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (80*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-80) + x0 + (80*x1)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 392.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-80) + x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp8, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp7, tmp26)
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c652iogxoerp2ygjgu4rmblhgcsey6czm3gupc7vc72binfn4k4n.py
# Source Nodes: [x_se_20], Original ATen: [aten.mean]
# x_se_20 => mean_5
triton_per_fused_mean_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_159', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctywt6tjahhnfv45jopex7beomwtpkx5ygnursjcrbuv7xkxjee7.py
# Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
# x_se_21 => convolution_79
# x_se_22 => relu_34
triton_poi_fused_convolution_relu_160 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_160', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5syodos5m5vjfuyl7xk424znenjbdo2gguuvsg54uefbocezrh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___se_gate, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___8_____1___se_gate => add_363, clamp_max_5, clamp_min_5, div_5
# x_se_23 => convolution_80
triton_poi_fused_convolution_hardsigmoid_161 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_161', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qguax224ikpscnsd5nl6vghbum6yikgoobreqiwc32ejlvon2n.py
# Source Nodes: [x_56], Original ATen: [aten.mul]
# x_56 => mul_488
triton_poi_fused_mul_162 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_162', 'mutated_arg_names': []},
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
    tmp1 = tl.load(in_ptr1 + (x0 + (960*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggalbj6qs74jts3dxcw57z7bsuvejmzmgus3kyucd3t2axwsxem.py
# Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___se_gate, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
# getattr_getattr_l__mod___blocks___8_____3___se_gate => add_406, clamp_max_6, clamp_min_6, div_6
# x_se_27 => convolution_90
triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_163 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_163', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp10 = -3.0
    tmp11 = tmp2 > tmp10
    tmp12 = tmp2 < tmp3
    tmp13 = tmp11 & tmp12
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zz/czz5al74g7z2g5zefflvluwmif6mdo44srcufvjvzhmtf7zywue4.py
# Source Nodes: [x_66], Original ATen: [aten.convolution]
# x_66 => convolution_93
triton_poi_fused_convolution_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_164', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czv35aw4x3gvrdkfqqw6dov265gzbpmtynf6cf2d7kjy2vocdtwc.py
# Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
# x_67 => var_mean_79
triton_red_fused__native_batch_norm_legit_functional_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_165', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv4mdntv7uuauhqnebi3afpzxhbk4n4wel2d43k6nrd5nfhuxys.py
# Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
# x_67 => add_419, add_420, add_421, mul_561, mul_562, mul_563, mul_564, mul_565, rsqrt_79, squeeze_238, var_mean_79
triton_per_fused__native_batch_norm_legit_functional_166 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_166', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp21 = 1.0025575447570332
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7o3yq2hxy5pkf45exrvtpw2ds5dzx5obtgkyvkmkjvoj2prm7tj.py
# Source Nodes: [x_67, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_67 => add_419, add_422, mul_560, mul_566, rsqrt_79, sub_79, var_mean_79
# x_72 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_167', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktumjnkeflm7us74e4rxwbd2megcq4k62s5a5wesv3a77uyrlkc.py
# Source Nodes: [pred, x_76, x_77], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward, aten.view]
# pred => view_1
# x_76 => convolution_94
# x_77 => relu_41
triton_poi_fused_convolution_relu_threshold_backward_view_168 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_view_168', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr1 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tv/ctvn3ixrkqynifgm6tcg2piiocumranj3vsc6w4ngnqr4odvdtkf.py
# Source Nodes: [x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_23 => convolution_80
triton_poi_fused_convolution_hardsigmoid_backward_169 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_169', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3kcsf7fxkbke2jm4epswbxnojs2sg3meu5y55hdfedwzzxu4fi.py
# Source Nodes: [x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_19 => convolution_68
triton_poi_fused_convolution_hardsigmoid_backward_170 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_170', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7orbicfme47bguapyerxx6xxcwqtoaamleoxhjzw7kbposlq6dt.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_11 => convolution_53
triton_poi_fused_convolution_hardsigmoid_backward_171 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_171', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxc3m36mjanaz7cuqar2hqgee3rkew4jftgow2kdafzsnzxp6iwk.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_7 => convolution_28
triton_poi_fused_convolution_hardsigmoid_backward_172 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_172', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csm762q2c4fjbwv3p3v5xnghytex7lazsdamx3odwh6j6gblp2mo.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_3 => convolution_20
triton_poi_fused_convolution_hardsigmoid_backward_173 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_173', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjko43mddcgjuudeb2rqtvzyblk4dnrnykhldn4rskkca42w4bpu.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add_418
triton_poi_fused_add_174 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_174', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513 = args
    args.clear()
    assert_size_stride(primals_1, (960, ), (1, ))
    assert_size_stride(primals_2, (960, ), (1, ))
    assert_size_stride(primals_3, (1000, 1280), (1280, 1))
    assert_size_stride(primals_4, (1000, ), (1, ))
    assert_size_stride(primals_5, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_12, (8, ), (1, ))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_15, (8, ), (1, ))
    assert_size_stride(primals_16, (8, ), (1, ))
    assert_size_stride(primals_17, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_19, (8, ), (1, ))
    assert_size_stride(primals_20, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_22, (24, ), (1, ))
    assert_size_stride(primals_23, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, ), (1, ))
    assert_size_stride(primals_26, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (12, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_30, (12, ), (1, ))
    assert_size_stride(primals_31, (12, ), (1, ))
    assert_size_stride(primals_32, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (12, ), (1, ))
    assert_size_stride(primals_34, (12, ), (1, ))
    assert_size_stride(primals_35, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (16, ), (1, ))
    assert_size_stride(primals_38, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_39, (24, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_41, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_42, (36, ), (1, ))
    assert_size_stride(primals_43, (36, ), (1, ))
    assert_size_stride(primals_44, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_45, (36, ), (1, ))
    assert_size_stride(primals_46, (36, ), (1, ))
    assert_size_stride(primals_47, (12, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_48, (12, ), (1, ))
    assert_size_stride(primals_49, (12, ), (1, ))
    assert_size_stride(primals_50, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (12, ), (1, ))
    assert_size_stride(primals_52, (12, ), (1, ))
    assert_size_stride(primals_53, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_54, (36, ), (1, ))
    assert_size_stride(primals_55, (36, ), (1, ))
    assert_size_stride(primals_56, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_57, (36, ), (1, ))
    assert_size_stride(primals_58, (36, ), (1, ))
    assert_size_stride(primals_59, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_60, (72, ), (1, ))
    assert_size_stride(primals_61, (72, ), (1, ))
    assert_size_stride(primals_62, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_63, (20, ), (1, ))
    assert_size_stride(primals_64, (72, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_65, (72, ), (1, ))
    assert_size_stride(primals_66, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_67, (20, ), (1, ))
    assert_size_stride(primals_68, (20, ), (1, ))
    assert_size_stride(primals_69, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_70, (20, ), (1, ))
    assert_size_stride(primals_71, (20, ), (1, ))
    assert_size_stride(primals_72, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (24, ), (1, ))
    assert_size_stride(primals_75, (40, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_76, (40, ), (1, ))
    assert_size_stride(primals_77, (40, ), (1, ))
    assert_size_stride(primals_78, (60, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_79, (60, ), (1, ))
    assert_size_stride(primals_80, (60, ), (1, ))
    assert_size_stride(primals_81, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_82, (60, ), (1, ))
    assert_size_stride(primals_83, (60, ), (1, ))
    assert_size_stride(primals_84, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_86, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_87, (120, ), (1, ))
    assert_size_stride(primals_88, (20, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_89, (20, ), (1, ))
    assert_size_stride(primals_90, (20, ), (1, ))
    assert_size_stride(primals_91, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (20, ), (1, ))
    assert_size_stride(primals_93, (20, ), (1, ))
    assert_size_stride(primals_94, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_95, (120, ), (1, ))
    assert_size_stride(primals_96, (120, ), (1, ))
    assert_size_stride(primals_97, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (120, ), (1, ))
    assert_size_stride(primals_99, (120, ), (1, ))
    assert_size_stride(primals_100, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (240, ), (1, ))
    assert_size_stride(primals_102, (240, ), (1, ))
    assert_size_stride(primals_103, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_104, (40, ), (1, ))
    assert_size_stride(primals_105, (40, ), (1, ))
    assert_size_stride(primals_106, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (40, ), (1, ))
    assert_size_stride(primals_108, (40, ), (1, ))
    assert_size_stride(primals_109, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (40, ), (1, ))
    assert_size_stride(primals_111, (40, ), (1, ))
    assert_size_stride(primals_112, (80, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_113, (80, ), (1, ))
    assert_size_stride(primals_114, (80, ), (1, ))
    assert_size_stride(primals_115, (100, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_116, (100, ), (1, ))
    assert_size_stride(primals_117, (100, ), (1, ))
    assert_size_stride(primals_118, (100, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (100, ), (1, ))
    assert_size_stride(primals_120, (100, ), (1, ))
    assert_size_stride(primals_121, (40, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_122, (40, ), (1, ))
    assert_size_stride(primals_123, (40, ), (1, ))
    assert_size_stride(primals_124, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (40, ), (1, ))
    assert_size_stride(primals_126, (40, ), (1, ))
    assert_size_stride(primals_127, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_128, (92, ), (1, ))
    assert_size_stride(primals_129, (92, ), (1, ))
    assert_size_stride(primals_130, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (92, ), (1, ))
    assert_size_stride(primals_132, (92, ), (1, ))
    assert_size_stride(primals_133, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (40, ), (1, ))
    assert_size_stride(primals_135, (40, ), (1, ))
    assert_size_stride(primals_136, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (40, ), (1, ))
    assert_size_stride(primals_138, (40, ), (1, ))
    assert_size_stride(primals_139, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_140, (92, ), (1, ))
    assert_size_stride(primals_141, (92, ), (1, ))
    assert_size_stride(primals_142, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (92, ), (1, ))
    assert_size_stride(primals_144, (92, ), (1, ))
    assert_size_stride(primals_145, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_146, (40, ), (1, ))
    assert_size_stride(primals_147, (40, ), (1, ))
    assert_size_stride(primals_148, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (40, ), (1, ))
    assert_size_stride(primals_150, (40, ), (1, ))
    assert_size_stride(primals_151, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_152, (240, ), (1, ))
    assert_size_stride(primals_153, (240, ), (1, ))
    assert_size_stride(primals_154, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (240, ), (1, ))
    assert_size_stride(primals_156, (240, ), (1, ))
    assert_size_stride(primals_157, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_158, (120, ), (1, ))
    assert_size_stride(primals_159, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_160, (480, ), (1, ))
    assert_size_stride(primals_161, (56, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_162, (56, ), (1, ))
    assert_size_stride(primals_163, (56, ), (1, ))
    assert_size_stride(primals_164, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_165, (56, ), (1, ))
    assert_size_stride(primals_166, (56, ), (1, ))
    assert_size_stride(primals_167, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (80, ), (1, ))
    assert_size_stride(primals_169, (80, ), (1, ))
    assert_size_stride(primals_170, (112, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_171, (112, ), (1, ))
    assert_size_stride(primals_172, (112, ), (1, ))
    assert_size_stride(primals_173, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_174, (336, ), (1, ))
    assert_size_stride(primals_175, (336, ), (1, ))
    assert_size_stride(primals_176, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_177, (336, ), (1, ))
    assert_size_stride(primals_178, (336, ), (1, ))
    assert_size_stride(primals_179, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_180, (168, ), (1, ))
    assert_size_stride(primals_181, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_182, (672, ), (1, ))
    assert_size_stride(primals_183, (56, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_184, (56, ), (1, ))
    assert_size_stride(primals_185, (56, ), (1, ))
    assert_size_stride(primals_186, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_187, (56, ), (1, ))
    assert_size_stride(primals_188, (56, ), (1, ))
    assert_size_stride(primals_189, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_190, (336, ), (1, ))
    assert_size_stride(primals_191, (336, ), (1, ))
    assert_size_stride(primals_192, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (336, ), (1, ))
    assert_size_stride(primals_194, (336, ), (1, ))
    assert_size_stride(primals_195, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (672, ), (1, ))
    assert_size_stride(primals_197, (672, ), (1, ))
    assert_size_stride(primals_198, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_199, (168, ), (1, ))
    assert_size_stride(primals_200, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_201, (672, ), (1, ))
    assert_size_stride(primals_202, (80, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_203, (80, ), (1, ))
    assert_size_stride(primals_204, (80, ), (1, ))
    assert_size_stride(primals_205, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (80, ), (1, ))
    assert_size_stride(primals_207, (80, ), (1, ))
    assert_size_stride(primals_208, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_209, (112, ), (1, ))
    assert_size_stride(primals_210, (112, ), (1, ))
    assert_size_stride(primals_211, (160, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_212, (160, ), (1, ))
    assert_size_stride(primals_213, (160, ), (1, ))
    assert_size_stride(primals_214, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_215, (480, ), (1, ))
    assert_size_stride(primals_216, (480, ), (1, ))
    assert_size_stride(primals_217, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (480, ), (1, ))
    assert_size_stride(primals_219, (480, ), (1, ))
    assert_size_stride(primals_220, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_221, (80, ), (1, ))
    assert_size_stride(primals_222, (80, ), (1, ))
    assert_size_stride(primals_223, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_224, (80, ), (1, ))
    assert_size_stride(primals_225, (80, ), (1, ))
    assert_size_stride(primals_226, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_227, (480, ), (1, ))
    assert_size_stride(primals_228, (480, ), (1, ))
    assert_size_stride(primals_229, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_230, (480, ), (1, ))
    assert_size_stride(primals_231, (480, ), (1, ))
    assert_size_stride(primals_232, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_234, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_235, (960, ), (1, ))
    assert_size_stride(primals_236, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_239, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (80, ), (1, ))
    assert_size_stride(primals_241, (80, ), (1, ))
    assert_size_stride(primals_242, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_245, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (480, ), (1, ))
    assert_size_stride(primals_247, (480, ), (1, ))
    assert_size_stride(primals_248, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_249, (80, ), (1, ))
    assert_size_stride(primals_250, (80, ), (1, ))
    assert_size_stride(primals_251, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_252, (80, ), (1, ))
    assert_size_stride(primals_253, (80, ), (1, ))
    assert_size_stride(primals_254, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_255, (480, ), (1, ))
    assert_size_stride(primals_256, (480, ), (1, ))
    assert_size_stride(primals_257, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_258, (480, ), (1, ))
    assert_size_stride(primals_259, (480, ), (1, ))
    assert_size_stride(primals_260, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_261, (240, ), (1, ))
    assert_size_stride(primals_262, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_263, (960, ), (1, ))
    assert_size_stride(primals_264, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_265, (80, ), (1, ))
    assert_size_stride(primals_266, (80, ), (1, ))
    assert_size_stride(primals_267, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_268, (80, ), (1, ))
    assert_size_stride(primals_269, (80, ), (1, ))
    assert_size_stride(primals_270, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_271, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_272, (1280, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (960, ), (1, ))
    assert_size_stride(primals_275, (960, ), (1, ))
    assert_size_stride(primals_276, (16, ), (1, ))
    assert_size_stride(primals_277, (16, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (8, ), (1, ))
    assert_size_stride(primals_280, (8, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (8, ), (1, ))
    assert_size_stride(primals_283, (8, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (8, ), (1, ))
    assert_size_stride(primals_286, (8, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (8, ), (1, ))
    assert_size_stride(primals_289, (8, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (24, ), (1, ))
    assert_size_stride(primals_292, (24, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (24, ), (1, ))
    assert_size_stride(primals_295, (24, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (48, ), (1, ))
    assert_size_stride(primals_298, (48, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (12, ), (1, ))
    assert_size_stride(primals_301, (12, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (12, ), (1, ))
    assert_size_stride(primals_304, (12, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (16, ), (1, ))
    assert_size_stride(primals_307, (16, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (24, ), (1, ))
    assert_size_stride(primals_310, (24, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (36, ), (1, ))
    assert_size_stride(primals_313, (36, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (36, ), (1, ))
    assert_size_stride(primals_316, (36, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (12, ), (1, ))
    assert_size_stride(primals_319, (12, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (12, ), (1, ))
    assert_size_stride(primals_322, (12, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (36, ), (1, ))
    assert_size_stride(primals_325, (36, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (36, ), (1, ))
    assert_size_stride(primals_328, (36, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (72, ), (1, ))
    assert_size_stride(primals_331, (72, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (20, ), (1, ))
    assert_size_stride(primals_334, (20, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (20, ), (1, ))
    assert_size_stride(primals_337, (20, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (24, ), (1, ))
    assert_size_stride(primals_340, (24, ), (1, ))
    assert_size_stride(primals_341, (), ())
    assert_size_stride(primals_342, (40, ), (1, ))
    assert_size_stride(primals_343, (40, ), (1, ))
    assert_size_stride(primals_344, (), ())
    assert_size_stride(primals_345, (60, ), (1, ))
    assert_size_stride(primals_346, (60, ), (1, ))
    assert_size_stride(primals_347, (), ())
    assert_size_stride(primals_348, (60, ), (1, ))
    assert_size_stride(primals_349, (60, ), (1, ))
    assert_size_stride(primals_350, (), ())
    assert_size_stride(primals_351, (20, ), (1, ))
    assert_size_stride(primals_352, (20, ), (1, ))
    assert_size_stride(primals_353, (), ())
    assert_size_stride(primals_354, (20, ), (1, ))
    assert_size_stride(primals_355, (20, ), (1, ))
    assert_size_stride(primals_356, (), ())
    assert_size_stride(primals_357, (120, ), (1, ))
    assert_size_stride(primals_358, (120, ), (1, ))
    assert_size_stride(primals_359, (), ())
    assert_size_stride(primals_360, (120, ), (1, ))
    assert_size_stride(primals_361, (120, ), (1, ))
    assert_size_stride(primals_362, (), ())
    assert_size_stride(primals_363, (240, ), (1, ))
    assert_size_stride(primals_364, (240, ), (1, ))
    assert_size_stride(primals_365, (), ())
    assert_size_stride(primals_366, (40, ), (1, ))
    assert_size_stride(primals_367, (40, ), (1, ))
    assert_size_stride(primals_368, (), ())
    assert_size_stride(primals_369, (40, ), (1, ))
    assert_size_stride(primals_370, (40, ), (1, ))
    assert_size_stride(primals_371, (), ())
    assert_size_stride(primals_372, (40, ), (1, ))
    assert_size_stride(primals_373, (40, ), (1, ))
    assert_size_stride(primals_374, (), ())
    assert_size_stride(primals_375, (80, ), (1, ))
    assert_size_stride(primals_376, (80, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (100, ), (1, ))
    assert_size_stride(primals_379, (100, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (100, ), (1, ))
    assert_size_stride(primals_382, (100, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (40, ), (1, ))
    assert_size_stride(primals_385, (40, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (40, ), (1, ))
    assert_size_stride(primals_388, (40, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (92, ), (1, ))
    assert_size_stride(primals_391, (92, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (92, ), (1, ))
    assert_size_stride(primals_394, (92, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (40, ), (1, ))
    assert_size_stride(primals_397, (40, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (40, ), (1, ))
    assert_size_stride(primals_400, (40, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (92, ), (1, ))
    assert_size_stride(primals_403, (92, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (92, ), (1, ))
    assert_size_stride(primals_406, (92, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (40, ), (1, ))
    assert_size_stride(primals_409, (40, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (40, ), (1, ))
    assert_size_stride(primals_412, (40, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (240, ), (1, ))
    assert_size_stride(primals_415, (240, ), (1, ))
    assert_size_stride(primals_416, (), ())
    assert_size_stride(primals_417, (240, ), (1, ))
    assert_size_stride(primals_418, (240, ), (1, ))
    assert_size_stride(primals_419, (), ())
    assert_size_stride(primals_420, (56, ), (1, ))
    assert_size_stride(primals_421, (56, ), (1, ))
    assert_size_stride(primals_422, (), ())
    assert_size_stride(primals_423, (56, ), (1, ))
    assert_size_stride(primals_424, (56, ), (1, ))
    assert_size_stride(primals_425, (), ())
    assert_size_stride(primals_426, (80, ), (1, ))
    assert_size_stride(primals_427, (80, ), (1, ))
    assert_size_stride(primals_428, (), ())
    assert_size_stride(primals_429, (112, ), (1, ))
    assert_size_stride(primals_430, (112, ), (1, ))
    assert_size_stride(primals_431, (), ())
    assert_size_stride(primals_432, (336, ), (1, ))
    assert_size_stride(primals_433, (336, ), (1, ))
    assert_size_stride(primals_434, (), ())
    assert_size_stride(primals_435, (336, ), (1, ))
    assert_size_stride(primals_436, (336, ), (1, ))
    assert_size_stride(primals_437, (), ())
    assert_size_stride(primals_438, (56, ), (1, ))
    assert_size_stride(primals_439, (56, ), (1, ))
    assert_size_stride(primals_440, (), ())
    assert_size_stride(primals_441, (56, ), (1, ))
    assert_size_stride(primals_442, (56, ), (1, ))
    assert_size_stride(primals_443, (), ())
    assert_size_stride(primals_444, (336, ), (1, ))
    assert_size_stride(primals_445, (336, ), (1, ))
    assert_size_stride(primals_446, (), ())
    assert_size_stride(primals_447, (336, ), (1, ))
    assert_size_stride(primals_448, (336, ), (1, ))
    assert_size_stride(primals_449, (), ())
    assert_size_stride(primals_450, (672, ), (1, ))
    assert_size_stride(primals_451, (672, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (80, ), (1, ))
    assert_size_stride(primals_454, (80, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (80, ), (1, ))
    assert_size_stride(primals_457, (80, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (112, ), (1, ))
    assert_size_stride(primals_460, (112, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (160, ), (1, ))
    assert_size_stride(primals_463, (160, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (480, ), (1, ))
    assert_size_stride(primals_466, (480, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (480, ), (1, ))
    assert_size_stride(primals_469, (480, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (80, ), (1, ))
    assert_size_stride(primals_472, (80, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (80, ), (1, ))
    assert_size_stride(primals_475, (80, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (480, ), (1, ))
    assert_size_stride(primals_478, (480, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (480, ), (1, ))
    assert_size_stride(primals_481, (480, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (80, ), (1, ))
    assert_size_stride(primals_484, (80, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (80, ), (1, ))
    assert_size_stride(primals_487, (80, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (480, ), (1, ))
    assert_size_stride(primals_490, (480, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (480, ), (1, ))
    assert_size_stride(primals_493, (480, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (80, ), (1, ))
    assert_size_stride(primals_496, (80, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (80, ), (1, ))
    assert_size_stride(primals_499, (80, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (480, ), (1, ))
    assert_size_stride(primals_502, (480, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (480, ), (1, ))
    assert_size_stride(primals_505, (480, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (80, ), (1, ))
    assert_size_stride(primals_508, (80, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (80, ), (1, ))
    assert_size_stride(primals_511, (80, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_5, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_5
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_513, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_513
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf3 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf4 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 16, 1, 1, 784), (12544, 1, 12544, 12544, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf3, buf4, buf5, buf6, 12544, 128, grid=grid(12544), stream=stream0)
        buf7 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1, 16, 1, 1, 7), (112, 1, 112, 112, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf4, buf5, buf6, buf7, buf8, buf9, 112, 112, grid=grid(112), stream=stream0)
        del buf4
        del buf5
        del buf6
        buf10 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf13 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_276, primals_277, buf10, buf11, buf13, primals_276, primals_277, 16, 7, grid=grid(16), stream=stream0)
        del primals_276
        del primals_277
        buf14 = reinterpret_tensor(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf3, buf10, buf11, primals_6, primals_7, buf14, 1605632, grid=grid(1605632), stream=stream0)
        del primals_7
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 8, 112, 112), (100352, 12544, 112, 1))
        buf16 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf15, buf16, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf17 = empty_strided((1, 8, 1, 1, 784), (6272, 1, 6272, 6272, 8), device='cuda', dtype=torch.float32)
        buf18 = empty_strided((1, 8, 1, 1, 784), (6272, 1, 6272, 6272, 8), device='cuda', dtype=torch.float32)
        buf19 = empty_strided((1, 8, 1, 1, 784), (6272, 1, 6272, 6272, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf16, buf17, buf18, buf19, 6272, 128, grid=grid(6272), stream=stream0)
        buf20 = empty_strided((1, 8, 1, 1, 7), (56, 1, 56, 56, 8), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 8, 1, 1, 7), (56, 1, 56, 56, 8), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((1, 8, 1, 1, 7), (56, 1, 56, 56, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf17, buf18, buf19, buf20, buf21, buf22, 56, 112, grid=grid(56), stream=stream0)
        buf23 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf26 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf20, buf21, buf22, primals_279, primals_280, buf23, buf24, buf26, primals_279, primals_280, 8, 7, grid=grid(8), stream=stream0)
        del primals_279
        del primals_280
        buf27 = reinterpret_tensor(buf15, (8, 8, 112, 112), (100352, 1, 896, 8), 0); del buf15  # reuse
        buf42 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        buf41 = reinterpret_tensor(buf42, (8, 8, 112, 112), (200704, 1, 1792, 16), 0)  # alias
        # Source Nodes: [cat_63, getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1, x1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11.run(buf16, buf23, buf24, primals_9, primals_10, buf27, buf41, 802816, grid=grid(802816), stream=stream0)
        del primals_10
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf28, (8, 8, 112, 112), (100352, 12544, 112, 1))
        buf29 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf28, buf29, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf30 = buf19; del buf19  # reuse
        buf31 = buf18; del buf18  # reuse
        buf32 = buf17; del buf17  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf29, buf30, buf31, buf32, 6272, 128, grid=grid(6272), stream=stream0)
        buf33 = buf22; del buf22  # reuse
        buf34 = buf21; del buf21  # reuse
        buf35 = buf20; del buf20  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf30, buf31, buf32, buf33, buf34, buf35, 56, 112, grid=grid(56), stream=stream0)
        buf36 = buf24; del buf24  # reuse
        buf37 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf39 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf33, buf34, buf35, primals_282, primals_283, buf36, buf37, buf39, primals_282, primals_283, 8, 7, grid=grid(8), stream=stream0)
        del primals_282
        del primals_283
        buf40 = reinterpret_tensor(buf42, (8, 8, 112, 112), (200704, 1, 1792, 16), 8)  # alias
        buf965 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1, x2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_12.run(buf29, buf36, buf37, primals_12, primals_13, buf40, buf965, 802816, grid=grid(802816), stream=stream0)
        del primals_13
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 8, 112, 112), (100352, 12544, 112, 1))
        buf44 = reinterpret_tensor(buf28, (8, 8, 112, 112), (100352, 1, 896, 8), 0); del buf28  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf43, buf44, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf45 = buf32; del buf32  # reuse
        buf46 = buf31; del buf31  # reuse
        buf47 = buf30; del buf30  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf44, buf45, buf46, buf47, 6272, 128, grid=grid(6272), stream=stream0)
        buf48 = buf35; del buf35  # reuse
        buf49 = buf34; del buf34  # reuse
        buf50 = buf33; del buf33  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf45, buf46, buf47, buf48, buf49, buf50, 56, 112, grid=grid(56), stream=stream0)
        buf51 = buf37; del buf37  # reuse
        buf52 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf54 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf48, buf49, buf50, primals_285, primals_286, buf51, buf52, buf54, primals_285, primals_286, 8, 7, grid=grid(8), stream=stream0)
        del primals_285
        del primals_286
        buf55 = reinterpret_tensor(buf43, (8, 8, 112, 112), (100352, 1, 896, 8), 0); del buf43  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_13.run(buf44, buf51, buf52, primals_15, primals_16, buf55, 802816, grid=grid(802816), stream=stream0)
        del primals_16
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf56, (8, 8, 112, 112), (100352, 12544, 112, 1))
        buf57 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf56, buf57, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del buf56
        buf58 = buf47; del buf47  # reuse
        buf59 = buf46; del buf46  # reuse
        buf60 = buf45; del buf45  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf57, buf58, buf59, buf60, 6272, 128, grid=grid(6272), stream=stream0)
        buf61 = buf50; del buf50  # reuse
        buf62 = buf49; del buf49  # reuse
        buf63 = buf48; del buf48  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf58, buf59, buf60, buf61, buf62, buf63, 56, 112, grid=grid(56), stream=stream0)
        del buf58
        del buf59
        del buf60
        buf64 = buf52; del buf52  # reuse
        buf65 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf67 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf61, buf62, buf63, primals_288, primals_289, buf64, buf65, buf67, primals_288, primals_289, 8, 7, grid=grid(8), stream=stream0)
        del primals_288
        del primals_289
        buf68 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_62, shortcut_1], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_14.run(buf55, buf57, buf64, buf65, primals_18, primals_19, buf14, buf68, 1605632, grid=grid(1605632), stream=stream0)
        del buf65
        del primals_19
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 24, 112, 112), (301056, 12544, 112, 1))
        buf70 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf69, buf70, 192, 12544, grid=grid(192, 12544), stream=stream0)
        buf71 = empty_strided((1, 24, 1, 1, 784), (18816, 1, 18816, 18816, 24), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 24, 1, 1, 784), (18816, 1, 18816, 18816, 24), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((1, 24, 1, 1, 784), (18816, 1, 18816, 18816, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf70, buf71, buf72, buf73, 18816, 128, grid=grid(18816), stream=stream0)
        buf74 = empty_strided((1, 24, 1, 1, 7), (168, 1, 168, 168, 24), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1, 24, 1, 1, 7), (168, 1, 168, 168, 24), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((1, 24, 1, 1, 7), (168, 1, 168, 168, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf71, buf72, buf73, buf74, buf75, buf76, 168, 112, grid=grid(168), stream=stream0)
        buf77 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf80 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_18.run(buf74, buf75, buf76, primals_291, primals_292, buf77, buf78, buf80, primals_291, primals_292, 24, 7, grid=grid(24), stream=stream0)
        del primals_291
        del primals_292
        buf81 = reinterpret_tensor(buf69, (8, 24, 112, 112), (301056, 1, 2688, 24), 0); del buf69  # reuse
        buf96 = empty_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        buf95 = reinterpret_tensor(buf96, (8, 24, 112, 112), (602112, 1, 5376, 48), 0)  # alias
        # Source Nodes: [cat_61, getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1, x1_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_19.run(buf70, buf77, buf78, primals_21, primals_22, buf81, buf95, 2408448, grid=grid(2408448), stream=stream0)
        del primals_22
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf82, (8, 24, 112, 112), (301056, 12544, 112, 1))
        buf83 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf82, buf83, 192, 12544, grid=grid(192, 12544), stream=stream0)
        del buf82
        buf84 = buf73; del buf73  # reuse
        buf85 = buf72; del buf72  # reuse
        buf86 = buf71; del buf71  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf83, buf84, buf85, buf86, 18816, 128, grid=grid(18816), stream=stream0)
        buf87 = buf76; del buf76  # reuse
        buf88 = buf75; del buf75  # reuse
        buf89 = buf74; del buf74  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf84, buf85, buf86, buf87, buf88, buf89, 168, 112, grid=grid(168), stream=stream0)
        del buf84
        del buf85
        del buf86
        buf90 = buf78; del buf78  # reuse
        buf91 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf93 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_18.run(buf87, buf88, buf89, primals_294, primals_295, buf90, buf91, buf93, primals_294, primals_295, 24, 7, grid=grid(24), stream=stream0)
        del buf87
        del buf88
        del buf89
        del primals_294
        del primals_295
        buf94 = reinterpret_tensor(buf96, (8, 24, 112, 112), (602112, 1, 5376, 48), 24)  # alias
        buf964 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1, x2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_20.run(buf83, buf90, buf91, primals_24, primals_25, buf94, buf964, 2408448, grid=grid(2408448), stream=stream0)
        del primals_25
        # Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf97, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf98 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf97, buf98, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf99 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        buf100 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        buf101 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_22.run(buf98, buf99, buf100, buf101, 9408, 128, grid=grid(9408), stream=stream0)
        buf102 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf99, buf100, buf101, buf102, buf103, buf104, 96, 98, grid=grid(96), stream=stream0)
        del buf100
        del buf101
        del buf99
        buf105 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf106 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf108 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_24.run(buf102, buf103, buf104, primals_297, primals_298, buf105, buf106, buf108, primals_297, primals_298, 48, 2, grid=grid(48), stream=stream0)
        del buf102
        del buf103
        del buf104
        del primals_297
        del primals_298
        buf109 = reinterpret_tensor(buf97, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf97  # reuse
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_25.run(buf98, buf105, buf106, primals_27, primals_28, buf109, 1204224, grid=grid(1204224), stream=stream0)
        del primals_28
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 12, 56, 56), (37632, 3136, 56, 1))
        buf111 = empty_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf110, buf111, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf112 = empty_strided((1, 12, 1, 1, 196), (2352, 1, 2352, 2352, 12), device='cuda', dtype=torch.float32)
        buf113 = empty_strided((1, 12, 1, 1, 196), (2352, 1, 2352, 2352, 12), device='cuda', dtype=torch.float32)
        buf114 = empty_strided((1, 12, 1, 1, 196), (2352, 1, 2352, 2352, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf111, buf112, buf113, buf114, 2352, 128, grid=grid(2352), stream=stream0)
        buf115 = reinterpret_tensor(buf91, (1, 12, 1, 1, 2), (24, 1, 24, 24, 12), 0); del buf91  # reuse
        buf116 = empty_strided((1, 12, 1, 1, 2), (24, 1, 24, 24, 12), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((1, 12, 1, 1, 2), (24, 1, 24, 24, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf112, buf113, buf114, buf115, buf116, buf117, 24, 98, grid=grid(24), stream=stream0)
        buf118 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cuda', dtype=torch.float32)
        buf121 = empty((12, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf115, buf116, buf117, primals_300, primals_301, buf118, buf119, buf121, primals_300, primals_301, 12, 2, grid=grid(12), stream=stream0)
        del primals_300
        del primals_301
        buf122 = reinterpret_tensor(buf110, (8, 12, 56, 56), (37632, 1, 672, 12), 0); del buf110  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_30.run(buf111, buf118, buf119, primals_30, primals_31, buf122, 301056, grid=grid(301056), stream=stream0)
        del primals_31
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
        assert_size_stride(buf123, (8, 12, 56, 56), (37632, 3136, 56, 1))
        buf124 = empty_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf123, buf124, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf125 = buf114; del buf114  # reuse
        buf126 = buf113; del buf113  # reuse
        buf127 = buf112; del buf112  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf124, buf125, buf126, buf127, 2352, 128, grid=grid(2352), stream=stream0)
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        buf130 = buf115; del buf115  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf125, buf126, buf127, buf128, buf129, buf130, 24, 98, grid=grid(24), stream=stream0)
        buf131 = buf119; del buf119  # reuse
        buf132 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cuda', dtype=torch.float32)
        buf134 = empty((12, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf128, buf129, buf130, primals_303, primals_304, buf131, buf132, buf134, primals_303, primals_304, 12, 2, grid=grid(12), stream=stream0)
        del primals_303
        del primals_304
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_0], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf68, primals_35, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf135, (8, 16, 56, 56), (50176, 3136, 56, 1))
        buf136 = empty_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_31.run(buf135, buf136, 128, 3136, grid=grid(128, 3136), stream=stream0)
        buf137 = empty_strided((1, 16, 1, 1, 196), (3136, 1, 3136, 3136, 16), device='cuda', dtype=torch.float32)
        buf138 = empty_strided((1, 16, 1, 1, 196), (3136, 1, 3136, 3136, 16), device='cuda', dtype=torch.float32)
        buf139 = empty_strided((1, 16, 1, 1, 196), (3136, 1, 3136, 3136, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf136, buf137, buf138, buf139, 3136, 128, grid=grid(3136), stream=stream0)
        buf140 = empty_strided((1, 16, 1, 1, 2), (32, 1, 32, 32, 16), device='cuda', dtype=torch.float32)
        buf141 = empty_strided((1, 16, 1, 1, 2), (32, 1, 32, 32, 16), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((1, 16, 1, 1, 2), (32, 1, 32, 32, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf137, buf138, buf139, buf140, buf141, buf142, 32, 98, grid=grid(32), stream=stream0)
        del buf137
        del buf138
        del buf139
        buf143 = buf11; del buf11  # reuse
        buf144 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf146 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf140, buf141, buf142, primals_306, primals_307, buf143, buf144, buf146, primals_306, primals_307, 16, 2, grid=grid(16), stream=stream0)
        del buf140
        del buf141
        del buf142
        del primals_306
        del primals_307
        buf147 = reinterpret_tensor(buf135, (8, 16, 56, 56), (50176, 1, 896, 16), 0); del buf135  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_35.run(buf136, buf143, buf144, primals_36, primals_37, buf147, 401408, grid=grid(401408), stream=stream0)
        del buf144
        del primals_37
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_2], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf149 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf148, buf149, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf150 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf151 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf152 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf149, buf150, buf151, buf152, 4704, 128, grid=grid(4704), stream=stream0)
        buf153 = reinterpret_tensor(buf106, (1, 24, 1, 1, 2), (48, 1, 48, 48, 24), 0); del buf106  # reuse
        buf154 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf150, buf151, buf152, buf153, buf154, buf155, 48, 98, grid=grid(48), stream=stream0)
        del buf150
        del buf151
        del buf152
        buf156 = reinterpret_tensor(buf130, (1, 24, 1, 1), (24, 1, 24, 24), 0); del buf130  # reuse
        buf157 = reinterpret_tensor(buf129, (1, 24, 1, 1), (24, 1, 24, 24), 0); del buf129  # reuse
        buf159 = reinterpret_tensor(buf128, (24, ), (1, ), 0); del buf128  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf153, buf154, buf155, primals_309, primals_310, buf156, buf157, buf159, primals_309, primals_310, 24, 2, grid=grid(24), stream=stream0)
        del buf153
        del buf154
        del buf155
        del primals_309
        del primals_310
        buf160 = reinterpret_tensor(buf148, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf148  # reuse
        # Source Nodes: [cat_60, getattr_getattr_l__mod___blocks___1_____0___shortcut_3, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_40.run(buf122, buf124, buf131, buf132, primals_33, primals_34, buf149, buf156, buf157, primals_39, primals_40, buf160, 602112, grid=grid(602112), stream=stream0)
        del primals_34
        del primals_40
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 36, 56, 56), (112896, 3136, 56, 1))
        buf162 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf161, buf162, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf163 = empty_strided((1, 36, 1, 1, 196), (7056, 1, 7056, 7056, 36), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((1, 36, 1, 1, 196), (7056, 1, 7056, 7056, 36), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((1, 36, 1, 1, 196), (7056, 1, 7056, 7056, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf162, buf163, buf164, buf165, 7056, 128, grid=grid(7056), stream=stream0)
        buf166 = empty_strided((1, 36, 1, 1, 2), (72, 1, 72, 72, 36), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 36, 1, 1, 2), (72, 1, 72, 72, 36), device='cuda', dtype=torch.float32)
        buf168 = empty_strided((1, 36, 1, 1, 2), (72, 1, 72, 72, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_43.run(buf163, buf164, buf165, buf166, buf167, buf168, 72, 98, grid=grid(72), stream=stream0)
        buf169 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf172 = empty((36, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_44.run(buf166, buf167, buf168, primals_312, primals_313, buf169, buf170, buf172, primals_312, primals_313, 36, 2, grid=grid(36), stream=stream0)
        del primals_312
        del primals_313
        buf173 = reinterpret_tensor(buf161, (8, 36, 56, 56), (112896, 1, 2016, 36), 0); del buf161  # reuse
        buf188 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        buf187 = reinterpret_tensor(buf188, (8, 36, 56, 56), (225792, 1, 4032, 72), 0)  # alias
        # Source Nodes: [cat_59, getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1, x1_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_45.run(buf162, buf169, buf170, primals_42, primals_43, buf173, buf187, 903168, grid=grid(903168), stream=stream0)
        del primals_43
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf174, (8, 36, 56, 56), (112896, 3136, 56, 1))
        buf175 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf174, buf175, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf176 = buf165; del buf165  # reuse
        buf177 = buf164; del buf164  # reuse
        buf178 = buf163; del buf163  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf175, buf176, buf177, buf178, 7056, 128, grid=grid(7056), stream=stream0)
        buf179 = buf168; del buf168  # reuse
        buf180 = buf167; del buf167  # reuse
        buf181 = buf166; del buf166  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_43.run(buf176, buf177, buf178, buf179, buf180, buf181, 72, 98, grid=grid(72), stream=stream0)
        buf182 = buf170; del buf170  # reuse
        buf183 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf185 = empty((36, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_44.run(buf179, buf180, buf181, primals_315, primals_316, buf182, buf183, buf185, primals_315, primals_316, 36, 2, grid=grid(36), stream=stream0)
        del primals_315
        del primals_316
        buf186 = reinterpret_tensor(buf188, (8, 36, 56, 56), (225792, 1, 4032, 72), 36)  # alias
        buf963 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1, x2_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_46.run(buf175, buf182, buf183, primals_45, primals_46, buf186, buf963, 903168, grid=grid(903168), stream=stream0)
        del primals_46
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 12, 56, 56), (37632, 3136, 56, 1))
        buf190 = reinterpret_tensor(buf123, (8, 12, 56, 56), (37632, 1, 672, 12), 0); del buf123  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf189, buf190, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf191 = buf127; del buf127  # reuse
        buf192 = buf126; del buf126  # reuse
        buf193 = buf125; del buf125  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf190, buf191, buf192, buf193, 2352, 128, grid=grid(2352), stream=stream0)
        buf194 = reinterpret_tensor(buf157, (1, 12, 1, 1, 2), (24, 1, 24, 24, 12), 0); del buf157  # reuse
        buf195 = empty_strided((1, 12, 1, 1, 2), (24, 1, 24, 24, 12), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((1, 12, 1, 1, 2), (24, 1, 24, 24, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf191, buf192, buf193, buf194, buf195, buf196, 24, 98, grid=grid(24), stream=stream0)
        buf197 = buf132; del buf132  # reuse
        buf198 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cuda', dtype=torch.float32)
        buf200 = empty((12, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf194, buf195, buf196, primals_318, primals_319, buf197, buf198, buf200, primals_318, primals_319, 12, 2, grid=grid(12), stream=stream0)
        del primals_318
        del primals_319
        buf201 = reinterpret_tensor(buf189, (8, 12, 56, 56), (37632, 1, 672, 12), 0); del buf189  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_30.run(buf190, buf197, buf198, primals_48, primals_49, buf201, 301056, grid=grid(301056), stream=stream0)
        del primals_49
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
        assert_size_stride(buf202, (8, 12, 56, 56), (37632, 3136, 56, 1))
        buf203 = empty_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_26.run(buf202, buf203, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del buf202
        buf204 = buf193; del buf193  # reuse
        buf205 = buf192; del buf192  # reuse
        buf206 = buf191; del buf191  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf203, buf204, buf205, buf206, 2352, 128, grid=grid(2352), stream=stream0)
        buf207 = buf196; del buf196  # reuse
        buf208 = buf195; del buf195  # reuse
        buf209 = buf194; del buf194  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf204, buf205, buf206, buf207, buf208, buf209, 24, 98, grid=grid(24), stream=stream0)
        del buf204
        del buf205
        del buf206
        buf210 = buf198; del buf198  # reuse
        buf211 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cuda', dtype=torch.float32)
        buf213 = empty((12, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_29.run(buf207, buf208, buf209, primals_321, primals_322, buf210, buf211, buf213, primals_321, primals_322, 12, 2, grid=grid(12), stream=stream0)
        del primals_321
        del primals_322
        buf214 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_58, shortcut_3], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_47.run(buf201, buf203, buf210, buf211, primals_51, primals_52, buf160, buf214, 602112, grid=grid(602112), stream=stream0)
        del buf211
        del primals_52
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 36, 56, 56), (112896, 3136, 56, 1))
        buf216 = reinterpret_tensor(buf174, (8, 36, 56, 56), (112896, 1, 2016, 36), 0); del buf174  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf215, buf216, 288, 3136, grid=grid(288, 3136), stream=stream0)
        buf217 = buf178; del buf178  # reuse
        buf218 = buf177; del buf177  # reuse
        buf219 = buf176; del buf176  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf216, buf217, buf218, buf219, 7056, 128, grid=grid(7056), stream=stream0)
        buf220 = buf181; del buf181  # reuse
        buf221 = buf180; del buf180  # reuse
        buf222 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_43.run(buf217, buf218, buf219, buf220, buf221, buf222, 72, 98, grid=grid(72), stream=stream0)
        buf223 = buf183; del buf183  # reuse
        buf224 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf226 = empty((36, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_44.run(buf220, buf221, buf222, primals_324, primals_325, buf223, buf224, buf226, primals_324, primals_325, 36, 2, grid=grid(36), stream=stream0)
        del primals_324
        del primals_325
        buf227 = reinterpret_tensor(buf215, (8, 36, 56, 56), (112896, 1, 2016, 36), 0); del buf215  # reuse
        buf242 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        buf241 = reinterpret_tensor(buf242, (8, 36, 56, 56), (225792, 1, 4032, 72), 0)  # alias
        # Source Nodes: [cat_57, getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1, x1_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_45.run(buf216, buf223, buf224, primals_54, primals_55, buf227, buf241, 903168, grid=grid(903168), stream=stream0)
        del primals_55
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
        assert_size_stride(buf228, (8, 36, 56, 56), (112896, 3136, 56, 1))
        buf229 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf228, buf229, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del buf228
        buf230 = buf219; del buf219  # reuse
        buf231 = buf218; del buf218  # reuse
        buf232 = buf217; del buf217  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf229, buf230, buf231, buf232, 7056, 128, grid=grid(7056), stream=stream0)
        buf233 = buf222; del buf222  # reuse
        buf234 = buf221; del buf221  # reuse
        buf235 = buf220; del buf220  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_43.run(buf230, buf231, buf232, buf233, buf234, buf235, 72, 98, grid=grid(72), stream=stream0)
        del buf230
        del buf231
        del buf232
        buf236 = buf224; del buf224  # reuse
        buf237 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf239 = empty((36, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_44.run(buf233, buf234, buf235, primals_327, primals_328, buf236, buf237, buf239, primals_327, primals_328, 36, 2, grid=grid(36), stream=stream0)
        del primals_327
        del primals_328
        buf240 = reinterpret_tensor(buf242, (8, 36, 56, 56), (225792, 1, 4032, 72), 36)  # alias
        buf962 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1, x2_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_46.run(buf229, buf236, buf237, primals_57, primals_58, buf240, buf962, 903168, grid=grid(903168), stream=stream0)
        del buf237
        del primals_58
        # Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_59, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf243, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf244 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf243, buf244, 576, 784, grid=grid(576, 784), stream=stream0)
        buf245 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_49.run(buf244, buf245, buf246, buf247, 3528, 128, grid=grid(3528), stream=stream0)
        buf248 = reinterpret_tensor(buf235, (1, 72, 1, 1), (72, 1, 72, 72), 0); del buf235  # reuse
        buf249 = reinterpret_tensor(buf234, (1, 72, 1, 1), (72, 1, 72, 72), 0); del buf234  # reuse
        buf251 = reinterpret_tensor(buf233, (72, ), (1, ), 0); del buf233  # reuse
        # Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_50.run(buf245, buf246, buf247, primals_330, primals_331, buf248, buf249, buf251, primals_330, primals_331, 72, 49, grid=grid(72), stream=stream0)
        del buf245
        del buf246
        del buf247
        del primals_330
        del primals_331
        buf252 = reinterpret_tensor(buf243, (8, 72, 28, 28), (56448, 1, 2016, 72), 0); del buf243  # reuse
        # Source Nodes: [x_16], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_51.run(buf244, buf248, buf249, primals_60, primals_61, buf252, 451584, grid=grid(451584), stream=stream0)
        del buf249
        del primals_61
        buf253 = empty_strided((8, 72, 1, 1, 7), (504, 1, 4032, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_red_fused_mean_52.run(buf252, buf253, 4032, 112, grid=grid(4032), stream=stream0)
        buf254 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf255 = reinterpret_tensor(buf254, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf254  # reuse
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_per_fused_mean_53.run(buf255, buf253, 576, 7, grid=grid(576), stream=stream0)
        del buf253
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 20, 1, 1), (20, 1, 1, 1))
        buf257 = reinterpret_tensor(buf256, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf256  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_54.run(buf257, primals_63, 160, grid=grid(160), stream=stream0)
        del primals_63
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 72, 1, 1), (72, 1, 1, 1))
        buf259 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_55.run(buf258, primals_65, buf259, 576, grid=grid(576), stream=stream0)
        buf260 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.mul]
        triton_poi_fused_mul_56.run(buf252, buf259, buf260, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 20, 28, 28), (15680, 784, 28, 1))
        buf262 = empty_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf261, buf262, 160, 784, grid=grid(160, 784), stream=stream0)
        buf263 = empty_strided((1, 20, 1, 1, 49), (980, 1, 980, 980, 20), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((1, 20, 1, 1, 49), (980, 1, 980, 980, 20), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((1, 20, 1, 1, 49), (980, 1, 980, 980, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf262, buf263, buf264, buf265, 980, 128, grid=grid(980), stream=stream0)
        buf266 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        buf267 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        buf269 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf263, buf264, buf265, primals_333, primals_334, buf266, buf267, buf269, primals_333, primals_334, 20, 49, grid=grid(20), stream=stream0)
        del primals_333
        del primals_334
        buf270 = reinterpret_tensor(buf261, (8, 20, 28, 28), (15680, 1, 560, 20), 0); del buf261  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_60.run(buf262, buf266, buf267, primals_67, primals_68, buf270, 125440, grid=grid(125440), stream=stream0)
        del primals_68
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf271, (8, 20, 28, 28), (15680, 784, 28, 1))
        buf272 = empty_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf271, buf272, 160, 784, grid=grid(160, 784), stream=stream0)
        buf273 = buf265; del buf265  # reuse
        buf274 = buf264; del buf264  # reuse
        buf275 = buf263; del buf263  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf272, buf273, buf274, buf275, 980, 128, grid=grid(980), stream=stream0)
        buf276 = buf267; del buf267  # reuse
        buf277 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        buf279 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf273, buf274, buf275, primals_336, primals_337, buf276, buf277, buf279, primals_336, primals_337, 20, 49, grid=grid(20), stream=stream0)
        del primals_336
        del primals_337
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_0], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf214, primals_72, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf280, (8, 24, 28, 28), (18816, 784, 28, 1))
        buf281 = empty_strided((8, 24, 28, 28), (18816, 1, 672, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf280, buf281, 192, 784, grid=grid(192, 784), stream=stream0)
        buf282 = empty_strided((1, 24, 1, 1, 49), (1176, 1, 1176, 1176, 24), device='cuda', dtype=torch.float32)
        buf283 = empty_strided((1, 24, 1, 1, 49), (1176, 1, 1176, 1176, 24), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((1, 24, 1, 1, 49), (1176, 1, 1176, 1176, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf281, buf282, buf283, buf284, 1176, 128, grid=grid(1176), stream=stream0)
        buf285 = reinterpret_tensor(buf209, (1, 24, 1, 1), (24, 1, 24, 24), 0); del buf209  # reuse
        buf286 = reinterpret_tensor(buf208, (1, 24, 1, 1), (24, 1, 24, 24), 0); del buf208  # reuse
        buf288 = reinterpret_tensor(buf207, (24, ), (1, ), 0); del buf207  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf282, buf283, buf284, primals_339, primals_340, buf285, buf286, buf288, primals_339, primals_340, 24, 49, grid=grid(24), stream=stream0)
        del buf282
        del buf283
        del buf284
        del primals_339
        del primals_340
        buf289 = reinterpret_tensor(buf280, (8, 24, 28, 28), (18816, 1, 672, 24), 0); del buf280  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_64.run(buf281, buf285, buf286, primals_73, primals_74, buf289, 150528, grid=grid(150528), stream=stream0)
        del buf286
        del primals_74
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_2], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf291 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf290, buf291, 320, 784, grid=grid(320, 784), stream=stream0)
        buf292 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf293 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf294 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf291, buf292, buf293, buf294, 1960, 128, grid=grid(1960), stream=stream0)
        buf295 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf296 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf298 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf292, buf293, buf294, primals_342, primals_343, buf295, buf296, buf298, primals_342, primals_343, 40, 49, grid=grid(40), stream=stream0)
        del buf292
        del buf293
        del buf294
        del primals_342
        del primals_343
        buf299 = reinterpret_tensor(buf290, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf290  # reuse
        # Source Nodes: [cat_56, getattr_getattr_l__mod___blocks___3_____0___shortcut_3, shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_68.run(buf270, buf272, buf276, buf277, primals_70, primals_71, buf291, buf295, buf296, primals_76, primals_77, buf299, 250880, grid=grid(250880), stream=stream0)
        del primals_71
        del primals_77
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf301 = empty_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf300, buf301, 480, 784, grid=grid(480, 784), stream=stream0)
        buf302 = empty_strided((1, 60, 1, 1, 49), (2940, 1, 2940, 2940, 60), device='cuda', dtype=torch.float32)
        buf303 = empty_strided((1, 60, 1, 1, 49), (2940, 1, 2940, 2940, 60), device='cuda', dtype=torch.float32)
        buf304 = empty_strided((1, 60, 1, 1, 49), (2940, 1, 2940, 2940, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf301, buf302, buf303, buf304, 2940, 128, grid=grid(2940), stream=stream0)
        buf305 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cuda', dtype=torch.float32)
        buf306 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cuda', dtype=torch.float32)
        buf308 = empty((60, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf302, buf303, buf304, primals_345, primals_346, buf305, buf306, buf308, primals_345, primals_346, 60, 49, grid=grid(60), stream=stream0)
        del primals_345
        del primals_346
        buf309 = reinterpret_tensor(buf300, (8, 60, 28, 28), (47040, 1, 1680, 60), 0); del buf300  # reuse
        buf321 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        buf320 = reinterpret_tensor(buf321, (8, 60, 28, 28), (94080, 1, 3360, 120), 0)  # alias
        # Source Nodes: [cat_55, getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1, x1_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_72.run(buf301, buf305, buf306, primals_79, primals_80, buf309, buf320, 376320, grid=grid(376320), stream=stream0)
        del primals_80
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf310, (8, 60, 28, 28), (47040, 784, 28, 1))
        buf311 = empty_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf310, buf311, 480, 784, grid=grid(480, 784), stream=stream0)
        buf312 = buf304; del buf304  # reuse
        buf313 = buf303; del buf303  # reuse
        buf314 = buf302; del buf302  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf311, buf312, buf313, buf314, 2940, 128, grid=grid(2940), stream=stream0)
        buf315 = buf306; del buf306  # reuse
        buf316 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cuda', dtype=torch.float32)
        buf318 = empty((60, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf312, buf313, buf314, primals_348, primals_349, buf315, buf316, buf318, primals_348, primals_349, 60, 49, grid=grid(60), stream=stream0)
        del buf312
        del buf313
        del buf314
        del primals_348
        del primals_349
        buf319 = reinterpret_tensor(buf321, (8, 60, 28, 28), (94080, 1, 3360, 120), 60)  # alias
        buf960 = empty_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1, x2_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_73.run(buf311, buf315, buf316, primals_82, primals_83, buf319, buf960, 376320, grid=grid(376320), stream=stream0)
        del buf316
        del primals_83
        buf322 = empty_strided((8, 120, 1, 1, 7), (840, 1, 6720, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_red_fused_mean_74.run(buf321, buf322, 6720, 112, grid=grid(6720), stream=stream0)
        buf323 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf324 = reinterpret_tensor(buf323, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf323  # reuse
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_per_fused_mean_75.run(buf324, buf322, 960, 7, grid=grid(960), stream=stream0)
        del buf322
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf324, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (8, 32, 1, 1), (32, 1, 1, 1))
        buf326 = reinterpret_tensor(buf325, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf325  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_76.run(buf326, primals_85, 256, grid=grid(256), stream=stream0)
        del primals_85
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 120, 1, 1), (120, 1, 1, 1))
        buf328 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_77.run(buf327, primals_87, buf328, 960, grid=grid(960), stream=stream0)
        buf329 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.mul]
        triton_poi_fused_mul_78.run(buf321, buf328, buf329, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 20, 28, 28), (15680, 784, 28, 1))
        buf331 = reinterpret_tensor(buf271, (8, 20, 28, 28), (15680, 1, 560, 20), 0); del buf271  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf330, buf331, 160, 784, grid=grid(160, 784), stream=stream0)
        buf332 = buf275; del buf275  # reuse
        buf333 = buf274; del buf274  # reuse
        buf334 = buf273; del buf273  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf331, buf332, buf333, buf334, 980, 128, grid=grid(980), stream=stream0)
        buf335 = buf277; del buf277  # reuse
        buf336 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        buf338 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf332, buf333, buf334, primals_351, primals_352, buf335, buf336, buf338, primals_351, primals_352, 20, 49, grid=grid(20), stream=stream0)
        del primals_351
        del primals_352
        buf339 = reinterpret_tensor(buf330, (8, 20, 28, 28), (15680, 1, 560, 20), 0); del buf330  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_60.run(buf331, buf335, buf336, primals_89, primals_90, buf339, 125440, grid=grid(125440), stream=stream0)
        del primals_90
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
        assert_size_stride(buf340, (8, 20, 28, 28), (15680, 784, 28, 1))
        buf341 = empty_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf340, buf341, 160, 784, grid=grid(160, 784), stream=stream0)
        buf342 = buf334; del buf334  # reuse
        buf343 = buf333; del buf333  # reuse
        buf344 = buf332; del buf332  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf341, buf342, buf343, buf344, 980, 128, grid=grid(980), stream=stream0)
        buf345 = buf336; del buf336  # reuse
        buf346 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        buf348 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf342, buf343, buf344, primals_354, primals_355, buf345, buf346, buf348, primals_354, primals_355, 20, 49, grid=grid(20), stream=stream0)
        del buf342
        del buf343
        del buf344
        del primals_354
        del primals_355
        buf349 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_54, shortcut_5], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_79.run(buf339, buf341, buf345, buf346, primals_92, primals_93, buf299, buf349, 250880, grid=grid(250880), stream=stream0)
        del buf346
        del primals_93
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf351 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf350, buf351, 960, 784, grid=grid(960, 784), stream=stream0)
        buf352 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf353 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf354 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf351, buf352, buf353, buf354, 5880, 128, grid=grid(5880), stream=stream0)
        buf355 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf356 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf358 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf352, buf353, buf354, primals_357, primals_358, buf355, buf356, buf358, primals_357, primals_358, 120, 49, grid=grid(120), stream=stream0)
        del primals_357
        del primals_358
        buf359 = reinterpret_tensor(buf350, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf350  # reuse
        buf371 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        buf370 = reinterpret_tensor(buf371, (8, 120, 28, 28), (188160, 1, 6720, 240), 0)  # alias
        # Source Nodes: [cat_53, getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1, x1_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_83.run(buf351, buf355, buf356, primals_95, primals_96, buf359, buf370, 752640, grid=grid(752640), stream=stream0)
        del primals_96
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf360, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf361 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_80.run(buf360, buf361, 960, 784, grid=grid(960, 784), stream=stream0)
        buf362 = buf354; del buf354  # reuse
        buf363 = buf353; del buf353  # reuse
        buf364 = buf352; del buf352  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf361, buf362, buf363, buf364, 5880, 128, grid=grid(5880), stream=stream0)
        buf365 = buf356; del buf356  # reuse
        buf366 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf368 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf362, buf363, buf364, primals_360, primals_361, buf365, buf366, buf368, primals_360, primals_361, 120, 49, grid=grid(120), stream=stream0)
        del buf362
        del buf363
        del buf364
        del primals_360
        del primals_361
        buf369 = reinterpret_tensor(buf371, (8, 120, 28, 28), (188160, 1, 6720, 240), 120)  # alias
        buf958 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1, x2_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_84.run(buf361, buf365, buf366, primals_98, primals_99, buf369, buf958, 752640, grid=grid(752640), stream=stream0)
        del buf366
        del primals_99
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_100, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf372, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf373 = reinterpret_tensor(buf310, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf310  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf372, buf373, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf374 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf375 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf373, buf374, buf375, buf376, 3120, 121, grid=grid(3120), stream=stream0)
        buf377 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf378 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf380 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf374, buf375, buf376, primals_363, primals_364, buf377, buf378, buf380, primals_363, primals_364, 240, 13, grid=grid(240), stream=stream0)
        del primals_363
        del primals_364
        buf381 = reinterpret_tensor(buf372, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf372  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_88.run(buf373, buf377, buf378, primals_101, primals_102, buf381, 376320, grid=grid(376320), stream=stream0)
        del primals_102
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf383 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf382, buf383, 320, 196, grid=grid(320, 196), stream=stream0)
        buf384 = empty_strided((1, 40, 1, 1, 13), (520, 1, 520, 520, 40), device='cuda', dtype=torch.float32)
        buf385 = empty_strided((1, 40, 1, 1, 13), (520, 1, 520, 520, 40), device='cuda', dtype=torch.float32)
        buf386 = empty_strided((1, 40, 1, 1, 13), (520, 1, 520, 520, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf383, buf384, buf385, buf386, 520, 121, grid=grid(520), stream=stream0)
        buf387 = buf296; del buf296  # reuse
        buf388 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf390 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf384, buf385, buf386, primals_366, primals_367, buf387, buf388, buf390, primals_366, primals_367, 40, 13, grid=grid(40), stream=stream0)
        del primals_366
        del primals_367
        buf391 = reinterpret_tensor(buf382, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf382  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf383, buf387, buf388, primals_104, primals_105, buf391, 62720, grid=grid(62720), stream=stream0)
        del primals_105
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf392, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf393 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf392, buf393, 320, 196, grid=grid(320, 196), stream=stream0)
        buf394 = buf386; del buf386  # reuse
        buf395 = buf385; del buf385  # reuse
        buf396 = buf384; del buf384  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf393, buf394, buf395, buf396, 520, 121, grid=grid(520), stream=stream0)
        buf397 = buf388; del buf388  # reuse
        buf398 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf400 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf394, buf395, buf396, primals_369, primals_370, buf397, buf398, buf400, primals_369, primals_370, 40, 13, grid=grid(40), stream=stream0)
        del primals_369
        del primals_370
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_0], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf349, primals_109, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf401, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf402 = reinterpret_tensor(buf392, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf392  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf401, buf402, 320, 196, grid=grid(320, 196), stream=stream0)
        buf403 = buf396; del buf396  # reuse
        buf404 = buf395; del buf395  # reuse
        buf405 = buf394; del buf394  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf402, buf403, buf404, buf405, 520, 121, grid=grid(520), stream=stream0)
        buf406 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf407 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf409 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf403, buf404, buf405, primals_372, primals_373, buf406, buf407, buf409, primals_372, primals_373, 40, 13, grid=grid(40), stream=stream0)
        del primals_372
        del primals_373
        buf410 = reinterpret_tensor(buf401, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf401  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf402, buf406, buf407, primals_110, primals_111, buf410, 62720, grid=grid(62720), stream=stream0)
        del primals_111
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_2], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf412 = reinterpret_tensor(buf340, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf340  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf411, buf412, 640, 196, grid=grid(640, 196), stream=stream0)
        buf413 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf414 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf415 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_94.run(buf412, buf413, buf414, buf415, 1040, 121, grid=grid(1040), stream=stream0)
        buf416 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf417 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf419 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_95.run(buf413, buf414, buf415, primals_375, primals_376, buf416, buf417, buf419, primals_375, primals_376, 80, 13, grid=grid(80), stream=stream0)
        del primals_375
        del primals_376
        buf420 = reinterpret_tensor(buf411, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf411  # reuse
        # Source Nodes: [cat_52, getattr_getattr_l__mod___blocks___5_____0___shortcut_3, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_96.run(buf391, buf393, buf397, buf398, primals_107, primals_108, buf412, buf416, buf417, primals_113, primals_114, buf420, 125440, grid=grid(125440), stream=stream0)
        del primals_108
        del primals_114
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (8, 100, 14, 14), (19600, 196, 14, 1))
        buf422 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf421, buf422, 800, 196, grid=grid(800, 196), stream=stream0)
        buf423 = empty_strided((1, 100, 1, 1, 13), (1300, 1, 1300, 1300, 100), device='cuda', dtype=torch.float32)
        buf424 = empty_strided((1, 100, 1, 1, 13), (1300, 1, 1300, 1300, 100), device='cuda', dtype=torch.float32)
        buf425 = empty_strided((1, 100, 1, 1, 13), (1300, 1, 1300, 1300, 100), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf422, buf423, buf424, buf425, 1300, 121, grid=grid(1300), stream=stream0)
        buf426 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cuda', dtype=torch.float32)
        buf427 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cuda', dtype=torch.float32)
        buf429 = empty((100, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf423, buf424, buf425, primals_378, primals_379, buf426, buf427, buf429, primals_378, primals_379, 100, 13, grid=grid(100), stream=stream0)
        del primals_378
        del primals_379
        buf430 = reinterpret_tensor(buf421, (8, 100, 14, 14), (19600, 1, 1400, 100), 0); del buf421  # reuse
        buf442 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        buf441 = reinterpret_tensor(buf442, (8, 100, 14, 14), (39200, 1, 2800, 200), 0)  # alias
        # Source Nodes: [cat_51, getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1, x1_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_100.run(buf422, buf426, buf427, primals_116, primals_117, buf430, buf441, 156800, grid=grid(156800), stream=stream0)
        del primals_117
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=100, bias=None)
        assert_size_stride(buf431, (8, 100, 14, 14), (19600, 196, 14, 1))
        buf432 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf431, buf432, 800, 196, grid=grid(800, 196), stream=stream0)
        del buf431
        buf433 = buf425; del buf425  # reuse
        buf434 = buf424; del buf424  # reuse
        buf435 = buf423; del buf423  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf432, buf433, buf434, buf435, 1300, 121, grid=grid(1300), stream=stream0)
        buf436 = buf427; del buf427  # reuse
        buf437 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cuda', dtype=torch.float32)
        buf439 = empty((100, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf433, buf434, buf435, primals_381, primals_382, buf436, buf437, buf439, primals_381, primals_382, 100, 13, grid=grid(100), stream=stream0)
        del buf433
        del buf434
        del buf435
        del primals_381
        del primals_382
        buf440 = reinterpret_tensor(buf442, (8, 100, 14, 14), (39200, 1, 2800, 200), 100)  # alias
        buf957 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1, x2_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_101.run(buf432, buf436, buf437, primals_119, primals_120, buf440, buf957, 156800, grid=grid(156800), stream=stream0)
        del buf437
        del primals_120
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf444 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf443, buf444, 320, 196, grid=grid(320, 196), stream=stream0)
        buf445 = buf405; del buf405  # reuse
        buf446 = buf404; del buf404  # reuse
        buf447 = buf403; del buf403  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf444, buf445, buf446, buf447, 520, 121, grid=grid(520), stream=stream0)
        buf448 = buf398; del buf398  # reuse
        buf449 = buf407; del buf407  # reuse
        buf451 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf445, buf446, buf447, primals_384, primals_385, buf448, buf449, buf451, primals_384, primals_385, 40, 13, grid=grid(40), stream=stream0)
        del primals_384
        del primals_385
        buf452 = reinterpret_tensor(buf443, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf443  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf444, buf448, buf449, primals_122, primals_123, buf452, 62720, grid=grid(62720), stream=stream0)
        del primals_123
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf453, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf454 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf453, buf454, 320, 196, grid=grid(320, 196), stream=stream0)
        buf455 = buf447; del buf447  # reuse
        buf456 = buf446; del buf446  # reuse
        buf457 = buf445; del buf445  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf454, buf455, buf456, buf457, 520, 121, grid=grid(520), stream=stream0)
        buf458 = buf449; del buf449  # reuse
        buf459 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf461 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf455, buf456, buf457, primals_387, primals_388, buf458, buf459, buf461, primals_387, primals_388, 40, 13, grid=grid(40), stream=stream0)
        del primals_387
        del primals_388
        buf462 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_50, shortcut_7], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_102.run(buf452, buf454, buf458, buf459, primals_125, primals_126, buf420, buf462, 125440, grid=grid(125440), stream=stream0)
        del primals_126
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (8, 92, 14, 14), (18032, 196, 14, 1))
        buf464 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf463, buf464, 736, 196, grid=grid(736, 196), stream=stream0)
        buf465 = empty_strided((1, 92, 1, 1, 13), (1196, 1, 1196, 1196, 92), device='cuda', dtype=torch.float32)
        buf466 = empty_strided((1, 92, 1, 1, 13), (1196, 1, 1196, 1196, 92), device='cuda', dtype=torch.float32)
        buf467 = empty_strided((1, 92, 1, 1, 13), (1196, 1, 1196, 1196, 92), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf464, buf465, buf466, buf467, 1196, 121, grid=grid(1196), stream=stream0)
        buf468 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cuda', dtype=torch.float32)
        buf469 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cuda', dtype=torch.float32)
        buf471 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf465, buf466, buf467, primals_390, primals_391, buf468, buf469, buf471, primals_390, primals_391, 92, 13, grid=grid(92), stream=stream0)
        del primals_390
        del primals_391
        buf472 = reinterpret_tensor(buf463, (8, 92, 14, 14), (18032, 1, 1288, 92), 0); del buf463  # reuse
        buf484 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        buf483 = reinterpret_tensor(buf484, (8, 92, 14, 14), (36064, 1, 2576, 184), 0)  # alias
        # Source Nodes: [cat_49, getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1, x1_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_106.run(buf464, buf468, buf469, primals_128, primals_129, buf472, buf483, 144256, grid=grid(144256), stream=stream0)
        del primals_129
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
        assert_size_stride(buf473, (8, 92, 14, 14), (18032, 196, 14, 1))
        buf474 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf473, buf474, 736, 196, grid=grid(736, 196), stream=stream0)
        buf475 = buf467; del buf467  # reuse
        buf476 = buf466; del buf466  # reuse
        buf477 = buf465; del buf465  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf474, buf475, buf476, buf477, 1196, 121, grid=grid(1196), stream=stream0)
        buf478 = buf469; del buf469  # reuse
        buf479 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cuda', dtype=torch.float32)
        buf481 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf475, buf476, buf477, primals_393, primals_394, buf478, buf479, buf481, primals_393, primals_394, 92, 13, grid=grid(92), stream=stream0)
        del primals_393
        del primals_394
        buf482 = reinterpret_tensor(buf484, (8, 92, 14, 14), (36064, 1, 2576, 184), 92)  # alias
        buf956 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1, x2_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_107.run(buf474, buf478, buf479, primals_131, primals_132, buf482, buf956, 144256, grid=grid(144256), stream=stream0)
        del primals_132
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf484, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf485, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf486 = reinterpret_tensor(buf453, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf453  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf485, buf486, 320, 196, grid=grid(320, 196), stream=stream0)
        buf487 = buf457; del buf457  # reuse
        buf488 = buf456; del buf456  # reuse
        buf489 = buf455; del buf455  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf486, buf487, buf488, buf489, 520, 121, grid=grid(520), stream=stream0)
        buf490 = buf459; del buf459  # reuse
        buf491 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf493 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf487, buf488, buf489, primals_396, primals_397, buf490, buf491, buf493, primals_396, primals_397, 40, 13, grid=grid(40), stream=stream0)
        del primals_396
        del primals_397
        buf494 = reinterpret_tensor(buf485, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf485  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf486, buf490, buf491, primals_134, primals_135, buf494, 62720, grid=grid(62720), stream=stream0)
        del primals_135
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf495, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf496 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf495, buf496, 320, 196, grid=grid(320, 196), stream=stream0)
        buf497 = buf489; del buf489  # reuse
        buf498 = buf488; del buf488  # reuse
        buf499 = buf487; del buf487  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf496, buf497, buf498, buf499, 520, 121, grid=grid(520), stream=stream0)
        buf500 = buf491; del buf491  # reuse
        buf501 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf503 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf497, buf498, buf499, primals_399, primals_400, buf500, buf501, buf503, primals_399, primals_400, 40, 13, grid=grid(40), stream=stream0)
        del primals_399
        del primals_400
        buf504 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_48, shortcut_8], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_102.run(buf494, buf496, buf500, buf501, primals_137, primals_138, buf462, buf504, 125440, grid=grid(125440), stream=stream0)
        del primals_138
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (8, 92, 14, 14), (18032, 196, 14, 1))
        buf506 = reinterpret_tensor(buf473, (8, 92, 14, 14), (18032, 1, 1288, 92), 0); del buf473  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf505, buf506, 736, 196, grid=grid(736, 196), stream=stream0)
        buf507 = buf477; del buf477  # reuse
        buf508 = buf476; del buf476  # reuse
        buf509 = buf475; del buf475  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf506, buf507, buf508, buf509, 1196, 121, grid=grid(1196), stream=stream0)
        buf510 = buf479; del buf479  # reuse
        buf511 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cuda', dtype=torch.float32)
        buf513 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf507, buf508, buf509, primals_402, primals_403, buf510, buf511, buf513, primals_402, primals_403, 92, 13, grid=grid(92), stream=stream0)
        del primals_402
        del primals_403
        buf514 = reinterpret_tensor(buf505, (8, 92, 14, 14), (18032, 1, 1288, 92), 0); del buf505  # reuse
        buf526 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        buf525 = reinterpret_tensor(buf526, (8, 92, 14, 14), (36064, 1, 2576, 184), 0)  # alias
        # Source Nodes: [cat_47, getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1, x1_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_106.run(buf506, buf510, buf511, primals_140, primals_141, buf514, buf525, 144256, grid=grid(144256), stream=stream0)
        del primals_141
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
        assert_size_stride(buf515, (8, 92, 14, 14), (18032, 196, 14, 1))
        buf516 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf515, buf516, 736, 196, grid=grid(736, 196), stream=stream0)
        del buf515
        buf517 = buf509; del buf509  # reuse
        buf518 = buf508; del buf508  # reuse
        buf519 = buf507; del buf507  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf516, buf517, buf518, buf519, 1196, 121, grid=grid(1196), stream=stream0)
        buf520 = buf511; del buf511  # reuse
        buf521 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cuda', dtype=torch.float32)
        buf523 = empty((92, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf517, buf518, buf519, primals_405, primals_406, buf520, buf521, buf523, primals_405, primals_406, 92, 13, grid=grid(92), stream=stream0)
        del buf517
        del buf518
        del buf519
        del primals_405
        del primals_406
        buf524 = reinterpret_tensor(buf526, (8, 92, 14, 14), (36064, 1, 2576, 184), 92)  # alias
        buf955 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1, x2_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_107.run(buf516, buf520, buf521, primals_143, primals_144, buf524, buf955, 144256, grid=grid(144256), stream=stream0)
        del buf521
        del primals_144
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf528 = reinterpret_tensor(buf495, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf495  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf527, buf528, 320, 196, grid=grid(320, 196), stream=stream0)
        buf529 = buf499; del buf499  # reuse
        buf530 = buf498; del buf498  # reuse
        buf531 = buf497; del buf497  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf528, buf529, buf530, buf531, 520, 121, grid=grid(520), stream=stream0)
        buf532 = buf501; del buf501  # reuse
        buf533 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf535 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf529, buf530, buf531, primals_408, primals_409, buf532, buf533, buf535, primals_408, primals_409, 40, 13, grid=grid(40), stream=stream0)
        del primals_408
        del primals_409
        buf536 = reinterpret_tensor(buf527, (8, 40, 14, 14), (7840, 1, 560, 40), 0); del buf527  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_92.run(buf528, buf532, buf533, primals_146, primals_147, buf536, 62720, grid=grid(62720), stream=stream0)
        del primals_147
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf537 = extern_kernels.convolution(buf536, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf537, (8, 40, 14, 14), (7840, 196, 14, 1))
        buf538 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf537, buf538, 320, 196, grid=grid(320, 196), stream=stream0)
        buf539 = buf531; del buf531  # reuse
        buf540 = buf530; del buf530  # reuse
        buf541 = buf529; del buf529  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf538, buf539, buf540, buf541, 520, 121, grid=grid(520), stream=stream0)
        buf542 = buf533; del buf533  # reuse
        buf543 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf545 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_91.run(buf539, buf540, buf541, primals_411, primals_412, buf542, buf543, buf545, primals_411, primals_412, 40, 13, grid=grid(40), stream=stream0)
        del buf539
        del buf540
        del buf541
        del primals_411
        del primals_412
        buf546 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_46, shortcut_9], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_102.run(buf536, buf538, buf542, buf543, primals_149, primals_150, buf504, buf546, 125440, grid=grid(125440), stream=stream0)
        del buf543
        del primals_150
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf548 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf547, buf548, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf549 = buf376; del buf376  # reuse
        buf550 = buf375; del buf375  # reuse
        buf551 = buf374; del buf374  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf548, buf549, buf550, buf551, 3120, 121, grid=grid(3120), stream=stream0)
        buf552 = buf378; del buf378  # reuse
        buf553 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf555 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf549, buf550, buf551, primals_414, primals_415, buf552, buf553, buf555, primals_414, primals_415, 240, 13, grid=grid(240), stream=stream0)
        del primals_414
        del primals_415
        buf556 = reinterpret_tensor(buf547, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf547  # reuse
        buf568 = reinterpret_tensor(buf360, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf360  # reuse
        buf567 = reinterpret_tensor(buf568, (8, 240, 14, 14), (94080, 1, 6720, 480), 0)  # alias
        # Source Nodes: [cat_45, getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1, x1_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_108.run(buf548, buf552, buf553, primals_152, primals_153, buf556, buf567, 376320, grid=grid(376320), stream=stream0)
        del primals_153
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf557, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf558 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf557, buf558, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf559 = buf551; del buf551  # reuse
        buf560 = buf550; del buf550  # reuse
        buf561 = buf549; del buf549  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf558, buf559, buf560, buf561, 3120, 121, grid=grid(3120), stream=stream0)
        buf562 = buf553; del buf553  # reuse
        buf563 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf565 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf559, buf560, buf561, primals_417, primals_418, buf562, buf563, buf565, primals_417, primals_418, 240, 13, grid=grid(240), stream=stream0)
        del buf559
        del buf560
        del buf561
        del primals_417
        del primals_418
        buf566 = reinterpret_tensor(buf568, (8, 240, 14, 14), (94080, 1, 6720, 480), 240)  # alias
        buf954 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1, x2_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_109.run(buf558, buf562, buf563, primals_155, primals_156, buf566, buf954, 376320, grid=grid(376320), stream=stream0)
        del buf563
        del primals_156
        buf569 = empty_strided((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_red_fused_mean_110.run(buf568, buf569, 7680, 98, grid=grid(7680), stream=stream0)
        buf570 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf571 = reinterpret_tensor(buf570, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf570  # reuse
        # Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_per_fused_mean_111.run(buf571, buf569, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (8, 120, 1, 1), (120, 1, 1, 1))
        buf573 = reinterpret_tensor(buf572, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf572  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_112.run(buf573, primals_158, 960, grid=grid(960), stream=stream0)
        del primals_158
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 480, 1, 1), (480, 1, 1, 1))
        buf575 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___se_gate, x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_113.run(buf574, primals_160, buf575, 3840, grid=grid(3840), stream=stream0)
        buf576 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten.mul]
        triton_poi_fused_mul_114.run(buf568, buf575, buf576, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf576, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf578 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf577, buf578, 448, 196, grid=grid(448, 196), stream=stream0)
        buf579 = empty_strided((1, 56, 1, 1, 13), (728, 1, 728, 728, 56), device='cuda', dtype=torch.float32)
        buf580 = empty_strided((1, 56, 1, 1, 13), (728, 1, 728, 728, 56), device='cuda', dtype=torch.float32)
        buf581 = empty_strided((1, 56, 1, 1, 13), (728, 1, 728, 728, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf578, buf579, buf580, buf581, 728, 121, grid=grid(728), stream=stream0)
        buf582 = reinterpret_tensor(buf63, (1, 56, 1, 1), (56, 1, 56, 56), 0); del buf63  # reuse
        buf583 = reinterpret_tensor(buf62, (1, 56, 1, 1), (56, 1, 56, 56), 0); del buf62  # reuse
        buf585 = reinterpret_tensor(buf61, (56, ), (1, ), 0); del buf61  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf579, buf580, buf581, primals_420, primals_421, buf582, buf583, buf585, primals_420, primals_421, 56, 13, grid=grid(56), stream=stream0)
        del primals_420
        del primals_421
        buf586 = reinterpret_tensor(buf577, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf577  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_118.run(buf578, buf582, buf583, primals_162, primals_163, buf586, 87808, grid=grid(87808), stream=stream0)
        del primals_163
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf587 = extern_kernels.convolution(buf586, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf587, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf588 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf587, buf588, 448, 196, grid=grid(448, 196), stream=stream0)
        buf589 = buf581; del buf581  # reuse
        buf590 = buf580; del buf580  # reuse
        buf591 = buf579; del buf579  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf588, buf589, buf590, buf591, 728, 121, grid=grid(728), stream=stream0)
        buf592 = buf583; del buf583  # reuse
        buf593 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf595 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf589, buf590, buf591, primals_423, primals_424, buf592, buf593, buf595, primals_423, primals_424, 56, 13, grid=grid(56), stream=stream0)
        del primals_423
        del primals_424
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_0], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf546, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf596, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf597 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_93.run(buf596, buf597, 640, 196, grid=grid(640, 196), stream=stream0)
        buf598 = buf415; del buf415  # reuse
        buf599 = buf414; del buf414  # reuse
        buf600 = buf413; del buf413  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_94.run(buf597, buf598, buf599, buf600, 1040, 121, grid=grid(1040), stream=stream0)
        buf601 = buf417; del buf417  # reuse
        buf602 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf604 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_95.run(buf598, buf599, buf600, primals_426, primals_427, buf601, buf602, buf604, primals_426, primals_427, 80, 13, grid=grid(80), stream=stream0)
        del buf598
        del buf599
        del buf600
        del primals_426
        del primals_427
        buf605 = reinterpret_tensor(buf596, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf596  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_119.run(buf597, buf601, buf602, primals_168, primals_169, buf605, 125440, grid=grid(125440), stream=stream0)
        del primals_169
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_2], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf607 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_120.run(buf606, buf607, 896, 196, grid=grid(896, 196), stream=stream0)
        buf608 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf609 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf610 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_121.run(buf607, buf608, buf609, buf610, 1456, 121, grid=grid(1456), stream=stream0)
        buf611 = reinterpret_tensor(buf9, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf9  # reuse
        buf612 = reinterpret_tensor(buf8, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf8  # reuse
        buf614 = reinterpret_tensor(buf7, (112, ), (1, ), 0); del buf7  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_122.run(buf608, buf609, buf610, primals_429, primals_430, buf611, buf612, buf614, primals_429, primals_430, 112, 13, grid=grid(112), stream=stream0)
        del buf608
        del buf609
        del buf610
        del primals_429
        del primals_430
        buf615 = reinterpret_tensor(buf606, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf606  # reuse
        # Source Nodes: [cat_44, getattr_getattr_l__mod___blocks___6_____3___shortcut_3, shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_123.run(buf586, buf588, buf592, buf593, primals_165, primals_166, buf607, buf611, buf612, primals_171, primals_172, buf615, 175616, grid=grid(175616), stream=stream0)
        del primals_166
        del primals_172
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf616 = extern_kernels.convolution(buf615, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf616, (8, 336, 14, 14), (65856, 196, 14, 1))
        buf617 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf616, buf617, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf618 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf619 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf620 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_125.run(buf617, buf618, buf619, buf620, 4368, 121, grid=grid(4368), stream=stream0)
        buf621 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf622 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf624 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_126.run(buf618, buf619, buf620, primals_432, primals_433, buf621, buf622, buf624, primals_432, primals_433, 336, 13, grid=grid(336), stream=stream0)
        del primals_432
        del primals_433
        buf625 = reinterpret_tensor(buf616, (8, 336, 14, 14), (65856, 1, 4704, 336), 0); del buf616  # reuse
        buf637 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        buf636 = reinterpret_tensor(buf637, (8, 336, 14, 14), (131712, 1, 9408, 672), 0)  # alias
        # Source Nodes: [cat_43, getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1, x1_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_127.run(buf617, buf621, buf622, primals_174, primals_175, buf625, buf636, 526848, grid=grid(526848), stream=stream0)
        del primals_175
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf626, (8, 336, 14, 14), (65856, 196, 14, 1))
        buf627 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf626, buf627, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf628 = buf620; del buf620  # reuse
        buf629 = buf619; del buf619  # reuse
        buf630 = buf618; del buf618  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_125.run(buf627, buf628, buf629, buf630, 4368, 121, grid=grid(4368), stream=stream0)
        buf631 = buf622; del buf622  # reuse
        buf632 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf634 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_126.run(buf628, buf629, buf630, primals_435, primals_436, buf631, buf632, buf634, primals_435, primals_436, 336, 13, grid=grid(336), stream=stream0)
        del primals_435
        del primals_436
        buf635 = reinterpret_tensor(buf637, (8, 336, 14, 14), (131712, 1, 9408, 672), 336)  # alias
        buf952 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1, x2_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_128.run(buf627, buf631, buf632, primals_177, primals_178, buf635, buf952, 526848, grid=grid(526848), stream=stream0)
        del primals_178
        buf638 = empty_strided((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_red_fused_mean_129.run(buf637, buf638, 10752, 98, grid=grid(10752), stream=stream0)
        buf639 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf640 = reinterpret_tensor(buf639, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf639  # reuse
        # Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_per_fused_mean_130.run(buf640, buf638, 5376, 2, grid=grid(5376), stream=stream0)
        del buf638
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (8, 168, 1, 1), (168, 1, 1, 1))
        buf642 = reinterpret_tensor(buf641, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf641  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_131.run(buf642, primals_180, 1344, grid=grid(1344), stream=stream0)
        del primals_180
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (8, 672, 1, 1), (672, 1, 1, 1))
        buf644 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___se_gate, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_132.run(buf643, primals_182, buf644, 5376, grid=grid(5376), stream=stream0)
        buf645 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.mul]
        triton_poi_fused_mul_133.run(buf637, buf644, buf645, 1053696, grid=grid(1053696), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf645, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf647 = reinterpret_tensor(buf587, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf587  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf646, buf647, 448, 196, grid=grid(448, 196), stream=stream0)
        buf648 = buf591; del buf591  # reuse
        buf649 = buf590; del buf590  # reuse
        buf650 = buf589; del buf589  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf647, buf648, buf649, buf650, 728, 121, grid=grid(728), stream=stream0)
        buf651 = buf593; del buf593  # reuse
        buf652 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf654 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf648, buf649, buf650, primals_438, primals_439, buf651, buf652, buf654, primals_438, primals_439, 56, 13, grid=grid(56), stream=stream0)
        del primals_438
        del primals_439
        buf655 = reinterpret_tensor(buf646, (8, 56, 14, 14), (10976, 1, 784, 56), 0); del buf646  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_118.run(buf647, buf651, buf652, primals_184, primals_185, buf655, 87808, grid=grid(87808), stream=stream0)
        del primals_185
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
        assert_size_stride(buf656, (8, 56, 14, 14), (10976, 196, 14, 1))
        buf657 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_115.run(buf656, buf657, 448, 196, grid=grid(448, 196), stream=stream0)
        del buf656
        buf658 = buf650; del buf650  # reuse
        buf659 = buf649; del buf649  # reuse
        buf660 = buf648; del buf648  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf657, buf658, buf659, buf660, 728, 121, grid=grid(728), stream=stream0)
        buf661 = buf652; del buf652  # reuse
        buf662 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf664 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_117.run(buf658, buf659, buf660, primals_441, primals_442, buf661, buf662, buf664, primals_441, primals_442, 56, 13, grid=grid(56), stream=stream0)
        del buf658
        del buf659
        del buf660
        del primals_441
        del primals_442
        buf665 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_42, shortcut_11], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_134.run(buf655, buf657, buf661, buf662, primals_187, primals_188, buf615, buf665, 175616, grid=grid(175616), stream=stream0)
        del buf662
        del primals_188
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (8, 336, 14, 14), (65856, 196, 14, 1))
        buf667 = reinterpret_tensor(buf626, (8, 336, 14, 14), (65856, 1, 4704, 336), 0); del buf626  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf666, buf667, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf668 = buf630; del buf630  # reuse
        buf669 = buf629; del buf629  # reuse
        buf670 = buf628; del buf628  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_125.run(buf667, buf668, buf669, buf670, 4368, 121, grid=grid(4368), stream=stream0)
        buf671 = buf632; del buf632  # reuse
        buf672 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf674 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_126.run(buf668, buf669, buf670, primals_444, primals_445, buf671, buf672, buf674, primals_444, primals_445, 336, 13, grid=grid(336), stream=stream0)
        del primals_444
        del primals_445
        buf675 = reinterpret_tensor(buf666, (8, 336, 14, 14), (65856, 1, 4704, 336), 0); del buf666  # reuse
        buf687 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        buf686 = reinterpret_tensor(buf687, (8, 336, 14, 14), (131712, 1, 9408, 672), 0)  # alias
        # Source Nodes: [cat_41, getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1, x1_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_127.run(buf667, buf671, buf672, primals_190, primals_191, buf675, buf686, 526848, grid=grid(526848), stream=stream0)
        del primals_191
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf676, (8, 336, 14, 14), (65856, 196, 14, 1))
        buf677 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_124.run(buf676, buf677, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del buf676
        buf678 = buf670; del buf670  # reuse
        buf679 = buf669; del buf669  # reuse
        buf680 = buf668; del buf668  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_125.run(buf677, buf678, buf679, buf680, 4368, 121, grid=grid(4368), stream=stream0)
        buf681 = buf672; del buf672  # reuse
        buf682 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf684 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_126.run(buf678, buf679, buf680, primals_447, primals_448, buf681, buf682, buf684, primals_447, primals_448, 336, 13, grid=grid(336), stream=stream0)
        del buf678
        del buf679
        del buf680
        del primals_447
        del primals_448
        buf685 = reinterpret_tensor(buf687, (8, 336, 14, 14), (131712, 1, 9408, 672), 336)  # alias
        buf950 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1, x2_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_128.run(buf677, buf681, buf682, primals_193, primals_194, buf685, buf950, 526848, grid=grid(526848), stream=stream0)
        del buf682
        del primals_194
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf688 = extern_kernels.convolution(buf687, primals_195, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf688, (8, 672, 7, 7), (32928, 49, 7, 1))
        buf689 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_135.run(buf688, buf689, 5376, 49, grid=grid(5376, 49), stream=stream0)
        buf690 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf691 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf692 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_136.run(buf689, buf690, buf691, buf692, 2688, 98, grid=grid(2688), stream=stream0)
        buf693 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf694 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf696 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_137.run(buf690, buf691, buf692, primals_450, primals_451, buf693, buf694, buf696, primals_450, primals_451, 672, 4, grid=grid(672), stream=stream0)
        del buf690
        del buf691
        del buf692
        del primals_450
        del primals_451
        buf697 = reinterpret_tensor(buf688, (8, 672, 7, 7), (32928, 1, 4704, 672), 0); del buf688  # reuse
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_138.run(buf689, buf693, buf694, primals_196, primals_197, buf697, 263424, grid=grid(263424), stream=stream0)
        del buf694
        del primals_197
        buf698 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf699 = reinterpret_tensor(buf698, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf698  # reuse
        # Source Nodes: [x_se_16], Original ATen: [aten.mean]
        triton_per_fused_mean_139.run(buf699, buf697, 5376, 49, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (8, 168, 1, 1), (168, 1, 1, 1))
        buf701 = reinterpret_tensor(buf700, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf700  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_131.run(buf701, primals_199, 1344, grid=grid(1344), stream=stream0)
        del primals_199
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf702 = extern_kernels.convolution(buf701, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf702, (8, 672, 1, 1), (672, 1, 1, 1))
        buf703 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___se_gate, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_132.run(buf702, primals_201, buf703, 5376, grid=grid(5376), stream=stream0)
        buf704 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.mul]
        triton_poi_fused_mul_140.run(buf697, buf703, buf704, 263424, grid=grid(263424), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf705 = extern_kernels.convolution(buf704, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf705, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf706 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf705, buf706, 640, 49, grid=grid(640, 49), stream=stream0)
        buf707 = empty_strided((1, 80, 1, 1, 4), (320, 1, 320, 320, 80), device='cuda', dtype=torch.float32)
        buf708 = empty_strided((1, 80, 1, 1, 4), (320, 1, 320, 320, 80), device='cuda', dtype=torch.float32)
        buf709 = empty_strided((1, 80, 1, 1, 4), (320, 1, 320, 320, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf706, buf707, buf708, buf709, 320, 98, grid=grid(320), stream=stream0)
        buf710 = buf602; del buf602  # reuse
        buf711 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf713 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf707, buf708, buf709, primals_453, primals_454, buf710, buf711, buf713, primals_453, primals_454, 80, 4, grid=grid(80), stream=stream0)
        del primals_453
        del primals_454
        buf714 = reinterpret_tensor(buf705, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf705  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_144.run(buf706, buf710, buf711, primals_203, primals_204, buf714, 31360, grid=grid(31360), stream=stream0)
        del primals_204
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf714, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf715, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf716 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf715, buf716, 640, 49, grid=grid(640, 49), stream=stream0)
        buf717 = buf709; del buf709  # reuse
        buf718 = buf708; del buf708  # reuse
        buf719 = buf707; del buf707  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf716, buf717, buf718, buf719, 320, 98, grid=grid(320), stream=stream0)
        buf720 = buf711; del buf711  # reuse
        buf721 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf723 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf717, buf718, buf719, primals_456, primals_457, buf720, buf721, buf723, primals_456, primals_457, 80, 4, grid=grid(80), stream=stream0)
        del primals_456
        del primals_457
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_0], Original ATen: [aten.convolution]
        buf724 = extern_kernels.convolution(buf665, primals_208, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
        assert_size_stride(buf724, (8, 112, 7, 7), (5488, 49, 7, 1))
        buf725 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_145.run(buf724, buf725, 896, 49, grid=grid(896, 49), stream=stream0)
        buf726 = empty_strided((1, 112, 1, 1, 4), (448, 1, 448, 448, 112), device='cuda', dtype=torch.float32)
        buf727 = empty_strided((1, 112, 1, 1, 4), (448, 1, 448, 448, 112), device='cuda', dtype=torch.float32)
        buf728 = empty_strided((1, 112, 1, 1, 4), (448, 1, 448, 448, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_146.run(buf725, buf726, buf727, buf728, 448, 98, grid=grid(448), stream=stream0)
        buf729 = buf612; del buf612  # reuse
        buf730 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf732 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_147.run(buf726, buf727, buf728, primals_459, primals_460, buf729, buf730, buf732, primals_459, primals_460, 112, 4, grid=grid(112), stream=stream0)
        del buf726
        del buf727
        del buf728
        del primals_459
        del primals_460
        buf733 = reinterpret_tensor(buf724, (8, 112, 7, 7), (5488, 1, 784, 112), 0); del buf724  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_148.run(buf725, buf729, buf730, primals_209, primals_210, buf733, 43904, grid=grid(43904), stream=stream0)
        del buf730
        del primals_210
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_2], Original ATen: [aten.convolution]
        buf734 = extern_kernels.convolution(buf733, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf734, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf735 = reinterpret_tensor(buf537, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf537  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_149.run(buf734, buf735, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf736 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        buf737 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        buf738 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_150.run(buf735, buf736, buf737, buf738, 640, 98, grid=grid(640), stream=stream0)
        buf739 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf740 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf742 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_151.run(buf736, buf737, buf738, primals_462, primals_463, buf739, buf740, buf742, primals_462, primals_463, 160, 4, grid=grid(160), stream=stream0)
        del buf736
        del buf737
        del buf738
        del primals_462
        del primals_463
        buf743 = reinterpret_tensor(buf734, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf734  # reuse
        # Source Nodes: [cat_40, getattr_getattr_l__mod___blocks___7_____0___shortcut_3, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.cat]
        triton_poi_fused__native_batch_norm_legit_functional_add_cat_152.run(buf714, buf716, buf720, buf721, primals_206, primals_207, buf735, buf739, buf740, primals_212, primals_213, buf743, 62720, grid=grid(62720), stream=stream0)
        del buf740
        del primals_207
        del primals_213
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf744 = extern_kernels.convolution(buf743, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf744, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf745 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf744, buf745, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf746 = empty_strided((1, 480, 1, 1, 4), (1920, 1, 1920, 1920, 480), device='cuda', dtype=torch.float32)
        buf747 = empty_strided((1, 480, 1, 1, 4), (1920, 1, 1920, 1920, 480), device='cuda', dtype=torch.float32)
        buf748 = empty_strided((1, 480, 1, 1, 4), (1920, 1, 1920, 1920, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf745, buf746, buf747, buf748, 1920, 98, grid=grid(1920), stream=stream0)
        buf749 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf750 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf752 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf746, buf747, buf748, primals_465, primals_466, buf749, buf750, buf752, primals_465, primals_466, 480, 4, grid=grid(480), stream=stream0)
        del primals_465
        del primals_466
        buf753 = reinterpret_tensor(buf744, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf744  # reuse
        buf765 = reinterpret_tensor(buf557, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf557  # reuse
        buf764 = reinterpret_tensor(buf765, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
        # Source Nodes: [cat_39, getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1, x1_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_156.run(buf745, buf749, buf750, primals_215, primals_216, buf753, buf764, 188160, grid=grid(188160), stream=stream0)
        del primals_216
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf754 = extern_kernels.convolution(buf753, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf754, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf755 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf754, buf755, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf756 = buf748; del buf748  # reuse
        buf757 = buf747; del buf747  # reuse
        buf758 = buf746; del buf746  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf755, buf756, buf757, buf758, 1920, 98, grid=grid(1920), stream=stream0)
        buf759 = buf750; del buf750  # reuse
        buf760 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf762 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf756, buf757, buf758, primals_468, primals_469, buf759, buf760, buf762, primals_468, primals_469, 480, 4, grid=grid(480), stream=stream0)
        del primals_468
        del primals_469
        buf763 = reinterpret_tensor(buf765, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
        buf948 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1, x2_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_157.run(buf755, buf759, buf760, primals_218, primals_219, buf763, buf948, 188160, grid=grid(188160), stream=stream0)
        del primals_219
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf766 = extern_kernels.convolution(buf765, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf766, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf767 = reinterpret_tensor(buf715, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf715  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf766, buf767, 640, 49, grid=grid(640, 49), stream=stream0)
        buf768 = buf719; del buf719  # reuse
        buf769 = buf718; del buf718  # reuse
        buf770 = buf717; del buf717  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf767, buf768, buf769, buf770, 320, 98, grid=grid(320), stream=stream0)
        buf771 = buf721; del buf721  # reuse
        buf772 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf774 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf768, buf769, buf770, primals_471, primals_472, buf771, buf772, buf774, primals_471, primals_472, 80, 4, grid=grid(80), stream=stream0)
        del primals_471
        del primals_472
        buf775 = reinterpret_tensor(buf766, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf766  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_144.run(buf767, buf771, buf772, primals_221, primals_222, buf775, 31360, grid=grid(31360), stream=stream0)
        del primals_222
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf776 = extern_kernels.convolution(buf775, primals_223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf776, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf777 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf776, buf777, 640, 49, grid=grid(640, 49), stream=stream0)
        buf778 = buf770; del buf770  # reuse
        buf779 = buf769; del buf769  # reuse
        buf780 = buf768; del buf768  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf777, buf778, buf779, buf780, 320, 98, grid=grid(320), stream=stream0)
        buf781 = buf772; del buf772  # reuse
        buf782 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf784 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf778, buf779, buf780, primals_474, primals_475, buf781, buf782, buf784, primals_474, primals_475, 80, 4, grid=grid(80), stream=stream0)
        del primals_474
        del primals_475
        buf785 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_38, shortcut_13], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_158.run(buf775, buf777, buf781, buf782, primals_224, primals_225, buf743, buf785, 62720, grid=grid(62720), stream=stream0)
        del primals_225
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf786 = extern_kernels.convolution(buf785, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf786, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf787 = reinterpret_tensor(buf754, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf754  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf786, buf787, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf788 = buf758; del buf758  # reuse
        buf789 = buf757; del buf757  # reuse
        buf790 = buf756; del buf756  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf787, buf788, buf789, buf790, 1920, 98, grid=grid(1920), stream=stream0)
        buf791 = buf760; del buf760  # reuse
        buf792 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf794 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf788, buf789, buf790, primals_477, primals_478, buf791, buf792, buf794, primals_477, primals_478, 480, 4, grid=grid(480), stream=stream0)
        del primals_477
        del primals_478
        buf795 = reinterpret_tensor(buf786, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf786  # reuse
        buf807 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        buf806 = reinterpret_tensor(buf807, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
        # Source Nodes: [cat_37, getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1, x1_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_156.run(buf787, buf791, buf792, primals_227, primals_228, buf795, buf806, 188160, grid=grid(188160), stream=stream0)
        del primals_228
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf796 = extern_kernels.convolution(buf795, primals_229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf796, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf797 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf796, buf797, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf798 = buf790; del buf790  # reuse
        buf799 = buf789; del buf789  # reuse
        buf800 = buf788; del buf788  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf797, buf798, buf799, buf800, 1920, 98, grid=grid(1920), stream=stream0)
        buf801 = buf792; del buf792  # reuse
        buf802 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf804 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf798, buf799, buf800, primals_480, primals_481, buf801, buf802, buf804, primals_480, primals_481, 480, 4, grid=grid(480), stream=stream0)
        del primals_480
        del primals_481
        buf805 = reinterpret_tensor(buf807, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
        buf947 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1, x2_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_157.run(buf797, buf801, buf802, primals_230, primals_231, buf805, buf947, 188160, grid=grid(188160), stream=stream0)
        del primals_231
        buf808 = reinterpret_tensor(buf569, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf569  # reuse
        buf809 = reinterpret_tensor(buf808, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf808  # reuse
        # Source Nodes: [x_se_20], Original ATen: [aten.mean]
        triton_per_fused_mean_159.run(buf809, buf807, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf810 = extern_kernels.convolution(buf809, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf810, (8, 240, 1, 1), (240, 1, 1, 1))
        buf811 = reinterpret_tensor(buf810, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf810  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_160.run(buf811, primals_233, 1920, grid=grid(1920), stream=stream0)
        del primals_233
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf812 = extern_kernels.convolution(buf811, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf812, (8, 960, 1, 1), (960, 1, 1, 1))
        buf813 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___se_gate, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_161.run(buf812, primals_235, buf813, 7680, grid=grid(7680), stream=stream0)
        buf814 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.mul]
        triton_poi_fused_mul_162.run(buf807, buf813, buf814, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf815 = extern_kernels.convolution(buf814, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf815, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf816 = reinterpret_tensor(buf776, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf776  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf815, buf816, 640, 49, grid=grid(640, 49), stream=stream0)
        buf817 = buf780; del buf780  # reuse
        buf818 = buf779; del buf779  # reuse
        buf819 = buf778; del buf778  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf816, buf817, buf818, buf819, 320, 98, grid=grid(320), stream=stream0)
        buf820 = buf782; del buf782  # reuse
        buf821 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf823 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf817, buf818, buf819, primals_483, primals_484, buf820, buf821, buf823, primals_483, primals_484, 80, 4, grid=grid(80), stream=stream0)
        del primals_483
        del primals_484
        buf824 = reinterpret_tensor(buf815, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf815  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_144.run(buf816, buf820, buf821, primals_237, primals_238, buf824, 31360, grid=grid(31360), stream=stream0)
        del primals_238
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf825 = extern_kernels.convolution(buf824, primals_239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf825, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf826 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf825, buf826, 640, 49, grid=grid(640, 49), stream=stream0)
        buf827 = buf819; del buf819  # reuse
        buf828 = buf818; del buf818  # reuse
        buf829 = buf817; del buf817  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf826, buf827, buf828, buf829, 320, 98, grid=grid(320), stream=stream0)
        buf830 = buf821; del buf821  # reuse
        buf831 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf833 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf827, buf828, buf829, primals_486, primals_487, buf830, buf831, buf833, primals_486, primals_487, 80, 4, grid=grid(80), stream=stream0)
        del primals_486
        del primals_487
        buf834 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_36, shortcut_14], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_158.run(buf824, buf826, buf830, buf831, primals_240, primals_241, buf785, buf834, 62720, grid=grid(62720), stream=stream0)
        del primals_241
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf835 = extern_kernels.convolution(buf834, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf836 = reinterpret_tensor(buf796, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf796  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf835, buf836, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf837 = buf800; del buf800  # reuse
        buf838 = buf799; del buf799  # reuse
        buf839 = buf798; del buf798  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf836, buf837, buf838, buf839, 1920, 98, grid=grid(1920), stream=stream0)
        buf840 = buf802; del buf802  # reuse
        buf841 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf843 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf837, buf838, buf839, primals_489, primals_490, buf840, buf841, buf843, primals_489, primals_490, 480, 4, grid=grid(480), stream=stream0)
        del primals_489
        del primals_490
        buf844 = reinterpret_tensor(buf835, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf835  # reuse
        buf856 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        buf855 = reinterpret_tensor(buf856, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
        # Source Nodes: [cat_35, getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1, x1_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_156.run(buf836, buf840, buf841, primals_243, primals_244, buf844, buf855, 188160, grid=grid(188160), stream=stream0)
        del primals_244
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf845 = extern_kernels.convolution(buf844, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf845, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf846 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf845, buf846, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf847 = buf839; del buf839  # reuse
        buf848 = buf838; del buf838  # reuse
        buf849 = buf837; del buf837  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf846, buf847, buf848, buf849, 1920, 98, grid=grid(1920), stream=stream0)
        buf850 = buf841; del buf841  # reuse
        buf851 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf853 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf847, buf848, buf849, primals_492, primals_493, buf850, buf851, buf853, primals_492, primals_493, 480, 4, grid=grid(480), stream=stream0)
        del primals_492
        del primals_493
        buf854 = reinterpret_tensor(buf856, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
        buf945 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1, x2_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_157.run(buf846, buf850, buf851, primals_246, primals_247, buf854, buf945, 188160, grid=grid(188160), stream=stream0)
        del primals_247
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf857 = extern_kernels.convolution(buf856, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf857, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf858 = reinterpret_tensor(buf825, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf825  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf857, buf858, 640, 49, grid=grid(640, 49), stream=stream0)
        buf859 = buf829; del buf829  # reuse
        buf860 = buf828; del buf828  # reuse
        buf861 = buf827; del buf827  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf858, buf859, buf860, buf861, 320, 98, grid=grid(320), stream=stream0)
        buf862 = buf831; del buf831  # reuse
        buf863 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf865 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf859, buf860, buf861, primals_495, primals_496, buf862, buf863, buf865, primals_495, primals_496, 80, 4, grid=grid(80), stream=stream0)
        del primals_495
        del primals_496
        buf866 = reinterpret_tensor(buf857, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf857  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_144.run(buf858, buf862, buf863, primals_249, primals_250, buf866, 31360, grid=grid(31360), stream=stream0)
        del primals_250
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf867 = extern_kernels.convolution(buf866, primals_251, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf867, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf868 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf867, buf868, 640, 49, grid=grid(640, 49), stream=stream0)
        buf869 = buf861; del buf861  # reuse
        buf870 = buf860; del buf860  # reuse
        buf871 = buf859; del buf859  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf868, buf869, buf870, buf871, 320, 98, grid=grid(320), stream=stream0)
        buf872 = buf863; del buf863  # reuse
        buf873 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf875 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf869, buf870, buf871, primals_498, primals_499, buf872, buf873, buf875, primals_498, primals_499, 80, 4, grid=grid(80), stream=stream0)
        del primals_498
        del primals_499
        buf876 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_34, shortcut_15], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_158.run(buf866, buf868, buf872, buf873, primals_252, primals_253, buf834, buf876, 62720, grid=grid(62720), stream=stream0)
        del primals_253
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        buf877 = extern_kernels.convolution(buf876, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf877, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf878 = reinterpret_tensor(buf845, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf845  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf877, buf878, 3840, 49, grid=grid(3840, 49), stream=stream0)
        buf879 = buf849; del buf849  # reuse
        buf880 = buf848; del buf848  # reuse
        buf881 = buf847; del buf847  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf878, buf879, buf880, buf881, 1920, 98, grid=grid(1920), stream=stream0)
        buf882 = buf851; del buf851  # reuse
        buf883 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf885 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf879, buf880, buf881, primals_501, primals_502, buf882, buf883, buf885, primals_501, primals_502, 480, 4, grid=grid(480), stream=stream0)
        del primals_501
        del primals_502
        buf886 = reinterpret_tensor(buf877, (8, 480, 7, 7), (23520, 1, 3360, 480), 0); del buf877  # reuse
        buf898 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        buf897 = reinterpret_tensor(buf898, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
        # Source Nodes: [cat_33, getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1, x1_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_156.run(buf878, buf882, buf883, primals_255, primals_256, buf886, buf897, 188160, grid=grid(188160), stream=stream0)
        del primals_256
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        buf887 = extern_kernels.convolution(buf886, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf887, (8, 480, 7, 7), (23520, 49, 7, 1))
        buf888 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_153.run(buf887, buf888, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del buf887
        buf889 = buf881; del buf881  # reuse
        buf890 = buf880; del buf880  # reuse
        buf891 = buf879; del buf879  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_154.run(buf888, buf889, buf890, buf891, 1920, 98, grid=grid(1920), stream=stream0)
        buf892 = buf883; del buf883  # reuse
        buf893 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf895 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_155.run(buf889, buf890, buf891, primals_504, primals_505, buf892, buf893, buf895, primals_504, primals_505, 480, 4, grid=grid(480), stream=stream0)
        del buf889
        del buf890
        del buf891
        del primals_504
        del primals_505
        buf896 = reinterpret_tensor(buf898, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
        buf944 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1, x2_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_157.run(buf888, buf892, buf893, primals_258, primals_259, buf896, buf944, 188160, grid=grid(188160), stream=stream0)
        del buf893
        del primals_259
        buf899 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cuda', dtype=torch.float32)
        buf900 = reinterpret_tensor(buf899, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf899  # reuse
        # Source Nodes: [x_se_24], Original ATen: [aten.mean]
        triton_per_fused_mean_159.run(buf900, buf898, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf901 = extern_kernels.convolution(buf900, primals_260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf901, (8, 240, 1, 1), (240, 1, 1, 1))
        buf902 = reinterpret_tensor(buf901, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf901  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_160.run(buf902, primals_261, 1920, grid=grid(1920), stream=stream0)
        del primals_261
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf903 = extern_kernels.convolution(buf902, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf903, (8, 960, 1, 1), (960, 1, 1, 1))
        buf904 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf943 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___se_gate, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_163.run(buf903, primals_263, buf904, buf943, 7680, grid=grid(7680), stream=stream0)
        del primals_263
        buf905 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten.mul]
        triton_poi_fused_mul_162.run(buf898, buf904, buf905, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        buf906 = extern_kernels.convolution(buf905, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf906, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf907 = reinterpret_tensor(buf867, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf867  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf906, buf907, 640, 49, grid=grid(640, 49), stream=stream0)
        buf908 = buf871; del buf871  # reuse
        buf909 = buf870; del buf870  # reuse
        buf910 = buf869; del buf869  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf907, buf908, buf909, buf910, 320, 98, grid=grid(320), stream=stream0)
        buf911 = buf873; del buf873  # reuse
        buf912 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf914 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf908, buf909, buf910, primals_507, primals_508, buf911, buf912, buf914, primals_507, primals_508, 80, 4, grid=grid(80), stream=stream0)
        del primals_507
        del primals_508
        buf915 = reinterpret_tensor(buf906, (8, 80, 7, 7), (3920, 1, 560, 80), 0); del buf906  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_144.run(buf907, buf911, buf912, primals_265, primals_266, buf915, 31360, grid=grid(31360), stream=stream0)
        del primals_266
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        buf916 = extern_kernels.convolution(buf915, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf916, (8, 80, 7, 7), (3920, 49, 7, 1))
        buf917 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_141.run(buf916, buf917, 640, 49, grid=grid(640, 49), stream=stream0)
        del buf916
        buf918 = buf910; del buf910  # reuse
        buf919 = buf909; del buf909  # reuse
        buf920 = buf908; del buf908  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_142.run(buf917, buf918, buf919, buf920, 320, 98, grid=grid(320), stream=stream0)
        buf921 = buf912; del buf912  # reuse
        buf922 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf924 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_143.run(buf918, buf919, buf920, primals_510, primals_511, buf921, buf922, buf924, primals_510, primals_511, 80, 4, grid=grid(80), stream=stream0)
        del buf918
        del buf919
        del buf920
        del primals_510
        del primals_511
        buf925 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_32, shortcut_16], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_158.run(buf915, buf917, buf921, buf922, primals_268, primals_269, buf876, buf925, 62720, grid=grid(62720), stream=stream0)
        del buf922
        del primals_269
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf926 = extern_kernels.convolution(buf925, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf926, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf927 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_164.run(buf926, buf927, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf928 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf929 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf930 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_165.run(buf927, buf928, buf929, buf930, 3840, 98, grid=grid(3840), stream=stream0)
        buf931 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf932 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf934 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_166.run(buf928, buf929, buf930, primals_275, primals_274, buf931, buf932, buf934, primals_275, primals_274, 960, 4, grid=grid(960), stream=stream0)
        del buf928
        del buf929
        del buf930
        del primals_274
        del primals_275
        buf935 = reinterpret_tensor(buf926, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf926  # reuse
        buf942 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_67, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_167.run(buf927, buf931, buf932, primals_1, primals_2, buf935, buf942, 376320, grid=grid(376320), stream=stream0)
        del buf932
        del primals_2
        buf936 = reinterpret_tensor(buf903, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf903  # reuse
        buf937 = reinterpret_tensor(buf936, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf936  # reuse
        # Source Nodes: [x_73], Original ATen: [aten.mean]
        triton_per_fused_mean_159.run(buf937, buf935, 7680, 49, grid=grid(7680), stream=stream0)
        del buf935
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf938 = extern_kernels.convolution(buf937, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf938, (8, 1280, 1, 1), (1280, 1, 1, 1))
        buf939 = empty((8, 1280), device='cuda', dtype=torch.float32)
        buf941 = empty_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.bool)
        # Source Nodes: [pred, x_76, x_77], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward, aten.view]
        triton_poi_fused_convolution_relu_threshold_backward_view_168.run(buf938, primals_272, buf939, buf941, 10240, grid=grid(10240), stream=stream0)
        del buf938
        del primals_272
        buf940 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_4, buf939, reinterpret_tensor(primals_3, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf940)
        del primals_4
        buf946 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_169.run(buf812, primals_235, buf946, 7680, grid=grid(7680), stream=stream0)
        del buf812
        del primals_235
        buf949 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_170.run(buf702, primals_201, buf949, 5376, grid=grid(5376), stream=stream0)
        del buf702
        del primals_201
        buf951 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_170.run(buf643, primals_182, buf951, 5376, grid=grid(5376), stream=stream0)
        del buf643
        del primals_182
        buf953 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_171.run(buf574, primals_160, buf953, 3840, grid=grid(3840), stream=stream0)
        del buf574
        del primals_160
        buf959 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_172.run(buf327, primals_87, buf959, 960, grid=grid(960), stream=stream0)
        del buf327
        del primals_87
        buf961 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_173.run(buf258, primals_65, buf961, 576, grid=grid(576), stream=stream0)
        del buf258
        del primals_65
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_278, primals_278, 1, grid=grid(1), stream=stream0)
        del primals_278
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_281, primals_281, 1, grid=grid(1), stream=stream0)
        del primals_281
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_284, primals_284, 1, grid=grid(1), stream=stream0)
        del primals_284
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_287, primals_287, 1, grid=grid(1), stream=stream0)
        del primals_287
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_290, primals_290, 1, grid=grid(1), stream=stream0)
        del primals_290
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_293, primals_293, 1, grid=grid(1), stream=stream0)
        del primals_293
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_296, primals_296, 1, grid=grid(1), stream=stream0)
        del primals_296
        # Source Nodes: [x_8], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_299, primals_299, 1, grid=grid(1), stream=stream0)
        del primals_299
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_302, primals_302, 1, grid=grid(1), stream=stream0)
        del primals_302
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_305, primals_305, 1, grid=grid(1), stream=stream0)
        del primals_305
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_308, primals_308, 1, grid=grid(1), stream=stream0)
        del primals_308
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_3], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_311, primals_311, 1, grid=grid(1), stream=stream0)
        del primals_311
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_314, primals_314, 1, grid=grid(1), stream=stream0)
        del primals_314
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_317, primals_317, 1, grid=grid(1), stream=stream0)
        del primals_317
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_320, primals_320, 1, grid=grid(1), stream=stream0)
        del primals_320
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_323, primals_323, 1, grid=grid(1), stream=stream0)
        del primals_323
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_326, primals_326, 1, grid=grid(1), stream=stream0)
        del primals_326
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_329, primals_329, 1, grid=grid(1), stream=stream0)
        del primals_329
        # Source Nodes: [x_16], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_332, primals_332, 1, grid=grid(1), stream=stream0)
        del primals_332
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_335, primals_335, 1, grid=grid(1), stream=stream0)
        del primals_335
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_338, primals_338, 1, grid=grid(1), stream=stream0)
        del primals_338
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_341, primals_341, 1, grid=grid(1), stream=stream0)
        del primals_341
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_3], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_344, primals_344, 1, grid=grid(1), stream=stream0)
        del primals_344
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_347, primals_347, 1, grid=grid(1), stream=stream0)
        del primals_347
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_350, primals_350, 1, grid=grid(1), stream=stream0)
        del primals_350
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_353, primals_353, 1, grid=grid(1), stream=stream0)
        del primals_353
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_356, primals_356, 1, grid=grid(1), stream=stream0)
        del primals_356
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_359, primals_359, 1, grid=grid(1), stream=stream0)
        del primals_359
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_362, primals_362, 1, grid=grid(1), stream=stream0)
        del primals_362
        # Source Nodes: [x_26], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_365, primals_365, 1, grid=grid(1), stream=stream0)
        del primals_365
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_368, primals_368, 1, grid=grid(1), stream=stream0)
        del primals_368
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_371, primals_371, 1, grid=grid(1), stream=stream0)
        del primals_371
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_374, primals_374, 1, grid=grid(1), stream=stream0)
        del primals_374
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_3], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_377, primals_377, 1, grid=grid(1), stream=stream0)
        del primals_377
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_380, primals_380, 1, grid=grid(1), stream=stream0)
        del primals_380
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_383, primals_383, 1, grid=grid(1), stream=stream0)
        del primals_383
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_386, primals_386, 1, grid=grid(1), stream=stream0)
        del primals_386
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_389, primals_389, 1, grid=grid(1), stream=stream0)
        del primals_389
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_392, primals_392, 1, grid=grid(1), stream=stream0)
        del primals_392
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_395, primals_395, 1, grid=grid(1), stream=stream0)
        del primals_395
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_398, primals_398, 1, grid=grid(1), stream=stream0)
        del primals_398
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_401, primals_401, 1, grid=grid(1), stream=stream0)
        del primals_401
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_404, primals_404, 1, grid=grid(1), stream=stream0)
        del primals_404
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_407, primals_407, 1, grid=grid(1), stream=stream0)
        del primals_407
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_410, primals_410, 1, grid=grid(1), stream=stream0)
        del primals_410
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_413, primals_413, 1, grid=grid(1), stream=stream0)
        del primals_413
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_416, primals_416, 1, grid=grid(1), stream=stream0)
        del primals_416
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_419, primals_419, 1, grid=grid(1), stream=stream0)
        del primals_419
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_422, primals_422, 1, grid=grid(1), stream=stream0)
        del primals_422
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_425, primals_425, 1, grid=grid(1), stream=stream0)
        del primals_425
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_428, primals_428, 1, grid=grid(1), stream=stream0)
        del primals_428
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_3], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_431, primals_431, 1, grid=grid(1), stream=stream0)
        del primals_431
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_434, primals_434, 1, grid=grid(1), stream=stream0)
        del primals_434
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_437, primals_437, 1, grid=grid(1), stream=stream0)
        del primals_437
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_440, primals_440, 1, grid=grid(1), stream=stream0)
        del primals_440
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_443, primals_443, 1, grid=grid(1), stream=stream0)
        del primals_443
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_446, primals_446, 1, grid=grid(1), stream=stream0)
        del primals_446
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_449, primals_449, 1, grid=grid(1), stream=stream0)
        del primals_449
        # Source Nodes: [x_48], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_452, primals_452, 1, grid=grid(1), stream=stream0)
        del primals_452
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_455, primals_455, 1, grid=grid(1), stream=stream0)
        del primals_455
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_458, primals_458, 1, grid=grid(1), stream=stream0)
        del primals_458
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_461, primals_461, 1, grid=grid(1), stream=stream0)
        del primals_461
        # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_3], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_464, primals_464, 1, grid=grid(1), stream=stream0)
        del primals_464
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_467, primals_467, 1, grid=grid(1), stream=stream0)
        del primals_467
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_470, primals_470, 1, grid=grid(1), stream=stream0)
        del primals_470
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_473, primals_473, 1, grid=grid(1), stream=stream0)
        del primals_473
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_476, primals_476, 1, grid=grid(1), stream=stream0)
        del primals_476
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_479, primals_479, 1, grid=grid(1), stream=stream0)
        del primals_479
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_482, primals_482, 1, grid=grid(1), stream=stream0)
        del primals_482
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_485, primals_485, 1, grid=grid(1), stream=stream0)
        del primals_485
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_488, primals_488, 1, grid=grid(1), stream=stream0)
        del primals_488
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_491, primals_491, 1, grid=grid(1), stream=stream0)
        del primals_491
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_494, primals_494, 1, grid=grid(1), stream=stream0)
        del primals_494
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_497, primals_497, 1, grid=grid(1), stream=stream0)
        del primals_497
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_500, primals_500, 1, grid=grid(1), stream=stream0)
        del primals_500
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_503, primals_503, 1, grid=grid(1), stream=stream0)
        del primals_503
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_506, primals_506, 1, grid=grid(1), stream=stream0)
        del primals_506
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_509, primals_509, 1, grid=grid(1), stream=stream0)
        del primals_509
        # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_1], Original ATen: [aten.add]
        triton_poi_fused_add_174.run(primals_512, primals_512, 1, grid=grid(1), stream=stream0)
        del primals_512
        return (buf940, primals_1, buf0, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_234, primals_236, primals_237, primals_239, primals_240, primals_242, primals_243, primals_245, primals_246, primals_248, primals_249, primals_251, primals_252, primals_254, primals_255, primals_257, primals_258, primals_260, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, buf1, buf3, buf13, buf14, buf16, buf26, buf27, buf29, buf39, buf42, buf44, buf54, buf55, buf57, buf67, buf68, buf70, buf80, buf81, buf83, buf93, buf96, buf98, buf108, buf109, buf111, buf121, buf122, buf124, buf134, buf136, buf146, buf147, buf149, buf159, buf160, buf162, buf172, buf173, buf175, buf185, buf188, buf190, buf200, buf201, buf203, buf213, buf214, buf216, buf226, buf227, buf229, buf239, buf242, buf244, buf251, buf252, buf255, buf257, buf259, buf260, buf262, buf269, buf270, buf272, buf279, buf281, buf288, buf289, buf291, buf298, buf299, buf301, buf308, buf309, buf311, buf318, buf321, buf324, buf326, buf328, buf329, buf331, buf338, buf339, buf341, buf348, buf349, buf351, buf358, buf359, buf361, buf368, buf371, buf373, buf380, buf381, buf383, buf390, buf391, buf393, buf400, buf402, buf409, buf410, buf412, buf419, buf420, buf422, buf429, buf430, buf432, buf439, buf442, buf444, buf451, buf452, buf454, buf461, buf462, buf464, buf471, buf472, buf474, buf481, buf484, buf486, buf493, buf494, buf496, buf503, buf504, buf506, buf513, buf514, buf516, buf523, buf526, buf528, buf535, buf536, buf538, buf545, buf546, buf548, buf555, buf556, buf558, buf565, buf568, buf571, buf573, buf575, buf576, buf578, buf585, buf586, buf588, buf595, buf597, buf604, buf605, buf607, buf614, buf615, buf617, buf624, buf625, buf627, buf634, buf637, buf640, buf642, buf644, buf645, buf647, buf654, buf655, buf657, buf664, buf665, buf667, buf674, buf675, buf677, buf684, buf687, buf689, buf696, buf697, buf699, buf701, buf703, buf704, buf706, buf713, buf714, buf716, buf723, buf725, buf732, buf733, buf735, buf742, buf743, buf745, buf752, buf753, buf755, buf762, buf765, buf767, buf774, buf775, buf777, buf784, buf785, buf787, buf794, buf795, buf797, buf804, buf807, buf809, buf811, buf813, buf814, buf816, buf823, buf824, buf826, buf833, buf834, buf836, buf843, buf844, buf846, buf853, buf856, buf858, buf865, buf866, buf868, buf875, buf876, buf878, buf885, buf886, buf888, buf895, buf898, buf900, buf902, buf904, buf905, buf907, buf914, buf915, buf917, buf924, buf925, buf927, buf934, buf937, buf939, reinterpret_tensor(primals_3, (1000, 1280), (1280, 1), 0), buf941, buf942, reinterpret_tensor(buf931, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf921, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf911, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf943, buf944, reinterpret_tensor(buf892, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf882, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf872, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf862, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf945, reinterpret_tensor(buf850, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf840, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf830, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf820, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf946, buf947, reinterpret_tensor(buf801, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf791, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf781, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf771, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf948, reinterpret_tensor(buf759, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf749, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf739, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf729, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf720, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf710, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf949, reinterpret_tensor(buf693, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf950, reinterpret_tensor(buf681, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf671, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf661, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf651, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf951, buf952, reinterpret_tensor(buf631, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf621, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf611, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf601, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf592, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf582, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf953, buf954, reinterpret_tensor(buf562, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf552, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf542, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf532, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf955, reinterpret_tensor(buf520, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf510, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf500, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf490, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf956, reinterpret_tensor(buf478, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf468, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf458, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf448, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf957, reinterpret_tensor(buf436, (1, 100, 1, 1), (100, 1, 1, 1), 0), reinterpret_tensor(buf426, (1, 100, 1, 1), (100, 1, 1, 1), 0), reinterpret_tensor(buf416, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf397, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf377, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf958, reinterpret_tensor(buf365, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf355, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf345, (1, 20, 1, 1), (20, 1, 1, 1), 0), reinterpret_tensor(buf335, (1, 20, 1, 1), (20, 1, 1, 1), 0), buf959, buf960, reinterpret_tensor(buf315, (1, 60, 1, 1), (60, 1, 1, 1), 0), reinterpret_tensor(buf305, (1, 60, 1, 1), (60, 1, 1, 1), 0), reinterpret_tensor(buf295, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf285, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf276, (1, 20, 1, 1), (20, 1, 1, 1), 0), reinterpret_tensor(buf266, (1, 20, 1, 1), (20, 1, 1, 1), 0), buf961, reinterpret_tensor(buf248, (1, 72, 1, 1), (72, 1, 1, 1), 0), buf962, reinterpret_tensor(buf236, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf223, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf210, (1, 12, 1, 1), (12, 1, 1, 1), 0), reinterpret_tensor(buf197, (1, 12, 1, 1), (12, 1, 1, 1), 0), buf963, reinterpret_tensor(buf182, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf169, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf156, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf143, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf131, (1, 12, 1, 1), (12, 1, 1, 1), 0), reinterpret_tensor(buf118, (1, 12, 1, 1), (12, 1, 1, 1), 0), reinterpret_tensor(buf105, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf964, reinterpret_tensor(buf90, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf77, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf64, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf51, (1, 8, 1, 1), (8, 1, 1, 1), 0), buf965, reinterpret_tensor(buf36, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf23, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((12, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((12, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((72, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((40, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((60, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((20, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((80, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((100, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((100, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((40, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((56, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((112, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((56, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((80, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((160, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_324 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_330 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_369 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_372 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_375 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_378 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_384 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_387 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_390 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_399 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_402 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_405 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_408 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_414 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_417 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_420 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_423 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_426 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_429 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_432 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_435 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_438 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_441 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_444 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_447 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_450 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ghostnet_100', benchmark_compiled_module)
