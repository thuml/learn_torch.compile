
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


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hmo7ocdhxrvyfqdptwp4atshykxixtry5dymlfcfep4uy7hwol.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# shortcut => add_5, clamp_max, clamp_min, div, mul_7
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_hardswish_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4ch72eed5dm2j2arwvlwp4m5lyeucil3trcuwcmerzao72zlau3.py
# Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_6 => add_10, add_7, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
# x_9 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrtnwo5d333ujdl527saox7n66g4z5ak3cgfnb55g5rbev7wiyh.py
# Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_1 => add_16
# x_12 => add_12, add_15, mul_15, mul_21, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_add_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybbgkyhbdgdxavsdxekdwyj4hlemtymyjvpk6ro4xhfgdbskqaf.py
# Source Nodes: [x_17], Original ATen: [aten.convolution]
# x_17 => convolution_3
triton_poi_fused_convolution_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgeprlj22hjmg6vmmsfmtfkwczkog5f7vt4v4wjvrgfiysnj3nua.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_18 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qo/cqoe6ltbju3zn6cenuz7pxoxxsbcvzoxqytagapd54jnwqavdb6x.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_18 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clofujfsjmmvn6jqud47c5ya6mks7agkgl6pp6uyqzzlrrmnxyaa.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_18 => add_18, add_19, add_20, mul_23, mul_24, mul_25, mul_26, mul_27, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxz2wiuk5smbt527yyyjio7mp3hf4klcobayprnyhtzlc77hsvhi.py
# Source Nodes: [x_18, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_18 => add_18, add_21, mul_22, mul_28, rsqrt_3, sub_3, var_mean_3
# x_21 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvqtc3q6epyyngqqo7e6lod3gkxqprsh7x5w4eokfs4cn3dexy6.py
# Source Nodes: [x_22], Original ATen: [aten.convolution]
# x_22 => convolution_4
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfinvbzpjunzzq3r5q5xsdqswma65uolj62bz3aizrwjupbabdq7.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lh/clh7k7oucmimddd5axglaavnjzzl5i35hkjhyzhkzjeqb653mluc.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (6272*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nz/cnzmsgf46qn22ujwewviuuqfolyiciijbe734jva7xi622gh5d4v.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => add_23, add_24, add_25, mul_30, mul_31, mul_32, mul_33, mul_34, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/ta/cta5gkt3stkym2mpclcnco7gwgaop5k7lckdcacyf5csnoffs4tj.py
# Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_23 => add_23, add_26, mul_29, mul_35, rsqrt_4, sub_4, var_mean_4
# x_26 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/rz/crzxh336sdvokvbw6llqye47rlohhg2wd3qayq7swgnc4w5x7dxb.py
# Source Nodes: [x_28], Original ATen: [aten.convolution]
# x_28 => convolution_5
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvxcqpc6zsvdpwrxmx5jkpikupw3xoz5iirws2rd7ag2qjpljho.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
# x_29 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ea/ceawb4o3ipg4b5rqfllu6g46karsymzcfeymemd4spgjt3sdqb4r.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
# x_29 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7d4me7ww3huk64kemwwgdaxiwmhrmpe723cpg4qii6uldotokq.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
# x_29 => add_28, add_29, add_30, mul_37, mul_38, mul_39, mul_40, mul_41, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/es/cesw7pyw2gzy6lvzdomcoxh4hdpkduaatycah4kv2ota6ytvpp2l.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
# x_29 => add_28, add_31, mul_36, mul_42, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hj/chjlogtkt2yadfcvhy7j4sq4n6o2hrvm7l6cx5y3ke23fholieeu.py
# Source Nodes: [x_33], Original ATen: [aten.convolution]
# x_33 => convolution_6
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


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7p7jbvwub4oseswk2pifoooa5zwoyfeyp7pnc7fsgncbdwycyq.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/or/corwmxt3ttctpz3t36fhficlb5bwkwzme35zspujmep6ekix2zfw.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => var_mean_6
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


# kernel path: /tmp/torchinductor_youkaichao/yl/cylngflwkgbncxudb24pjppwyywdmjfckm2jaqthkxhjaibcvunh.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => add_33, add_34, add_35, mul_44, mul_45, mul_46, mul_47, mul_48, rsqrt_6, squeeze_19, var_mean_6
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/oi/coii2hpqtjaae6nl755fbh5mebetvozcmq6mgix2eaet6ssvuey3.py
# Source Nodes: [x_34, x_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_34 => add_33, add_36, mul_43, mul_49, rsqrt_6, sub_6, var_mean_6
# x_37 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/uw/cuw2nf3a5n2a5uhsp44bhosastwfb5lwbppylwrdya2b4qcfowxa.py
# Source Nodes: [shortcut_3, x_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_47
# x_45 => add_43, add_46, mul_57, mul_63, rsqrt_8, sub_8, var_mean_8
triton_poi_fused__native_batch_norm_legit_functional_add_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zp/czprcrudtgs7jw46ly6mtp3g5bp2gk32morfgoh5n753ysdh3kit.py
# Source Nodes: [x_55], Original ATen: [aten.convolution]
# x_55 => convolution_10
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4e/c4ep5uwthvphstyya4ps7aknt7rohtxnmcrktapssmih4e2jjqse.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
# x_56 => var_mean_10
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5yyumf2rj7jpwkeb6zx37pfal7r6dxfgljdluyjxvfrv4htlyh.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
# x_56 => add_54, add_55, add_56, mul_72, mul_73, mul_74, mul_75, mul_76, rsqrt_10, squeeze_31, var_mean_10
triton_per_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ls/clsls33xwcwljd3gj55tcuak45nnsl5ergaqx2367l7acju5j7zh.py
# Source Nodes: [x_56, x_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_56 => add_54, add_57, mul_71, mul_77, rsqrt_10, sub_10, var_mean_10
# x_59 => relu_6
triton_poi_fused__native_batch_norm_legit_functional_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5ep3vmo7qcd32cvznhq3cltebi34i4bdx3raceisyms3zvtt2c.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_red_fused_mean_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_34', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcdfiyaxqbuvsyndvc7xaocgwxsie5vxqutsmxh5zawdgv4n4w4.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_per_fused_mean_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_35', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2ndvz4omogy4oeztcvwbaczokiee6it3y5fe5d2gtjceclawa5.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_11
# x_se_2 => relu_7
triton_poi_fused_convolution_relu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgi6n26yp2vdvd5o7g7bcrm75kxbtgmudfank33xoqfj63bukcjt.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => add_58, clamp_max_1, clamp_min_1, div_1
# x_se_3 => convolution_12
triton_poi_fused_convolution_hardsigmoid_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_37', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch65jbpsgyhfyg7gvom5vzo34s4bsqv7arshtr7ytx2zqr3k7ful.py
# Source Nodes: [x_60], Original ATen: [aten.mul]
# x_60 => mul_78
triton_poi_fused_mul_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_38', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jm/cjma2nesjsdpgn3g7m353qz6yh4qvbqda2y3ljelu24vqawbpibq.py
# Source Nodes: [x_61], Original ATen: [aten.convolution]
# x_61 => convolution_13
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4rovemrbosxbqvj54tverubudu5ntr6tfsuqvi4raj6ctywub2.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zn/cznevranqshzbns65anl7swloaqywipb6enjlsfed3yyyvrgwirb.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => add_60, add_61, add_62, mul_80, mul_81, mul_82, mul_83, mul_84, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/dn/cdn3skp472h6afjs6a62ifknonszezxsvf6nq74c62ko5uvsfkiv.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
# x_62 => add_60, add_63, mul_79, mul_85, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7s/c7slx65omenunsfrg6zvwtde7l2stvjzy3yfmez3jqhmk7ua3mta.py
# Source Nodes: [x_66], Original ATen: [aten.convolution]
# x_66 => convolution_14
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/66/c66hbabh3v2y7zasqpifmxhjyypywfrwfbkfao4xlgkgjnw3gf3g.py
# Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
# x_67 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yq3rivxpjmz3cpqbebcargxuwt42h6lucigef57cuypue45ngt.py
# Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
# x_67 => add_65, add_66, add_67, mul_87, mul_88, mul_89, mul_90, mul_91, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/qd/cqduikbqpytmuoqznq6dccs67q3yjqk5z7dka4cqc7unhrbqxsyq.py
# Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_67 => add_65, add_68, mul_86, mul_92, rsqrt_12, sub_12, var_mean_12
# x_70 => relu_8
triton_poi_fused__native_batch_norm_legit_functional_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_46', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35b3hatox7ld4oimq4z3h2wm4rf3oltax743gow5mlmfgpyvzdp.py
# Source Nodes: [x_se_4], Original ATen: [aten.mean]
# x_se_4 => mean_1
triton_red_fused_mean_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_47', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6k5h4f4jr56emxtmhdrvn2tntgzoabjdk5w7iquzd6a2xg4vc7.py
# Source Nodes: [x_se_4], Original ATen: [aten.mean]
# x_se_4 => mean_1
triton_per_fused_mean_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_48', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/ay/caympqrtrinhk2mkufilognvhz7n7t5idn3e3jbvi6k7cfqnadbk.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
# x_se_5 => convolution_16
# x_se_6 => relu_10
triton_poi_fused_convolution_relu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_49', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2ino6tannk6prljoyvxegbki7xieo42b6dj27v75tgmrn6dfgv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => add_74, clamp_max_2, clamp_min_2, div_2
# x_se_7 => convolution_17
triton_poi_fused_convolution_hardsigmoid_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_50', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5x/c5x5nqx3544evv62mz33eyx6igfkzpnto2osaucibzdygsrtm776.py
# Source Nodes: [x_76], Original ATen: [aten.mul]
# x_76 => mul_100
triton_poi_fused_mul_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_51', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqcxejrmh43xozv4yqhyj2sqdlwcmn7f6lwmrls7xscixlffvhy.py
# Source Nodes: [shortcut_5, x_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_5 => add_80
# x_78 => add_76, add_79, mul_101, mul_107, rsqrt_14, sub_14, var_mean_14
triton_poi_fused__native_batch_norm_legit_functional_add_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_52', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ol/col2pbq34hnlghmx23jdd6iiyfogo55tg27m2wnjw4esewreafvo.py
# Source Nodes: [x_100], Original ATen: [aten.convolution]
# x_100 => convolution_24
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


# kernel path: /tmp/torchinductor_youkaichao/3c/c3csfuq57lturyo2qj4qb6ypdahpspeniaingllugjwcplvxanyy.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
# x_101 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xu/cxuwiddwpzshhqspyqhc6liuurbjko6iyjhjvkascej64gmihflq.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
# x_101 => add_100, add_101, add_99, mul_131, mul_132, mul_133, mul_134, mul_135, rsqrt_18, squeeze_55, var_mean_18
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


# kernel path: /tmp/torchinductor_youkaichao/3b/c3b4mpd3v6mcmknautbc3f2qfsgkx65pc56tzcbblc6qdf62hu65.py
# Source Nodes: [x_101, x_104], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_101 => add_102, add_99, mul_130, mul_136, rsqrt_18, sub_18, var_mean_18
# x_104 => add_103, clamp_max_4, clamp_min_4, div_4, mul_137
triton_poi_fused__native_batch_norm_legit_functional_hardswish_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_56', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknbdz66fzstzwlp4folxwb5b2wnlkgspjzxpldnknqn6v766ezb.py
# Source Nodes: [x_105], Original ATen: [aten.convolution]
# x_105 => convolution_25
triton_poi_fused_convolution_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_57', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cnckdnmotu4viov2fcaws4k2sdr2h355dpf7was47sympv2dxvzn.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => var_mean_19
triton_red_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zm/czmri5noxfhyg2473pue5sev27tvvrjvwlorf5ks5sdkxxxzwyox.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => add_105, add_106, add_107, mul_139, mul_140, mul_141, mul_142, mul_143, rsqrt_19, squeeze_58, var_mean_19
triton_per_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/7k/c7k56ln455xzbi5wfbh7ofw5xrsh7vdcvjygjcqidfuiwju62aje.py
# Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_106 => add_105, add_108, mul_138, mul_144, rsqrt_19, sub_19, var_mean_19
# x_109 => add_109, clamp_max_5, clamp_min_5, div_5, mul_145
triton_poi_fused__native_batch_norm_legit_functional_hardswish_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_60', 'mutated_arg_names': []},
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufl5dxzh5avtl7mcuuererthbfu4og53me5tx3mhcvdbbmzbaal.py
# Source Nodes: [x_111], Original ATen: [aten.convolution]
# x_111 => convolution_26
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xy/cxy4hwz5nsajapienfmqp7j4sy3rrkg6ed7mbvv5lw5jrng6ynyy.py
# Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
# x_112 => var_mean_20
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jy/cjyayy7uvij7irtbewf2qs3i27gth46pzrr24mjodqgi55rqesf3.py
# Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
# x_112 => add_111, add_112, add_113, mul_147, mul_148, mul_149, mul_150, mul_151, rsqrt_20, squeeze_61, var_mean_20
triton_per_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/wv/cwveezgh55igiuutkgzir3mb7575wvg262aq6qh26dhgqxara4p6.py
# Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
# x_112 => add_111, add_114, mul_146, mul_152, rsqrt_20, sub_20, var_mean_20
triton_poi_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/o3/co3paayjvk5flwziqxff3alwicqdh4vbwawdcufrydlpcnr6boeq.py
# Source Nodes: [x_116], Original ATen: [aten.convolution]
# x_116 => convolution_27
triton_poi_fused_convolution_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1600
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 200
    y1 = (yindex // 200)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (200*x2) + (39200*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywjoveojgmevrrgt4acol2wwg5pmh3uaaxug4kbd7lqbimq6pdh.py
# Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
# x_117 => var_mean_21
triton_red_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2600
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 200)
    x0 = xindex % 200
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
        tmp3 = tl.load(in_ptr0 + (x0 + (200*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2w/c2wv5tgnqemx3gb5cn542nvhxcsrocsu2g4se46uks23pv5vkazv.py
# Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
# x_117 => add_116, add_117, add_118, mul_154, mul_155, mul_156, mul_157, mul_158, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_67', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (200*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wj/cwj2lnwnayxyhzulfqjh344552x6wlhlbj23d5afx7bxv5v2seyb.py
# Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_117 => add_116, add_119, mul_153, mul_159, rsqrt_21, sub_21, var_mean_21
# x_120 => add_120, clamp_max_6, clamp_min_6, div_6, mul_160
triton_poi_fused__native_batch_norm_legit_functional_hardswish_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fjjepewkfevopsizifqxuzvv47ibfziuyonr3ht4b2lumiuvxy.py
# Source Nodes: [shortcut_8, x_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_8 => add_132
# x_128 => add_128, add_131, mul_169, mul_175, rsqrt_23, sub_23, var_mean_23
triton_poi_fused__native_batch_norm_legit_functional_add_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_69', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/am/camiv2biu5aom2e4jqiy74rdzbqqaigrhr6kd55zccsxuhcyioxc.py
# Source Nodes: [x_133], Original ATen: [aten.convolution]
# x_133 => convolution_30
triton_poi_fused_convolution_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (184*x2) + (36064*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdgymo7x3t3dhyz7sv7dvyuqngctwpebrm6bac2msaejawbwd36.py
# Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
# x_134 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2392
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 184)
    x0 = xindex % 184
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
        tmp3 = tl.load(in_ptr0 + (x0 + (184*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zg/czgfi73mwa2y3kkwticdoqylwkicidfeb4r6fl3ucsoxqh64hnko.py
# Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
# x_134 => add_134, add_135, add_136, mul_177, mul_178, mul_179, mul_180, mul_181, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_72', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (184*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (184*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (184*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bc/cbcvatqcjcg67k73u2lbjhe6opoll3vjx72zmqnebr6mitl3g4gh.py
# Source Nodes: [x_134, x_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_134 => add_134, add_137, mul_176, mul_182, rsqrt_24, sub_24, var_mean_24
# x_137 => add_138, clamp_max_8, clamp_min_8, div_8, mul_183
triton_poi_fused__native_batch_norm_legit_functional_hardswish_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 184
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7ohuza4azx27ucexieop6gqc3pi5pcdmozz3azghoyjrpgywor.py
# Source Nodes: [x_167], Original ATen: [aten.convolution]
# x_167 => convolution_36
triton_poi_fused_convolution_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jk/cjkm6koss3et36drpcsatqj222j2bux3vu47limkzy67nx3qe2fm.py
# Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
# x_168 => var_mean_30
triton_red_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/34/c345ysuood7hzi7pabecjiovjdox4ddmo6xcx7vsnncp5fxltljz.py
# Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
# x_168 => add_170, add_171, add_172, mul_223, mul_224, mul_225, mul_226, mul_227, rsqrt_30, squeeze_91, var_mean_30
triton_per_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/rb/crbm57ylkzfkbg47dsjo6slfkysram2aw5dfn5oqx2hdy2dv6pts.py
# Source Nodes: [x_168, x_171], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_168 => add_170, add_173, mul_222, mul_228, rsqrt_30, sub_30, var_mean_30
# x_171 => add_174, clamp_max_12, clamp_min_12, div_12, mul_229
triton_poi_fused__native_batch_norm_legit_functional_hardswish_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_77', 'mutated_arg_names': []},
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogp37neauk4v72cg5gzwwrgs7it3qdlbeconmo6ehp7icelkj7f.py
# Source Nodes: [x_se_12], Original ATen: [aten.mean]
# x_se_12 => mean_3
triton_red_fused_mean_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_78', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2kgnasijr2gbd7gcsf4squajjfj64am4fq3eipve5647kaexfr.py
# Source Nodes: [x_se_12], Original ATen: [aten.mean]
# x_se_12 => mean_3
triton_per_fused_mean_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_79', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcnrsp2ouw4albyoxg277argl7dddas5olfkmfy5iiyy2dxd34a.py
# Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
# x_se_13 => convolution_38
# x_se_14 => relu_14
triton_poi_fused_convolution_relu_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_80', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvlii43u2qfhof5niynvn4eik2dstokz4exe6m4mqepxrlxopf5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => add_181, clamp_max_14, clamp_min_14, div_14
# x_se_15 => convolution_39
triton_poi_fused_convolution_hardsigmoid_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_81', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/il/cilyg2hl5e4r36bponhdfrsavvsaectk5cy3zitxvk5blbj4qjoo.py
# Source Nodes: [x_177], Original ATen: [aten.mul]
# x_177 => mul_238
triton_poi_fused_mul_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_82', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgirndwbyfj3ar46gpvvvzkjzhyvxe2qde3kp6i74qy7umwzb66.py
# Source Nodes: [x_178], Original ATen: [aten.convolution]
# x_178 => convolution_40
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcgchicil35afujiqze4lh3t6ksm5mspozdta3q6snp62bw2t7e.py
# Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
# x_179 => var_mean_32
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs36s7oughs6ze7fvzegy7idj2vgjiwidjiqhd4jofgqhu5ka4xn.py
# Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
# x_179 => add_183, add_184, add_185, mul_240, mul_241, mul_242, mul_243, mul_244, rsqrt_32, squeeze_97, var_mean_32
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_85', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/aw/cawi6k6p7rnewpiqknin4scktvgnefhnkxfmwaqdcjhzp2twrfjw.py
# Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
# x_179 => add_183, add_186, mul_239, mul_245, rsqrt_32, sub_32, var_mean_32
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33aehev6dssuxqxyvnt5xwgpbso2ye6p7baxfxsregpa4z5rnvk.py
# Source Nodes: [x_183], Original ATen: [aten.convolution]
# x_183 => convolution_41
triton_poi_fused_convolution_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_87', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/op/copfa5kp6nnl7wkafzkgshddgoacttwrxi57u3mgtvmtc7xdclu3.py
# Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
# x_184 => var_mean_33
triton_red_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5udma2ckebbclj74urcm73tns3kgxax7dubak2t7p3lsjbc5kj.py
# Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
# x_184 => add_188, add_189, add_190, mul_247, mul_248, mul_249, mul_250, mul_251, rsqrt_33, squeeze_100, var_mean_33
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3rtwsn56nbqk2s24sxfmu5udlov2assmjaa2if35qtvkm556vv.py
# Source Nodes: [x_184, x_187], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_184 => add_188, add_191, mul_246, mul_252, rsqrt_33, sub_33, var_mean_33
# x_187 => add_192, clamp_max_15, clamp_min_15, div_15, mul_253
triton_poi_fused__native_batch_norm_legit_functional_hardswish_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_90', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/clebk2igw6zt4g2cpieuhptqrngumpwrepakk57rvbl7iy7qumht.py
# Source Nodes: [x_se_16], Original ATen: [aten.mean]
# x_se_16 => mean_4
triton_red_fused_mean_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_91', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6wpphvhvmdwhxlu7fgerdab4emiurwxe7bhuowqhmzr5zouhoh.py
# Source Nodes: [x_se_16], Original ATen: [aten.mean]
# x_se_16 => mean_4
triton_per_fused_mean_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_92', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/da/cda7z4ac4vyiwtklrffbmpqouag6yyt2hvcyvkln7r4d74gct4je.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
# x_se_17 => convolution_43
# x_se_18 => relu_15
triton_poi_fused_convolution_relu_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_93', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/an/canwrqmj7lpx7do4rqvaqgwda4is6gkva5vqfh22o77xw4ll3ujb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => add_199, clamp_max_17, clamp_min_17, div_17
# x_se_19 => convolution_44
triton_poi_fused_convolution_hardsigmoid_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_94', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcviwamxxcjhrrgckw6so52vkadqu53zwi6v4ye77ba3y2yxbjf.py
# Source Nodes: [x_193], Original ATen: [aten.mul]
# x_193 => mul_262
triton_poi_fused_mul_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_95', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ie/cienqll2763c2nvbymj4i3syfebbzslf5qf5vl5cmfdqn2eibdin.py
# Source Nodes: [shortcut_12, x_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_12 => add_205
# x_195 => add_201, add_204, mul_263, mul_269, rsqrt_35, sub_35, var_mean_35
triton_poi_fused__native_batch_norm_legit_functional_add_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_96', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3e2b5mwxsq2sd5tpqtbxvaio3dpa2l7nz2nm43ubgy3w5nkfhk.py
# Source Nodes: [x_205], Original ATen: [aten.convolution]
# x_205 => convolution_47
triton_poi_fused_convolution_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_97', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vp/cvp56liwlyzufkb3oy2em3roee4nrp2cjvj5hgja7dv25tluzlkp.py
# Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
# x_206 => var_mean_37
triton_red_fused__native_batch_norm_legit_functional_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhydfwab5xcpyvnkhq4nimortmfnf43f7momll3wmj252reehsn.py
# Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
# x_206 => add_213, add_214, add_215, mul_279, mul_280, mul_281, mul_282, mul_283, rsqrt_37, squeeze_112, var_mean_37
triton_per_fused__native_batch_norm_legit_functional_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_99', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfbtedhce7dmzq5phtors4mkw52hb4o5bq6rwxrqxkddf32vcn2.py
# Source Nodes: [x_206, x_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_206 => add_213, add_216, mul_278, mul_284, rsqrt_37, sub_37, var_mean_37
# x_209 => add_217, clamp_max_19, clamp_min_19, div_19, mul_285
triton_poi_fused__native_batch_norm_legit_functional_hardswish_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwrssauh3l5w57x5eq73wngpk4hhijaorfiipdxzc2atsyzr3an.py
# Source Nodes: [x_se_20], Original ATen: [aten.mean]
# x_se_20 => mean_5
triton_per_fused_mean_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_101', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuss7iceovmonvi6egj42dggjdt34nd3573z5kp7yrknfl7lhrx.py
# Source Nodes: [x_210], Original ATen: [aten.mul]
# x_210 => mul_286
triton_poi_fused_mul_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_102', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr7a6g75xbjiuhq56c2z7rpa4ye37sq3ycmmqtpht5aq5xpilx7.py
# Source Nodes: [x_211], Original ATen: [aten.convolution]
# x_211 => convolution_50
triton_poi_fused_convolution_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_103', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/te/ctenqy4gw3rieimaecen3yt3unl5y4nabmxx2qiwj3mzqkq2zqsd.py
# Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
# x_212 => var_mean_38
triton_red_fused__native_batch_norm_legit_functional_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_104', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgdikgpwilc2ooryivi2iwu56aqmnew6tnnnc64ztbbkejze2ht.py
# Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
# x_212 => add_220, add_221, add_222, mul_288, mul_289, mul_290, mul_291, mul_292, rsqrt_38, squeeze_115, var_mean_38
triton_per_fused__native_batch_norm_legit_functional_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_105', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwmceoe6xpne2i7sq6ubovxw5c3vfztchokwfiwcv2o4rojmddg.py
# Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
# x_212 => add_220, add_223, mul_287, mul_293, rsqrt_38, sub_38, var_mean_38
triton_poi_fused__native_batch_norm_legit_functional_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_106', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ua/cuacclud346mzutbidldawjciuwh4nkx5qt73sbbtiz2gdgv2df7.py
# Source Nodes: [x_216], Original ATen: [aten.convolution]
# x_216 => convolution_51
triton_poi_fused_convolution_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_107', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qe/cqeynicnbbv6c22szni6qar6jcejj7oc6vsfvfvcebizm376uarg.py
# Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
# x_217 => var_mean_39
triton_red_fused__native_batch_norm_legit_functional_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_108', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcr5auonb26olp4mc7jjzydifmvug4b3jqafb6c4lfsllfjgqzv.py
# Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
# x_217 => add_225, add_226, add_227, mul_295, mul_296, mul_297, mul_298, mul_299, rsqrt_39, squeeze_118, var_mean_39
triton_per_fused__native_batch_norm_legit_functional_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_109', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/nr/cnra43mtpcy67i6iv5o7jmi5b3iumeocajanewbdnshsu7g26ijt.py
# Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_217 => add_225, add_228, mul_294, mul_300, rsqrt_39, sub_39, var_mean_39
# x_220 => add_229, clamp_max_21, clamp_min_21, div_21, mul_301
triton_poi_fused__native_batch_norm_legit_functional_hardswish_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_110', 'mutated_arg_names': []},
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
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sd/csdeb6jjyqbybpebqc6bl4qbi2vbfpg3mhgdaonf6e2553fbovhk.py
# Source Nodes: [x_se_24], Original ATen: [aten.mean]
# x_se_24 => mean_6
triton_per_fused_mean_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_111', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rl4jqmdqfcejejd3riivivtaycpjzkovq72swq6f6ibddy7y4l.py
# Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
# x_se_25 => convolution_53
# x_se_26 => relu_17
triton_poi_fused_convolution_relu_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_112', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdu44je4tgm2gfyr6uo67eb2hsac7z7y7gnjl4jqsb4cgnferu6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => add_236, clamp_max_23, clamp_min_23, div_23
# x_se_27 => convolution_54
triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_113', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/o3/co3f2erkxvh4rywe6zkuobxyaoldlikr6e6qynol7ocz35yt7szj.py
# Source Nodes: [x_226], Original ATen: [aten.mul]
# x_226 => mul_310
triton_poi_fused_mul_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_114', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wc/cwchgw2bgw5ft44wskxajfdbypflyyvifcojytk7gptyledndugt.py
# Source Nodes: [shortcut_14, x_228], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_14 => add_242
# x_228 => add_238, add_241, mul_311, mul_317, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_add_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_115', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xvv7iwus264rh2ytmkvqbxa2uwvrgwz3gmxmzxyaivf6moxvbc.py
# Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
# x_251 => add_263, add_266, mul_342, mul_348, rsqrt_45, sub_45, var_mean_45
triton_poi_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/34/c34gph5wemwd775dhfa2x4rwhmsg3efhmcydn6ls2wde75db2u2i.py
# Source Nodes: [x_256, x_257], Original ATen: [aten.hardswish, aten.mean]
# x_256 => add_267, clamp_max_27, clamp_min_27, div_27, mul_349
# x_257 => mean_8
triton_per_fused_hardswish_mean_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_117', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtyo3n2ssija2jx5sckkch65hkqenb4o6epcssrsihgruq2rhg7.py
# Source Nodes: [pred, x_260, x_261], Original ATen: [aten.convolution, aten.hardswish, aten.view]
# pred => view_1
# x_260 => convolution_62
# x_261 => add_268, clamp_max_28, clamp_min_28, div_28, mul_350
triton_poi_fused_convolution_hardswish_view_118 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_view_118', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ka/cka4piqxdjh2brajurpbwxp4yvibzmrkxujlpiqsrynnthiu2slo.py
# Source Nodes: [x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_23 => convolution_49
triton_poi_fused_convolution_hardsigmoid_backward_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_119', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cc/cccm3xowtjkpjdvdw3a22z4hv26otn7tuhw6mroubn2slb7k6g67.py
# Source Nodes: [x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_15 => convolution_39
triton_poi_fused_convolution_hardsigmoid_backward_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_120', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3c/c3crtsyhv5dz6rwma2fdklxiqcbbabqikhvrod7dfuss7ktydbmk.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_11 => convolution_22
triton_poi_fused_convolution_hardsigmoid_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_121', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sp/cspg3sqsvjuvpmhkzfhnpuzrtmgj7rjknvsxw4x4pdyylcvw6ljl.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_3 => convolution_12
triton_poi_fused_convolution_hardsigmoid_backward_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_122', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4xpg3mpfhn77hfeq4c3qs5rxh6hi4yrcpqwo3rh42hkgfnw5e4.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_123', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
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
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_24, (40, ), (1, ))
    assert_size_stride(primals_25, (120, ), (1, ))
    assert_size_stride(primals_26, (120, ), (1, ))
    assert_size_stride(primals_27, (120, ), (1, ))
    assert_size_stride(primals_28, (120, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_30, (40, ), (1, ))
    assert_size_stride(primals_31, (120, ), (1, ))
    assert_size_stride(primals_32, (120, ), (1, ))
    assert_size_stride(primals_33, (120, ), (1, ))
    assert_size_stride(primals_34, (120, ), (1, ))
    assert_size_stride(primals_35, (40, ), (1, ))
    assert_size_stride(primals_36, (40, ), (1, ))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_38, (240, ), (1, ))
    assert_size_stride(primals_39, (240, ), (1, ))
    assert_size_stride(primals_40, (240, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_42, (80, ), (1, ))
    assert_size_stride(primals_43, (200, ), (1, ))
    assert_size_stride(primals_44, (200, ), (1, ))
    assert_size_stride(primals_45, (200, ), (1, ))
    assert_size_stride(primals_46, (200, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_48, (80, ), (1, ))
    assert_size_stride(primals_49, (184, ), (1, ))
    assert_size_stride(primals_50, (184, ), (1, ))
    assert_size_stride(primals_51, (184, ), (1, ))
    assert_size_stride(primals_52, (184, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_54, (80, ), (1, ))
    assert_size_stride(primals_55, (184, ), (1, ))
    assert_size_stride(primals_56, (184, ), (1, ))
    assert_size_stride(primals_57, (184, ), (1, ))
    assert_size_stride(primals_58, (184, ), (1, ))
    assert_size_stride(primals_59, (80, ), (1, ))
    assert_size_stride(primals_60, (80, ), (1, ))
    assert_size_stride(primals_61, (480, ), (1, ))
    assert_size_stride(primals_62, (480, ), (1, ))
    assert_size_stride(primals_63, (480, ), (1, ))
    assert_size_stride(primals_64, (480, ), (1, ))
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
    assert_size_stride(primals_77, (160, ), (1, ))
    assert_size_stride(primals_78, (160, ), (1, ))
    assert_size_stride(primals_79, (960, ), (1, ))
    assert_size_stride(primals_80, (960, ), (1, ))
    assert_size_stride(primals_81, (960, ), (1, ))
    assert_size_stride(primals_82, (960, ), (1, ))
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
    assert_size_stride(primals_93, (1000, 1280), (1280, 1))
    assert_size_stride(primals_94, (1000, ), (1, ))
    assert_size_stride(primals_95, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_96, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_97, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_98, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_99, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_101, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_102, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_104, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_105, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_107, (24, ), (1, ))
    assert_size_stride(primals_108, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_109, (72, ), (1, ))
    assert_size_stride(primals_110, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_111, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_112, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_114, (32, ), (1, ))
    assert_size_stride(primals_115, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_116, (120, ), (1, ))
    assert_size_stride(primals_117, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_118, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_119, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_120, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_123, (120, ), (1, ))
    assert_size_stride(primals_124, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_125, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_126, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_127, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_128, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_129, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_131, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_132, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_135, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_137, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_138, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_139, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_140, (120, ), (1, ))
    assert_size_stride(primals_141, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_142, (480, ), (1, ))
    assert_size_stride(primals_143, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_144, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_145, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_147, (168, ), (1, ))
    assert_size_stride(primals_148, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_149, (672, ), (1, ))
    assert_size_stride(primals_150, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_151, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_152, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_154, (168, ), (1, ))
    assert_size_stride(primals_155, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_156, (672, ), (1, ))
    assert_size_stride(primals_157, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_158, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_159, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_160, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_161, (240, ), (1, ))
    assert_size_stride(primals_162, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_163, (960, ), (1, ))
    assert_size_stride(primals_164, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_165, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_166, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_167, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_168, (240, ), (1, ))
    assert_size_stride(primals_169, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_170, (960, ), (1, ))
    assert_size_stride(primals_171, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_172, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_174, (1280, ), (1, ))
    assert_size_stride(primals_175, (), ())
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_177, (16, ), (1, ))
    assert_size_stride(primals_178, (), ())
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_180, (16, ), (1, ))
    assert_size_stride(primals_181, (), ())
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_183, (16, ), (1, ))
    assert_size_stride(primals_184, (), ())
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (), ())
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (), ())
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (), ())
    assert_size_stride(primals_194, (72, ), (1, ))
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (), ())
    assert_size_stride(primals_197, (72, ), (1, ))
    assert_size_stride(primals_198, (72, ), (1, ))
    assert_size_stride(primals_199, (), ())
    assert_size_stride(primals_200, (24, ), (1, ))
    assert_size_stride(primals_201, (24, ), (1, ))
    assert_size_stride(primals_202, (), ())
    assert_size_stride(primals_203, (72, ), (1, ))
    assert_size_stride(primals_204, (72, ), (1, ))
    assert_size_stride(primals_205, (), ())
    assert_size_stride(primals_206, (72, ), (1, ))
    assert_size_stride(primals_207, (72, ), (1, ))
    assert_size_stride(primals_208, (), ())
    assert_size_stride(primals_209, (40, ), (1, ))
    assert_size_stride(primals_210, (40, ), (1, ))
    assert_size_stride(primals_211, (), ())
    assert_size_stride(primals_212, (120, ), (1, ))
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (), ())
    assert_size_stride(primals_215, (120, ), (1, ))
    assert_size_stride(primals_216, (120, ), (1, ))
    assert_size_stride(primals_217, (), ())
    assert_size_stride(primals_218, (40, ), (1, ))
    assert_size_stride(primals_219, (40, ), (1, ))
    assert_size_stride(primals_220, (), ())
    assert_size_stride(primals_221, (120, ), (1, ))
    assert_size_stride(primals_222, (120, ), (1, ))
    assert_size_stride(primals_223, (), ())
    assert_size_stride(primals_224, (120, ), (1, ))
    assert_size_stride(primals_225, (120, ), (1, ))
    assert_size_stride(primals_226, (), ())
    assert_size_stride(primals_227, (40, ), (1, ))
    assert_size_stride(primals_228, (40, ), (1, ))
    assert_size_stride(primals_229, (), ())
    assert_size_stride(primals_230, (240, ), (1, ))
    assert_size_stride(primals_231, (240, ), (1, ))
    assert_size_stride(primals_232, (), ())
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_234, (240, ), (1, ))
    assert_size_stride(primals_235, (), ())
    assert_size_stride(primals_236, (80, ), (1, ))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (), ())
    assert_size_stride(primals_239, (200, ), (1, ))
    assert_size_stride(primals_240, (200, ), (1, ))
    assert_size_stride(primals_241, (), ())
    assert_size_stride(primals_242, (200, ), (1, ))
    assert_size_stride(primals_243, (200, ), (1, ))
    assert_size_stride(primals_244, (), ())
    assert_size_stride(primals_245, (80, ), (1, ))
    assert_size_stride(primals_246, (80, ), (1, ))
    assert_size_stride(primals_247, (), ())
    assert_size_stride(primals_248, (184, ), (1, ))
    assert_size_stride(primals_249, (184, ), (1, ))
    assert_size_stride(primals_250, (), ())
    assert_size_stride(primals_251, (184, ), (1, ))
    assert_size_stride(primals_252, (184, ), (1, ))
    assert_size_stride(primals_253, (), ())
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_255, (80, ), (1, ))
    assert_size_stride(primals_256, (), ())
    assert_size_stride(primals_257, (184, ), (1, ))
    assert_size_stride(primals_258, (184, ), (1, ))
    assert_size_stride(primals_259, (), ())
    assert_size_stride(primals_260, (184, ), (1, ))
    assert_size_stride(primals_261, (184, ), (1, ))
    assert_size_stride(primals_262, (), ())
    assert_size_stride(primals_263, (80, ), (1, ))
    assert_size_stride(primals_264, (80, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (480, ), (1, ))
    assert_size_stride(primals_267, (480, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_270, (480, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_273, (112, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_276, (672, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (672, ), (1, ))
    assert_size_stride(primals_279, (672, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (112, ), (1, ))
    assert_size_stride(primals_282, (112, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (672, ), (1, ))
    assert_size_stride(primals_285, (672, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (672, ), (1, ))
    assert_size_stride(primals_288, (672, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (160, ), (1, ))
    assert_size_stride(primals_291, (160, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (960, ), (1, ))
    assert_size_stride(primals_294, (960, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_297, (960, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_300, (160, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (960, ), (1, ))
    assert_size_stride(primals_303, (960, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_306, (960, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_309, (160, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (960, ), (1, ))
    assert_size_stride(primals_312, (960, ), (1, ))
    assert_size_stride(primals_313, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_95, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_95
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_313, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_313
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
        buf10 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf13 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_176, primals_177, buf10, buf11, buf13, primals_176, primals_177, 16, 7, grid=grid(16), stream=stream0)
        del primals_176
        del primals_177
        buf14 = reinterpret_tensor(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        buf15 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_6.run(buf3, buf10, buf11, primals_1, primals_2, buf14, buf15, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf16, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf17 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf16, buf17, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf18 = buf6; del buf6  # reuse
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf17, buf18, buf19, buf20, 12544, 128, grid=grid(12544), stream=stream0)
        buf21 = buf9; del buf9  # reuse
        buf22 = buf8; del buf8  # reuse
        buf23 = buf7; del buf7  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf18, buf19, buf20, buf21, buf22, buf23, 112, 112, grid=grid(112), stream=stream0)
        buf24 = buf11; del buf11  # reuse
        buf25 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf27 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf21, buf22, buf23, primals_179, primals_180, buf24, buf25, buf27, primals_179, primals_180, 16, 7, grid=grid(16), stream=stream0)
        del primals_179
        del primals_180
        buf28 = reinterpret_tensor(buf16, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf16  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_7.run(buf17, buf24, buf25, primals_3, primals_4, buf28, 1605632, grid=grid(1605632), stream=stream0)
        del primals_4
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf30 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf29, buf30, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf31 = buf20; del buf20  # reuse
        buf32 = buf19; del buf19  # reuse
        buf33 = buf18; del buf18  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf30, buf31, buf32, buf33, 12544, 128, grid=grid(12544), stream=stream0)
        buf34 = buf23; del buf23  # reuse
        buf35 = buf22; del buf22  # reuse
        buf36 = buf21; del buf21  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf31, buf32, buf33, buf34, buf35, buf36, 112, 112, grid=grid(112), stream=stream0)
        buf37 = buf25; del buf25  # reuse
        buf38 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf40 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf34, buf35, buf36, primals_182, primals_183, buf37, buf38, buf40, primals_182, primals_183, 16, 7, grid=grid(16), stream=stream0)
        del primals_182
        del primals_183
        buf41 = reinterpret_tensor(buf29, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf29  # reuse
        # Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_8.run(buf30, buf37, buf38, primals_5, primals_6, buf15, buf41, 1605632, grid=grid(1605632), stream=stream0)
        del buf38
        del primals_6
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf43 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf42, buf43, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf44 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf43, buf44, buf45, buf46, 50176, 128, grid=grid(50176), stream=stream0)
        buf47 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf44, buf45, buf46, buf47, buf48, buf49, 448, 112, grid=grid(448), stream=stream0)
        del buf44
        del buf45
        del buf46
        buf50 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf53 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf47, buf48, buf49, primals_185, primals_186, buf50, buf51, buf53, primals_185, primals_186, 64, 7, grid=grid(64), stream=stream0)
        del buf47
        del buf48
        del buf49
        del primals_185
        del primals_186
        buf54 = reinterpret_tensor(buf42, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf42  # reuse
        # Source Nodes: [x_18, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_13.run(buf43, buf50, buf51, primals_7, primals_8, buf54, 6422528, grid=grid(6422528), stream=stream0)
        del primals_8
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_99, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf55, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf56 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf55, buf56, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf57 = reinterpret_tensor(buf33, (1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), 0); del buf33  # reuse
        buf58 = reinterpret_tensor(buf32, (1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), 0); del buf32  # reuse
        buf59 = reinterpret_tensor(buf31, (1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), 0); del buf31  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf56, buf57, buf58, buf59, 12544, 128, grid=grid(12544), stream=stream0)
        buf60 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf57, buf58, buf59, buf60, buf61, buf62, 128, 98, grid=grid(128), stream=stream0)
        del buf57
        del buf58
        del buf59
        buf63 = buf51; del buf51  # reuse
        buf64 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf66 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_17.run(buf60, buf61, buf62, primals_188, primals_189, buf63, buf64, buf66, primals_188, primals_189, 64, 2, grid=grid(64), stream=stream0)
        del buf60
        del buf61
        del buf62
        del primals_188
        del primals_189
        buf67 = reinterpret_tensor(buf55, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf55  # reuse
        # Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_18.run(buf56, buf63, buf64, primals_9, primals_10, buf67, 1605632, grid=grid(1605632), stream=stream0)
        del buf64
        del primals_10
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf69 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf68, buf69, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf70 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf69, buf70, buf71, buf72, 4704, 128, grid=grid(4704), stream=stream0)
        buf73 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf70, buf71, buf72, buf73, buf74, buf75, 48, 98, grid=grid(48), stream=stream0)
        buf76 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf77 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf79 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf73, buf74, buf75, primals_191, primals_192, buf76, buf77, buf79, primals_191, primals_192, 24, 2, grid=grid(24), stream=stream0)
        del primals_191
        del primals_192
        buf80 = reinterpret_tensor(buf68, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf68  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_23.run(buf69, buf76, buf77, primals_11, primals_12, buf80, 602112, grid=grid(602112), stream=stream0)
        del primals_12
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf82 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf81, buf82, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf83 = empty_strided((1, 72, 1, 1, 196), (14112, 1, 14112, 14112, 72), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((1, 72, 1, 1, 196), (14112, 1, 14112, 14112, 72), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((1, 72, 1, 1, 196), (14112, 1, 14112, 14112, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf82, buf83, buf84, buf85, 14112, 128, grid=grid(14112), stream=stream0)
        buf86 = empty_strided((1, 72, 1, 1, 2), (144, 1, 144, 144, 72), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((1, 72, 1, 1, 2), (144, 1, 144, 144, 72), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((1, 72, 1, 1, 2), (144, 1, 144, 144, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf83, buf84, buf85, buf86, buf87, buf88, 144, 98, grid=grid(144), stream=stream0)
        buf89 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf92 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf86, buf87, buf88, primals_194, primals_195, buf89, buf90, buf92, primals_194, primals_195, 72, 2, grid=grid(72), stream=stream0)
        del primals_194
        del primals_195
        buf93 = reinterpret_tensor(buf81, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf81  # reuse
        # Source Nodes: [x_34, x_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf82, buf89, buf90, primals_13, primals_14, buf93, 1806336, grid=grid(1806336), stream=stream0)
        del primals_14
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf94, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf95 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf94, buf95, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf96 = buf85; del buf85  # reuse
        buf97 = buf84; del buf84  # reuse
        buf98 = buf83; del buf83  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf95, buf96, buf97, buf98, 14112, 128, grid=grid(14112), stream=stream0)
        buf99 = buf88; del buf88  # reuse
        buf100 = buf87; del buf87  # reuse
        buf101 = buf86; del buf86  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf96, buf97, buf98, buf99, buf100, buf101, 144, 98, grid=grid(144), stream=stream0)
        buf102 = buf90; del buf90  # reuse
        buf103 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf105 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf99, buf100, buf101, primals_197, primals_198, buf102, buf103, buf105, primals_197, primals_198, 72, 2, grid=grid(72), stream=stream0)
        del primals_197
        del primals_198
        buf106 = reinterpret_tensor(buf94, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf94  # reuse
        # Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf95, buf102, buf103, primals_15, primals_16, buf106, 1806336, grid=grid(1806336), stream=stream0)
        del primals_16
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf108 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf107, buf108, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf109 = buf72; del buf72  # reuse
        buf110 = buf71; del buf71  # reuse
        buf111 = buf70; del buf70  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf108, buf109, buf110, buf111, 4704, 128, grid=grid(4704), stream=stream0)
        buf112 = buf75; del buf75  # reuse
        buf113 = buf74; del buf74  # reuse
        buf114 = buf73; del buf73  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf109, buf110, buf111, buf112, buf113, buf114, 48, 98, grid=grid(48), stream=stream0)
        del buf109
        del buf110
        del buf111
        buf115 = buf77; del buf77  # reuse
        buf116 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf118 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf112, buf113, buf114, primals_200, primals_201, buf115, buf116, buf118, primals_200, primals_201, 24, 2, grid=grid(24), stream=stream0)
        del buf112
        del buf113
        del buf114
        del primals_200
        del primals_201
        buf119 = reinterpret_tensor(buf107, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf107  # reuse
        # Source Nodes: [shortcut_3, x_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_29.run(buf108, buf115, buf116, primals_17, primals_18, buf80, buf119, 602112, grid=grid(602112), stream=stream0)
        del buf116
        del primals_18
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 72, 56, 56), (225792, 3136, 56, 1))
        buf121 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf120, buf121, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf122 = buf98; del buf98  # reuse
        buf123 = buf97; del buf97  # reuse
        buf124 = buf96; del buf96  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf121, buf122, buf123, buf124, 14112, 128, grid=grid(14112), stream=stream0)
        buf125 = buf99; del buf99  # reuse
        buf126 = buf101; del buf101  # reuse
        buf127 = buf100; del buf100  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf122, buf123, buf124, buf125, buf126, buf127, 144, 98, grid=grid(144), stream=stream0)
        del buf122
        del buf123
        del buf124
        buf128 = buf103; del buf103  # reuse
        buf129 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf131 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf125, buf126, buf127, primals_203, primals_204, buf128, buf129, buf131, primals_203, primals_204, 72, 2, grid=grid(72), stream=stream0)
        del buf125
        del buf126
        del buf127
        del primals_203
        del primals_204
        buf132 = reinterpret_tensor(buf120, (8, 72, 56, 56), (225792, 1, 4032, 72), 0); del buf120  # reuse
        # Source Nodes: [x_51, x_54], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf121, buf128, buf129, primals_19, primals_20, buf132, 1806336, grid=grid(1806336), stream=stream0)
        del primals_20
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_105, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf133, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf134 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf133, buf134, 576, 784, grid=grid(576, 784), stream=stream0)
        buf135 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        buf136 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((1, 72, 1, 1, 49), (3528, 1, 3528, 3528, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf134, buf135, buf136, buf137, 3528, 128, grid=grid(3528), stream=stream0)
        buf138 = buf129; del buf129  # reuse
        buf139 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf141 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_32.run(buf135, buf136, buf137, primals_206, primals_207, buf138, buf139, buf141, primals_206, primals_207, 72, 49, grid=grid(72), stream=stream0)
        del buf135
        del buf136
        del buf137
        del primals_206
        del primals_207
        buf142 = reinterpret_tensor(buf133, (8, 72, 28, 28), (56448, 1, 2016, 72), 0); del buf133  # reuse
        # Source Nodes: [x_56, x_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_33.run(buf134, buf138, buf139, primals_21, primals_22, buf142, 451584, grid=grid(451584), stream=stream0)
        del buf139
        del primals_22
        buf143 = empty_strided((8, 72, 1, 1, 7), (504, 1, 4032, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_red_fused_mean_34.run(buf142, buf143, 4032, 112, grid=grid(4032), stream=stream0)
        buf144 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf145 = reinterpret_tensor(buf144, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf144  # reuse
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_per_fused_mean_35.run(buf145, buf143, 576, 7, grid=grid(576), stream=stream0)
        del buf143
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 24, 1, 1), (24, 1, 1, 1))
        buf147 = reinterpret_tensor(buf146, (8, 24, 1, 1), (24, 1, 24, 24), 0); del buf146  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_36.run(buf147, primals_107, 192, grid=grid(192), stream=stream0)
        del primals_107
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 72, 1, 1), (72, 1, 1, 1))
        buf149 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_37.run(buf148, primals_109, buf149, 576, grid=grid(576), stream=stream0)
        buf150 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.mul]
        triton_poi_fused_mul_38.run(buf142, buf149, buf150, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf152 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf151, buf152, 320, 784, grid=grid(320, 784), stream=stream0)
        buf153 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf154 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf152, buf153, buf154, buf155, 1960, 128, grid=grid(1960), stream=stream0)
        buf156 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf157 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf159 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf153, buf154, buf155, primals_209, primals_210, buf156, buf157, buf159, primals_209, primals_210, 40, 49, grid=grid(40), stream=stream0)
        del primals_209
        del primals_210
        buf160 = reinterpret_tensor(buf151, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf151  # reuse
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_42.run(buf152, buf156, buf157, primals_23, primals_24, buf160, 250880, grid=grid(250880), stream=stream0)
        del primals_24
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf162 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf161, buf162, 960, 784, grid=grid(960, 784), stream=stream0)
        buf163 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf162, buf163, buf164, buf165, 5880, 128, grid=grid(5880), stream=stream0)
        buf166 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf169 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf163, buf164, buf165, primals_212, primals_213, buf166, buf167, buf169, primals_212, primals_213, 120, 49, grid=grid(120), stream=stream0)
        del primals_212
        del primals_213
        buf170 = reinterpret_tensor(buf161, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf161  # reuse
        # Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf162, buf166, buf167, primals_25, primals_26, buf170, 752640, grid=grid(752640), stream=stream0)
        del primals_26
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_112, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf171, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf172 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf171, buf172, 960, 784, grid=grid(960, 784), stream=stream0)
        buf173 = buf165; del buf165  # reuse
        buf174 = buf164; del buf164  # reuse
        buf175 = buf163; del buf163  # reuse
        # Source Nodes: [x_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf172, buf173, buf174, buf175, 5880, 128, grid=grid(5880), stream=stream0)
        buf176 = buf167; del buf167  # reuse
        buf177 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf179 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf173, buf174, buf175, primals_215, primals_216, buf176, buf177, buf179, primals_215, primals_216, 120, 49, grid=grid(120), stream=stream0)
        del primals_215
        del primals_216
        buf180 = reinterpret_tensor(buf171, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf171  # reuse
        # Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf172, buf176, buf177, primals_27, primals_28, buf180, 752640, grid=grid(752640), stream=stream0)
        del primals_28
        buf181 = empty_strided((8, 120, 1, 1, 7), (840, 1, 6720, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_red_fused_mean_47.run(buf180, buf181, 6720, 112, grid=grid(6720), stream=stream0)
        buf182 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf183 = reinterpret_tensor(buf182, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf182  # reuse
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_per_fused_mean_48.run(buf183, buf181, 960, 7, grid=grid(960), stream=stream0)
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 32, 1, 1), (32, 1, 1, 1))
        buf185 = reinterpret_tensor(buf184, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf184  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_49.run(buf185, primals_114, 256, grid=grid(256), stream=stream0)
        del primals_114
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 120, 1, 1), (120, 1, 1, 1))
        buf187 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_50.run(buf186, primals_116, buf187, 960, grid=grid(960), stream=stream0)
        buf188 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.mul]
        triton_poi_fused_mul_51.run(buf180, buf187, buf188, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf190 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf189, buf190, 320, 784, grid=grid(320, 784), stream=stream0)
        buf191 = buf155; del buf155  # reuse
        buf192 = buf154; del buf154  # reuse
        buf193 = buf153; del buf153  # reuse
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf190, buf191, buf192, buf193, 1960, 128, grid=grid(1960), stream=stream0)
        buf194 = buf157; del buf157  # reuse
        buf195 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf197 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf191, buf192, buf193, primals_218, primals_219, buf194, buf195, buf197, primals_218, primals_219, 40, 49, grid=grid(40), stream=stream0)
        del primals_218
        del primals_219
        buf198 = reinterpret_tensor(buf189, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf189  # reuse
        # Source Nodes: [shortcut_5, x_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_52.run(buf190, buf194, buf195, primals_29, primals_30, buf160, buf198, 250880, grid=grid(250880), stream=stream0)
        del primals_30
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf200 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf199, buf200, 960, 784, grid=grid(960, 784), stream=stream0)
        buf201 = buf175; del buf175  # reuse
        buf202 = buf174; del buf174  # reuse
        buf203 = buf173; del buf173  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf200, buf201, buf202, buf203, 5880, 128, grid=grid(5880), stream=stream0)
        buf204 = buf177; del buf177  # reuse
        buf205 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf207 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf201, buf202, buf203, primals_221, primals_222, buf204, buf205, buf207, primals_221, primals_222, 120, 49, grid=grid(120), stream=stream0)
        del primals_221
        del primals_222
        buf208 = reinterpret_tensor(buf199, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf199  # reuse
        # Source Nodes: [x_84, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf200, buf204, buf205, primals_31, primals_32, buf208, 752640, grid=grid(752640), stream=stream0)
        del primals_32
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_119, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf209, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf210 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf209, buf210, 960, 784, grid=grid(960, 784), stream=stream0)
        buf211 = buf203; del buf203  # reuse
        buf212 = buf202; del buf202  # reuse
        buf213 = buf201; del buf201  # reuse
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf210, buf211, buf212, buf213, 5880, 128, grid=grid(5880), stream=stream0)
        buf214 = buf205; del buf205  # reuse
        buf215 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf217 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf211, buf212, buf213, primals_224, primals_225, buf214, buf215, buf217, primals_224, primals_225, 120, 49, grid=grid(120), stream=stream0)
        del buf211
        del buf212
        del buf213
        del primals_224
        del primals_225
        buf218 = reinterpret_tensor(buf209, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf209  # reuse
        # Source Nodes: [x_89, x_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf210, buf214, buf215, primals_33, primals_34, buf218, 752640, grid=grid(752640), stream=stream0)
        del buf215
        del primals_34
        buf219 = buf181; del buf181  # reuse
        # Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_red_fused_mean_47.run(buf218, buf219, 6720, 112, grid=grid(6720), stream=stream0)
        buf220 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf221 = reinterpret_tensor(buf220, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf220  # reuse
        # Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_per_fused_mean_48.run(buf221, buf219, 960, 7, grid=grid(960), stream=stream0)
        del buf219
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 32, 1, 1), (32, 1, 1, 1))
        buf223 = reinterpret_tensor(buf222, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf222  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_49.run(buf223, primals_121, 256, grid=grid(256), stream=stream0)
        del primals_121
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 120, 1, 1), (120, 1, 1, 1))
        buf225 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_50.run(buf224, primals_123, buf225, 960, grid=grid(960), stream=stream0)
        buf226 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.mul]
        triton_poi_fused_mul_51.run(buf218, buf225, buf226, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf228 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf227, buf228, 320, 784, grid=grid(320, 784), stream=stream0)
        buf229 = buf193; del buf193  # reuse
        buf230 = buf192; del buf192  # reuse
        buf231 = buf191; del buf191  # reuse
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf228, buf229, buf230, buf231, 1960, 128, grid=grid(1960), stream=stream0)
        buf232 = buf195; del buf195  # reuse
        buf233 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf235 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf229, buf230, buf231, primals_227, primals_228, buf232, buf233, buf235, primals_227, primals_228, 40, 49, grid=grid(40), stream=stream0)
        del buf229
        del buf230
        del buf231
        del primals_227
        del primals_228
        buf236 = reinterpret_tensor(buf227, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf227  # reuse
        # Source Nodes: [shortcut_6, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_52.run(buf228, buf232, buf233, primals_35, primals_36, buf198, buf236, 250880, grid=grid(250880), stream=stream0)
        del buf233
        del primals_36
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 240, 28, 28), (188160, 784, 28, 1))
        buf238 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf237, buf238, 1920, 784, grid=grid(1920, 784), stream=stream0)
        buf239 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf240 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        buf241 = empty_strided((1, 240, 1, 1, 49), (11760, 1, 11760, 11760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf238, buf239, buf240, buf241, 11760, 128, grid=grid(11760), stream=stream0)
        buf242 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf243 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf245 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_55.run(buf239, buf240, buf241, primals_230, primals_231, buf242, buf243, buf245, primals_230, primals_231, 240, 49, grid=grid(240), stream=stream0)
        del buf239
        del buf240
        del buf241
        del primals_230
        del primals_231
        buf246 = reinterpret_tensor(buf237, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf237  # reuse
        buf247 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_104], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_56.run(buf238, buf242, buf243, primals_37, primals_38, buf246, buf247, 1505280, grid=grid(1505280), stream=stream0)
        del primals_38
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_126, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf248, (8, 240, 14, 14), (47040, 196, 14, 1))
        buf249 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf248, buf249, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf250 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf251 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        buf252 = empty_strided((1, 240, 1, 1, 13), (3120, 1, 3120, 3120, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf249, buf250, buf251, buf252, 3120, 121, grid=grid(3120), stream=stream0)
        buf253 = buf243; del buf243  # reuse
        buf254 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cuda', dtype=torch.float32)
        buf256 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf250, buf251, buf252, primals_233, primals_234, buf253, buf254, buf256, primals_233, primals_234, 240, 13, grid=grid(240), stream=stream0)
        del buf250
        del buf251
        del buf252
        del primals_233
        del primals_234
        buf257 = reinterpret_tensor(buf248, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf248  # reuse
        buf258 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_60.run(buf249, buf253, buf254, primals_39, primals_40, buf257, buf258, 376320, grid=grid(376320), stream=stream0)
        del buf254
        del primals_40
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf260 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf259, buf260, 640, 196, grid=grid(640, 196), stream=stream0)
        buf261 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf262 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        buf263 = empty_strided((1, 80, 1, 1, 13), (1040, 1, 1040, 1040, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf260, buf261, buf262, buf263, 1040, 121, grid=grid(1040), stream=stream0)
        buf264 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf267 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf261, buf262, buf263, primals_236, primals_237, buf264, buf265, buf267, primals_236, primals_237, 80, 13, grid=grid(80), stream=stream0)
        del primals_236
        del primals_237
        buf268 = reinterpret_tensor(buf259, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf259  # reuse
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_64.run(buf260, buf264, buf265, primals_41, primals_42, buf268, 125440, grid=grid(125440), stream=stream0)
        del primals_42
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 200, 14, 14), (39200, 196, 14, 1))
        buf270 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf269, buf270, 1600, 196, grid=grid(1600, 196), stream=stream0)
        buf271 = empty_strided((1, 200, 1, 1, 13), (2600, 1, 2600, 2600, 200), device='cuda', dtype=torch.float32)
        buf272 = empty_strided((1, 200, 1, 1, 13), (2600, 1, 2600, 2600, 200), device='cuda', dtype=torch.float32)
        buf273 = empty_strided((1, 200, 1, 1, 13), (2600, 1, 2600, 2600, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf270, buf271, buf272, buf273, 2600, 121, grid=grid(2600), stream=stream0)
        buf274 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf275 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf277 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf271, buf272, buf273, primals_239, primals_240, buf274, buf275, buf277, primals_239, primals_240, 200, 13, grid=grid(200), stream=stream0)
        del primals_239
        del primals_240
        buf278 = reinterpret_tensor(buf269, (8, 200, 14, 14), (39200, 1, 2800, 200), 0); del buf269  # reuse
        buf279 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_68.run(buf270, buf274, buf275, primals_43, primals_44, buf278, buf279, 313600, grid=grid(313600), stream=stream0)
        del primals_44
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf280, (8, 200, 14, 14), (39200, 196, 14, 1))
        buf281 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_65.run(buf280, buf281, 1600, 196, grid=grid(1600, 196), stream=stream0)
        buf282 = buf273; del buf273  # reuse
        buf283 = buf272; del buf272  # reuse
        buf284 = buf271; del buf271  # reuse
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf281, buf282, buf283, buf284, 2600, 121, grid=grid(2600), stream=stream0)
        buf285 = buf275; del buf275  # reuse
        buf286 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf288 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf282, buf283, buf284, primals_242, primals_243, buf285, buf286, buf288, primals_242, primals_243, 200, 13, grid=grid(200), stream=stream0)
        del buf282
        del buf283
        del buf284
        del primals_242
        del primals_243
        buf289 = reinterpret_tensor(buf280, (8, 200, 14, 14), (39200, 1, 2800, 200), 0); del buf280  # reuse
        buf290 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122, x_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_68.run(buf281, buf285, buf286, primals_45, primals_46, buf289, buf290, 313600, grid=grid(313600), stream=stream0)
        del buf286
        del primals_46
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf292 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf291, buf292, 640, 196, grid=grid(640, 196), stream=stream0)
        buf293 = buf263; del buf263  # reuse
        buf294 = buf262; del buf262  # reuse
        buf295 = buf261; del buf261  # reuse
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf292, buf293, buf294, buf295, 1040, 121, grid=grid(1040), stream=stream0)
        buf296 = buf265; del buf265  # reuse
        buf297 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf299 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf293, buf294, buf295, primals_245, primals_246, buf296, buf297, buf299, primals_245, primals_246, 80, 13, grid=grid(80), stream=stream0)
        del primals_245
        del primals_246
        buf300 = reinterpret_tensor(buf291, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf291  # reuse
        # Source Nodes: [shortcut_8, x_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_69.run(buf292, buf296, buf297, primals_47, primals_48, buf268, buf300, 125440, grid=grid(125440), stream=stream0)
        del primals_48
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 184, 14, 14), (36064, 196, 14, 1))
        buf302 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf301, buf302, 1472, 196, grid=grid(1472, 196), stream=stream0)
        buf303 = empty_strided((1, 184, 1, 1, 13), (2392, 1, 2392, 2392, 184), device='cuda', dtype=torch.float32)
        buf304 = empty_strided((1, 184, 1, 1, 13), (2392, 1, 2392, 2392, 184), device='cuda', dtype=torch.float32)
        buf305 = empty_strided((1, 184, 1, 1, 13), (2392, 1, 2392, 2392, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_71.run(buf302, buf303, buf304, buf305, 2392, 121, grid=grid(2392), stream=stream0)
        buf306 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf309 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_72.run(buf303, buf304, buf305, primals_248, primals_249, buf306, buf307, buf309, primals_248, primals_249, 184, 13, grid=grid(184), stream=stream0)
        del primals_248
        del primals_249
        buf310 = reinterpret_tensor(buf301, (8, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf301  # reuse
        buf311 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134, x_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_73.run(buf302, buf306, buf307, primals_49, primals_50, buf310, buf311, 288512, grid=grid(288512), stream=stream0)
        del primals_50
        # Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf312, (8, 184, 14, 14), (36064, 196, 14, 1))
        buf313 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf312, buf313, 1472, 196, grid=grid(1472, 196), stream=stream0)
        buf314 = buf305; del buf305  # reuse
        buf315 = buf304; del buf304  # reuse
        buf316 = buf303; del buf303  # reuse
        # Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_71.run(buf313, buf314, buf315, buf316, 2392, 121, grid=grid(2392), stream=stream0)
        buf317 = buf307; del buf307  # reuse
        buf318 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf320 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_72.run(buf314, buf315, buf316, primals_251, primals_252, buf317, buf318, buf320, primals_251, primals_252, 184, 13, grid=grid(184), stream=stream0)
        del primals_251
        del primals_252
        buf321 = reinterpret_tensor(buf312, (8, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf312  # reuse
        buf322 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139, x_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_73.run(buf313, buf317, buf318, primals_51, primals_52, buf321, buf322, 288512, grid=grid(288512), stream=stream0)
        del primals_52
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf324 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf323, buf324, 640, 196, grid=grid(640, 196), stream=stream0)
        buf325 = buf295; del buf295  # reuse
        buf326 = buf294; del buf294  # reuse
        buf327 = buf293; del buf293  # reuse
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf324, buf325, buf326, buf327, 1040, 121, grid=grid(1040), stream=stream0)
        buf328 = buf297; del buf297  # reuse
        buf329 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf331 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf325, buf326, buf327, primals_254, primals_255, buf328, buf329, buf331, primals_254, primals_255, 80, 13, grid=grid(80), stream=stream0)
        del primals_254
        del primals_255
        buf332 = reinterpret_tensor(buf323, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf323  # reuse
        # Source Nodes: [shortcut_9, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_69.run(buf324, buf328, buf329, primals_53, primals_54, buf300, buf332, 125440, grid=grid(125440), stream=stream0)
        del primals_54
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 184, 14, 14), (36064, 196, 14, 1))
        buf334 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf333, buf334, 1472, 196, grid=grid(1472, 196), stream=stream0)
        buf335 = buf316; del buf316  # reuse
        buf336 = buf315; del buf315  # reuse
        buf337 = buf314; del buf314  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_71.run(buf334, buf335, buf336, buf337, 2392, 121, grid=grid(2392), stream=stream0)
        buf338 = buf318; del buf318  # reuse
        buf339 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf341 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_72.run(buf335, buf336, buf337, primals_257, primals_258, buf338, buf339, buf341, primals_257, primals_258, 184, 13, grid=grid(184), stream=stream0)
        del primals_257
        del primals_258
        buf342 = reinterpret_tensor(buf333, (8, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf333  # reuse
        buf343 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_73.run(buf334, buf338, buf339, primals_55, primals_56, buf342, buf343, 288512, grid=grid(288512), stream=stream0)
        del primals_56
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf344, (8, 184, 14, 14), (36064, 196, 14, 1))
        buf345 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf344, buf345, 1472, 196, grid=grid(1472, 196), stream=stream0)
        buf346 = buf337; del buf337  # reuse
        buf347 = buf336; del buf336  # reuse
        buf348 = buf335; del buf335  # reuse
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_71.run(buf345, buf346, buf347, buf348, 2392, 121, grid=grid(2392), stream=stream0)
        buf349 = buf339; del buf339  # reuse
        buf350 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf352 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_72.run(buf346, buf347, buf348, primals_260, primals_261, buf349, buf350, buf352, primals_260, primals_261, 184, 13, grid=grid(184), stream=stream0)
        del buf346
        del buf347
        del buf348
        del primals_260
        del primals_261
        buf353 = reinterpret_tensor(buf344, (8, 184, 14, 14), (36064, 1, 2576, 184), 0); del buf344  # reuse
        buf354 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_73.run(buf345, buf349, buf350, primals_57, primals_58, buf353, buf354, 288512, grid=grid(288512), stream=stream0)
        del buf350
        del primals_58
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 80, 14, 14), (15680, 196, 14, 1))
        buf356 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf355, buf356, 640, 196, grid=grid(640, 196), stream=stream0)
        buf357 = buf327; del buf327  # reuse
        buf358 = buf326; del buf326  # reuse
        buf359 = buf325; del buf325  # reuse
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf356, buf357, buf358, buf359, 1040, 121, grid=grid(1040), stream=stream0)
        buf360 = buf329; del buf329  # reuse
        buf361 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cuda', dtype=torch.float32)
        buf363 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf357, buf358, buf359, primals_263, primals_264, buf360, buf361, buf363, primals_263, primals_264, 80, 13, grid=grid(80), stream=stream0)
        del buf357
        del buf358
        del buf359
        del primals_263
        del primals_264
        buf364 = reinterpret_tensor(buf355, (8, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf355  # reuse
        # Source Nodes: [shortcut_10, x_162], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_69.run(buf356, buf360, buf361, primals_59, primals_60, buf332, buf364, 125440, grid=grid(125440), stream=stream0)
        del buf361
        del primals_60
        # Source Nodes: [x_167], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf366 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf365, buf366, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf367 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        buf369 = empty_strided((1, 480, 1, 1, 13), (6240, 1, 6240, 6240, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf366, buf367, buf368, buf369, 6240, 121, grid=grid(6240), stream=stream0)
        buf370 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf371 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf373 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf367, buf368, buf369, primals_266, primals_267, buf370, buf371, buf373, primals_266, primals_267, 480, 13, grid=grid(480), stream=stream0)
        del primals_266
        del primals_267
        buf374 = reinterpret_tensor(buf365, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf365  # reuse
        buf375 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168, x_171], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_77.run(buf366, buf370, buf371, primals_61, primals_62, buf374, buf375, 752640, grid=grid(752640), stream=stream0)
        del primals_62
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf376, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf377 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_74.run(buf376, buf377, 3840, 196, grid=grid(3840, 196), stream=stream0)
        buf378 = buf369; del buf369  # reuse
        buf379 = buf368; del buf368  # reuse
        buf380 = buf367; del buf367  # reuse
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_75.run(buf377, buf378, buf379, buf380, 6240, 121, grid=grid(6240), stream=stream0)
        buf381 = buf371; del buf371  # reuse
        buf382 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf384 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_76.run(buf378, buf379, buf380, primals_269, primals_270, buf381, buf382, buf384, primals_269, primals_270, 480, 13, grid=grid(480), stream=stream0)
        del buf378
        del buf379
        del buf380
        del primals_269
        del primals_270
        buf385 = reinterpret_tensor(buf376, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf376  # reuse
        buf386 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_77.run(buf377, buf381, buf382, primals_63, primals_64, buf385, buf386, 752640, grid=grid(752640), stream=stream0)
        del buf382
        del primals_64
        buf387 = empty_strided((8, 480, 1, 1, 2), (960, 1, 7680, 7680, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_red_fused_mean_78.run(buf386, buf387, 7680, 98, grid=grid(7680), stream=stream0)
        buf388 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf389 = reinterpret_tensor(buf388, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf388  # reuse
        # Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_per_fused_mean_79.run(buf389, buf387, 3840, 2, grid=grid(3840), stream=stream0)
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (8, 120, 1, 1), (120, 1, 1, 1))
        buf391 = reinterpret_tensor(buf390, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf390  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_80.run(buf391, primals_140, 960, grid=grid(960), stream=stream0)
        del primals_140
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 480, 1, 1), (480, 1, 1, 1))
        buf393 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_81.run(buf392, primals_142, buf393, 3840, grid=grid(3840), stream=stream0)
        buf394 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.mul]
        triton_poi_fused_mul_82.run(buf386, buf393, buf394, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_178], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf395, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf396 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf395, buf396, 896, 196, grid=grid(896, 196), stream=stream0)
        buf397 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf398 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf399 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf396, buf397, buf398, buf399, 1456, 121, grid=grid(1456), stream=stream0)
        buf400 = reinterpret_tensor(buf36, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf36  # reuse
        buf401 = reinterpret_tensor(buf35, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf35  # reuse
        buf403 = reinterpret_tensor(buf34, (112, ), (1, ), 0); del buf34  # reuse
        # Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf397, buf398, buf399, primals_272, primals_273, buf400, buf401, buf403, primals_272, primals_273, 112, 13, grid=grid(112), stream=stream0)
        del primals_272
        del primals_273
        buf404 = reinterpret_tensor(buf395, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf395  # reuse
        # Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_86.run(buf396, buf400, buf401, primals_65, primals_66, buf404, 175616, grid=grid(175616), stream=stream0)
        del primals_66
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf406 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_87.run(buf405, buf406, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf407 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        buf408 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        buf409 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf406, buf407, buf408, buf409, 8736, 121, grid=grid(8736), stream=stream0)
        buf410 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf411 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf413 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf407, buf408, buf409, primals_275, primals_276, buf410, buf411, buf413, primals_275, primals_276, 672, 13, grid=grid(672), stream=stream0)
        del primals_275
        del primals_276
        buf414 = reinterpret_tensor(buf405, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf405  # reuse
        buf415 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184, x_187], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_90.run(buf406, buf410, buf411, primals_67, primals_68, buf414, buf415, 1053696, grid=grid(1053696), stream=stream0)
        del primals_68
        # Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf416, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf417 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_87.run(buf416, buf417, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf418 = buf409; del buf409  # reuse
        buf419 = buf408; del buf408  # reuse
        buf420 = buf407; del buf407  # reuse
        # Source Nodes: [x_189], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf417, buf418, buf419, buf420, 8736, 121, grid=grid(8736), stream=stream0)
        buf421 = buf411; del buf411  # reuse
        buf422 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf424 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf418, buf419, buf420, primals_278, primals_279, buf421, buf422, buf424, primals_278, primals_279, 672, 13, grid=grid(672), stream=stream0)
        del primals_278
        del primals_279
        buf425 = reinterpret_tensor(buf416, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf416  # reuse
        buf426 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189, x_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_90.run(buf417, buf421, buf422, primals_69, primals_70, buf425, buf426, 1053696, grid=grid(1053696), stream=stream0)
        del primals_70
        buf427 = empty_strided((8, 672, 1, 1, 2), (1344, 1, 10752, 10752, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_16], Original ATen: [aten.mean]
        triton_red_fused_mean_91.run(buf426, buf427, 10752, 98, grid=grid(10752), stream=stream0)
        buf428 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf429 = reinterpret_tensor(buf428, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf428  # reuse
        # Source Nodes: [x_se_16], Original ATen: [aten.mean]
        triton_per_fused_mean_92.run(buf429, buf427, 5376, 2, grid=grid(5376), stream=stream0)
        del buf427
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 168, 1, 1), (168, 1, 1, 1))
        buf431 = reinterpret_tensor(buf430, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf430  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_93.run(buf431, primals_147, 1344, grid=grid(1344), stream=stream0)
        del primals_147
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 672, 1, 1), (672, 1, 1, 1))
        buf433 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_94.run(buf432, primals_149, buf433, 5376, grid=grid(5376), stream=stream0)
        buf434 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.mul]
        triton_poi_fused_mul_95.run(buf426, buf433, buf434, 1053696, grid=grid(1053696), stream=stream0)
        # Source Nodes: [x_194], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf436 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_83.run(buf435, buf436, 896, 196, grid=grid(896, 196), stream=stream0)
        buf437 = buf399; del buf399  # reuse
        buf438 = buf398; del buf398  # reuse
        buf439 = buf397; del buf397  # reuse
        # Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf436, buf437, buf438, buf439, 1456, 121, grid=grid(1456), stream=stream0)
        buf440 = buf401; del buf401  # reuse
        buf441 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf443 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_85.run(buf437, buf438, buf439, primals_281, primals_282, buf440, buf441, buf443, primals_281, primals_282, 112, 13, grid=grid(112), stream=stream0)
        del buf437
        del buf438
        del buf439
        del primals_281
        del primals_282
        buf444 = reinterpret_tensor(buf435, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf435  # reuse
        # Source Nodes: [shortcut_12, x_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_96.run(buf436, buf440, buf441, primals_71, primals_72, buf404, buf444, 175616, grid=grid(175616), stream=stream0)
        del buf441
        del primals_72
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf446 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_87.run(buf445, buf446, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf447 = buf420; del buf420  # reuse
        buf448 = buf419; del buf419  # reuse
        buf449 = buf418; del buf418  # reuse
        # Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_88.run(buf446, buf447, buf448, buf449, 8736, 121, grid=grid(8736), stream=stream0)
        buf450 = buf422; del buf422  # reuse
        buf451 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf453 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_89.run(buf447, buf448, buf449, primals_284, primals_285, buf450, buf451, buf453, primals_284, primals_285, 672, 13, grid=grid(672), stream=stream0)
        del buf447
        del buf448
        del buf449
        del primals_284
        del primals_285
        buf454 = reinterpret_tensor(buf445, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf445  # reuse
        buf455 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201, x_204], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_90.run(buf446, buf450, buf451, primals_73, primals_74, buf454, buf455, 1053696, grid=grid(1053696), stream=stream0)
        del primals_74
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_152, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf456, (8, 672, 7, 7), (32928, 49, 7, 1))
        buf457 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_97.run(buf456, buf457, 5376, 49, grid=grid(5376, 49), stream=stream0)
        buf458 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf459 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf460 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_98.run(buf457, buf458, buf459, buf460, 2688, 98, grid=grid(2688), stream=stream0)
        buf461 = buf451; del buf451  # reuse
        buf462 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf464 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_99.run(buf458, buf459, buf460, primals_287, primals_288, buf461, buf462, buf464, primals_287, primals_288, 672, 4, grid=grid(672), stream=stream0)
        del buf458
        del buf459
        del buf460
        del primals_287
        del primals_288
        buf465 = reinterpret_tensor(buf456, (8, 672, 7, 7), (32928, 1, 4704, 672), 0); del buf456  # reuse
        buf466 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206, x_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_100.run(buf457, buf461, buf462, primals_75, primals_76, buf465, buf466, 263424, grid=grid(263424), stream=stream0)
        del buf462
        del primals_76
        buf467 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf468 = reinterpret_tensor(buf467, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf467  # reuse
        # Source Nodes: [x_se_20], Original ATen: [aten.mean]
        triton_per_fused_mean_101.run(buf468, buf466, 5376, 49, grid=grid(5376), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf468, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (8, 168, 1, 1), (168, 1, 1, 1))
        buf470 = reinterpret_tensor(buf469, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf469  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_93.run(buf470, primals_154, 1344, grid=grid(1344), stream=stream0)
        del primals_154
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 672, 1, 1), (672, 1, 1, 1))
        buf472 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_94.run(buf471, primals_156, buf472, 5376, grid=grid(5376), stream=stream0)
        buf473 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.mul]
        triton_poi_fused_mul_102.run(buf466, buf472, buf473, 263424, grid=grid(263424), stream=stream0)
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf475 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf474, buf475, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf476 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        buf477 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        buf478 = empty_strided((1, 160, 1, 1, 4), (640, 1, 640, 640, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf475, buf476, buf477, buf478, 640, 98, grid=grid(640), stream=stream0)
        buf479 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf480 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf482 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf476, buf477, buf478, primals_290, primals_291, buf479, buf480, buf482, primals_290, primals_291, 160, 4, grid=grid(160), stream=stream0)
        del primals_290
        del primals_291
        buf483 = reinterpret_tensor(buf474, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf474  # reuse
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_106.run(buf475, buf479, buf480, primals_77, primals_78, buf483, 62720, grid=grid(62720), stream=stream0)
        del primals_78
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf485 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf484, buf485, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf486 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf487 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        buf488 = empty_strided((1, 960, 1, 1, 4), (3840, 1, 3840, 3840, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf485, buf486, buf487, buf488, 3840, 98, grid=grid(3840), stream=stream0)
        buf489 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf490 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf492 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf486, buf487, buf488, primals_293, primals_294, buf489, buf490, buf492, primals_293, primals_294, 960, 4, grid=grid(960), stream=stream0)
        del primals_293
        del primals_294
        buf493 = reinterpret_tensor(buf484, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf484  # reuse
        buf494 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_110.run(buf485, buf489, buf490, primals_79, primals_80, buf493, buf494, 376320, grid=grid(376320), stream=stream0)
        del primals_80
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_159, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf495, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf496 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf495, buf496, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf497 = buf488; del buf488  # reuse
        buf498 = buf487; del buf487  # reuse
        buf499 = buf486; del buf486  # reuse
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf496, buf497, buf498, buf499, 3840, 98, grid=grid(3840), stream=stream0)
        buf500 = buf490; del buf490  # reuse
        buf501 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf503 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf497, buf498, buf499, primals_296, primals_297, buf500, buf501, buf503, primals_296, primals_297, 960, 4, grid=grid(960), stream=stream0)
        del primals_296
        del primals_297
        buf504 = reinterpret_tensor(buf495, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf495  # reuse
        buf505 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_110.run(buf496, buf500, buf501, primals_81, primals_82, buf504, buf505, 376320, grid=grid(376320), stream=stream0)
        del primals_82
        buf506 = reinterpret_tensor(buf387, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf387  # reuse
        buf507 = reinterpret_tensor(buf506, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf506  # reuse
        # Source Nodes: [x_se_24], Original ATen: [aten.mean]
        triton_per_fused_mean_111.run(buf507, buf505, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (8, 240, 1, 1), (240, 1, 1, 1))
        buf509 = reinterpret_tensor(buf508, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf508  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_112.run(buf509, primals_161, 1920, grid=grid(1920), stream=stream0)
        del primals_161
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (8, 960, 1, 1), (960, 1, 1, 1))
        buf511 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf579 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_113.run(buf510, primals_163, buf511, buf579, 7680, grid=grid(7680), stream=stream0)
        del primals_163
        buf512 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.mul]
        triton_poi_fused_mul_114.run(buf505, buf511, buf512, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf514 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf513, buf514, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf515 = buf478; del buf478  # reuse
        buf516 = buf477; del buf477  # reuse
        buf517 = buf476; del buf476  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf514, buf515, buf516, buf517, 640, 98, grid=grid(640), stream=stream0)
        buf518 = buf480; del buf480  # reuse
        buf519 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf521 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf515, buf516, buf517, primals_299, primals_300, buf518, buf519, buf521, primals_299, primals_300, 160, 4, grid=grid(160), stream=stream0)
        del primals_299
        del primals_300
        buf522 = reinterpret_tensor(buf513, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf513  # reuse
        # Source Nodes: [shortcut_14, x_228], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_115.run(buf514, buf518, buf519, primals_83, primals_84, buf483, buf522, 62720, grid=grid(62720), stream=stream0)
        del primals_84
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf524 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf523, buf524, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf525 = buf499; del buf499  # reuse
        buf526 = buf498; del buf498  # reuse
        buf527 = buf497; del buf497  # reuse
        # Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf524, buf525, buf526, buf527, 3840, 98, grid=grid(3840), stream=stream0)
        buf528 = buf501; del buf501  # reuse
        buf529 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf531 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf525, buf526, buf527, primals_302, primals_303, buf528, buf529, buf531, primals_302, primals_303, 960, 4, grid=grid(960), stream=stream0)
        del primals_302
        del primals_303
        buf532 = reinterpret_tensor(buf523, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf523  # reuse
        buf533 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234, x_237], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_110.run(buf524, buf528, buf529, primals_85, primals_86, buf532, buf533, 376320, grid=grid(376320), stream=stream0)
        del primals_86
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_166, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf534, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf535 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf534, buf535, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf536 = buf527; del buf527  # reuse
        buf537 = buf526; del buf526  # reuse
        buf538 = buf525; del buf525  # reuse
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf535, buf536, buf537, buf538, 3840, 98, grid=grid(3840), stream=stream0)
        buf539 = buf529; del buf529  # reuse
        buf540 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf542 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf536, buf537, buf538, primals_305, primals_306, buf539, buf540, buf542, primals_305, primals_306, 960, 4, grid=grid(960), stream=stream0)
        del primals_305
        del primals_306
        buf543 = reinterpret_tensor(buf534, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf534  # reuse
        buf544 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239, x_242], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_110.run(buf535, buf539, buf540, primals_87, primals_88, buf543, buf544, 376320, grid=grid(376320), stream=stream0)
        del primals_88
        buf545 = reinterpret_tensor(buf510, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf510  # reuse
        buf546 = reinterpret_tensor(buf545, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf545  # reuse
        # Source Nodes: [x_se_28], Original ATen: [aten.mean]
        triton_per_fused_mean_111.run(buf546, buf544, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf547 = extern_kernels.convolution(buf546, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf547, (8, 240, 1, 1), (240, 1, 1, 1))
        buf548 = reinterpret_tensor(buf547, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf547  # reuse
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_112.run(buf548, primals_168, 1920, grid=grid(1920), stream=stream0)
        del primals_168
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf549, (8, 960, 1, 1), (960, 1, 1, 1))
        buf550 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf578 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_113.run(buf549, primals_170, buf550, buf578, 7680, grid=grid(7680), stream=stream0)
        del primals_170
        buf551 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten.mul]
        triton_poi_fused_mul_114.run(buf544, buf550, buf551, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (8, 160, 7, 7), (7840, 49, 7, 1))
        buf553 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_103.run(buf552, buf553, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf554 = buf517; del buf517  # reuse
        buf555 = buf516; del buf516  # reuse
        buf556 = buf515; del buf515  # reuse
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_104.run(buf553, buf554, buf555, buf556, 640, 98, grid=grid(640), stream=stream0)
        buf557 = buf519; del buf519  # reuse
        buf558 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf560 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_105.run(buf554, buf555, buf556, primals_308, primals_309, buf557, buf558, buf560, primals_308, primals_309, 160, 4, grid=grid(160), stream=stream0)
        del buf554
        del buf555
        del buf556
        del primals_308
        del primals_309
        buf561 = reinterpret_tensor(buf552, (8, 160, 7, 7), (7840, 1, 1120, 160), 0); del buf552  # reuse
        # Source Nodes: [shortcut_15, x_245], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_115.run(buf553, buf557, buf558, primals_89, primals_90, buf522, buf561, 62720, grid=grid(62720), stream=stream0)
        del buf558
        del primals_90
        # Source Nodes: [x_250], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf563 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_107.run(buf562, buf563, 7680, 49, grid=grid(7680, 49), stream=stream0)
        buf564 = buf538; del buf538  # reuse
        buf565 = buf537; del buf537  # reuse
        buf566 = buf536; del buf536  # reuse
        # Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_108.run(buf563, buf564, buf565, buf566, 3840, 98, grid=grid(3840), stream=stream0)
        buf567 = buf540; del buf540  # reuse
        buf568 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf570 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_109.run(buf564, buf565, buf566, primals_311, primals_312, buf567, buf568, buf570, primals_311, primals_312, 960, 4, grid=grid(960), stream=stream0)
        del buf564
        del buf565
        del buf566
        del primals_311
        del primals_312
        buf571 = reinterpret_tensor(buf562, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf562  # reuse
        # Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_116.run(buf563, buf567, buf568, primals_91, primals_92, buf571, 376320, grid=grid(376320), stream=stream0)
        del buf568
        del primals_92
        buf572 = reinterpret_tensor(buf549, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf549  # reuse
        buf573 = reinterpret_tensor(buf572, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf572  # reuse
        # Source Nodes: [x_256, x_257], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_117.run(buf573, buf571, 7680, 49, grid=grid(7680), stream=stream0)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 1280, 1, 1), (1280, 1, 1, 1))
        buf575 = reinterpret_tensor(buf574, (8, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf574  # reuse
        buf576 = empty((8, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred, x_260, x_261], Original ATen: [aten.convolution, aten.hardswish, aten.view]
        triton_poi_fused_convolution_hardswish_view_118.run(buf575, primals_174, buf576, 10240, grid=grid(10240), stream=stream0)
        del primals_174
        buf577 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_94, buf576, reinterpret_tensor(primals_93, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf577)
        del primals_94
        buf580 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_119.run(buf471, primals_156, buf580, 5376, grid=grid(5376), stream=stream0)
        del buf471
        del primals_156
        buf581 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_119.run(buf432, primals_149, buf581, 5376, grid=grid(5376), stream=stream0)
        del buf432
        del primals_149
        buf582 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_120.run(buf392, primals_142, buf582, 3840, grid=grid(3840), stream=stream0)
        del buf392
        del primals_142
        buf583 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_121.run(buf224, primals_123, buf583, 960, grid=grid(960), stream=stream0)
        del buf224
        del primals_123
        buf584 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_121.run(buf186, primals_116, buf584, 960, grid=grid(960), stream=stream0)
        del buf186
        del primals_116
        buf585 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_122.run(buf148, primals_109, buf585, 576, grid=grid(576), stream=stream0)
        del buf148
        del primals_109
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_175, primals_175, 1, grid=grid(1), stream=stream0)
        del primals_175
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_178, primals_178, 1, grid=grid(1), stream=stream0)
        del primals_178
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_181, primals_181, 1, grid=grid(1), stream=stream0)
        del primals_181
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_184, primals_184, 1, grid=grid(1), stream=stream0)
        del primals_184
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_187, primals_187, 1, grid=grid(1), stream=stream0)
        del primals_187
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_190, primals_190, 1, grid=grid(1), stream=stream0)
        del primals_190
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_193, primals_193, 1, grid=grid(1), stream=stream0)
        del primals_193
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_196, primals_196, 1, grid=grid(1), stream=stream0)
        del primals_196
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_199, primals_199, 1, grid=grid(1), stream=stream0)
        del primals_199
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_202, primals_202, 1, grid=grid(1), stream=stream0)
        del primals_202
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_205, primals_205, 1, grid=grid(1), stream=stream0)
        del primals_205
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_208, primals_208, 1, grid=grid(1), stream=stream0)
        del primals_208
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_211, primals_211, 1, grid=grid(1), stream=stream0)
        del primals_211
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_214, primals_214, 1, grid=grid(1), stream=stream0)
        del primals_214
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_217, primals_217, 1, grid=grid(1), stream=stream0)
        del primals_217
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_220, primals_220, 1, grid=grid(1), stream=stream0)
        del primals_220
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_223, primals_223, 1, grid=grid(1), stream=stream0)
        del primals_223
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_226, primals_226, 1, grid=grid(1), stream=stream0)
        del primals_226
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_229, primals_229, 1, grid=grid(1), stream=stream0)
        del primals_229
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_232, primals_232, 1, grid=grid(1), stream=stream0)
        del primals_232
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_235, primals_235, 1, grid=grid(1), stream=stream0)
        del primals_235
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_238, primals_238, 1, grid=grid(1), stream=stream0)
        del primals_238
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_241, primals_241, 1, grid=grid(1), stream=stream0)
        del primals_241
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_244, primals_244, 1, grid=grid(1), stream=stream0)
        del primals_244
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_247, primals_247, 1, grid=grid(1), stream=stream0)
        del primals_247
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_250, primals_250, 1, grid=grid(1), stream=stream0)
        del primals_250
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_253, primals_253, 1, grid=grid(1), stream=stream0)
        del primals_253
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_256, primals_256, 1, grid=grid(1), stream=stream0)
        del primals_256
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_259, primals_259, 1, grid=grid(1), stream=stream0)
        del primals_259
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_262, primals_262, 1, grid=grid(1), stream=stream0)
        del primals_262
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_265, primals_265, 1, grid=grid(1), stream=stream0)
        del primals_265
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_268, primals_268, 1, grid=grid(1), stream=stream0)
        del primals_268
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_271, primals_271, 1, grid=grid(1), stream=stream0)
        del primals_271
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_274, primals_274, 1, grid=grid(1), stream=stream0)
        del primals_274
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_277, primals_277, 1, grid=grid(1), stream=stream0)
        del primals_277
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_280, primals_280, 1, grid=grid(1), stream=stream0)
        del primals_280
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_283, primals_283, 1, grid=grid(1), stream=stream0)
        del primals_283
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_286, primals_286, 1, grid=grid(1), stream=stream0)
        del primals_286
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_289, primals_289, 1, grid=grid(1), stream=stream0)
        del primals_289
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_292, primals_292, 1, grid=grid(1), stream=stream0)
        del primals_292
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_123.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        return (buf577, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, buf0, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_169, primals_171, primals_172, primals_173, buf1, buf3, buf13, buf14, buf15, buf17, buf27, buf28, buf30, buf40, buf41, buf43, buf53, buf54, buf56, buf66, buf67, buf69, buf79, buf80, buf82, buf92, buf93, buf95, buf105, buf106, buf108, buf118, buf119, buf121, buf131, buf132, buf134, buf141, buf142, buf145, buf147, buf149, buf150, buf152, buf159, buf160, buf162, buf169, buf170, buf172, buf179, buf180, buf183, buf185, buf187, buf188, buf190, buf197, buf198, buf200, buf207, buf208, buf210, buf217, buf218, buf221, buf223, buf225, buf226, buf228, buf235, buf236, buf238, buf245, buf246, buf247, buf249, buf256, buf257, buf258, buf260, buf267, buf268, buf270, buf277, buf278, buf279, buf281, buf288, buf289, buf290, buf292, buf299, buf300, buf302, buf309, buf310, buf311, buf313, buf320, buf321, buf322, buf324, buf331, buf332, buf334, buf341, buf342, buf343, buf345, buf352, buf353, buf354, buf356, buf363, buf364, buf366, buf373, buf374, buf375, buf377, buf384, buf385, buf386, buf389, buf391, buf393, buf394, buf396, buf403, buf404, buf406, buf413, buf414, buf415, buf417, buf424, buf425, buf426, buf429, buf431, buf433, buf434, buf436, buf443, buf444, buf446, buf453, buf454, buf455, buf457, buf464, buf465, buf466, buf468, buf470, buf472, buf473, buf475, buf482, buf483, buf485, buf492, buf493, buf494, buf496, buf503, buf504, buf505, buf507, buf509, buf511, buf512, buf514, buf521, buf522, buf524, buf531, buf532, buf533, buf535, buf542, buf543, buf544, buf546, buf548, buf550, buf551, buf553, buf560, buf561, buf563, buf570, buf571, buf573, buf575, buf576, reinterpret_tensor(primals_93, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf567, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf557, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf578, reinterpret_tensor(buf539, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf528, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf518, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf579, reinterpret_tensor(buf500, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf489, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf479, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf580, reinterpret_tensor(buf461, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf450, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf440, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf581, reinterpret_tensor(buf421, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf410, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf400, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf582, reinterpret_tensor(buf381, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf370, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf360, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf349, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf338, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf328, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf317, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf306, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf296, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf285, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf274, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf264, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf253, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf242, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf232, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf583, reinterpret_tensor(buf214, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf204, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf194, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf584, reinterpret_tensor(buf176, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf166, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf156, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf585, reinterpret_tensor(buf138, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf128, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf115, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf102, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf89, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf76, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf37, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_23 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_77 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_93 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_176 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_179 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_182 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_191 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_194 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_200 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_203 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_209 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_215 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_218 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_221 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_224 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_227 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_230 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_233 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_236 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_239 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_242 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_245 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_248 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_251 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_254 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_257 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_260 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_263 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_269 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_275 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_278 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_284 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_293 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv3_large_100', benchmark_compiled_module)
