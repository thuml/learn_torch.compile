
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


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxi55w5nc6najitl4qwloyvysdjyhmwfqnhbif6sfffw7o3jgj3.py
# Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_1 => add_17
# x_12 => add_13, add_16, mul_16, mul_22, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_add_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ub/cubauvvyd2jgcejxvnaaaqfrvztv7ahcrp7hobow5cnn4jfesxfq.py
# Source Nodes: [x_29], Original ATen: [aten.convolution]
# x_29 => convolution_5
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lzqcdium3oopngkzhf6gk7na5amo6ucglcgzfqtwkcytuxcbd5.py
# Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
# x_30 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fgcpxif5rvldjmt6za2rzeeinsjhn7hmayb4apnbbtgqss42yf.py
# Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
# x_30 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gy/cgydivbitpaeqg7jklkdbs3wvsmfsj7btcihfai5wnog344biejc.py
# Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
# x_30 => add_31, add_32, add_33, mul_39, mul_40, mul_41, mul_42, mul_43, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/hh/chh3xfa7ppv3ndwkb2m6gb6ggi67bqct6ytbd7r4sdoy7j2wbs5w.py
# Source Nodes: [x_30, x_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_30 => add_31, add_34, mul_38, mul_44, rsqrt_5, sub_5, var_mean_5
# x_33 => add_35, clamp_max_3, clamp_min_3, div_3, mul_45
triton_poi_fused__native_batch_norm_legit_functional_hardswish_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzelboz6drxtfze6bzuawfqmxlo24ftkbqsinwgopvrf365mggw.py
# Source Nodes: [x_34], Original ATen: [aten.convolution]
# x_34 => convolution_6
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/he/chejlgtxembqnb7vh7lhyosp7ugxz7ezs5f3yahixbm6jhx6i6rv.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => var_mean_6
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


# kernel path: /tmp/torchinductor_youkaichao/ij/cijhm5wbvt5trabahw4o7l3obxpjn43jkugyilw6cva4v5f3bxjg.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => var_mean_6
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


# kernel path: /tmp/torchinductor_youkaichao/wu/cwubcmvn5xpqplbtiind76t2iuojpysjaf7hsgvhi5qs7nikoqu5.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => add_37, add_38, add_39, mul_47, mul_48, mul_49, mul_50, mul_51, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqygjkccmsl7gsgfz6n3elc5zclntslprc5r7dhxsl5r32xa4zq.py
# Source Nodes: [x_35, x_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_35 => add_37, add_40, mul_46, mul_52, rsqrt_6, sub_6, var_mean_6
# x_38 => add_41, clamp_max_4, clamp_min_4, div_4, mul_53
triton_poi_fused__native_batch_norm_legit_functional_hardswish_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/ve/cvegdhuap23u3u6e55apqogd22sy6qq5p64vx3tsmzrtoriebkhm.py
# Source Nodes: [x_40], Original ATen: [aten.convolution]
# x_40 => convolution_7
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ek/cekpfi4muj4ljcaghbvdz6d4s2smlvplgpd3vtw72ltmg7uooppr.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
# x_41 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4e/c4eaibn3tjfoqe5uhs27zprheup2am54t2gshk7rdwhxwb5mpou6.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
# x_41 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fl/cfl6bxwjwyvnzorjg6d5zm3wp43jybvfnz5wjhb4sxw3ip4hh3ln.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
# x_41 => add_43, add_44, add_45, mul_55, mul_56, mul_57, mul_58, mul_59, rsqrt_7, squeeze_22, var_mean_7
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ar/caraboy4hfzcsmv6oiuekkv26lfkbc4x4hfiftilgr3hpen6hqlc.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
# x_41 => add_43, add_46, mul_54, mul_60, rsqrt_7, sub_7, var_mean_7
triton_poi_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxrexynxntw36us34bzzc5mxs2ycwolwywdswxqdwo3bmnzvnp7.py
# Source Nodes: [x_45], Original ATen: [aten.convolution]
# x_45 => convolution_8
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/un/cunof2kjohx2r7vwlcdh4sgpdq2tggqqcjjf6he7afxhfi6dnhae.py
# Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
# x_46 => var_mean_8
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


# kernel path: /tmp/torchinductor_youkaichao/7w/c7w5zf6rsyal56hxfigvdu46gnpwhhyfkpqu5bzb2czj6tp7qdog.py
# Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
# x_46 => var_mean_8
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


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5j3cgsmwzsiew4lxjxurkuflwr44rpjgh6rbedhwphgodlo4qd.py
# Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
# x_46 => add_48, add_49, add_50, mul_62, mul_63, mul_64, mul_65, mul_66, rsqrt_8, squeeze_25, var_mean_8
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/eg/ceg6dsbdb2sa5up5unkb7ccjcc45jr7eiboxwrpvr253mdriuv6y.py
# Source Nodes: [x_46, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_46 => add_48, add_51, mul_61, mul_67, rsqrt_8, sub_8, var_mean_8
# x_49 => add_52, clamp_max_5, clamp_min_5, div_5, mul_68
triton_poi_fused__native_batch_norm_legit_functional_hardswish_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/ma/cma7whrisdj4g7wig3chkh6s7b6jobgjrbx76hcialpl7epffot3.py
# Source Nodes: [shortcut_4, x_57], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_4 => add_64
# x_57 => add_60, add_63, mul_77, mul_83, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_add_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ve/cve6ybsdfilrt5dnx2tlrdbfnhsrxiygcu3ephah7idoyvbi5v6d.py
# Source Nodes: [x_96], Original ATen: [aten.convolution]
# x_96 => convolution_17
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyyqykorp53izj44bug4n6wex46bezqbh2g2zhvt5yzz7qr7h6f.py
# Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
# x_97 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/me/cmeif4ejjsbesmdp2v2jesrwsjbgf77kiz5bripbjdt2nlzkkfzk.py
# Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
# x_97 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/td/ctd5tjcwjoimlipdlmw2mszditp6fucjksmrpifc4npdmaaj5tta.py
# Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
# x_97 => add_102, add_103, add_104, mul_131, mul_132, mul_133, mul_134, mul_135, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kppvmud77bi6emjtals47rcm2gamsrc6dce4z376ur6mioqpp5.py
# Source Nodes: [x_100, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_100 => add_106, clamp_max_11, clamp_min_11, div_11, mul_137
# x_97 => add_102, add_105, mul_130, mul_136, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_hardswish_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4mmijhaqagcqvepllfnnlfnl2ch4e7ozuqeb5d6dthigtsesyr.py
# Source Nodes: [x_101], Original ATen: [aten.convolution]
# x_101 => convolution_18
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp72crkqqgposhxvqezm6rgirwbqdlps7eh7ulbkh4ojpijju7cy.py
# Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
# x_102 => var_mean_18
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrvfgqku7j6mhbecyr2cooprjlms7n22x3kbsunfxif74iq7h6n.py
# Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
# x_102 => add_108, add_109, add_110, mul_139, mul_140, mul_141, mul_142, mul_143, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/yn/cynsz3zgpmceacpq2qwds4dghm6xmgojp4dppnzlpalxqqo2dciq.py
# Source Nodes: [x_102, x_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_102 => add_108, add_111, mul_138, mul_144, rsqrt_18, sub_18, var_mean_18
# x_105 => add_112, clamp_max_12, clamp_min_12, div_12, mul_145
triton_poi_fused__native_batch_norm_legit_functional_hardswish_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_37', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tksicfa7j3votawtvvqe2ba6kqn5razfkftnyk77suyzltd77i.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_red_fused_mean_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_38', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvrphtajbygodxpkippl5xjpqr2u66zfuqmgohhvcki5bldenbb.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_per_fused_mean_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_39', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/oa/coaajpozv3t742cr6yjt2ji6im27u3v4ohov3bahfzyllcfvswt6.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.hardswish]
# x_se_1 => convolution_19
# x_se_2 => add_113, clamp_max_13, clamp_min_13, div_13, mul_146
triton_poi_fused_convolution_hardswish_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_40', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7w4cra4ybr4nppmvclrni2lnj227azi7uv2eh5dm56ober64vz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => add_114, clamp_max_14, clamp_min_14, div_14
# x_se_3 => convolution_20
triton_poi_fused_convolution_hardsigmoid_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_41', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/st/cst2xnvrqf527sc4j64h6cb5efgbdpydnzs3deuh32mhxq27l2ik.py
# Source Nodes: [x_106], Original ATen: [aten.mul]
# x_106 => mul_147
triton_poi_fused_mul_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_42', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dbfpgugw6ep3vfmaudjo6hx7ya7c5hayk4xttvf7asrvkz2r2m.py
# Source Nodes: [x_107], Original ATen: [aten.convolution]
# x_107 => convolution_21
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/iz/cizspwfv5pfqcjx2dfkb7czgy6cxg4aceuusvrirfnn6eeglojdu.py
# Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
# x_108 => var_mean_19
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czvqmeuiqzbrght4awziqvi2fot2mqlipcyhslaux44zt7ar2wby.py
# Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
# x_108 => add_116, add_117, add_118, mul_149, mul_150, mul_151, mul_152, mul_153, rsqrt_19, squeeze_58, var_mean_19
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/2a/c2a4hlfxl5lvni4s72nhe2fuyz4qxlu2n3xeuumhe72bd64zzvsi.py
# Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
# x_108 => add_116, add_119, mul_148, mul_154, rsqrt_19, sub_19, var_mean_19
triton_poi_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2rgmrgt2yh6h7jzqaskvgdbbiqezjvpzbdvmhluv5bzwc4rcxy.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.hardswish]
# x_se_5 => convolution_24
# x_se_6 => add_132, clamp_max_17, clamp_min_17, div_17, mul_171
triton_poi_fused_convolution_hardswish_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5eucbn6o52ekhnwhmcjcox5eqomqkczinl3jufn2uzyqnvq4bv.py
# Source Nodes: [shortcut_8, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_8 => add_139
# x_124 => add_135, add_138, mul_173, mul_179, rsqrt_22, sub_22, var_mean_22
triton_poi_fused__native_batch_norm_legit_functional_add_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_48', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/aq/caqqrcz5fbhrt7bszz7rlvxopw7w5ynun6e2czoi6lmfr43o7s5t.py
# Source Nodes: [x_180], Original ATen: [aten.convolution]
# x_180 => convolution_42
triton_poi_fused_convolution_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1600
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (200*x2) + (156800*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/cal6dwba22lekaeckjmu23t66jrxgarvff2z3ump5lm5efhxw4ph.py
# Source Nodes: [x_181], Original ATen: [aten._native_batch_norm_legit_functional]
# x_181 => var_mean_32
triton_red_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9800
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (200*r2) + (25600*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtkznyrrdxcbvviui5jxdkhndrh6fixa3gtb6hqn37qqnadgsqj.py
# Source Nodes: [x_181], Original ATen: [aten._native_batch_norm_legit_functional]
# x_181 => add_201, add_202, add_203, mul_256, mul_257, mul_258, mul_259, mul_260, rsqrt_32, squeeze_97, var_mean_32
triton_per_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/lv/clv253jjwadcfwnb63pw6ckfszyhtbj37ge7hr6nrkp744r4cxbm.py
# Source Nodes: [x_181, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_181 => add_201, add_204, mul_255, mul_261, rsqrt_32, sub_32, var_mean_32
# x_184 => add_205, clamp_max_31, clamp_min_31, div_31, mul_262
triton_poi_fused__native_batch_norm_legit_functional_hardswish_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
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
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jx/cjxtp2lwhpzkdkrpyfmn27vvtfm6fbdd4znwc24wnu3si2fsisia.py
# Source Nodes: [x_185], Original ATen: [aten.convolution]
# x_185 => convolution_43
triton_poi_fused_convolution_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/et/cetvw2dljgdrhlibvrutbyepbmormgnaqo7q6cxguljdvmruisof.py
# Source Nodes: [x_186], Original ATen: [aten._native_batch_norm_legit_functional]
# x_186 => var_mean_33
triton_red_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/oj/cojo5eptjtcwheyvebosi5j4j75lpwfd2cppopqot7hqj6odcour.py
# Source Nodes: [x_186], Original ATen: [aten._native_batch_norm_legit_functional]
# x_186 => add_207, add_208, add_209, mul_264, mul_265, mul_266, mul_267, mul_268, rsqrt_33, squeeze_100, var_mean_33
triton_per_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jyylcwqjo6r2m2apbxmgrfqic3ml63x5z4mtndfb3d2onvbens.py
# Source Nodes: [x_186, x_189], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_186 => add_207, add_210, mul_263, mul_269, rsqrt_33, sub_33, var_mean_33
# x_189 => add_211, clamp_max_32, clamp_min_32, div_32, mul_270
triton_poi_fused__native_batch_norm_legit_functional_hardswish_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_56', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rw/crwdbv4kxbvpyzczkczkwqvrnlhqzcddrwop6q2tqwhswsez7g3s.py
# Source Nodes: [x_191], Original ATen: [aten.convolution]
# x_191 => convolution_44
triton_poi_fused_convolution_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (72*x2) + (14112*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chf5izmfwhg73luv2an7gv65h6oaysbx5iq47kzux4rcyu3wdbvt.py
# Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
# x_192 => var_mean_34
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 936
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 72)
    x0 = xindex % 72
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
        tmp3 = tl.load(in_ptr0 + (x0 + (72*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xe/cxec2ivavjgmkrk7gkwddy2244sdvoscb4zb4lq3cv7zcq6lpuoa.py
# Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
# x_192 => add_213, add_214, add_215, mul_272, mul_273, mul_274, mul_275, mul_276, rsqrt_34, squeeze_103, var_mean_34
triton_per_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvcfm3kjx3e7sxri4w5o32s37hpcabq3ms3kt7rd6q5r6xki4ue.py
# Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
# x_192 => add_213, add_216, mul_271, mul_277, rsqrt_34, sub_34, var_mean_34
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
    xnumel = 112896
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


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4t5wyq2lpjjvz46bjogidhkry3h3a2dwg4xkgtolqfh4mbiqug.py
# Source Nodes: [x_196], Original ATen: [aten.convolution]
# x_196 => convolution_45
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 216
    y1 = (yindex // 216)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (216*x2) + (42336*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv52koa4umv5r2j5m7zexqqp5ubwqkkqlk5jwxladuvzdgod5hiz.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => var_mean_35
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2808
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 216)
    x0 = xindex % 216
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
        tmp3 = tl.load(in_ptr0 + (x0 + (216*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/du/cdusgvy4iy6j4x5vtnqnqhyqrhufqqvay5dze3grn2dza7zx645i.py
# Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
# x_197 => add_218, add_219, add_220, mul_279, mul_280, mul_281, mul_282, mul_283, rsqrt_35, squeeze_106, var_mean_35
triton_per_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 216
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (216*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (216*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (216*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/iz/cizmdfitcrcaacaaqkwrg4ugdj72deb6rhpalivldw4caqmi6nqr.py
# Source Nodes: [x_197, x_200], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_197 => add_218, add_221, mul_278, mul_284, rsqrt_35, sub_35, var_mean_35
# x_200 => add_222, clamp_max_33, clamp_min_33, div_33, mul_285
triton_poi_fused__native_batch_norm_legit_functional_hardswish_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 338688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 216
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7doeldvh6ct3a2bf2f524ta7gyqu77tn6sok2cqjons5z7vsbr2.py
# Source Nodes: [shortcut_13, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_13 => add_234
# x_208 => add_230, add_233, mul_294, mul_300, rsqrt_37, sub_37, var_mean_37
triton_poi_fused__native_batch_norm_legit_functional_add_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
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


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnyk64u3glasepiebrcbwklld34lluh2dioigcgbbptydvryo7u.py
# Source Nodes: [x_264], Original ATen: [aten.convolution]
# x_264 => convolution_57
triton_poi_fused_convolution_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2880
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 360
    y1 = (yindex // 360)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (360*x2) + (70560*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tb/ctbmvfa2vietcjaefanwo5rzoeqr5kwxt54cilza3lb7fueqwcbm.py
# Source Nodes: [x_265], Original ATen: [aten._native_batch_norm_legit_functional]
# x_265 => var_mean_47
triton_red_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4680
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 360)
    x0 = xindex % 360
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
        tmp3 = tl.load(in_ptr0 + (x0 + (360*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gaf27y2zaenudv4jehp2pwpenzrd5u4kujaglw6tku6lms4tvi.py
# Source Nodes: [x_265], Original ATen: [aten._native_batch_norm_legit_functional]
# x_265 => add_290, add_291, add_292, mul_371, mul_372, mul_373, mul_374, mul_375, rsqrt_47, squeeze_142, var_mean_47
triton_per_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 360
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (360*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (360*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (360*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrfl3lg5o4uirqzeo2i4rct3aiokuinvpbnt42n4nz6tavatf2h.py
# Source Nodes: [x_265, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_265 => add_290, add_293, mul_370, mul_376, rsqrt_47, sub_47, var_mean_47
# x_268 => add_294, clamp_max_41, clamp_min_41, div_41, mul_377
triton_poi_fused__native_batch_norm_legit_functional_hardswish_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 564480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 360
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


# kernel path: /tmp/torchinductor_youkaichao/un/cun4voqpcolrpzgyxotksfmqe2voz6rzxecgm3bthbzt2fpymqk2.py
# Source Nodes: [x_se_20], Original ATen: [aten.mean]
# x_se_20 => mean_5
triton_red_fused_mean_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 360
    x1 = (xindex // 360)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (360*r2) + (35280*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrvkvxbb3skfii5bzvkwboyf3vfrudrtvxs4wwywfokh32bojou.py
# Source Nodes: [x_se_20], Original ATen: [aten.mean]
# x_se_20 => mean_5
triton_per_fused_mean_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_71', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2880
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 360
    x1 = (xindex // 360)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (360*r2) + (720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvtzjc2pj67amqg4dezgm76vzeqhl3yfw43vbomvajjc5skhnp7.py
# Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.hardswish]
# x_se_21 => convolution_59
# x_se_22 => add_301, clamp_max_43, clamp_min_43, div_43, mul_386
triton_poi_fused_convolution_hardswish_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chltytqrm6bcn4whhunt6wycbln6olwnsu5lnitniztvql44mut5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => add_302, clamp_max_44, clamp_min_44, div_44
# x_se_23 => convolution_60
triton_poi_fused_convolution_hardsigmoid_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 360
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


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7fwxfevpnnppdno7ogfjln2wtwrixc6mrmtvtdf7tqhwwokark.py
# Source Nodes: [x_274], Original ATen: [aten.mul]
# x_274 => mul_387
triton_poi_fused_mul_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 564480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 360
    x2 = (xindex // 70560)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (360*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65c5gmvr7rnrwxdc6cf7up6k7opl7xtpfl2plj3lv6cm7tnofjq.py
# Source Nodes: [x_275], Original ATen: [aten.convolution]
# x_275 => convolution_61
triton_poi_fused_convolution_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_75', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfegh2sj3hzpfhjc2x3z7eoqmkse67xclpq7hrmsw2rxpuh4ssxl.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
# x_276 => var_mean_49
triton_red_fused__native_batch_norm_legit_functional_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1560
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
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
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrxzneaniehbtc2gmjgs5yigqvbbfszkjwfjs7vaqd7uwuyuqd7.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
# x_276 => add_304, add_305, add_306, mul_389, mul_390, mul_391, mul_392, mul_393, rsqrt_49, squeeze_148, var_mean_49
triton_per_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/a4/ca44htwfnesf22x43emdzim3567kq2jidbbspeyaebbc7uwko4eg.py
# Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
# x_276 => add_304, add_307, mul_388, mul_394, rsqrt_49, sub_49, var_mean_49
triton_poi_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
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


# kernel path: /tmp/torchinductor_youkaichao/ke/ckerypzwaqpaxb4gz7ze7rbamup4t25ij6ccvped6pcblwurtsgx.py
# Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.hardswish]
# x_se_25 => convolution_64
# x_se_26 => add_320, clamp_max_47, clamp_min_47, div_47, mul_411
triton_poi_fused_convolution_hardswish_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2lfronqlblxoqlvexayba5ff5lim274ybgavdgmxtiy65f6tzu.py
# Source Nodes: [shortcut_18, x_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_18 => add_327
# x_292 => add_323, add_326, mul_413, mul_419, rsqrt_52, sub_52, var_mean_52
triton_poi_fused__native_batch_norm_legit_functional_add_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jfxl7cy6t56hzkipsayf6phr2zmlvq6j3tn3lljonf4ibkkghg.py
# Source Nodes: [x_365], Original ATen: [aten.convolution]
# x_365 => convolution_87
triton_poi_fused_convolution_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5760
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 720
    y1 = (yindex // 720)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (720*x2) + (141120*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cdu2lqwnm737etijgir7dxjjte6dgrryfufyylao6t75fn3pz3j4.py
# Source Nodes: [x_366], Original ATen: [aten._native_batch_norm_legit_functional]
# x_366 => var_mean_65
triton_red_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9360
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 720)
    x0 = xindex % 720
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
        tmp3 = tl.load(in_ptr0 + (x0 + (720*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6ct54m3eeu7uq6nifmnnz3wphfonh3o7s7x2qqwplaivvu2yeg.py
# Source Nodes: [x_366], Original ATen: [aten._native_batch_norm_legit_functional]
# x_366 => add_409, add_410, add_411, mul_521, mul_522, mul_523, mul_524, mul_525, rsqrt_65, squeeze_196, var_mean_65
triton_per_fused__native_batch_norm_legit_functional_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_83', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (720*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dy/cdy5ct2zwmnaxeho3lszu6fk3fbbkbge2yo6mdyo7sayyyuhewe7.py
# Source Nodes: [x_366, x_369], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_366 => add_409, add_412, mul_520, mul_526, rsqrt_65, sub_65, var_mean_65
# x_369 => add_413, clamp_max_65, clamp_min_65, div_65, mul_527
triton_poi_fused__native_batch_norm_legit_functional_hardswish_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1128960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 720
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3hxy76fh4xqkcps7dmh6msvodgxwtaf2jz3w55n4qtpp2we7fa.py
# Source Nodes: [x_370], Original ATen: [aten.convolution]
# x_370 => convolution_88
triton_poi_fused_convolution_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5760
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 720
    y1 = (yindex // 720)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (720*x2) + (35280*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5t4ugkeg6eypwfvtwcwzb3d2v6pefmxgl7dcecbpfy2xwyydiku.py
# Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
# x_371 => var_mean_66
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
    xnumel = 2880
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 720
    x1 = (xindex // 720)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (720*r2) + (70560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wpjed6nxekbz5ovu4kqj5r5tdsovlvmyzx264ceejatz6wbl6d.py
# Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
# x_371 => add_415, add_416, add_417, mul_529, mul_530, mul_531, mul_532, mul_533, rsqrt_66, squeeze_199, var_mean_66
triton_per_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (720*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (720*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ry/crym2nfbrvnd3qgsivfyfy5gk6r3ymdcrhdeiipizgizosfsesxp.py
# Source Nodes: [x_371, x_374], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_371 => add_415, add_418, mul_528, mul_534, rsqrt_66, sub_66, var_mean_66
# x_374 => add_419, clamp_max_66, clamp_min_66, div_66, mul_535
triton_poi_fused__native_batch_norm_legit_functional_hardswish_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 282240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 720
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


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2o36g67qbbnt2whvkwvuid2v5kcqhlxlbi2hiivuqjyv4yeiwh.py
# Source Nodes: [x_se_44], Original ATen: [aten.mean]
# x_se_44 => mean_11
triton_per_fused_mean_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_89', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5760
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 720
    x1 = (xindex // 720)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (720*r2) + (35280*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdnxwgzrrk6bho7za3kxffxfoeq74c3bmu7ol3bgx3dho5ufkzf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => add_421, clamp_max_68, clamp_min_68, div_68
# x_se_47 => convolution_90
triton_poi_fused_convolution_hardsigmoid_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 720
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


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvngeaarsddrvfhcagi7i3cwck7dxownma7c5dtpevovccmbt2k.py
# Source Nodes: [x_375], Original ATen: [aten.mul]
# x_375 => mul_537
triton_poi_fused_mul_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 282240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 720
    x2 = (xindex // 35280)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (720*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspfescsnewx6iw445pj6ojjatpovc4az7dqvhjww4v74uffecbk.py
# Source Nodes: [x_376], Original ATen: [aten.convolution]
# x_376 => convolution_91
triton_poi_fused_convolution_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (184*x2) + (9016*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2fofsdna33vgnirqboz2cehkhaymlq5fnoh4fxnrllcfr2zutr.py
# Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
# x_377 => var_mean_67
triton_red_fused__native_batch_norm_legit_functional_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 184
    x1 = (xindex // 184)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (184*r2) + (18032*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/th/cthjbouopnp3wic57ec75uzc4x2m73rkkqg5y6cqxiqtze5ulvdc.py
# Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
# x_377 => add_423, add_424, add_425, mul_539, mul_540, mul_541, mul_542, mul_543, rsqrt_67, squeeze_202, var_mean_67
triton_per_fused__native_batch_norm_legit_functional_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_94', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/q6/cq6ma2s3xl4pyochds5gimsqtpo45umnr4ir4seffzepmwprtfhy.py
# Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
# x_377 => add_423, add_426, mul_538, mul_544, rsqrt_67, sub_67, var_mean_67
triton_poi_fused__native_batch_norm_legit_functional_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72128
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


# kernel path: /tmp/torchinductor_youkaichao/ih/cihofu3gt7vyirbtxtfegixb56vmgjhba55pyvywgh6mqm7x4nnm.py
# Source Nodes: [x_381], Original ATen: [aten.convolution]
# x_381 => convolution_92
triton_poi_fused_convolution_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5888
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 736
    y1 = (yindex // 736)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (736*x2) + (36064*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctjqn7zwzxmhgbzix6ncop3fy7gqpp4una3tkyhtdunnxhtf6wp4.py
# Source Nodes: [x_382], Original ATen: [aten._native_batch_norm_legit_functional]
# x_382 => var_mean_68
triton_red_fused__native_batch_norm_legit_functional_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 736
    x1 = (xindex // 736)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (736*r2) + (72128*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4a/c4alekzwl3ol7stj3jdquch2ybcfpziguh24b56q2aokcbruhflq.py
# Source Nodes: [x_382], Original ATen: [aten._native_batch_norm_legit_functional]
# x_382 => add_428, add_429, add_430, mul_546, mul_547, mul_548, mul_549, mul_550, rsqrt_68, squeeze_205, var_mean_68
triton_per_fused__native_batch_norm_legit_functional_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_98', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (736*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (736*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (736*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xu/cxudhg75n3vp7eijspjht2dcy6w33d6nh7hwmz4mkrcb3pe2dji5.py
# Source Nodes: [x_382, x_385], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_382 => add_428, add_431, mul_545, mul_551, rsqrt_68, sub_68, var_mean_68
# x_385 => add_432, clamp_max_69, clamp_min_69, div_69, mul_552
triton_poi_fused__native_batch_norm_legit_functional_hardswish_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 736
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


# kernel path: /tmp/torchinductor_youkaichao/fv/cfviao6bbzsmghyxkeiksh3iwekf3nvvshb6diojdlccz4ixwntx.py
# Source Nodes: [x_se_48], Original ATen: [aten.mean]
# x_se_48 => mean_12
triton_per_fused_mean_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_100', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 736
    x1 = (xindex // 736)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (736*r2) + (36064*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/cio24kumodhnieppycfzijhnvxiumgwzatbodds47t5ljfeplvf6.py
# Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.hardswish]
# x_se_49 => convolution_94
# x_se_50 => add_439, clamp_max_71, clamp_min_71, div_71, mul_561
triton_poi_fused_convolution_hardswish_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_101', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cqu4ahwfcczfduzibwmmavwkg6tium6lyuoal3f4jxr5zqqnqgqu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_se_51], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => add_440, clamp_max_72, clamp_min_72, div_72
# x_se_51 => convolution_95
triton_poi_fused_convolution_hardsigmoid_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 736
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


# kernel path: /tmp/torchinductor_youkaichao/e7/ce75s3en4kmg7wwbrnebahzn76ukav6n32pucdonnfucdfisfpxn.py
# Source Nodes: [x_391], Original ATen: [aten.mul]
# x_391 => mul_562
triton_poi_fused_mul_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 736
    x2 = (xindex // 36064)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (736*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/cantojklko52grha4z5zrefm55ghnf7tuwd5klu75n64x4frogxq.py
# Source Nodes: [shortcut_24, x_393], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_24 => add_446
# x_393 => add_442, add_445, mul_563, mul_569, rsqrt_70, sub_70, var_mean_70
triton_poi_fused__native_batch_norm_legit_functional_add_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72128
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


# kernel path: /tmp/torchinductor_youkaichao/xr/cxritv43n5uwb33mk5togv3vhkjse7q6myxrud37i546e6zkp5a2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____5___se_gate, x_se_67], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
# getattr_getattr_l__mod___blocks___5_____5___se_gate => add_520, clamp_max_88, clamp_min_88, div_88
# x_se_67 => convolution_115
triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 736
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


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmegvhi4sgvhqzusr7fxuhyqlfkqbukbf4t36dnsdfe7kpgkplv.py
# Source Nodes: [x_466], Original ATen: [aten.convolution]
# x_466 => convolution_117
triton_poi_fused_convolution_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8832
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1104
    y1 = (yindex // 1104)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1104*x2) + (54096*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3ahhywa2djyyidqw3zqqolwrqzrk2qjjftgdoxo3rpiebpmhk5l.py
# Source Nodes: [x_467], Original ATen: [aten._native_batch_norm_legit_functional]
# x_467 => var_mean_83
triton_red_fused__native_batch_norm_legit_functional_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4416
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1104*r2) + (108192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7wedutfcdwja6mtkkajabfaobu5xp6rpdxdapfcmcuuemfdgg6.py
# Source Nodes: [x_467], Original ATen: [aten._native_batch_norm_legit_functional]
# x_467 => add_528, add_529, add_530, mul_671, mul_672, mul_673, mul_674, mul_675, rsqrt_83, squeeze_250, var_mean_83
triton_per_fused__native_batch_norm_legit_functional_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_108', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1104
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1104*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1104*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bc/cbcp6peouzkirv3acgup63l6fvsn4tlk4da4eipihvnvcfnepqfe.py
# Source Nodes: [x_467, x_470], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# x_467 => add_528, add_531, mul_670, mul_676, rsqrt_83, sub_83, var_mean_83
# x_470 => add_532, clamp_max_89, clamp_min_89, div_89, mul_677
triton_poi_fused__native_batch_norm_legit_functional_hardswish_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1104
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


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyt77tgydh766zbrovaev6pdipbjv45jqhzzx7omfxbmgcicqlu.py
# Source Nodes: [x_se_68], Original ATen: [aten.mean]
# x_se_68 => mean_17
triton_per_fused_mean_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_110', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8832
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r2) + (54096*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4nrvlibdgfsqedycwr4j2dydk2d4ys37pv264dfzehfq3fbnae.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____6___se_gate, x_se_71], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
# getattr_getattr_l__mod___blocks___5_____6___se_gate => add_540, clamp_max_92, clamp_min_92, div_92
# x_se_71 => convolution_120
triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1104
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czz4h7rf3gdbdmuphb33edo4d7unxicuihi6bq4ps7ohnjr5d3z2.py
# Source Nodes: [x_476], Original ATen: [aten.mul]
# x_476 => mul_687
triton_poi_fused_mul_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 432768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1104
    x2 = (xindex // 54096)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (1104*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzcmgq4n5jo4ietqxmpa5r5nerlako4quvv4p3ed5sivj35g3ef.py
# Source Nodes: [x_477], Original ATen: [aten.convolution]
# x_477 => convolution_121
triton_poi_fused_convolution_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (224*x2) + (10976*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztjyu3euhmsnsdfdprsnji3hyqdvoki3paycn2o57buog3ncctp.py
# Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_functional]
# x_478 => var_mean_85
triton_red_fused__native_batch_norm_legit_functional_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 224
    x1 = (xindex // 224)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (224*r2) + (21952*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sz/cszlepmclqnwivayqx6qud4lmxbabizz6btt7gi7wiws3jg2pvkd.py
# Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_functional]
# x_478 => add_542, add_543, add_544, mul_689, mul_690, mul_691, mul_692, mul_693, rsqrt_85, squeeze_256, var_mean_85
triton_per_fused__native_batch_norm_legit_functional_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_115', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (224*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (224*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnakyywbkup5lk3riwwphx65cmtw4rob2zu4jvjbgmbvkki47ql.py
# Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_functional]
# x_478 => add_542, add_545, mul_688, mul_694, rsqrt_85, sub_85, var_mean_85
triton_poi_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3ec2b2gja23sgrlbflplj3r4lobkqzp4t4qx3io54bo3apglsc.py
# Source Nodes: [x_482], Original ATen: [aten.convolution]
# x_482 => convolution_122
triton_poi_fused_convolution_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_117', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10752
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1344
    y1 = (yindex // 1344)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1344*x2) + (65856*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7wgn3aagd37p5ujyhqzcz6wikuhvgicl5a5gevryh7ef3xo2ni.py
# Source Nodes: [x_483], Original ATen: [aten._native_batch_norm_legit_functional]
# x_483 => var_mean_86
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
    xnumel = 5376
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1344
    x1 = (xindex // 1344)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1344*r2) + (131712*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/o2/co2un2rd3ocstntoz4rfbh2qbrwvgyqku6jqqxadapcrweoieil4.py
# Source Nodes: [x_483], Original ATen: [aten._native_batch_norm_legit_functional]
# x_483 => add_547, add_548, add_549, mul_696, mul_697, mul_698, mul_699, mul_700, rsqrt_86, squeeze_259, var_mean_86
triton_per_fused__native_batch_norm_legit_functional_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_119', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1344*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1344*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1344*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5artcy5eggm5mhilcyoes4gz5c2gnc3ljsafa25o5zdv6seyzx.py
# Source Nodes: [x_483], Original ATen: [aten._native_batch_norm_legit_functional]
# x_483 => add_547, add_550, mul_695, mul_701, rsqrt_86, sub_86, var_mean_86
triton_poi_fused__native_batch_norm_legit_functional_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1344
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


# kernel path: /tmp/torchinductor_youkaichao/qe/cqegqdo3pxcu34gq3nv2rq5oijk2lfd66llfu457dcl7b2ojx5bn.py
# Source Nodes: [x_488, x_489], Original ATen: [aten.hardswish, aten.mean]
# x_488 => add_551, clamp_max_93, clamp_min_93, div_93, mul_702
# x_489 => mean_18
triton_per_fused_hardswish_mean_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_121', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1344
    x1 = (xindex // 1344)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1344*r2) + (65856*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czo6h64x3sfz4s7kmrsyayri47dogs7li7op6dyayxaauetfhyos.py
# Source Nodes: [pred, x_493], Original ATen: [aten.hardswish, aten.view]
# pred => view_1
# x_493 => add_552, clamp_max_94, clamp_min_94, div_94, mul_703
triton_poi_fused_hardswish_view_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_view_122', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cruslnmgeemfycmbt4aqzjgbd2muul52hwt6oolgdoxfgxbqqe72.py
# Source Nodes: [x_se_63], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_63 => convolution_110
triton_poi_fused_convolution_hardsigmoid_backward_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 736
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


# kernel path: /tmp/torchinductor_youkaichao/re/crekwfldpxkrjvrjggx442nvewdaefpx4dapxjdtqtexjnpczt3y.py
# Source Nodes: [x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_47 => convolution_90
triton_poi_fused_convolution_hardsigmoid_backward_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 720
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


# kernel path: /tmp/torchinductor_youkaichao/ni/cni4e2llhweo5lszrh5lbba2ekinpimvhlmb77y2owhuvxgtv6xz.py
# Source Nodes: [x_se_43], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_43 => convolution_85
triton_poi_fused_convolution_hardsigmoid_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_125', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 360
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


# kernel path: /tmp/torchinductor_youkaichao/rk/crkovksqyxa7icxfyefl4uqpvddozn4hhgshmnjleinpmjys2c3n.py
# Source Nodes: [x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_19 => convolution_40
triton_poi_fused_convolution_hardsigmoid_backward_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_126', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhyqyl2rt47yvbnrvm472fomd5mdzjtqam5xzkski2sy6noknw7.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_127', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (24, ), (1, ))
    assert_size_stride(primals_16, (24, ), (1, ))
    assert_size_stride(primals_17, (48, ), (1, ))
    assert_size_stride(primals_18, (48, ), (1, ))
    assert_size_stride(primals_19, (48, ), (1, ))
    assert_size_stride(primals_20, (48, ), (1, ))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_22, (24, ), (1, ))
    assert_size_stride(primals_23, (48, ), (1, ))
    assert_size_stride(primals_24, (48, ), (1, ))
    assert_size_stride(primals_25, (48, ), (1, ))
    assert_size_stride(primals_26, (48, ), (1, ))
    assert_size_stride(primals_27, (24, ), (1, ))
    assert_size_stride(primals_28, (24, ), (1, ))
    assert_size_stride(primals_29, (48, ), (1, ))
    assert_size_stride(primals_30, (48, ), (1, ))
    assert_size_stride(primals_31, (48, ), (1, ))
    assert_size_stride(primals_32, (48, ), (1, ))
    assert_size_stride(primals_33, (24, ), (1, ))
    assert_size_stride(primals_34, (24, ), (1, ))
    assert_size_stride(primals_35, (120, ), (1, ))
    assert_size_stride(primals_36, (120, ), (1, ))
    assert_size_stride(primals_37, (120, ), (1, ))
    assert_size_stride(primals_38, (120, ), (1, ))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_40, (40, ), (1, ))
    assert_size_stride(primals_41, (120, ), (1, ))
    assert_size_stride(primals_42, (120, ), (1, ))
    assert_size_stride(primals_43, (120, ), (1, ))
    assert_size_stride(primals_44, (120, ), (1, ))
    assert_size_stride(primals_45, (40, ), (1, ))
    assert_size_stride(primals_46, (40, ), (1, ))
    assert_size_stride(primals_47, (120, ), (1, ))
    assert_size_stride(primals_48, (120, ), (1, ))
    assert_size_stride(primals_49, (120, ), (1, ))
    assert_size_stride(primals_50, (120, ), (1, ))
    assert_size_stride(primals_51, (40, ), (1, ))
    assert_size_stride(primals_52, (40, ), (1, ))
    assert_size_stride(primals_53, (120, ), (1, ))
    assert_size_stride(primals_54, (120, ), (1, ))
    assert_size_stride(primals_55, (120, ), (1, ))
    assert_size_stride(primals_56, (120, ), (1, ))
    assert_size_stride(primals_57, (40, ), (1, ))
    assert_size_stride(primals_58, (40, ), (1, ))
    assert_size_stride(primals_59, (120, ), (1, ))
    assert_size_stride(primals_60, (120, ), (1, ))
    assert_size_stride(primals_61, (120, ), (1, ))
    assert_size_stride(primals_62, (120, ), (1, ))
    assert_size_stride(primals_63, (40, ), (1, ))
    assert_size_stride(primals_64, (40, ), (1, ))
    assert_size_stride(primals_65, (200, ), (1, ))
    assert_size_stride(primals_66, (200, ), (1, ))
    assert_size_stride(primals_67, (200, ), (1, ))
    assert_size_stride(primals_68, (200, ), (1, ))
    assert_size_stride(primals_69, (72, ), (1, ))
    assert_size_stride(primals_70, (72, ), (1, ))
    assert_size_stride(primals_71, (216, ), (1, ))
    assert_size_stride(primals_72, (216, ), (1, ))
    assert_size_stride(primals_73, (216, ), (1, ))
    assert_size_stride(primals_74, (216, ), (1, ))
    assert_size_stride(primals_75, (72, ), (1, ))
    assert_size_stride(primals_76, (72, ), (1, ))
    assert_size_stride(primals_77, (216, ), (1, ))
    assert_size_stride(primals_78, (216, ), (1, ))
    assert_size_stride(primals_79, (216, ), (1, ))
    assert_size_stride(primals_80, (216, ), (1, ))
    assert_size_stride(primals_81, (72, ), (1, ))
    assert_size_stride(primals_82, (72, ), (1, ))
    assert_size_stride(primals_83, (216, ), (1, ))
    assert_size_stride(primals_84, (216, ), (1, ))
    assert_size_stride(primals_85, (216, ), (1, ))
    assert_size_stride(primals_86, (216, ), (1, ))
    assert_size_stride(primals_87, (72, ), (1, ))
    assert_size_stride(primals_88, (72, ), (1, ))
    assert_size_stride(primals_89, (216, ), (1, ))
    assert_size_stride(primals_90, (216, ), (1, ))
    assert_size_stride(primals_91, (216, ), (1, ))
    assert_size_stride(primals_92, (216, ), (1, ))
    assert_size_stride(primals_93, (72, ), (1, ))
    assert_size_stride(primals_94, (72, ), (1, ))
    assert_size_stride(primals_95, (360, ), (1, ))
    assert_size_stride(primals_96, (360, ), (1, ))
    assert_size_stride(primals_97, (360, ), (1, ))
    assert_size_stride(primals_98, (360, ), (1, ))
    assert_size_stride(primals_99, (120, ), (1, ))
    assert_size_stride(primals_100, (120, ), (1, ))
    assert_size_stride(primals_101, (360, ), (1, ))
    assert_size_stride(primals_102, (360, ), (1, ))
    assert_size_stride(primals_103, (360, ), (1, ))
    assert_size_stride(primals_104, (360, ), (1, ))
    assert_size_stride(primals_105, (120, ), (1, ))
    assert_size_stride(primals_106, (120, ), (1, ))
    assert_size_stride(primals_107, (360, ), (1, ))
    assert_size_stride(primals_108, (360, ), (1, ))
    assert_size_stride(primals_109, (360, ), (1, ))
    assert_size_stride(primals_110, (360, ), (1, ))
    assert_size_stride(primals_111, (120, ), (1, ))
    assert_size_stride(primals_112, (120, ), (1, ))
    assert_size_stride(primals_113, (360, ), (1, ))
    assert_size_stride(primals_114, (360, ), (1, ))
    assert_size_stride(primals_115, (360, ), (1, ))
    assert_size_stride(primals_116, (360, ), (1, ))
    assert_size_stride(primals_117, (120, ), (1, ))
    assert_size_stride(primals_118, (120, ), (1, ))
    assert_size_stride(primals_119, (360, ), (1, ))
    assert_size_stride(primals_120, (360, ), (1, ))
    assert_size_stride(primals_121, (360, ), (1, ))
    assert_size_stride(primals_122, (360, ), (1, ))
    assert_size_stride(primals_123, (120, ), (1, ))
    assert_size_stride(primals_124, (120, ), (1, ))
    assert_size_stride(primals_125, (360, ), (1, ))
    assert_size_stride(primals_126, (360, ), (1, ))
    assert_size_stride(primals_127, (360, ), (1, ))
    assert_size_stride(primals_128, (360, ), (1, ))
    assert_size_stride(primals_129, (120, ), (1, ))
    assert_size_stride(primals_130, (120, ), (1, ))
    assert_size_stride(primals_131, (720, ), (1, ))
    assert_size_stride(primals_132, (720, ), (1, ))
    assert_size_stride(primals_133, (720, ), (1, ))
    assert_size_stride(primals_134, (720, ), (1, ))
    assert_size_stride(primals_135, (184, ), (1, ))
    assert_size_stride(primals_136, (184, ), (1, ))
    assert_size_stride(primals_137, (736, ), (1, ))
    assert_size_stride(primals_138, (736, ), (1, ))
    assert_size_stride(primals_139, (736, ), (1, ))
    assert_size_stride(primals_140, (736, ), (1, ))
    assert_size_stride(primals_141, (184, ), (1, ))
    assert_size_stride(primals_142, (184, ), (1, ))
    assert_size_stride(primals_143, (736, ), (1, ))
    assert_size_stride(primals_144, (736, ), (1, ))
    assert_size_stride(primals_145, (736, ), (1, ))
    assert_size_stride(primals_146, (736, ), (1, ))
    assert_size_stride(primals_147, (184, ), (1, ))
    assert_size_stride(primals_148, (184, ), (1, ))
    assert_size_stride(primals_149, (736, ), (1, ))
    assert_size_stride(primals_150, (736, ), (1, ))
    assert_size_stride(primals_151, (736, ), (1, ))
    assert_size_stride(primals_152, (736, ), (1, ))
    assert_size_stride(primals_153, (184, ), (1, ))
    assert_size_stride(primals_154, (184, ), (1, ))
    assert_size_stride(primals_155, (736, ), (1, ))
    assert_size_stride(primals_156, (736, ), (1, ))
    assert_size_stride(primals_157, (736, ), (1, ))
    assert_size_stride(primals_158, (736, ), (1, ))
    assert_size_stride(primals_159, (184, ), (1, ))
    assert_size_stride(primals_160, (184, ), (1, ))
    assert_size_stride(primals_161, (736, ), (1, ))
    assert_size_stride(primals_162, (736, ), (1, ))
    assert_size_stride(primals_163, (736, ), (1, ))
    assert_size_stride(primals_164, (736, ), (1, ))
    assert_size_stride(primals_165, (184, ), (1, ))
    assert_size_stride(primals_166, (184, ), (1, ))
    assert_size_stride(primals_167, (1104, ), (1, ))
    assert_size_stride(primals_168, (1104, ), (1, ))
    assert_size_stride(primals_169, (1104, ), (1, ))
    assert_size_stride(primals_170, (1104, ), (1, ))
    assert_size_stride(primals_171, (224, ), (1, ))
    assert_size_stride(primals_172, (224, ), (1, ))
    assert_size_stride(primals_173, (1344, ), (1, ))
    assert_size_stride(primals_174, (1344, ), (1, ))
    assert_size_stride(primals_175, (1000, 1984), (1984, 1))
    assert_size_stride(primals_176, (1000, ), (1, ))
    assert_size_stride(primals_177, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_178, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_179, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_180, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_181, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_182, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_183, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_184, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_185, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_186, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_187, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_188, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_189, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_190, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_191, (48, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_192, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_193, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_194, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_195, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (8, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_197, (8, ), (1, ))
    assert_size_stride(primals_198, (120, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_199, (120, ), (1, ))
    assert_size_stride(primals_200, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_201, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_202, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_203, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_204, (16, ), (1, ))
    assert_size_stride(primals_205, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_206, (120, ), (1, ))
    assert_size_stride(primals_207, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_208, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_209, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_210, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_211, (16, ), (1, ))
    assert_size_stride(primals_212, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_215, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_216, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_217, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_218, (16, ), (1, ))
    assert_size_stride(primals_219, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_220, (120, ), (1, ))
    assert_size_stride(primals_221, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_222, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_223, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_224, (16, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_225, (16, ), (1, ))
    assert_size_stride(primals_226, (120, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_227, (120, ), (1, ))
    assert_size_stride(primals_228, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_229, (200, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_230, (200, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_231, (72, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_232, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_233, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_234, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_235, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_236, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_237, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_238, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_239, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_241, (216, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_242, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (72, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(primals_244, (360, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_245, (360, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (24, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_247, (24, ), (1, ))
    assert_size_stride(primals_248, (360, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_249, (360, ), (1, ))
    assert_size_stride(primals_250, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_251, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_252, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_253, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_254, (32, ), (1, ))
    assert_size_stride(primals_255, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_256, (360, ), (1, ))
    assert_size_stride(primals_257, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_258, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_259, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_260, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_261, (32, ), (1, ))
    assert_size_stride(primals_262, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_263, (360, ), (1, ))
    assert_size_stride(primals_264, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_265, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_266, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_267, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_268, (32, ), (1, ))
    assert_size_stride(primals_269, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_270, (360, ), (1, ))
    assert_size_stride(primals_271, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_272, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_273, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_274, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_275, (32, ), (1, ))
    assert_size_stride(primals_276, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_277, (360, ), (1, ))
    assert_size_stride(primals_278, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_279, (360, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_280, (360, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_281, (32, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_282, (32, ), (1, ))
    assert_size_stride(primals_283, (360, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_284, (360, ), (1, ))
    assert_size_stride(primals_285, (120, 360, 1, 1), (360, 1, 1, 1))
    assert_size_stride(primals_286, (720, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_287, (720, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_288, (32, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (720, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_291, (720, ), (1, ))
    assert_size_stride(primals_292, (184, 720, 1, 1), (720, 1, 1, 1))
    assert_size_stride(primals_293, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_294, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_295, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_296, (48, ), (1, ))
    assert_size_stride(primals_297, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_298, (736, ), (1, ))
    assert_size_stride(primals_299, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_300, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_301, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_302, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_303, (48, ), (1, ))
    assert_size_stride(primals_304, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_305, (736, ), (1, ))
    assert_size_stride(primals_306, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_307, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_308, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_309, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_310, (48, ), (1, ))
    assert_size_stride(primals_311, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_312, (736, ), (1, ))
    assert_size_stride(primals_313, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_314, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_315, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_316, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_317, (48, ), (1, ))
    assert_size_stride(primals_318, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_319, (736, ), (1, ))
    assert_size_stride(primals_320, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_321, (736, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_322, (736, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_323, (48, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_324, (48, ), (1, ))
    assert_size_stride(primals_325, (736, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_326, (736, ), (1, ))
    assert_size_stride(primals_327, (184, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_328, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_329, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_330, (48, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_331, (48, ), (1, ))
    assert_size_stride(primals_332, (1104, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_333, (1104, ), (1, ))
    assert_size_stride(primals_334, (224, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_335, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_336, (1984, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (16, ), (1, ))
    assert_size_stride(primals_339, (16, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (16, ), (1, ))
    assert_size_stride(primals_342, (16, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (16, ), (1, ))
    assert_size_stride(primals_345, (16, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (16, ), (1, ))
    assert_size_stride(primals_348, (16, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (16, ), (1, ))
    assert_size_stride(primals_351, (16, ), (1, ))
    assert_size_stride(primals_352, (), ())
    assert_size_stride(primals_353, (64, ), (1, ))
    assert_size_stride(primals_354, (64, ), (1, ))
    assert_size_stride(primals_355, (), ())
    assert_size_stride(primals_356, (64, ), (1, ))
    assert_size_stride(primals_357, (64, ), (1, ))
    assert_size_stride(primals_358, (), ())
    assert_size_stride(primals_359, (24, ), (1, ))
    assert_size_stride(primals_360, (24, ), (1, ))
    assert_size_stride(primals_361, (), ())
    assert_size_stride(primals_362, (48, ), (1, ))
    assert_size_stride(primals_363, (48, ), (1, ))
    assert_size_stride(primals_364, (), ())
    assert_size_stride(primals_365, (48, ), (1, ))
    assert_size_stride(primals_366, (48, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (24, ), (1, ))
    assert_size_stride(primals_369, (24, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (48, ), (1, ))
    assert_size_stride(primals_372, (48, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (48, ), (1, ))
    assert_size_stride(primals_375, (48, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (24, ), (1, ))
    assert_size_stride(primals_378, (24, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (48, ), (1, ))
    assert_size_stride(primals_381, (48, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (48, ), (1, ))
    assert_size_stride(primals_384, (48, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (24, ), (1, ))
    assert_size_stride(primals_387, (24, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (120, ), (1, ))
    assert_size_stride(primals_390, (120, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (120, ), (1, ))
    assert_size_stride(primals_393, (120, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (40, ), (1, ))
    assert_size_stride(primals_396, (40, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (120, ), (1, ))
    assert_size_stride(primals_399, (120, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (120, ), (1, ))
    assert_size_stride(primals_402, (120, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (40, ), (1, ))
    assert_size_stride(primals_405, (40, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (120, ), (1, ))
    assert_size_stride(primals_408, (120, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (120, ), (1, ))
    assert_size_stride(primals_411, (120, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (40, ), (1, ))
    assert_size_stride(primals_414, (40, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (120, ), (1, ))
    assert_size_stride(primals_417, (120, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (120, ), (1, ))
    assert_size_stride(primals_420, (120, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (40, ), (1, ))
    assert_size_stride(primals_423, (40, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (120, ), (1, ))
    assert_size_stride(primals_426, (120, ), (1, ))
    assert_size_stride(primals_427, (), ())
    assert_size_stride(primals_428, (120, ), (1, ))
    assert_size_stride(primals_429, (120, ), (1, ))
    assert_size_stride(primals_430, (), ())
    assert_size_stride(primals_431, (40, ), (1, ))
    assert_size_stride(primals_432, (40, ), (1, ))
    assert_size_stride(primals_433, (), ())
    assert_size_stride(primals_434, (200, ), (1, ))
    assert_size_stride(primals_435, (200, ), (1, ))
    assert_size_stride(primals_436, (), ())
    assert_size_stride(primals_437, (200, ), (1, ))
    assert_size_stride(primals_438, (200, ), (1, ))
    assert_size_stride(primals_439, (), ())
    assert_size_stride(primals_440, (72, ), (1, ))
    assert_size_stride(primals_441, (72, ), (1, ))
    assert_size_stride(primals_442, (), ())
    assert_size_stride(primals_443, (216, ), (1, ))
    assert_size_stride(primals_444, (216, ), (1, ))
    assert_size_stride(primals_445, (), ())
    assert_size_stride(primals_446, (216, ), (1, ))
    assert_size_stride(primals_447, (216, ), (1, ))
    assert_size_stride(primals_448, (), ())
    assert_size_stride(primals_449, (72, ), (1, ))
    assert_size_stride(primals_450, (72, ), (1, ))
    assert_size_stride(primals_451, (), ())
    assert_size_stride(primals_452, (216, ), (1, ))
    assert_size_stride(primals_453, (216, ), (1, ))
    assert_size_stride(primals_454, (), ())
    assert_size_stride(primals_455, (216, ), (1, ))
    assert_size_stride(primals_456, (216, ), (1, ))
    assert_size_stride(primals_457, (), ())
    assert_size_stride(primals_458, (72, ), (1, ))
    assert_size_stride(primals_459, (72, ), (1, ))
    assert_size_stride(primals_460, (), ())
    assert_size_stride(primals_461, (216, ), (1, ))
    assert_size_stride(primals_462, (216, ), (1, ))
    assert_size_stride(primals_463, (), ())
    assert_size_stride(primals_464, (216, ), (1, ))
    assert_size_stride(primals_465, (216, ), (1, ))
    assert_size_stride(primals_466, (), ())
    assert_size_stride(primals_467, (72, ), (1, ))
    assert_size_stride(primals_468, (72, ), (1, ))
    assert_size_stride(primals_469, (), ())
    assert_size_stride(primals_470, (216, ), (1, ))
    assert_size_stride(primals_471, (216, ), (1, ))
    assert_size_stride(primals_472, (), ())
    assert_size_stride(primals_473, (216, ), (1, ))
    assert_size_stride(primals_474, (216, ), (1, ))
    assert_size_stride(primals_475, (), ())
    assert_size_stride(primals_476, (72, ), (1, ))
    assert_size_stride(primals_477, (72, ), (1, ))
    assert_size_stride(primals_478, (), ())
    assert_size_stride(primals_479, (360, ), (1, ))
    assert_size_stride(primals_480, (360, ), (1, ))
    assert_size_stride(primals_481, (), ())
    assert_size_stride(primals_482, (360, ), (1, ))
    assert_size_stride(primals_483, (360, ), (1, ))
    assert_size_stride(primals_484, (), ())
    assert_size_stride(primals_485, (120, ), (1, ))
    assert_size_stride(primals_486, (120, ), (1, ))
    assert_size_stride(primals_487, (), ())
    assert_size_stride(primals_488, (360, ), (1, ))
    assert_size_stride(primals_489, (360, ), (1, ))
    assert_size_stride(primals_490, (), ())
    assert_size_stride(primals_491, (360, ), (1, ))
    assert_size_stride(primals_492, (360, ), (1, ))
    assert_size_stride(primals_493, (), ())
    assert_size_stride(primals_494, (120, ), (1, ))
    assert_size_stride(primals_495, (120, ), (1, ))
    assert_size_stride(primals_496, (), ())
    assert_size_stride(primals_497, (360, ), (1, ))
    assert_size_stride(primals_498, (360, ), (1, ))
    assert_size_stride(primals_499, (), ())
    assert_size_stride(primals_500, (360, ), (1, ))
    assert_size_stride(primals_501, (360, ), (1, ))
    assert_size_stride(primals_502, (), ())
    assert_size_stride(primals_503, (120, ), (1, ))
    assert_size_stride(primals_504, (120, ), (1, ))
    assert_size_stride(primals_505, (), ())
    assert_size_stride(primals_506, (360, ), (1, ))
    assert_size_stride(primals_507, (360, ), (1, ))
    assert_size_stride(primals_508, (), ())
    assert_size_stride(primals_509, (360, ), (1, ))
    assert_size_stride(primals_510, (360, ), (1, ))
    assert_size_stride(primals_511, (), ())
    assert_size_stride(primals_512, (120, ), (1, ))
    assert_size_stride(primals_513, (120, ), (1, ))
    assert_size_stride(primals_514, (), ())
    assert_size_stride(primals_515, (360, ), (1, ))
    assert_size_stride(primals_516, (360, ), (1, ))
    assert_size_stride(primals_517, (), ())
    assert_size_stride(primals_518, (360, ), (1, ))
    assert_size_stride(primals_519, (360, ), (1, ))
    assert_size_stride(primals_520, (), ())
    assert_size_stride(primals_521, (120, ), (1, ))
    assert_size_stride(primals_522, (120, ), (1, ))
    assert_size_stride(primals_523, (), ())
    assert_size_stride(primals_524, (360, ), (1, ))
    assert_size_stride(primals_525, (360, ), (1, ))
    assert_size_stride(primals_526, (), ())
    assert_size_stride(primals_527, (360, ), (1, ))
    assert_size_stride(primals_528, (360, ), (1, ))
    assert_size_stride(primals_529, (), ())
    assert_size_stride(primals_530, (120, ), (1, ))
    assert_size_stride(primals_531, (120, ), (1, ))
    assert_size_stride(primals_532, (), ())
    assert_size_stride(primals_533, (720, ), (1, ))
    assert_size_stride(primals_534, (720, ), (1, ))
    assert_size_stride(primals_535, (), ())
    assert_size_stride(primals_536, (720, ), (1, ))
    assert_size_stride(primals_537, (720, ), (1, ))
    assert_size_stride(primals_538, (), ())
    assert_size_stride(primals_539, (184, ), (1, ))
    assert_size_stride(primals_540, (184, ), (1, ))
    assert_size_stride(primals_541, (), ())
    assert_size_stride(primals_542, (736, ), (1, ))
    assert_size_stride(primals_543, (736, ), (1, ))
    assert_size_stride(primals_544, (), ())
    assert_size_stride(primals_545, (736, ), (1, ))
    assert_size_stride(primals_546, (736, ), (1, ))
    assert_size_stride(primals_547, (), ())
    assert_size_stride(primals_548, (184, ), (1, ))
    assert_size_stride(primals_549, (184, ), (1, ))
    assert_size_stride(primals_550, (), ())
    assert_size_stride(primals_551, (736, ), (1, ))
    assert_size_stride(primals_552, (736, ), (1, ))
    assert_size_stride(primals_553, (), ())
    assert_size_stride(primals_554, (736, ), (1, ))
    assert_size_stride(primals_555, (736, ), (1, ))
    assert_size_stride(primals_556, (), ())
    assert_size_stride(primals_557, (184, ), (1, ))
    assert_size_stride(primals_558, (184, ), (1, ))
    assert_size_stride(primals_559, (), ())
    assert_size_stride(primals_560, (736, ), (1, ))
    assert_size_stride(primals_561, (736, ), (1, ))
    assert_size_stride(primals_562, (), ())
    assert_size_stride(primals_563, (736, ), (1, ))
    assert_size_stride(primals_564, (736, ), (1, ))
    assert_size_stride(primals_565, (), ())
    assert_size_stride(primals_566, (184, ), (1, ))
    assert_size_stride(primals_567, (184, ), (1, ))
    assert_size_stride(primals_568, (), ())
    assert_size_stride(primals_569, (736, ), (1, ))
    assert_size_stride(primals_570, (736, ), (1, ))
    assert_size_stride(primals_571, (), ())
    assert_size_stride(primals_572, (736, ), (1, ))
    assert_size_stride(primals_573, (736, ), (1, ))
    assert_size_stride(primals_574, (), ())
    assert_size_stride(primals_575, (184, ), (1, ))
    assert_size_stride(primals_576, (184, ), (1, ))
    assert_size_stride(primals_577, (), ())
    assert_size_stride(primals_578, (736, ), (1, ))
    assert_size_stride(primals_579, (736, ), (1, ))
    assert_size_stride(primals_580, (), ())
    assert_size_stride(primals_581, (736, ), (1, ))
    assert_size_stride(primals_582, (736, ), (1, ))
    assert_size_stride(primals_583, (), ())
    assert_size_stride(primals_584, (184, ), (1, ))
    assert_size_stride(primals_585, (184, ), (1, ))
    assert_size_stride(primals_586, (), ())
    assert_size_stride(primals_587, (1104, ), (1, ))
    assert_size_stride(primals_588, (1104, ), (1, ))
    assert_size_stride(primals_589, (), ())
    assert_size_stride(primals_590, (1104, ), (1, ))
    assert_size_stride(primals_591, (1104, ), (1, ))
    assert_size_stride(primals_592, (), ())
    assert_size_stride(primals_593, (224, ), (1, ))
    assert_size_stride(primals_594, (224, ), (1, ))
    assert_size_stride(primals_595, (), ())
    assert_size_stride(primals_596, (1344, ), (1, ))
    assert_size_stride(primals_597, (1344, ), (1, ))
    assert_size_stride(primals_598, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_177, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_177
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_598, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_598
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
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_338, primals_339, buf10, buf11, buf13, primals_338, primals_339, 16, 7, grid=grid(16), stream=stream0)
        del primals_338
        del primals_339
        buf14 = reinterpret_tensor(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        buf15 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_6.run(buf3, buf10, buf11, primals_1, primals_2, buf14, buf15, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
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
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf21, buf22, buf23, primals_341, primals_342, buf24, buf25, buf27, primals_341, primals_342, 16, 7, grid=grid(16), stream=stream0)
        del primals_341
        del primals_342
        buf28 = reinterpret_tensor(buf16, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf16  # reuse
        buf29 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_6.run(buf17, buf24, buf25, primals_3, primals_4, buf28, buf29, 1605632, grid=grid(1605632), stream=stream0)
        del primals_4
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf31 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf30, buf31, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf32 = buf20; del buf20  # reuse
        buf33 = buf19; del buf19  # reuse
        buf34 = buf18; del buf18  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf31, buf32, buf33, buf34, 12544, 128, grid=grid(12544), stream=stream0)
        buf35 = buf23; del buf23  # reuse
        buf36 = buf22; del buf22  # reuse
        buf37 = buf21; del buf21  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf32, buf33, buf34, buf35, buf36, buf37, 112, 112, grid=grid(112), stream=stream0)
        buf38 = buf25; del buf25  # reuse
        buf39 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf41 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf35, buf36, buf37, primals_344, primals_345, buf38, buf39, buf41, primals_344, primals_345, 16, 7, grid=grid(16), stream=stream0)
        del primals_344
        del primals_345
        buf42 = reinterpret_tensor(buf30, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf30  # reuse
        # Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_7.run(buf31, buf38, buf39, primals_5, primals_6, buf15, buf42, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf43, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf44 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf43, buf44, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf45 = buf34; del buf34  # reuse
        buf46 = buf33; del buf33  # reuse
        buf47 = buf32; del buf32  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf44, buf45, buf46, buf47, 12544, 128, grid=grid(12544), stream=stream0)
        buf48 = buf37; del buf37  # reuse
        buf49 = buf36; del buf36  # reuse
        buf50 = buf35; del buf35  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf45, buf46, buf47, buf48, buf49, buf50, 112, 112, grid=grid(112), stream=stream0)
        buf51 = buf39; del buf39  # reuse
        buf52 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf54 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf48, buf49, buf50, primals_347, primals_348, buf51, buf52, buf54, primals_347, primals_348, 16, 7, grid=grid(16), stream=stream0)
        del primals_347
        del primals_348
        buf55 = reinterpret_tensor(buf43, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf43  # reuse
        buf56 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18, x_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_6.run(buf44, buf51, buf52, primals_7, primals_8, buf55, buf56, 1605632, grid=grid(1605632), stream=stream0)
        del primals_8
        # Source Nodes: [x_23], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf58 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf57, buf58, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf59 = buf47; del buf47  # reuse
        buf60 = buf46; del buf46  # reuse
        buf61 = buf45; del buf45  # reuse
        # Source Nodes: [x_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf58, buf59, buf60, buf61, 12544, 128, grid=grid(12544), stream=stream0)
        buf62 = buf50; del buf50  # reuse
        buf63 = buf49; del buf49  # reuse
        buf64 = buf48; del buf48  # reuse
        # Source Nodes: [x_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf59, buf60, buf61, buf62, buf63, buf64, 112, 112, grid=grid(112), stream=stream0)
        buf65 = buf52; del buf52  # reuse
        buf66 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf68 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf62, buf63, buf64, primals_350, primals_351, buf65, buf66, buf68, primals_350, primals_351, 16, 7, grid=grid(16), stream=stream0)
        del buf62
        del buf63
        del buf64
        del primals_350
        del primals_351
        buf69 = reinterpret_tensor(buf57, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf57  # reuse
        # Source Nodes: [shortcut_2, x_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_7.run(buf58, buf65, buf66, primals_9, primals_10, buf42, buf69, 1605632, grid=grid(1605632), stream=stream0)
        del buf66
        del primals_10
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf71 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf70, buf71, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf72 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf71, buf72, buf73, buf74, 50176, 128, grid=grid(50176), stream=stream0)
        buf75 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf77 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf72, buf73, buf74, buf75, buf76, buf77, 448, 112, grid=grid(448), stream=stream0)
        del buf72
        del buf73
        del buf74
        buf78 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf79 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf81 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf75, buf76, buf77, primals_353, primals_354, buf78, buf79, buf81, primals_353, primals_354, 64, 7, grid=grid(64), stream=stream0)
        del buf75
        del buf76
        del buf77
        del primals_353
        del primals_354
        buf82 = reinterpret_tensor(buf70, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf70  # reuse
        buf83 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30, x_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_12.run(buf71, buf78, buf79, primals_11, primals_12, buf82, buf83, 6422528, grid=grid(6422528), stream=stream0)
        del primals_12
        # Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_183, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf84, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf85 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf84, buf85, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf86 = reinterpret_tensor(buf61, (1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), 0); del buf61  # reuse
        buf87 = reinterpret_tensor(buf60, (1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), 0); del buf60  # reuse
        buf88 = reinterpret_tensor(buf59, (1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), 0); del buf59  # reuse
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf85, buf86, buf87, buf88, 12544, 128, grid=grid(12544), stream=stream0)
        buf89 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf86, buf87, buf88, buf89, buf90, buf91, 128, 98, grid=grid(128), stream=stream0)
        del buf86
        del buf87
        del buf88
        buf92 = buf79; del buf79  # reuse
        buf93 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf95 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf89, buf90, buf91, primals_356, primals_357, buf92, buf93, buf95, primals_356, primals_357, 64, 2, grid=grid(64), stream=stream0)
        del primals_356
        del primals_357
        buf96 = reinterpret_tensor(buf84, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf84  # reuse
        buf97 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35, x_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_17.run(buf85, buf92, buf93, primals_13, primals_14, buf96, buf97, 1605632, grid=grid(1605632), stream=stream0)
        del primals_14
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf99 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf98, buf99, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf100 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf101 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf99, buf100, buf101, buf102, 4704, 128, grid=grid(4704), stream=stream0)
        buf103 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf100, buf101, buf102, buf103, buf104, buf105, 48, 98, grid=grid(48), stream=stream0)
        buf106 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf107 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf109 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf103, buf104, buf105, primals_359, primals_360, buf106, buf107, buf109, primals_359, primals_360, 24, 2, grid=grid(24), stream=stream0)
        del primals_359
        del primals_360
        buf110 = reinterpret_tensor(buf98, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf98  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_22.run(buf99, buf106, buf107, primals_15, primals_16, buf110, 602112, grid=grid(602112), stream=stream0)
        del primals_16
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf112 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf111, buf112, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf113 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        buf114 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        buf115 = empty_strided((1, 48, 1, 1, 196), (9408, 1, 9408, 9408, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf112, buf113, buf114, buf115, 9408, 128, grid=grid(9408), stream=stream0)
        buf116 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((1, 48, 1, 1, 2), (96, 1, 96, 96, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf113, buf114, buf115, buf116, buf117, buf118, 96, 98, grid=grid(96), stream=stream0)
        buf119 = reinterpret_tensor(buf105, (1, 48, 1, 1), (48, 1, 48, 48), 0); del buf105  # reuse
        buf120 = reinterpret_tensor(buf104, (1, 48, 1, 1), (48, 1, 48, 48), 0); del buf104  # reuse
        buf122 = reinterpret_tensor(buf103, (48, ), (1, ), 0); del buf103  # reuse
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf116, buf117, buf118, primals_362, primals_363, buf119, buf120, buf122, primals_362, primals_363, 48, 2, grid=grid(48), stream=stream0)
        del primals_362
        del primals_363
        buf123 = reinterpret_tensor(buf111, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf111  # reuse
        buf124 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_27.run(buf112, buf119, buf120, primals_17, primals_18, buf123, buf124, 1204224, grid=grid(1204224), stream=stream0)
        del primals_18
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_186, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf125, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf126 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf125, buf126, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf127 = buf115; del buf115  # reuse
        buf128 = buf114; del buf114  # reuse
        buf129 = buf113; del buf113  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf126, buf127, buf128, buf129, 9408, 128, grid=grid(9408), stream=stream0)
        buf130 = buf118; del buf118  # reuse
        buf131 = buf117; del buf117  # reuse
        buf132 = buf116; del buf116  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf127, buf128, buf129, buf130, buf131, buf132, 96, 98, grid=grid(96), stream=stream0)
        buf133 = buf120; del buf120  # reuse
        buf134 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf136 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf130, buf131, buf132, primals_365, primals_366, buf133, buf134, buf136, primals_365, primals_366, 48, 2, grid=grid(48), stream=stream0)
        del primals_365
        del primals_366
        buf137 = reinterpret_tensor(buf125, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf125  # reuse
        buf138 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51, x_54], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_27.run(buf126, buf133, buf134, primals_19, primals_20, buf137, buf138, 1204224, grid=grid(1204224), stream=stream0)
        del primals_20
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf140 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf139, buf140, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf141 = buf102; del buf102  # reuse
        buf142 = buf101; del buf101  # reuse
        buf143 = buf100; del buf100  # reuse
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf140, buf141, buf142, buf143, 4704, 128, grid=grid(4704), stream=stream0)
        buf144 = reinterpret_tensor(buf134, (1, 24, 1, 1, 2), (48, 1, 48, 48, 24), 0); del buf134  # reuse
        buf145 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf146 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf141, buf142, buf143, buf144, buf145, buf146, 48, 98, grid=grid(48), stream=stream0)
        buf147 = buf107; del buf107  # reuse
        buf148 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf150 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf144, buf145, buf146, primals_368, primals_369, buf147, buf148, buf150, primals_368, primals_369, 24, 2, grid=grid(24), stream=stream0)
        del primals_368
        del primals_369
        buf151 = reinterpret_tensor(buf139, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf139  # reuse
        # Source Nodes: [shortcut_4, x_57], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_28.run(buf140, buf147, buf148, primals_21, primals_22, buf110, buf151, 602112, grid=grid(602112), stream=stream0)
        del primals_22
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf153 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf152, buf153, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf154 = buf129; del buf129  # reuse
        buf155 = buf128; del buf128  # reuse
        buf156 = buf127; del buf127  # reuse
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf153, buf154, buf155, buf156, 9408, 128, grid=grid(9408), stream=stream0)
        buf157 = buf132; del buf132  # reuse
        buf158 = buf131; del buf131  # reuse
        buf159 = buf130; del buf130  # reuse
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf154, buf155, buf156, buf157, buf158, buf159, 96, 98, grid=grid(96), stream=stream0)
        buf160 = reinterpret_tensor(buf146, (1, 48, 1, 1), (48, 1, 48, 48), 0); del buf146  # reuse
        buf161 = reinterpret_tensor(buf145, (1, 48, 1, 1), (48, 1, 48, 48), 0); del buf145  # reuse
        buf163 = reinterpret_tensor(buf144, (48, ), (1, ), 0); del buf144  # reuse
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf157, buf158, buf159, primals_371, primals_372, buf160, buf161, buf163, primals_371, primals_372, 48, 2, grid=grid(48), stream=stream0)
        del primals_371
        del primals_372
        buf164 = reinterpret_tensor(buf152, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf152  # reuse
        buf165 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_27.run(buf153, buf160, buf161, primals_23, primals_24, buf164, buf165, 1204224, grid=grid(1204224), stream=stream0)
        del primals_24
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_189, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf166, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf167 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf166, buf167, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf168 = buf156; del buf156  # reuse
        buf169 = buf155; del buf155  # reuse
        buf170 = buf154; del buf154  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf167, buf168, buf169, buf170, 9408, 128, grid=grid(9408), stream=stream0)
        buf171 = buf159; del buf159  # reuse
        buf172 = buf158; del buf158  # reuse
        buf173 = buf157; del buf157  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf168, buf169, buf170, buf171, buf172, buf173, 96, 98, grid=grid(96), stream=stream0)
        buf174 = buf161; del buf161  # reuse
        buf175 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf177 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf171, buf172, buf173, primals_374, primals_375, buf174, buf175, buf177, primals_374, primals_375, 48, 2, grid=grid(48), stream=stream0)
        del primals_374
        del primals_375
        buf178 = reinterpret_tensor(buf166, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf166  # reuse
        buf179 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68, x_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_27.run(buf167, buf174, buf175, primals_25, primals_26, buf178, buf179, 1204224, grid=grid(1204224), stream=stream0)
        del primals_26
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf181 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf180, buf181, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf182 = buf143; del buf143  # reuse
        buf183 = buf142; del buf142  # reuse
        buf184 = buf141; del buf141  # reuse
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf181, buf182, buf183, buf184, 4704, 128, grid=grid(4704), stream=stream0)
        buf185 = reinterpret_tensor(buf175, (1, 24, 1, 1, 2), (48, 1, 48, 48, 24), 0); del buf175  # reuse
        buf186 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf187 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf182, buf183, buf184, buf185, buf186, buf187, 48, 98, grid=grid(48), stream=stream0)
        buf188 = buf148; del buf148  # reuse
        buf189 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf191 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf185, buf186, buf187, primals_377, primals_378, buf188, buf189, buf191, primals_377, primals_378, 24, 2, grid=grid(24), stream=stream0)
        del primals_377
        del primals_378
        buf192 = reinterpret_tensor(buf180, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf180  # reuse
        # Source Nodes: [shortcut_5, x_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_28.run(buf181, buf188, buf189, primals_27, primals_28, buf151, buf192, 602112, grid=grid(602112), stream=stream0)
        del primals_28
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf194 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf193, buf194, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf195 = buf170; del buf170  # reuse
        buf196 = buf169; del buf169  # reuse
        buf197 = buf168; del buf168  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf194, buf195, buf196, buf197, 9408, 128, grid=grid(9408), stream=stream0)
        buf198 = buf173; del buf173  # reuse
        buf199 = buf172; del buf172  # reuse
        buf200 = buf171; del buf171  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf195, buf196, buf197, buf198, buf199, buf200, 96, 98, grid=grid(96), stream=stream0)
        buf201 = reinterpret_tensor(buf187, (1, 48, 1, 1), (48, 1, 48, 48), 0); del buf187  # reuse
        buf202 = reinterpret_tensor(buf186, (1, 48, 1, 1), (48, 1, 48, 48), 0); del buf186  # reuse
        buf204 = reinterpret_tensor(buf185, (48, ), (1, ), 0); del buf185  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf198, buf199, buf200, primals_380, primals_381, buf201, buf202, buf204, primals_380, primals_381, 48, 2, grid=grid(48), stream=stream0)
        del primals_380
        del primals_381
        buf205 = reinterpret_tensor(buf193, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf193  # reuse
        buf206 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_27.run(buf194, buf201, buf202, primals_29, primals_30, buf205, buf206, 1204224, grid=grid(1204224), stream=stream0)
        del primals_30
        # Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_192, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf207, (8, 48, 56, 56), (150528, 3136, 56, 1))
        buf208 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf207, buf208, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf209 = buf197; del buf197  # reuse
        buf210 = buf196; del buf196  # reuse
        buf211 = buf195; del buf195  # reuse
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf208, buf209, buf210, buf211, 9408, 128, grid=grid(9408), stream=stream0)
        buf212 = buf200; del buf200  # reuse
        buf213 = buf199; del buf199  # reuse
        buf214 = buf198; del buf198  # reuse
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf209, buf210, buf211, buf212, buf213, buf214, 96, 98, grid=grid(96), stream=stream0)
        del buf209
        del buf210
        del buf211
        buf215 = buf202; del buf202  # reuse
        buf216 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        buf218 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf212, buf213, buf214, primals_383, primals_384, buf215, buf216, buf218, primals_383, primals_384, 48, 2, grid=grid(48), stream=stream0)
        del buf212
        del buf213
        del buf214
        del primals_383
        del primals_384
        buf219 = reinterpret_tensor(buf207, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf207  # reuse
        buf220 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_27.run(buf208, buf215, buf216, primals_31, primals_32, buf219, buf220, 1204224, grid=grid(1204224), stream=stream0)
        del primals_32
        # Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf222 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf221, buf222, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf223 = buf184; del buf184  # reuse
        buf224 = buf183; del buf183  # reuse
        buf225 = buf182; del buf182  # reuse
        # Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf222, buf223, buf224, buf225, 4704, 128, grid=grid(4704), stream=stream0)
        buf226 = reinterpret_tensor(buf216, (1, 24, 1, 1, 2), (48, 1, 48, 48, 24), 0); del buf216  # reuse
        buf227 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf228 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf223, buf224, buf225, buf226, buf227, buf228, 48, 98, grid=grid(48), stream=stream0)
        del buf223
        del buf224
        del buf225
        buf229 = buf189; del buf189  # reuse
        buf230 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf232 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf226, buf227, buf228, primals_386, primals_387, buf229, buf230, buf232, primals_386, primals_387, 24, 2, grid=grid(24), stream=stream0)
        del buf226
        del buf227
        del buf228
        del primals_386
        del primals_387
        buf233 = reinterpret_tensor(buf221, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf221  # reuse
        # Source Nodes: [shortcut_6, x_91], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_28.run(buf222, buf229, buf230, primals_33, primals_34, buf192, buf233, 602112, grid=grid(602112), stream=stream0)
        del buf230
        del primals_34
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 120, 56, 56), (376320, 3136, 56, 1))
        buf235 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf234, buf235, 960, 3136, grid=grid(960, 3136), stream=stream0)
        buf236 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        buf237 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((1, 120, 1, 1, 196), (23520, 1, 23520, 23520, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf235, buf236, buf237, buf238, 23520, 128, grid=grid(23520), stream=stream0)
        buf239 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        buf240 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        buf241 = empty_strided((1, 120, 1, 1, 2), (240, 1, 240, 240, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf236, buf237, buf238, buf239, buf240, buf241, 240, 98, grid=grid(240), stream=stream0)
        del buf236
        del buf237
        del buf238
        buf242 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf243 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf245 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_32.run(buf239, buf240, buf241, primals_389, primals_390, buf242, buf243, buf245, primals_389, primals_390, 120, 2, grid=grid(120), stream=stream0)
        del buf239
        del buf240
        del buf241
        del primals_389
        del primals_390
        buf246 = reinterpret_tensor(buf234, (8, 120, 56, 56), (376320, 1, 6720, 120), 0); del buf234  # reuse
        buf247 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_33.run(buf235, buf242, buf243, primals_35, primals_36, buf246, buf247, 3010560, grid=grid(3010560), stream=stream0)
        del primals_36
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_195, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf248, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf249 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf248, buf249, 960, 784, grid=grid(960, 784), stream=stream0)
        buf250 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf251 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        buf252 = empty_strided((1, 120, 1, 1, 49), (5880, 1, 5880, 5880, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf249, buf250, buf251, buf252, 5880, 128, grid=grid(5880), stream=stream0)
        buf253 = buf243; del buf243  # reuse
        buf254 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf256 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf250, buf251, buf252, primals_392, primals_393, buf253, buf254, buf256, primals_392, primals_393, 120, 49, grid=grid(120), stream=stream0)
        del primals_392
        del primals_393
        buf257 = reinterpret_tensor(buf248, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf248  # reuse
        buf258 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf249, buf253, buf254, primals_37, primals_38, buf257, buf258, 752640, grid=grid(752640), stream=stream0)
        del primals_38
        buf259 = empty_strided((8, 120, 1, 1, 7), (840, 1, 6720, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_red_fused_mean_38.run(buf258, buf259, 6720, 112, grid=grid(6720), stream=stream0)
        buf260 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf261 = reinterpret_tensor(buf260, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf260  # reuse
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_per_fused_mean_39.run(buf261, buf259, 960, 7, grid=grid(960), stream=stream0)
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 8, 1, 1), (8, 1, 1, 1))
        buf263 = reinterpret_tensor(buf262, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf262  # reuse
        buf264 = reinterpret_tensor(buf93, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf93  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_40.run(buf263, primals_197, buf264, 64, grid=grid(64), stream=stream0)
        del primals_197
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 120, 1, 1), (120, 1, 1, 1))
        buf266 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_41.run(buf265, primals_199, buf266, 960, grid=grid(960), stream=stream0)
        buf267 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten.mul]
        triton_poi_fused_mul_42.run(buf258, buf266, buf267, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf269 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf268, buf269, 320, 784, grid=grid(320, 784), stream=stream0)
        buf270 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf271 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        buf272 = empty_strided((1, 40, 1, 1, 49), (1960, 1, 1960, 1960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf269, buf270, buf271, buf272, 1960, 128, grid=grid(1960), stream=stream0)
        buf273 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf274 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf276 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf270, buf271, buf272, primals_395, primals_396, buf273, buf274, buf276, primals_395, primals_396, 40, 49, grid=grid(40), stream=stream0)
        del primals_395
        del primals_396
        buf277 = reinterpret_tensor(buf268, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf268  # reuse
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_46.run(buf269, buf273, buf274, primals_39, primals_40, buf277, 250880, grid=grid(250880), stream=stream0)
        del primals_40
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf279 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf278, buf279, 960, 784, grid=grid(960, 784), stream=stream0)
        buf280 = buf252; del buf252  # reuse
        buf281 = buf251; del buf251  # reuse
        buf282 = buf250; del buf250  # reuse
        # Source Nodes: [x_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf279, buf280, buf281, buf282, 5880, 128, grid=grid(5880), stream=stream0)
        buf283 = buf254; del buf254  # reuse
        buf284 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf286 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf280, buf281, buf282, primals_398, primals_399, buf283, buf284, buf286, primals_398, primals_399, 120, 49, grid=grid(120), stream=stream0)
        del primals_398
        del primals_399
        buf287 = reinterpret_tensor(buf278, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf278  # reuse
        buf288 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113, x_116], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf279, buf283, buf284, primals_41, primals_42, buf287, buf288, 752640, grid=grid(752640), stream=stream0)
        del primals_42
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_202, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf289, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf290 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf289, buf290, 960, 784, grid=grid(960, 784), stream=stream0)
        buf291 = buf282; del buf282  # reuse
        buf292 = buf281; del buf281  # reuse
        buf293 = buf280; del buf280  # reuse
        # Source Nodes: [x_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf290, buf291, buf292, buf293, 5880, 128, grid=grid(5880), stream=stream0)
        buf294 = buf284; del buf284  # reuse
        buf295 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf297 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf291, buf292, buf293, primals_401, primals_402, buf294, buf295, buf297, primals_401, primals_402, 120, 49, grid=grid(120), stream=stream0)
        del primals_401
        del primals_402
        buf298 = reinterpret_tensor(buf289, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf289  # reuse
        buf299 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118, x_121], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf290, buf294, buf295, primals_43, primals_44, buf298, buf299, 752640, grid=grid(752640), stream=stream0)
        del primals_44
        buf300 = buf259; del buf259  # reuse
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_red_fused_mean_38.run(buf299, buf300, 6720, 112, grid=grid(6720), stream=stream0)
        buf301 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf302 = reinterpret_tensor(buf301, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf301  # reuse
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_per_fused_mean_39.run(buf302, buf300, 960, 7, grid=grid(960), stream=stream0)
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 16, 1, 1), (16, 1, 1, 1))
        buf304 = reinterpret_tensor(buf303, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf303  # reuse
        buf305 = reinterpret_tensor(buf91, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf91  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_47.run(buf304, primals_204, buf305, 128, grid=grid(128), stream=stream0)
        del primals_204
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 120, 1, 1), (120, 1, 1, 1))
        buf307 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_41.run(buf306, primals_206, buf307, 960, grid=grid(960), stream=stream0)
        buf308 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.mul]
        triton_poi_fused_mul_42.run(buf299, buf307, buf308, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf310 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf309, buf310, 320, 784, grid=grid(320, 784), stream=stream0)
        buf311 = buf272; del buf272  # reuse
        buf312 = buf271; del buf271  # reuse
        buf313 = buf270; del buf270  # reuse
        # Source Nodes: [x_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf310, buf311, buf312, buf313, 1960, 128, grid=grid(1960), stream=stream0)
        buf314 = buf274; del buf274  # reuse
        buf315 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf317 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf311, buf312, buf313, primals_404, primals_405, buf314, buf315, buf317, primals_404, primals_405, 40, 49, grid=grid(40), stream=stream0)
        del primals_404
        del primals_405
        buf318 = reinterpret_tensor(buf309, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf309  # reuse
        # Source Nodes: [shortcut_8, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_48.run(buf310, buf314, buf315, primals_45, primals_46, buf277, buf318, 250880, grid=grid(250880), stream=stream0)
        del primals_46
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf320 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf319, buf320, 960, 784, grid=grid(960, 784), stream=stream0)
        buf321 = buf293; del buf293  # reuse
        buf322 = buf292; del buf292  # reuse
        buf323 = buf291; del buf291  # reuse
        # Source Nodes: [x_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf320, buf321, buf322, buf323, 5880, 128, grid=grid(5880), stream=stream0)
        buf324 = buf295; del buf295  # reuse
        buf325 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf327 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf321, buf322, buf323, primals_407, primals_408, buf324, buf325, buf327, primals_407, primals_408, 120, 49, grid=grid(120), stream=stream0)
        del primals_407
        del primals_408
        buf328 = reinterpret_tensor(buf319, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf319  # reuse
        buf329 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130, x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf320, buf324, buf325, primals_47, primals_48, buf328, buf329, 752640, grid=grid(752640), stream=stream0)
        del primals_48
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_209, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf330, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf331 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf330, buf331, 960, 784, grid=grid(960, 784), stream=stream0)
        buf332 = buf323; del buf323  # reuse
        buf333 = buf322; del buf322  # reuse
        buf334 = buf321; del buf321  # reuse
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf331, buf332, buf333, buf334, 5880, 128, grid=grid(5880), stream=stream0)
        buf335 = buf325; del buf325  # reuse
        buf336 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf338 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf332, buf333, buf334, primals_410, primals_411, buf335, buf336, buf338, primals_410, primals_411, 120, 49, grid=grid(120), stream=stream0)
        del primals_410
        del primals_411
        buf339 = reinterpret_tensor(buf330, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf330  # reuse
        buf340 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf331, buf335, buf336, primals_49, primals_50, buf339, buf340, 752640, grid=grid(752640), stream=stream0)
        del primals_50
        buf341 = buf300; del buf300  # reuse
        # Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_red_fused_mean_38.run(buf340, buf341, 6720, 112, grid=grid(6720), stream=stream0)
        buf342 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf343 = reinterpret_tensor(buf342, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf342  # reuse
        # Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_per_fused_mean_39.run(buf343, buf341, 960, 7, grid=grid(960), stream=stream0)
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (8, 16, 1, 1), (16, 1, 1, 1))
        buf345 = reinterpret_tensor(buf344, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf344  # reuse
        buf346 = reinterpret_tensor(buf90, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf90  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_47.run(buf345, primals_211, buf346, 128, grid=grid(128), stream=stream0)
        del primals_211
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 120, 1, 1), (120, 1, 1, 1))
        buf348 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_41.run(buf347, primals_213, buf348, 960, grid=grid(960), stream=stream0)
        buf349 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten.mul]
        triton_poi_fused_mul_42.run(buf340, buf348, buf349, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf351 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf350, buf351, 320, 784, grid=grid(320, 784), stream=stream0)
        buf352 = buf313; del buf313  # reuse
        buf353 = buf312; del buf312  # reuse
        buf354 = buf311; del buf311  # reuse
        # Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf351, buf352, buf353, buf354, 1960, 128, grid=grid(1960), stream=stream0)
        buf355 = buf315; del buf315  # reuse
        buf356 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf358 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf352, buf353, buf354, primals_413, primals_414, buf355, buf356, buf358, primals_413, primals_414, 40, 49, grid=grid(40), stream=stream0)
        del primals_413
        del primals_414
        buf359 = reinterpret_tensor(buf350, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf350  # reuse
        # Source Nodes: [shortcut_9, x_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_48.run(buf351, buf355, buf356, primals_51, primals_52, buf318, buf359, 250880, grid=grid(250880), stream=stream0)
        del primals_52
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf361 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf360, buf361, 960, 784, grid=grid(960, 784), stream=stream0)
        buf362 = buf334; del buf334  # reuse
        buf363 = buf333; del buf333  # reuse
        buf364 = buf332; del buf332  # reuse
        # Source Nodes: [x_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf361, buf362, buf363, buf364, 5880, 128, grid=grid(5880), stream=stream0)
        buf365 = buf336; del buf336  # reuse
        buf366 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf368 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf362, buf363, buf364, primals_416, primals_417, buf365, buf366, buf368, primals_416, primals_417, 120, 49, grid=grid(120), stream=stream0)
        del primals_416
        del primals_417
        buf369 = reinterpret_tensor(buf360, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf360  # reuse
        buf370 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147, x_150], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf361, buf365, buf366, primals_53, primals_54, buf369, buf370, 752640, grid=grid(752640), stream=stream0)
        del primals_54
        # Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_216, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf371, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf372 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf371, buf372, 960, 784, grid=grid(960, 784), stream=stream0)
        buf373 = buf364; del buf364  # reuse
        buf374 = buf363; del buf363  # reuse
        buf375 = buf362; del buf362  # reuse
        # Source Nodes: [x_152], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf372, buf373, buf374, buf375, 5880, 128, grid=grid(5880), stream=stream0)
        buf376 = buf366; del buf366  # reuse
        buf377 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf379 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf373, buf374, buf375, primals_419, primals_420, buf376, buf377, buf379, primals_419, primals_420, 120, 49, grid=grid(120), stream=stream0)
        del primals_419
        del primals_420
        buf380 = reinterpret_tensor(buf371, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf371  # reuse
        buf381 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152, x_155], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf372, buf376, buf377, primals_55, primals_56, buf380, buf381, 752640, grid=grid(752640), stream=stream0)
        del primals_56
        buf382 = buf341; del buf341  # reuse
        # Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_red_fused_mean_38.run(buf381, buf382, 6720, 112, grid=grid(6720), stream=stream0)
        buf383 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf384 = reinterpret_tensor(buf383, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf383  # reuse
        # Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_per_fused_mean_39.run(buf384, buf382, 960, 7, grid=grid(960), stream=stream0)
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (8, 16, 1, 1), (16, 1, 1, 1))
        buf386 = reinterpret_tensor(buf385, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf385  # reuse
        buf387 = reinterpret_tensor(buf89, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf89  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_47.run(buf386, primals_218, buf387, 128, grid=grid(128), stream=stream0)
        del primals_218
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf387, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (8, 120, 1, 1), (120, 1, 1, 1))
        buf389 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_41.run(buf388, primals_220, buf389, 960, grid=grid(960), stream=stream0)
        buf390 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.mul]
        triton_poi_fused_mul_42.run(buf381, buf389, buf390, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_157], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf392 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf391, buf392, 320, 784, grid=grid(320, 784), stream=stream0)
        buf393 = buf354; del buf354  # reuse
        buf394 = buf353; del buf353  # reuse
        buf395 = buf352; del buf352  # reuse
        # Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf392, buf393, buf394, buf395, 1960, 128, grid=grid(1960), stream=stream0)
        buf396 = buf356; del buf356  # reuse
        buf397 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf399 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf393, buf394, buf395, primals_422, primals_423, buf396, buf397, buf399, primals_422, primals_423, 40, 49, grid=grid(40), stream=stream0)
        del primals_422
        del primals_423
        buf400 = reinterpret_tensor(buf391, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf391  # reuse
        # Source Nodes: [shortcut_10, x_158], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_48.run(buf392, buf396, buf397, primals_57, primals_58, buf359, buf400, 250880, grid=grid(250880), stream=stream0)
        del primals_58
        # Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf402 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf401, buf402, 960, 784, grid=grid(960, 784), stream=stream0)
        buf403 = buf375; del buf375  # reuse
        buf404 = buf374; del buf374  # reuse
        buf405 = buf373; del buf373  # reuse
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf402, buf403, buf404, buf405, 5880, 128, grid=grid(5880), stream=stream0)
        buf406 = buf377; del buf377  # reuse
        buf407 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf409 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf403, buf404, buf405, primals_425, primals_426, buf406, buf407, buf409, primals_425, primals_426, 120, 49, grid=grid(120), stream=stream0)
        del primals_425
        del primals_426
        buf410 = reinterpret_tensor(buf401, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf401  # reuse
        buf411 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf402, buf406, buf407, primals_59, primals_60, buf410, buf411, 752640, grid=grid(752640), stream=stream0)
        del primals_60
        # Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_223, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf412, (8, 120, 28, 28), (94080, 784, 28, 1))
        buf413 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf412, buf413, 960, 784, grid=grid(960, 784), stream=stream0)
        buf414 = buf405; del buf405  # reuse
        buf415 = buf404; del buf404  # reuse
        buf416 = buf403; del buf403  # reuse
        # Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf413, buf414, buf415, buf416, 5880, 128, grid=grid(5880), stream=stream0)
        buf417 = buf407; del buf407  # reuse
        buf418 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf420 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf414, buf415, buf416, primals_428, primals_429, buf417, buf418, buf420, primals_428, primals_429, 120, 49, grid=grid(120), stream=stream0)
        del buf414
        del buf415
        del buf416
        del primals_428
        del primals_429
        buf421 = reinterpret_tensor(buf412, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf412  # reuse
        buf422 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169, x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_37.run(buf413, buf417, buf418, primals_61, primals_62, buf421, buf422, 752640, grid=grid(752640), stream=stream0)
        del primals_62
        buf423 = buf382; del buf382  # reuse
        # Source Nodes: [x_se_16], Original ATen: [aten.mean]
        triton_red_fused_mean_38.run(buf422, buf423, 6720, 112, grid=grid(6720), stream=stream0)
        buf424 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf425 = reinterpret_tensor(buf424, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf424  # reuse
        # Source Nodes: [x_se_16], Original ATen: [aten.mean]
        triton_per_fused_mean_39.run(buf425, buf423, 960, 7, grid=grid(960), stream=stream0)
        del buf423
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 16, 1, 1), (16, 1, 1, 1))
        buf427 = reinterpret_tensor(buf426, (8, 16, 1, 1), (16, 1, 16, 16), 0); del buf426  # reuse
        buf428 = empty_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_47.run(buf427, primals_225, buf428, 128, grid=grid(128), stream=stream0)
        del primals_225
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf428, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (8, 120, 1, 1), (120, 1, 1, 1))
        buf430 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____4___se_gate, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_41.run(buf429, primals_227, buf430, 960, grid=grid(960), stream=stream0)
        buf431 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten.mul]
        triton_poi_fused_mul_42.run(buf422, buf430, buf431, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 40, 28, 28), (31360, 784, 28, 1))
        buf433 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf432, buf433, 320, 784, grid=grid(320, 784), stream=stream0)
        buf434 = buf395; del buf395  # reuse
        buf435 = buf394; del buf394  # reuse
        buf436 = buf393; del buf393  # reuse
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf433, buf434, buf435, buf436, 1960, 128, grid=grid(1960), stream=stream0)
        buf437 = buf397; del buf397  # reuse
        buf438 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cuda', dtype=torch.float32)
        buf440 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf434, buf435, buf436, primals_431, primals_432, buf437, buf438, buf440, primals_431, primals_432, 40, 49, grid=grid(40), stream=stream0)
        del buf434
        del buf435
        del buf436
        del primals_431
        del primals_432
        buf441 = reinterpret_tensor(buf432, (8, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf432  # reuse
        # Source Nodes: [shortcut_11, x_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_48.run(buf433, buf437, buf438, primals_63, primals_64, buf400, buf441, 250880, grid=grid(250880), stream=stream0)
        del buf438
        del primals_64
        # Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 200, 28, 28), (156800, 784, 28, 1))
        buf443 = empty_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_49.run(buf442, buf443, 1600, 784, grid=grid(1600, 784), stream=stream0)
        buf444 = empty_strided((1, 200, 1, 1, 49), (9800, 1, 9800, 9800, 200), device='cuda', dtype=torch.float32)
        buf445 = empty_strided((1, 200, 1, 1, 49), (9800, 1, 9800, 9800, 200), device='cuda', dtype=torch.float32)
        buf446 = empty_strided((1, 200, 1, 1, 49), (9800, 1, 9800, 9800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf443, buf444, buf445, buf446, 9800, 128, grid=grid(9800), stream=stream0)
        buf447 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf448 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf450 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_51.run(buf444, buf445, buf446, primals_434, primals_435, buf447, buf448, buf450, primals_434, primals_435, 200, 49, grid=grid(200), stream=stream0)
        del buf444
        del buf445
        del buf446
        del primals_434
        del primals_435
        buf451 = reinterpret_tensor(buf442, (8, 200, 28, 28), (156800, 1, 5600, 200), 0); del buf442  # reuse
        buf452 = empty_strided((8, 200, 28, 28), (156800, 1, 5600, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_52.run(buf443, buf447, buf448, primals_65, primals_66, buf451, buf452, 1254400, grid=grid(1254400), stream=stream0)
        del primals_66
        # Source Nodes: [x_185], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_230, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf453, (8, 200, 14, 14), (39200, 196, 14, 1))
        buf454 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf453, buf454, 1600, 196, grid=grid(1600, 196), stream=stream0)
        buf455 = empty_strided((1, 200, 1, 1, 13), (2600, 1, 2600, 2600, 200), device='cuda', dtype=torch.float32)
        buf456 = empty_strided((1, 200, 1, 1, 13), (2600, 1, 2600, 2600, 200), device='cuda', dtype=torch.float32)
        buf457 = empty_strided((1, 200, 1, 1, 13), (2600, 1, 2600, 2600, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf454, buf455, buf456, buf457, 2600, 121, grid=grid(2600), stream=stream0)
        buf458 = buf448; del buf448  # reuse
        buf459 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf461 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_55.run(buf455, buf456, buf457, primals_437, primals_438, buf458, buf459, buf461, primals_437, primals_438, 200, 13, grid=grid(200), stream=stream0)
        del buf455
        del buf456
        del buf457
        del primals_437
        del primals_438
        buf462 = reinterpret_tensor(buf453, (8, 200, 14, 14), (39200, 1, 2800, 200), 0); del buf453  # reuse
        buf463 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_186, x_189], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_56.run(buf454, buf458, buf459, primals_67, primals_68, buf462, buf463, 313600, grid=grid(313600), stream=stream0)
        del buf459
        del primals_68
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (8, 72, 14, 14), (14112, 196, 14, 1))
        buf465 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf464, buf465, 576, 196, grid=grid(576, 196), stream=stream0)
        buf466 = empty_strided((1, 72, 1, 1, 13), (936, 1, 936, 936, 72), device='cuda', dtype=torch.float32)
        buf467 = empty_strided((1, 72, 1, 1, 13), (936, 1, 936, 936, 72), device='cuda', dtype=torch.float32)
        buf468 = empty_strided((1, 72, 1, 1, 13), (936, 1, 936, 936, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf465, buf466, buf467, buf468, 936, 121, grid=grid(936), stream=stream0)
        buf469 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf470 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf472 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf466, buf467, buf468, primals_440, primals_441, buf469, buf470, buf472, primals_440, primals_441, 72, 13, grid=grid(72), stream=stream0)
        del primals_440
        del primals_441
        buf473 = reinterpret_tensor(buf464, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf464  # reuse
        # Source Nodes: [x_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_60.run(buf465, buf469, buf470, primals_69, primals_70, buf473, 112896, grid=grid(112896), stream=stream0)
        del primals_70
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf475 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf474, buf475, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf476 = empty_strided((1, 216, 1, 1, 13), (2808, 1, 2808, 2808, 216), device='cuda', dtype=torch.float32)
        buf477 = empty_strided((1, 216, 1, 1, 13), (2808, 1, 2808, 2808, 216), device='cuda', dtype=torch.float32)
        buf478 = empty_strided((1, 216, 1, 1, 13), (2808, 1, 2808, 2808, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf475, buf476, buf477, buf478, 2808, 121, grid=grid(2808), stream=stream0)
        buf479 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf480 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf482 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf476, buf477, buf478, primals_443, primals_444, buf479, buf480, buf482, primals_443, primals_444, 216, 13, grid=grid(216), stream=stream0)
        del primals_443
        del primals_444
        buf483 = reinterpret_tensor(buf474, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf474  # reuse
        buf484 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197, x_200], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf475, buf479, buf480, primals_71, primals_72, buf483, buf484, 338688, grid=grid(338688), stream=stream0)
        del primals_72
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf484, primals_233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf485, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf486 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf485, buf486, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf487 = buf478; del buf478  # reuse
        buf488 = buf477; del buf477  # reuse
        buf489 = buf476; del buf476  # reuse
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf486, buf487, buf488, buf489, 2808, 121, grid=grid(2808), stream=stream0)
        buf490 = buf480; del buf480  # reuse
        buf491 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf493 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf487, buf488, buf489, primals_446, primals_447, buf490, buf491, buf493, primals_446, primals_447, 216, 13, grid=grid(216), stream=stream0)
        del primals_446
        del primals_447
        buf494 = reinterpret_tensor(buf485, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf485  # reuse
        buf495 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202, x_205], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf486, buf490, buf491, primals_73, primals_74, buf494, buf495, 338688, grid=grid(338688), stream=stream0)
        del primals_74
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (8, 72, 14, 14), (14112, 196, 14, 1))
        buf497 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf496, buf497, 576, 196, grid=grid(576, 196), stream=stream0)
        buf498 = buf468; del buf468  # reuse
        buf499 = buf467; del buf467  # reuse
        buf500 = buf466; del buf466  # reuse
        # Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf497, buf498, buf499, buf500, 936, 121, grid=grid(936), stream=stream0)
        buf501 = buf470; del buf470  # reuse
        buf502 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf504 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf498, buf499, buf500, primals_449, primals_450, buf501, buf502, buf504, primals_449, primals_450, 72, 13, grid=grid(72), stream=stream0)
        del primals_449
        del primals_450
        buf505 = reinterpret_tensor(buf496, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf496  # reuse
        # Source Nodes: [shortcut_13, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_65.run(buf497, buf501, buf502, primals_75, primals_76, buf473, buf505, 112896, grid=grid(112896), stream=stream0)
        del primals_76
        # Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf507 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf506, buf507, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf508 = buf489; del buf489  # reuse
        buf509 = buf488; del buf488  # reuse
        buf510 = buf487; del buf487  # reuse
        # Source Nodes: [x_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf507, buf508, buf509, buf510, 2808, 121, grid=grid(2808), stream=stream0)
        buf511 = buf491; del buf491  # reuse
        buf512 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf514 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf508, buf509, buf510, primals_452, primals_453, buf511, buf512, buf514, primals_452, primals_453, 216, 13, grid=grid(216), stream=stream0)
        del primals_452
        del primals_453
        buf515 = reinterpret_tensor(buf506, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf506  # reuse
        buf516 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214, x_217], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf507, buf511, buf512, primals_77, primals_78, buf515, buf516, 338688, grid=grid(338688), stream=stream0)
        del primals_78
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_236, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf517, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf518 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf517, buf518, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf519 = buf510; del buf510  # reuse
        buf520 = buf509; del buf509  # reuse
        buf521 = buf508; del buf508  # reuse
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf518, buf519, buf520, buf521, 2808, 121, grid=grid(2808), stream=stream0)
        buf522 = buf512; del buf512  # reuse
        buf523 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf525 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf519, buf520, buf521, primals_455, primals_456, buf522, buf523, buf525, primals_455, primals_456, 216, 13, grid=grid(216), stream=stream0)
        del primals_455
        del primals_456
        buf526 = reinterpret_tensor(buf517, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf517  # reuse
        buf527 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219, x_222], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf518, buf522, buf523, primals_79, primals_80, buf526, buf527, 338688, grid=grid(338688), stream=stream0)
        del primals_80
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (8, 72, 14, 14), (14112, 196, 14, 1))
        buf529 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf528, buf529, 576, 196, grid=grid(576, 196), stream=stream0)
        buf530 = buf500; del buf500  # reuse
        buf531 = buf499; del buf499  # reuse
        buf532 = buf498; del buf498  # reuse
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf529, buf530, buf531, buf532, 936, 121, grid=grid(936), stream=stream0)
        buf533 = buf502; del buf502  # reuse
        buf534 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf536 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf530, buf531, buf532, primals_458, primals_459, buf533, buf534, buf536, primals_458, primals_459, 72, 13, grid=grid(72), stream=stream0)
        del primals_458
        del primals_459
        buf537 = reinterpret_tensor(buf528, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf528  # reuse
        # Source Nodes: [shortcut_14, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_65.run(buf529, buf533, buf534, primals_81, primals_82, buf505, buf537, 112896, grid=grid(112896), stream=stream0)
        del primals_82
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf539 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf538, buf539, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf540 = buf521; del buf521  # reuse
        buf541 = buf520; del buf520  # reuse
        buf542 = buf519; del buf519  # reuse
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf539, buf540, buf541, buf542, 2808, 121, grid=grid(2808), stream=stream0)
        buf543 = buf523; del buf523  # reuse
        buf544 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf546 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf540, buf541, buf542, primals_461, primals_462, buf543, buf544, buf546, primals_461, primals_462, 216, 13, grid=grid(216), stream=stream0)
        del primals_461
        del primals_462
        buf547 = reinterpret_tensor(buf538, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf538  # reuse
        buf548 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231, x_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf539, buf543, buf544, primals_83, primals_84, buf547, buf548, 338688, grid=grid(338688), stream=stream0)
        del primals_84
        # Source Nodes: [x_235], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf549, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf550 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf549, buf550, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf551 = buf542; del buf542  # reuse
        buf552 = buf541; del buf541  # reuse
        buf553 = buf540; del buf540  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf550, buf551, buf552, buf553, 2808, 121, grid=grid(2808), stream=stream0)
        buf554 = buf544; del buf544  # reuse
        buf555 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf557 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf551, buf552, buf553, primals_464, primals_465, buf554, buf555, buf557, primals_464, primals_465, 216, 13, grid=grid(216), stream=stream0)
        del primals_464
        del primals_465
        buf558 = reinterpret_tensor(buf549, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf549  # reuse
        buf559 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236, x_239], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf550, buf554, buf555, primals_85, primals_86, buf558, buf559, 338688, grid=grid(338688), stream=stream0)
        del primals_86
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (8, 72, 14, 14), (14112, 196, 14, 1))
        buf561 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf560, buf561, 576, 196, grid=grid(576, 196), stream=stream0)
        buf562 = buf532; del buf532  # reuse
        buf563 = buf531; del buf531  # reuse
        buf564 = buf530; del buf530  # reuse
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf561, buf562, buf563, buf564, 936, 121, grid=grid(936), stream=stream0)
        buf565 = buf534; del buf534  # reuse
        buf566 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf568 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf562, buf563, buf564, primals_467, primals_468, buf565, buf566, buf568, primals_467, primals_468, 72, 13, grid=grid(72), stream=stream0)
        del primals_467
        del primals_468
        buf569 = reinterpret_tensor(buf560, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf560  # reuse
        # Source Nodes: [shortcut_15, x_242], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_65.run(buf561, buf565, buf566, primals_87, primals_88, buf537, buf569, 112896, grid=grid(112896), stream=stream0)
        del primals_88
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf571 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf570, buf571, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf572 = buf553; del buf553  # reuse
        buf573 = buf552; del buf552  # reuse
        buf574 = buf551; del buf551  # reuse
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf571, buf572, buf573, buf574, 2808, 121, grid=grid(2808), stream=stream0)
        buf575 = buf555; del buf555  # reuse
        buf576 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf578 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf572, buf573, buf574, primals_470, primals_471, buf575, buf576, buf578, primals_470, primals_471, 216, 13, grid=grid(216), stream=stream0)
        del primals_470
        del primals_471
        buf579 = reinterpret_tensor(buf570, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf570  # reuse
        buf580 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248, x_251], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf571, buf575, buf576, primals_89, primals_90, buf579, buf580, 338688, grid=grid(338688), stream=stream0)
        del primals_90
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf581, (8, 216, 14, 14), (42336, 196, 14, 1))
        buf582 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf581, buf582, 1728, 196, grid=grid(1728, 196), stream=stream0)
        buf583 = buf574; del buf574  # reuse
        buf584 = buf573; del buf573  # reuse
        buf585 = buf572; del buf572  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf582, buf583, buf584, buf585, 2808, 121, grid=grid(2808), stream=stream0)
        buf586 = buf576; del buf576  # reuse
        buf587 = empty_strided((1, 216, 1, 1), (216, 1, 216, 216), device='cuda', dtype=torch.float32)
        buf589 = empty((216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf583, buf584, buf585, primals_473, primals_474, buf586, buf587, buf589, primals_473, primals_474, 216, 13, grid=grid(216), stream=stream0)
        del buf583
        del buf584
        del buf585
        del primals_473
        del primals_474
        buf590 = reinterpret_tensor(buf581, (8, 216, 14, 14), (42336, 1, 3024, 216), 0); del buf581  # reuse
        buf591 = empty_strided((8, 216, 14, 14), (42336, 1, 3024, 216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253, x_256], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_64.run(buf582, buf586, buf587, primals_91, primals_92, buf590, buf591, 338688, grid=grid(338688), stream=stream0)
        del buf587
        del primals_92
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        buf592 = extern_kernels.convolution(buf591, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (8, 72, 14, 14), (14112, 196, 14, 1))
        buf593 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_57.run(buf592, buf593, 576, 196, grid=grid(576, 196), stream=stream0)
        buf594 = buf564; del buf564  # reuse
        buf595 = buf563; del buf563  # reuse
        buf596 = buf562; del buf562  # reuse
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf593, buf594, buf595, buf596, 936, 121, grid=grid(936), stream=stream0)
        buf597 = buf566; del buf566  # reuse
        buf598 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf600 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_59.run(buf594, buf595, buf596, primals_476, primals_477, buf597, buf598, buf600, primals_476, primals_477, 72, 13, grid=grid(72), stream=stream0)
        del buf594
        del buf595
        del buf596
        del primals_476
        del primals_477
        buf601 = reinterpret_tensor(buf592, (8, 72, 14, 14), (14112, 1, 1008, 72), 0); del buf592  # reuse
        # Source Nodes: [shortcut_16, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_65.run(buf593, buf597, buf598, primals_93, primals_94, buf569, buf601, 112896, grid=grid(112896), stream=stream0)
        del buf598
        del primals_94
        # Source Nodes: [x_264], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf603 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf602, buf603, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf604 = empty_strided((1, 360, 1, 1, 13), (4680, 1, 4680, 4680, 360), device='cuda', dtype=torch.float32)
        buf605 = empty_strided((1, 360, 1, 1, 13), (4680, 1, 4680, 4680, 360), device='cuda', dtype=torch.float32)
        buf606 = empty_strided((1, 360, 1, 1, 13), (4680, 1, 4680, 4680, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_265], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf603, buf604, buf605, buf606, 4680, 121, grid=grid(4680), stream=stream0)
        buf607 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf608 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf610 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_265], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf604, buf605, buf606, primals_479, primals_480, buf607, buf608, buf610, primals_479, primals_480, 360, 13, grid=grid(360), stream=stream0)
        del primals_479
        del primals_480
        buf611 = reinterpret_tensor(buf602, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf602  # reuse
        buf612 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_265, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf603, buf607, buf608, primals_95, primals_96, buf611, buf612, 564480, grid=grid(564480), stream=stream0)
        del primals_96
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf613, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf614 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf613, buf614, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf615 = buf606; del buf606  # reuse
        buf616 = buf605; del buf605  # reuse
        buf617 = buf604; del buf604  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf614, buf615, buf616, buf617, 4680, 121, grid=grid(4680), stream=stream0)
        buf618 = buf608; del buf608  # reuse
        buf619 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf621 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf615, buf616, buf617, primals_482, primals_483, buf618, buf619, buf621, primals_482, primals_483, 360, 13, grid=grid(360), stream=stream0)
        del primals_482
        del primals_483
        buf622 = reinterpret_tensor(buf613, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf613  # reuse
        buf623 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_270, x_273], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf614, buf618, buf619, primals_97, primals_98, buf622, buf623, 564480, grid=grid(564480), stream=stream0)
        del primals_98
        buf624 = empty_strided((8, 360, 1, 1, 2), (720, 1, 5760, 5760, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_20], Original ATen: [aten.mean]
        triton_red_fused_mean_70.run(buf623, buf624, 5760, 98, grid=grid(5760), stream=stream0)
        buf625 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf626 = reinterpret_tensor(buf625, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf625  # reuse
        # Source Nodes: [x_se_20], Original ATen: [aten.mean]
        triton_per_fused_mean_71.run(buf626, buf624, 2880, 2, grid=grid(2880), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (8, 24, 1, 1), (24, 1, 1, 1))
        buf628 = reinterpret_tensor(buf627, (8, 24, 1, 1), (24, 1, 24, 24), 0); del buf627  # reuse
        buf629 = empty_strided((8, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_72.run(buf628, primals_247, buf629, 192, grid=grid(192), stream=stream0)
        del primals_247
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (8, 360, 1, 1), (360, 1, 1, 1))
        buf631 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_73.run(buf630, primals_249, buf631, 2880, grid=grid(2880), stream=stream0)
        buf632 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf623, buf631, buf632, 564480, grid=grid(564480), stream=stream0)
        # Source Nodes: [x_275], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf634 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf633, buf634, 960, 196, grid=grid(960, 196), stream=stream0)
        buf635 = empty_strided((1, 120, 1, 1, 13), (1560, 1, 1560, 1560, 120), device='cuda', dtype=torch.float32)
        buf636 = empty_strided((1, 120, 1, 1, 13), (1560, 1, 1560, 1560, 120), device='cuda', dtype=torch.float32)
        buf637 = empty_strided((1, 120, 1, 1, 13), (1560, 1, 1560, 1560, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf634, buf635, buf636, buf637, 1560, 121, grid=grid(1560), stream=stream0)
        buf638 = buf418; del buf418  # reuse
        buf639 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf641 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf635, buf636, buf637, primals_485, primals_486, buf638, buf639, buf641, primals_485, primals_486, 120, 13, grid=grid(120), stream=stream0)
        del primals_485
        del primals_486
        buf642 = reinterpret_tensor(buf633, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf633  # reuse
        # Source Nodes: [x_276], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_78.run(buf634, buf638, buf639, primals_99, primals_100, buf642, 188160, grid=grid(188160), stream=stream0)
        del primals_100
        # Source Nodes: [x_280], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf644 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf643, buf644, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf645 = buf617; del buf617  # reuse
        buf646 = buf616; del buf616  # reuse
        buf647 = buf615; del buf615  # reuse
        # Source Nodes: [x_281], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf644, buf645, buf646, buf647, 4680, 121, grid=grid(4680), stream=stream0)
        buf648 = buf619; del buf619  # reuse
        buf649 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf651 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_281], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf645, buf646, buf647, primals_488, primals_489, buf648, buf649, buf651, primals_488, primals_489, 360, 13, grid=grid(360), stream=stream0)
        del primals_488
        del primals_489
        buf652 = reinterpret_tensor(buf643, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf643  # reuse
        buf653 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_281, x_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf644, buf648, buf649, primals_101, primals_102, buf652, buf653, 564480, grid=grid(564480), stream=stream0)
        del primals_102
        # Source Nodes: [x_285], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, primals_252, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf654, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf655 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf654, buf655, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf656 = buf647; del buf647  # reuse
        buf657 = buf646; del buf646  # reuse
        buf658 = buf645; del buf645  # reuse
        # Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf655, buf656, buf657, buf658, 4680, 121, grid=grid(4680), stream=stream0)
        buf659 = buf649; del buf649  # reuse
        buf660 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf662 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf656, buf657, buf658, primals_491, primals_492, buf659, buf660, buf662, primals_491, primals_492, 360, 13, grid=grid(360), stream=stream0)
        del primals_491
        del primals_492
        buf663 = reinterpret_tensor(buf654, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf654  # reuse
        buf664 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286, x_289], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf655, buf659, buf660, primals_103, primals_104, buf663, buf664, 564480, grid=grid(564480), stream=stream0)
        del primals_104
        buf665 = buf624; del buf624  # reuse
        # Source Nodes: [x_se_24], Original ATen: [aten.mean]
        triton_red_fused_mean_70.run(buf664, buf665, 5760, 98, grid=grid(5760), stream=stream0)
        buf666 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf667 = reinterpret_tensor(buf666, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf666  # reuse
        # Source Nodes: [x_se_24], Original ATen: [aten.mean]
        triton_per_fused_mean_71.run(buf667, buf665, 2880, 2, grid=grid(2880), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf668 = extern_kernels.convolution(buf667, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (8, 32, 1, 1), (32, 1, 1, 1))
        buf669 = reinterpret_tensor(buf668, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf668  # reuse
        buf670 = empty_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_79.run(buf669, primals_254, buf670, 256, grid=grid(256), stream=stream0)
        del primals_254
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf671 = extern_kernels.convolution(buf670, primals_255, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf671, (8, 360, 1, 1), (360, 1, 1, 1))
        buf672 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_73.run(buf671, primals_256, buf672, 2880, grid=grid(2880), stream=stream0)
        buf673 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf664, buf672, buf673, 564480, grid=grid(564480), stream=stream0)
        # Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf675 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf674, buf675, 960, 196, grid=grid(960, 196), stream=stream0)
        buf676 = buf637; del buf637  # reuse
        buf677 = buf636; del buf636  # reuse
        buf678 = buf635; del buf635  # reuse
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf675, buf676, buf677, buf678, 1560, 121, grid=grid(1560), stream=stream0)
        buf679 = buf639; del buf639  # reuse
        buf680 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf682 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf676, buf677, buf678, primals_494, primals_495, buf679, buf680, buf682, primals_494, primals_495, 120, 13, grid=grid(120), stream=stream0)
        del primals_494
        del primals_495
        buf683 = reinterpret_tensor(buf674, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf674  # reuse
        # Source Nodes: [shortcut_18, x_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf675, buf679, buf680, primals_105, primals_106, buf642, buf683, 188160, grid=grid(188160), stream=stream0)
        del primals_106
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf685 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf684, buf685, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf686 = buf658; del buf658  # reuse
        buf687 = buf657; del buf657  # reuse
        buf688 = buf656; del buf656  # reuse
        # Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf685, buf686, buf687, buf688, 4680, 121, grid=grid(4680), stream=stream0)
        buf689 = buf660; del buf660  # reuse
        buf690 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf692 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf686, buf687, buf688, primals_497, primals_498, buf689, buf690, buf692, primals_497, primals_498, 360, 13, grid=grid(360), stream=stream0)
        del primals_497
        del primals_498
        buf693 = reinterpret_tensor(buf684, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf684  # reuse
        buf694 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298, x_301], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf685, buf689, buf690, primals_107, primals_108, buf693, buf694, 564480, grid=grid(564480), stream=stream0)
        del primals_108
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        buf695 = extern_kernels.convolution(buf694, primals_259, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf695, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf696 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf695, buf696, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf697 = buf688; del buf688  # reuse
        buf698 = buf687; del buf687  # reuse
        buf699 = buf686; del buf686  # reuse
        # Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf696, buf697, buf698, buf699, 4680, 121, grid=grid(4680), stream=stream0)
        buf700 = buf690; del buf690  # reuse
        buf701 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf703 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf697, buf698, buf699, primals_500, primals_501, buf700, buf701, buf703, primals_500, primals_501, 360, 13, grid=grid(360), stream=stream0)
        del primals_500
        del primals_501
        buf704 = reinterpret_tensor(buf695, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf695  # reuse
        buf705 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303, x_306], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf696, buf700, buf701, primals_109, primals_110, buf704, buf705, 564480, grid=grid(564480), stream=stream0)
        del primals_110
        buf706 = buf665; del buf665  # reuse
        # Source Nodes: [x_se_28], Original ATen: [aten.mean]
        triton_red_fused_mean_70.run(buf705, buf706, 5760, 98, grid=grid(5760), stream=stream0)
        buf707 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf708 = reinterpret_tensor(buf707, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf707  # reuse
        # Source Nodes: [x_se_28], Original ATen: [aten.mean]
        triton_per_fused_mean_71.run(buf708, buf706, 2880, 2, grid=grid(2880), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf709 = extern_kernels.convolution(buf708, primals_260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf709, (8, 32, 1, 1), (32, 1, 1, 1))
        buf710 = reinterpret_tensor(buf709, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf709  # reuse
        buf711 = empty_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_79.run(buf710, primals_261, buf711, 256, grid=grid(256), stream=stream0)
        del primals_261
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf712 = extern_kernels.convolution(buf711, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf712, (8, 360, 1, 1), (360, 1, 1, 1))
        buf713 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_73.run(buf712, primals_263, buf713, 2880, grid=grid(2880), stream=stream0)
        buf714 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf705, buf713, buf714, 564480, grid=grid(564480), stream=stream0)
        # Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf714, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf716 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf715, buf716, 960, 196, grid=grid(960, 196), stream=stream0)
        buf717 = buf678; del buf678  # reuse
        buf718 = buf677; del buf677  # reuse
        buf719 = buf676; del buf676  # reuse
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf716, buf717, buf718, buf719, 1560, 121, grid=grid(1560), stream=stream0)
        buf720 = buf680; del buf680  # reuse
        buf721 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf723 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_309], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf717, buf718, buf719, primals_503, primals_504, buf720, buf721, buf723, primals_503, primals_504, 120, 13, grid=grid(120), stream=stream0)
        del primals_503
        del primals_504
        buf724 = reinterpret_tensor(buf715, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf715  # reuse
        # Source Nodes: [shortcut_19, x_309], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf716, buf720, buf721, primals_111, primals_112, buf683, buf724, 188160, grid=grid(188160), stream=stream0)
        del primals_112
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf725, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf726 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf725, buf726, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf727 = buf699; del buf699  # reuse
        buf728 = buf698; del buf698  # reuse
        buf729 = buf697; del buf697  # reuse
        # Source Nodes: [x_315], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf726, buf727, buf728, buf729, 4680, 121, grid=grid(4680), stream=stream0)
        buf730 = buf701; del buf701  # reuse
        buf731 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf733 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf727, buf728, buf729, primals_506, primals_507, buf730, buf731, buf733, primals_506, primals_507, 360, 13, grid=grid(360), stream=stream0)
        del primals_506
        del primals_507
        buf734 = reinterpret_tensor(buf725, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf725  # reuse
        buf735 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315, x_318], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf726, buf730, buf731, primals_113, primals_114, buf734, buf735, 564480, grid=grid(564480), stream=stream0)
        del primals_114
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        buf736 = extern_kernels.convolution(buf735, primals_266, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf736, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf737 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf736, buf737, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf738 = buf729; del buf729  # reuse
        buf739 = buf728; del buf728  # reuse
        buf740 = buf727; del buf727  # reuse
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf737, buf738, buf739, buf740, 4680, 121, grid=grid(4680), stream=stream0)
        buf741 = buf731; del buf731  # reuse
        buf742 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf744 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf738, buf739, buf740, primals_509, primals_510, buf741, buf742, buf744, primals_509, primals_510, 360, 13, grid=grid(360), stream=stream0)
        del primals_509
        del primals_510
        buf745 = reinterpret_tensor(buf736, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf736  # reuse
        buf746 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320, x_323], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf737, buf741, buf742, primals_115, primals_116, buf745, buf746, 564480, grid=grid(564480), stream=stream0)
        del primals_116
        buf747 = buf706; del buf706  # reuse
        # Source Nodes: [x_se_32], Original ATen: [aten.mean]
        triton_red_fused_mean_70.run(buf746, buf747, 5760, 98, grid=grid(5760), stream=stream0)
        buf748 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf749 = reinterpret_tensor(buf748, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf748  # reuse
        # Source Nodes: [x_se_32], Original ATen: [aten.mean]
        triton_per_fused_mean_71.run(buf749, buf747, 2880, 2, grid=grid(2880), stream=stream0)
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf750 = extern_kernels.convolution(buf749, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf750, (8, 32, 1, 1), (32, 1, 1, 1))
        buf751 = reinterpret_tensor(buf750, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf750  # reuse
        buf752 = empty_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_79.run(buf751, primals_268, buf752, 256, grid=grid(256), stream=stream0)
        del primals_268
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf753 = extern_kernels.convolution(buf752, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf753, (8, 360, 1, 1), (360, 1, 1, 1))
        buf754 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_se_35], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_73.run(buf753, primals_270, buf754, 2880, grid=grid(2880), stream=stream0)
        buf755 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf746, buf754, buf755, 564480, grid=grid(564480), stream=stream0)
        # Source Nodes: [x_325], Original ATen: [aten.convolution]
        buf756 = extern_kernels.convolution(buf755, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf756, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf757 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf756, buf757, 960, 196, grid=grid(960, 196), stream=stream0)
        buf758 = buf719; del buf719  # reuse
        buf759 = buf718; del buf718  # reuse
        buf760 = buf717; del buf717  # reuse
        # Source Nodes: [x_326], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf757, buf758, buf759, buf760, 1560, 121, grid=grid(1560), stream=stream0)
        buf761 = buf721; del buf721  # reuse
        buf762 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf764 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf758, buf759, buf760, primals_512, primals_513, buf761, buf762, buf764, primals_512, primals_513, 120, 13, grid=grid(120), stream=stream0)
        del primals_512
        del primals_513
        buf765 = reinterpret_tensor(buf756, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf756  # reuse
        # Source Nodes: [shortcut_20, x_326], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf757, buf761, buf762, primals_117, primals_118, buf724, buf765, 188160, grid=grid(188160), stream=stream0)
        del primals_118
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        buf766 = extern_kernels.convolution(buf765, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf766, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf767 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_331], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf766, buf767, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf768 = buf740; del buf740  # reuse
        buf769 = buf739; del buf739  # reuse
        buf770 = buf738; del buf738  # reuse
        # Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf767, buf768, buf769, buf770, 4680, 121, grid=grid(4680), stream=stream0)
        buf771 = buf742; del buf742  # reuse
        buf772 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf774 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf768, buf769, buf770, primals_515, primals_516, buf771, buf772, buf774, primals_515, primals_516, 360, 13, grid=grid(360), stream=stream0)
        del primals_515
        del primals_516
        buf775 = reinterpret_tensor(buf766, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf766  # reuse
        buf776 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf767, buf771, buf772, primals_119, primals_120, buf775, buf776, 564480, grid=grid(564480), stream=stream0)
        del primals_120
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        buf777 = extern_kernels.convolution(buf776, primals_273, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf777, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf778 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf777, buf778, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf779 = buf770; del buf770  # reuse
        buf780 = buf769; del buf769  # reuse
        buf781 = buf768; del buf768  # reuse
        # Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf778, buf779, buf780, buf781, 4680, 121, grid=grid(4680), stream=stream0)
        buf782 = buf772; del buf772  # reuse
        buf783 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf785 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf779, buf780, buf781, primals_518, primals_519, buf782, buf783, buf785, primals_518, primals_519, 360, 13, grid=grid(360), stream=stream0)
        del primals_518
        del primals_519
        buf786 = reinterpret_tensor(buf777, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf777  # reuse
        buf787 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_337, x_340], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf778, buf782, buf783, primals_121, primals_122, buf786, buf787, 564480, grid=grid(564480), stream=stream0)
        del primals_122
        buf788 = buf747; del buf747  # reuse
        # Source Nodes: [x_se_36], Original ATen: [aten.mean]
        triton_red_fused_mean_70.run(buf787, buf788, 5760, 98, grid=grid(5760), stream=stream0)
        buf789 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf790 = reinterpret_tensor(buf789, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf789  # reuse
        # Source Nodes: [x_se_36], Original ATen: [aten.mean]
        triton_per_fused_mean_71.run(buf790, buf788, 2880, 2, grid=grid(2880), stream=stream0)
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf791 = extern_kernels.convolution(buf790, primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf791, (8, 32, 1, 1), (32, 1, 1, 1))
        buf792 = reinterpret_tensor(buf791, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf791  # reuse
        buf793 = empty_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_79.run(buf792, primals_275, buf793, 256, grid=grid(256), stream=stream0)
        del primals_275
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf794 = extern_kernels.convolution(buf793, primals_276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf794, (8, 360, 1, 1), (360, 1, 1, 1))
        buf795 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____4___se_gate, x_se_39], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_73.run(buf794, primals_277, buf795, 2880, grid=grid(2880), stream=stream0)
        buf796 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf787, buf795, buf796, 564480, grid=grid(564480), stream=stream0)
        # Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf797 = extern_kernels.convolution(buf796, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf797, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf798 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_342], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf797, buf798, 960, 196, grid=grid(960, 196), stream=stream0)
        buf799 = buf760; del buf760  # reuse
        buf800 = buf759; del buf759  # reuse
        buf801 = buf758; del buf758  # reuse
        # Source Nodes: [x_343], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf798, buf799, buf800, buf801, 1560, 121, grid=grid(1560), stream=stream0)
        buf802 = buf762; del buf762  # reuse
        buf803 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf805 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf799, buf800, buf801, primals_521, primals_522, buf802, buf803, buf805, primals_521, primals_522, 120, 13, grid=grid(120), stream=stream0)
        del primals_521
        del primals_522
        buf806 = reinterpret_tensor(buf797, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf797  # reuse
        # Source Nodes: [shortcut_21, x_343], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf798, buf802, buf803, primals_123, primals_124, buf765, buf806, 188160, grid=grid(188160), stream=stream0)
        del primals_124
        # Source Nodes: [x_348], Original ATen: [aten.convolution]
        buf807 = extern_kernels.convolution(buf806, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf807, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf808 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_348], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf807, buf808, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf809 = buf781; del buf781  # reuse
        buf810 = buf780; del buf780  # reuse
        buf811 = buf779; del buf779  # reuse
        # Source Nodes: [x_349], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf808, buf809, buf810, buf811, 4680, 121, grid=grid(4680), stream=stream0)
        buf812 = buf783; del buf783  # reuse
        buf813 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf815 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_349], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf809, buf810, buf811, primals_524, primals_525, buf812, buf813, buf815, primals_524, primals_525, 360, 13, grid=grid(360), stream=stream0)
        del primals_524
        del primals_525
        buf816 = reinterpret_tensor(buf807, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf807  # reuse
        buf817 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_349, x_352], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf808, buf812, buf813, primals_125, primals_126, buf816, buf817, 564480, grid=grid(564480), stream=stream0)
        del primals_126
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        buf818 = extern_kernels.convolution(buf817, primals_280, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=360, bias=None)
        assert_size_stride(buf818, (8, 360, 14, 14), (70560, 196, 14, 1))
        buf819 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_353], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf818, buf819, 2880, 196, grid=grid(2880, 196), stream=stream0)
        buf820 = buf811; del buf811  # reuse
        buf821 = buf810; del buf810  # reuse
        buf822 = buf809; del buf809  # reuse
        # Source Nodes: [x_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_67.run(buf819, buf820, buf821, buf822, 4680, 121, grid=grid(4680), stream=stream0)
        buf823 = buf813; del buf813  # reuse
        buf824 = empty_strided((1, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        buf826 = empty((360, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_68.run(buf820, buf821, buf822, primals_527, primals_528, buf823, buf824, buf826, primals_527, primals_528, 360, 13, grid=grid(360), stream=stream0)
        del buf820
        del buf821
        del buf822
        del primals_527
        del primals_528
        buf827 = reinterpret_tensor(buf818, (8, 360, 14, 14), (70560, 1, 5040, 360), 0); del buf818  # reuse
        buf828 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_354, x_357], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_69.run(buf819, buf823, buf824, primals_127, primals_128, buf827, buf828, 564480, grid=grid(564480), stream=stream0)
        del buf824
        del primals_128
        buf829 = buf788; del buf788  # reuse
        # Source Nodes: [x_se_40], Original ATen: [aten.mean]
        triton_red_fused_mean_70.run(buf828, buf829, 5760, 98, grid=grid(5760), stream=stream0)
        buf830 = empty_strided((8, 360, 1, 1), (360, 1, 2880, 2880), device='cuda', dtype=torch.float32)
        buf831 = reinterpret_tensor(buf830, (8, 360, 1, 1), (360, 1, 360, 360), 0); del buf830  # reuse
        # Source Nodes: [x_se_40], Original ATen: [aten.mean]
        triton_per_fused_mean_71.run(buf831, buf829, 2880, 2, grid=grid(2880), stream=stream0)
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf832 = extern_kernels.convolution(buf831, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf832, (8, 32, 1, 1), (32, 1, 1, 1))
        buf833 = reinterpret_tensor(buf832, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf832  # reuse
        buf834 = empty_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_79.run(buf833, primals_282, buf834, 256, grid=grid(256), stream=stream0)
        del primals_282
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf835 = extern_kernels.convolution(buf834, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (8, 360, 1, 1), (360, 1, 1, 1))
        buf836 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____5___se_gate, x_se_43], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_73.run(buf835, primals_284, buf836, 2880, grid=grid(2880), stream=stream0)
        buf837 = empty_strided((8, 360, 14, 14), (70560, 1, 5040, 360), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_358], Original ATen: [aten.mul]
        triton_poi_fused_mul_74.run(buf828, buf836, buf837, 564480, grid=grid(564480), stream=stream0)
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        buf838 = extern_kernels.convolution(buf837, primals_285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf838, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf839 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf838, buf839, 960, 196, grid=grid(960, 196), stream=stream0)
        buf840 = buf801; del buf801  # reuse
        buf841 = buf800; del buf800  # reuse
        buf842 = buf799; del buf799  # reuse
        # Source Nodes: [x_360], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_76.run(buf839, buf840, buf841, buf842, 1560, 121, grid=grid(1560), stream=stream0)
        buf843 = buf803; del buf803  # reuse
        buf844 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.float32)
        buf846 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_360], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_77.run(buf840, buf841, buf842, primals_530, primals_531, buf843, buf844, buf846, primals_530, primals_531, 120, 13, grid=grid(120), stream=stream0)
        del buf840
        del buf841
        del buf842
        del primals_530
        del primals_531
        buf847 = reinterpret_tensor(buf838, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf838  # reuse
        # Source Nodes: [shortcut_22, x_360], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_80.run(buf839, buf843, buf844, primals_129, primals_130, buf806, buf847, 188160, grid=grid(188160), stream=stream0)
        del buf844
        del primals_130
        # Source Nodes: [x_365], Original ATen: [aten.convolution]
        buf848 = extern_kernels.convolution(buf847, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf848, (8, 720, 14, 14), (141120, 196, 14, 1))
        buf849 = empty_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_365], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf848, buf849, 5760, 196, grid=grid(5760, 196), stream=stream0)
        buf850 = empty_strided((1, 720, 1, 1, 13), (9360, 1, 9360, 9360, 720), device='cuda', dtype=torch.float32)
        buf851 = empty_strided((1, 720, 1, 1, 13), (9360, 1, 9360, 9360, 720), device='cuda', dtype=torch.float32)
        buf852 = empty_strided((1, 720, 1, 1, 13), (9360, 1, 9360, 9360, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_366], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf849, buf850, buf851, buf852, 9360, 121, grid=grid(9360), stream=stream0)
        buf853 = empty_strided((1, 720, 1, 1), (720, 1, 720, 720), device='cuda', dtype=torch.float32)
        buf854 = empty_strided((1, 720, 1, 1), (720, 1, 720, 720), device='cuda', dtype=torch.float32)
        buf856 = empty((720, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_366], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf850, buf851, buf852, primals_533, primals_534, buf853, buf854, buf856, primals_533, primals_534, 720, 13, grid=grid(720), stream=stream0)
        del buf850
        del buf851
        del buf852
        del primals_533
        del primals_534
        buf857 = reinterpret_tensor(buf848, (8, 720, 14, 14), (141120, 1, 10080, 720), 0); del buf848  # reuse
        buf858 = empty_strided((8, 720, 14, 14), (141120, 1, 10080, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_366, x_369], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_84.run(buf849, buf853, buf854, primals_131, primals_132, buf857, buf858, 1128960, grid=grid(1128960), stream=stream0)
        del primals_132
        # Source Nodes: [x_370], Original ATen: [aten.convolution]
        buf859 = extern_kernels.convolution(buf858, primals_287, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=720, bias=None)
        assert_size_stride(buf859, (8, 720, 7, 7), (35280, 49, 7, 1))
        buf860 = empty_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_85.run(buf859, buf860, 5760, 49, grid=grid(5760, 49), stream=stream0)
        buf861 = empty_strided((1, 720, 1, 1, 4), (2880, 1, 2880, 2880, 720), device='cuda', dtype=torch.float32)
        buf862 = empty_strided((1, 720, 1, 1, 4), (2880, 1, 2880, 2880, 720), device='cuda', dtype=torch.float32)
        buf863 = empty_strided((1, 720, 1, 1, 4), (2880, 1, 2880, 2880, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_86.run(buf860, buf861, buf862, buf863, 2880, 98, grid=grid(2880), stream=stream0)
        buf864 = buf854; del buf854  # reuse
        buf865 = empty_strided((1, 720, 1, 1), (720, 1, 720, 720), device='cuda', dtype=torch.float32)
        buf867 = empty((720, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_87.run(buf861, buf862, buf863, primals_536, primals_537, buf864, buf865, buf867, primals_536, primals_537, 720, 4, grid=grid(720), stream=stream0)
        del buf861
        del buf862
        del buf863
        del primals_536
        del primals_537
        buf868 = reinterpret_tensor(buf859, (8, 720, 7, 7), (35280, 1, 5040, 720), 0); del buf859  # reuse
        buf869 = empty_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371, x_374], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_88.run(buf860, buf864, buf865, primals_133, primals_134, buf868, buf869, 282240, grid=grid(282240), stream=stream0)
        del buf865
        del primals_134
        buf870 = reinterpret_tensor(buf829, (8, 720, 1, 1), (720, 1, 5760, 5760), 0); del buf829  # reuse
        buf871 = reinterpret_tensor(buf870, (8, 720, 1, 1), (720, 1, 720, 720), 0); del buf870  # reuse
        # Source Nodes: [x_se_44], Original ATen: [aten.mean]
        triton_per_fused_mean_89.run(buf871, buf869, 5760, 49, grid=grid(5760), stream=stream0)
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf871, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (8, 32, 1, 1), (32, 1, 1, 1))
        buf873 = reinterpret_tensor(buf872, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf872  # reuse
        buf874 = empty_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_79.run(buf873, primals_289, buf874, 256, grid=grid(256), stream=stream0)
        del primals_289
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf875 = extern_kernels.convolution(buf874, primals_290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf875, (8, 720, 1, 1), (720, 1, 1, 1))
        buf876 = empty_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_90.run(buf875, primals_291, buf876, 5760, grid=grid(5760), stream=stream0)
        buf877 = empty_strided((8, 720, 7, 7), (35280, 1, 5040, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_375], Original ATen: [aten.mul]
        triton_poi_fused_mul_91.run(buf869, buf876, buf877, 282240, grid=grid(282240), stream=stream0)
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        buf878 = extern_kernels.convolution(buf877, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf878, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf879 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf878, buf879, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf880 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf881 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf882 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_93.run(buf879, buf880, buf881, buf882, 736, 98, grid=grid(736), stream=stream0)
        buf883 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf884 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf886 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_94.run(buf880, buf881, buf882, primals_539, primals_540, buf883, buf884, buf886, primals_539, primals_540, 184, 4, grid=grid(184), stream=stream0)
        del primals_539
        del primals_540
        buf887 = reinterpret_tensor(buf878, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf878  # reuse
        # Source Nodes: [x_377], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_95.run(buf879, buf883, buf884, primals_135, primals_136, buf887, 72128, grid=grid(72128), stream=stream0)
        del primals_136
        # Source Nodes: [x_381], Original ATen: [aten.convolution]
        buf888 = extern_kernels.convolution(buf887, primals_293, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf888, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf889 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_381], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf888, buf889, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf890 = empty_strided((1, 736, 1, 1, 4), (2944, 1, 2944, 2944, 736), device='cuda', dtype=torch.float32)
        buf891 = empty_strided((1, 736, 1, 1, 4), (2944, 1, 2944, 2944, 736), device='cuda', dtype=torch.float32)
        buf892 = empty_strided((1, 736, 1, 1, 4), (2944, 1, 2944, 2944, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf889, buf890, buf891, buf892, 2944, 98, grid=grid(2944), stream=stream0)
        buf893 = reinterpret_tensor(buf882, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf882  # reuse
        buf894 = reinterpret_tensor(buf881, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf881  # reuse
        buf896 = reinterpret_tensor(buf880, (736, ), (1, ), 0); del buf880  # reuse
        # Source Nodes: [x_382], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf890, buf891, buf892, primals_542, primals_543, buf893, buf894, buf896, primals_542, primals_543, 736, 4, grid=grid(736), stream=stream0)
        del primals_542
        del primals_543
        buf897 = reinterpret_tensor(buf888, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf888  # reuse
        buf898 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382, x_385], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf889, buf893, buf894, primals_137, primals_138, buf897, buf898, 288512, grid=grid(288512), stream=stream0)
        del primals_138
        # Source Nodes: [x_386], Original ATen: [aten.convolution]
        buf899 = extern_kernels.convolution(buf898, primals_294, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf899, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf900 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_386], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf899, buf900, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf901 = buf892; del buf892  # reuse
        buf902 = buf891; del buf891  # reuse
        buf903 = buf890; del buf890  # reuse
        # Source Nodes: [x_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf900, buf901, buf902, buf903, 2944, 98, grid=grid(2944), stream=stream0)
        buf904 = buf894; del buf894  # reuse
        buf905 = empty_strided((1, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        buf907 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_387], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf901, buf902, buf903, primals_545, primals_546, buf904, buf905, buf907, primals_545, primals_546, 736, 4, grid=grid(736), stream=stream0)
        del primals_545
        del primals_546
        buf908 = reinterpret_tensor(buf899, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf899  # reuse
        buf909 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_387, x_390], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf900, buf904, buf905, primals_139, primals_140, buf908, buf909, 288512, grid=grid(288512), stream=stream0)
        del primals_140
        buf910 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf911 = reinterpret_tensor(buf910, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf910  # reuse
        # Source Nodes: [x_se_48], Original ATen: [aten.mean]
        triton_per_fused_mean_100.run(buf911, buf909, 5888, 49, grid=grid(5888), stream=stream0)
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf912 = extern_kernels.convolution(buf911, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf912, (8, 48, 1, 1), (48, 1, 1, 1))
        buf913 = reinterpret_tensor(buf912, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf912  # reuse
        buf914 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_101.run(buf913, primals_296, buf914, 384, grid=grid(384), stream=stream0)
        del primals_296
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf915 = extern_kernels.convolution(buf914, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf915, (8, 736, 1, 1), (736, 1, 1, 1))
        buf916 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_se_51], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_102.run(buf915, primals_298, buf916, 5888, grid=grid(5888), stream=stream0)
        buf917 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_391], Original ATen: [aten.mul]
        triton_poi_fused_mul_103.run(buf909, buf916, buf917, 288512, grid=grid(288512), stream=stream0)
        # Source Nodes: [x_392], Original ATen: [aten.convolution]
        buf918 = extern_kernels.convolution(buf917, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf918, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf919 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_392], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf918, buf919, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf920 = reinterpret_tensor(buf905, (1, 184, 1, 1, 4), (736, 1, 736, 736, 184), 0); del buf905  # reuse
        buf921 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf922 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_393], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_93.run(buf919, buf920, buf921, buf922, 736, 98, grid=grid(736), stream=stream0)
        buf923 = buf884; del buf884  # reuse
        buf924 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf926 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_393], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_94.run(buf920, buf921, buf922, primals_548, primals_549, buf923, buf924, buf926, primals_548, primals_549, 184, 4, grid=grid(184), stream=stream0)
        del primals_548
        del primals_549
        buf927 = reinterpret_tensor(buf918, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf918  # reuse
        # Source Nodes: [shortcut_24, x_393], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_104.run(buf919, buf923, buf924, primals_141, primals_142, buf887, buf927, 72128, grid=grid(72128), stream=stream0)
        del primals_142
        # Source Nodes: [x_398], Original ATen: [aten.convolution]
        buf928 = extern_kernels.convolution(buf927, primals_300, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf928, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf929 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_398], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf928, buf929, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf930 = buf903; del buf903  # reuse
        buf931 = buf902; del buf902  # reuse
        buf932 = buf901; del buf901  # reuse
        # Source Nodes: [x_399], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf929, buf930, buf931, buf932, 2944, 98, grid=grid(2944), stream=stream0)
        buf933 = reinterpret_tensor(buf922, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf922  # reuse
        buf934 = reinterpret_tensor(buf921, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf921  # reuse
        buf936 = reinterpret_tensor(buf920, (736, ), (1, ), 0); del buf920  # reuse
        # Source Nodes: [x_399], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf930, buf931, buf932, primals_551, primals_552, buf933, buf934, buf936, primals_551, primals_552, 736, 4, grid=grid(736), stream=stream0)
        del primals_551
        del primals_552
        buf937 = reinterpret_tensor(buf928, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf928  # reuse
        buf938 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_399, x_402], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf929, buf933, buf934, primals_143, primals_144, buf937, buf938, 288512, grid=grid(288512), stream=stream0)
        del primals_144
        # Source Nodes: [x_403], Original ATen: [aten.convolution]
        buf939 = extern_kernels.convolution(buf938, primals_301, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf939, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf940 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_403], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf939, buf940, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf941 = buf932; del buf932  # reuse
        buf942 = buf931; del buf931  # reuse
        buf943 = buf930; del buf930  # reuse
        # Source Nodes: [x_404], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf940, buf941, buf942, buf943, 2944, 98, grid=grid(2944), stream=stream0)
        buf944 = buf934; del buf934  # reuse
        buf945 = empty_strided((1, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        buf947 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_404], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf941, buf942, buf943, primals_554, primals_555, buf944, buf945, buf947, primals_554, primals_555, 736, 4, grid=grid(736), stream=stream0)
        del primals_554
        del primals_555
        buf948 = reinterpret_tensor(buf939, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf939  # reuse
        buf949 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_404, x_407], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf940, buf944, buf945, primals_145, primals_146, buf948, buf949, 288512, grid=grid(288512), stream=stream0)
        del primals_146
        buf950 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf951 = reinterpret_tensor(buf950, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf950  # reuse
        # Source Nodes: [x_se_52], Original ATen: [aten.mean]
        triton_per_fused_mean_100.run(buf951, buf949, 5888, 49, grid=grid(5888), stream=stream0)
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf952 = extern_kernels.convolution(buf951, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf952, (8, 48, 1, 1), (48, 1, 1, 1))
        buf953 = reinterpret_tensor(buf952, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf952  # reuse
        buf954 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_101.run(buf953, primals_303, buf954, 384, grid=grid(384), stream=stream0)
        del primals_303
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf955 = extern_kernels.convolution(buf954, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf955, (8, 736, 1, 1), (736, 1, 1, 1))
        buf956 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_se_55], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_102.run(buf955, primals_305, buf956, 5888, grid=grid(5888), stream=stream0)
        buf957 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_408], Original ATen: [aten.mul]
        triton_poi_fused_mul_103.run(buf949, buf956, buf957, 288512, grid=grid(288512), stream=stream0)
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        buf958 = extern_kernels.convolution(buf957, primals_306, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf958, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf959 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_409], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf958, buf959, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf960 = reinterpret_tensor(buf945, (1, 184, 1, 1, 4), (736, 1, 736, 736, 184), 0); del buf945  # reuse
        buf961 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf962 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_410], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_93.run(buf959, buf960, buf961, buf962, 736, 98, grid=grid(736), stream=stream0)
        buf963 = buf924; del buf924  # reuse
        buf964 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf966 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_410], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_94.run(buf960, buf961, buf962, primals_557, primals_558, buf963, buf964, buf966, primals_557, primals_558, 184, 4, grid=grid(184), stream=stream0)
        del primals_557
        del primals_558
        buf967 = reinterpret_tensor(buf958, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf958  # reuse
        # Source Nodes: [shortcut_25, x_410], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_104.run(buf959, buf963, buf964, primals_147, primals_148, buf927, buf967, 72128, grid=grid(72128), stream=stream0)
        del primals_148
        # Source Nodes: [x_415], Original ATen: [aten.convolution]
        buf968 = extern_kernels.convolution(buf967, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf968, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf969 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_415], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf968, buf969, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf970 = buf943; del buf943  # reuse
        buf971 = buf942; del buf942  # reuse
        buf972 = buf941; del buf941  # reuse
        # Source Nodes: [x_416], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf969, buf970, buf971, buf972, 2944, 98, grid=grid(2944), stream=stream0)
        buf973 = reinterpret_tensor(buf962, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf962  # reuse
        buf974 = reinterpret_tensor(buf961, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf961  # reuse
        buf976 = reinterpret_tensor(buf960, (736, ), (1, ), 0); del buf960  # reuse
        # Source Nodes: [x_416], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf970, buf971, buf972, primals_560, primals_561, buf973, buf974, buf976, primals_560, primals_561, 736, 4, grid=grid(736), stream=stream0)
        del primals_560
        del primals_561
        buf977 = reinterpret_tensor(buf968, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf968  # reuse
        buf978 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_416, x_419], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf969, buf973, buf974, primals_149, primals_150, buf977, buf978, 288512, grid=grid(288512), stream=stream0)
        del primals_150
        # Source Nodes: [x_420], Original ATen: [aten.convolution]
        buf979 = extern_kernels.convolution(buf978, primals_308, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf979, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf980 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_420], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf979, buf980, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf981 = buf972; del buf972  # reuse
        buf982 = buf971; del buf971  # reuse
        buf983 = buf970; del buf970  # reuse
        # Source Nodes: [x_421], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf980, buf981, buf982, buf983, 2944, 98, grid=grid(2944), stream=stream0)
        buf984 = buf974; del buf974  # reuse
        buf985 = empty_strided((1, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        buf987 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_421], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf981, buf982, buf983, primals_563, primals_564, buf984, buf985, buf987, primals_563, primals_564, 736, 4, grid=grid(736), stream=stream0)
        del primals_563
        del primals_564
        buf988 = reinterpret_tensor(buf979, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf979  # reuse
        buf989 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_421, x_424], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf980, buf984, buf985, primals_151, primals_152, buf988, buf989, 288512, grid=grid(288512), stream=stream0)
        del primals_152
        buf990 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf991 = reinterpret_tensor(buf990, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf990  # reuse
        # Source Nodes: [x_se_56], Original ATen: [aten.mean]
        triton_per_fused_mean_100.run(buf991, buf989, 5888, 49, grid=grid(5888), stream=stream0)
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf992 = extern_kernels.convolution(buf991, primals_309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf992, (8, 48, 1, 1), (48, 1, 1, 1))
        buf993 = reinterpret_tensor(buf992, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf992  # reuse
        buf994 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_101.run(buf993, primals_310, buf994, 384, grid=grid(384), stream=stream0)
        del primals_310
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf995 = extern_kernels.convolution(buf994, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf995, (8, 736, 1, 1), (736, 1, 1, 1))
        buf996 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_se_59], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_102.run(buf995, primals_312, buf996, 5888, grid=grid(5888), stream=stream0)
        buf997 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_425], Original ATen: [aten.mul]
        triton_poi_fused_mul_103.run(buf989, buf996, buf997, 288512, grid=grid(288512), stream=stream0)
        # Source Nodes: [x_426], Original ATen: [aten.convolution]
        buf998 = extern_kernels.convolution(buf997, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf998, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf999 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_426], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf998, buf999, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf1000 = reinterpret_tensor(buf985, (1, 184, 1, 1, 4), (736, 1, 736, 736, 184), 0); del buf985  # reuse
        buf1001 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf1002 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_427], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_93.run(buf999, buf1000, buf1001, buf1002, 736, 98, grid=grid(736), stream=stream0)
        buf1003 = buf964; del buf964  # reuse
        buf1004 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf1006 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_427], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_94.run(buf1000, buf1001, buf1002, primals_566, primals_567, buf1003, buf1004, buf1006, primals_566, primals_567, 184, 4, grid=grid(184), stream=stream0)
        del primals_566
        del primals_567
        buf1007 = reinterpret_tensor(buf998, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf998  # reuse
        # Source Nodes: [shortcut_26, x_427], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_104.run(buf999, buf1003, buf1004, primals_153, primals_154, buf967, buf1007, 72128, grid=grid(72128), stream=stream0)
        del primals_154
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf1008 = extern_kernels.convolution(buf1007, primals_314, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1008, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf1009 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf1008, buf1009, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf1010 = buf983; del buf983  # reuse
        buf1011 = buf982; del buf982  # reuse
        buf1012 = buf981; del buf981  # reuse
        # Source Nodes: [x_433], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf1009, buf1010, buf1011, buf1012, 2944, 98, grid=grid(2944), stream=stream0)
        buf1013 = reinterpret_tensor(buf1002, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf1002  # reuse
        buf1014 = reinterpret_tensor(buf1001, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf1001  # reuse
        buf1016 = reinterpret_tensor(buf1000, (736, ), (1, ), 0); del buf1000  # reuse
        # Source Nodes: [x_433], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf1010, buf1011, buf1012, primals_569, primals_570, buf1013, buf1014, buf1016, primals_569, primals_570, 736, 4, grid=grid(736), stream=stream0)
        del primals_569
        del primals_570
        buf1017 = reinterpret_tensor(buf1008, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf1008  # reuse
        buf1018 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_433, x_436], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf1009, buf1013, buf1014, primals_155, primals_156, buf1017, buf1018, 288512, grid=grid(288512), stream=stream0)
        del primals_156
        # Source Nodes: [x_437], Original ATen: [aten.convolution]
        buf1019 = extern_kernels.convolution(buf1018, primals_315, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf1019, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf1020 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_437], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf1019, buf1020, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf1021 = buf1012; del buf1012  # reuse
        buf1022 = buf1011; del buf1011  # reuse
        buf1023 = buf1010; del buf1010  # reuse
        # Source Nodes: [x_438], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf1020, buf1021, buf1022, buf1023, 2944, 98, grid=grid(2944), stream=stream0)
        buf1024 = buf1014; del buf1014  # reuse
        buf1025 = empty_strided((1, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        buf1027 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf1021, buf1022, buf1023, primals_572, primals_573, buf1024, buf1025, buf1027, primals_572, primals_573, 736, 4, grid=grid(736), stream=stream0)
        del primals_572
        del primals_573
        buf1028 = reinterpret_tensor(buf1019, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf1019  # reuse
        buf1029 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438, x_441], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf1020, buf1024, buf1025, primals_157, primals_158, buf1028, buf1029, 288512, grid=grid(288512), stream=stream0)
        del primals_158
        buf1030 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf1031 = reinterpret_tensor(buf1030, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf1030  # reuse
        # Source Nodes: [x_se_60], Original ATen: [aten.mean]
        triton_per_fused_mean_100.run(buf1031, buf1029, 5888, 49, grid=grid(5888), stream=stream0)
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf1032 = extern_kernels.convolution(buf1031, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1032, (8, 48, 1, 1), (48, 1, 1, 1))
        buf1033 = reinterpret_tensor(buf1032, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf1032  # reuse
        buf1034 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_101.run(buf1033, primals_317, buf1034, 384, grid=grid(384), stream=stream0)
        del primals_317
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf1035 = extern_kernels.convolution(buf1034, primals_318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1035, (8, 736, 1, 1), (736, 1, 1, 1))
        buf1036 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_se_63], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_102.run(buf1035, primals_319, buf1036, 5888, grid=grid(5888), stream=stream0)
        buf1037 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_442], Original ATen: [aten.mul]
        triton_poi_fused_mul_103.run(buf1029, buf1036, buf1037, 288512, grid=grid(288512), stream=stream0)
        # Source Nodes: [x_443], Original ATen: [aten.convolution]
        buf1038 = extern_kernels.convolution(buf1037, primals_320, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1038, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf1039 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_443], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf1038, buf1039, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf1040 = reinterpret_tensor(buf1025, (1, 184, 1, 1, 4), (736, 1, 736, 736, 184), 0); del buf1025  # reuse
        buf1041 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf1042 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_444], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_93.run(buf1039, buf1040, buf1041, buf1042, 736, 98, grid=grid(736), stream=stream0)
        buf1043 = buf1004; del buf1004  # reuse
        buf1044 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf1046 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_444], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_94.run(buf1040, buf1041, buf1042, primals_575, primals_576, buf1043, buf1044, buf1046, primals_575, primals_576, 184, 4, grid=grid(184), stream=stream0)
        del primals_575
        del primals_576
        buf1047 = reinterpret_tensor(buf1038, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf1038  # reuse
        # Source Nodes: [shortcut_27, x_444], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_104.run(buf1039, buf1043, buf1044, primals_159, primals_160, buf1007, buf1047, 72128, grid=grid(72128), stream=stream0)
        del primals_160
        # Source Nodes: [x_449], Original ATen: [aten.convolution]
        buf1048 = extern_kernels.convolution(buf1047, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1048, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf1049 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_449], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf1048, buf1049, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf1050 = buf1023; del buf1023  # reuse
        buf1051 = buf1022; del buf1022  # reuse
        buf1052 = buf1021; del buf1021  # reuse
        # Source Nodes: [x_450], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf1049, buf1050, buf1051, buf1052, 2944, 98, grid=grid(2944), stream=stream0)
        buf1053 = reinterpret_tensor(buf1042, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf1042  # reuse
        buf1054 = reinterpret_tensor(buf1041, (1, 736, 1, 1), (736, 1, 736, 736), 0); del buf1041  # reuse
        buf1056 = reinterpret_tensor(buf1040, (736, ), (1, ), 0); del buf1040  # reuse
        # Source Nodes: [x_450], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf1050, buf1051, buf1052, primals_578, primals_579, buf1053, buf1054, buf1056, primals_578, primals_579, 736, 4, grid=grid(736), stream=stream0)
        del primals_578
        del primals_579
        buf1057 = reinterpret_tensor(buf1048, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf1048  # reuse
        buf1058 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_450, x_453], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf1049, buf1053, buf1054, primals_161, primals_162, buf1057, buf1058, 288512, grid=grid(288512), stream=stream0)
        del primals_162
        # Source Nodes: [x_454], Original ATen: [aten.convolution]
        buf1059 = extern_kernels.convolution(buf1058, primals_322, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=736, bias=None)
        assert_size_stride(buf1059, (8, 736, 7, 7), (36064, 49, 7, 1))
        buf1060 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_454], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_96.run(buf1059, buf1060, 5888, 49, grid=grid(5888, 49), stream=stream0)
        buf1061 = buf1052; del buf1052  # reuse
        buf1062 = buf1051; del buf1051  # reuse
        buf1063 = buf1050; del buf1050  # reuse
        # Source Nodes: [x_455], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf1060, buf1061, buf1062, buf1063, 2944, 98, grid=grid(2944), stream=stream0)
        buf1064 = buf1054; del buf1054  # reuse
        buf1065 = empty_strided((1, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        buf1067 = empty((736, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_455], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_98.run(buf1061, buf1062, buf1063, primals_581, primals_582, buf1064, buf1065, buf1067, primals_581, primals_582, 736, 4, grid=grid(736), stream=stream0)
        del buf1061
        del buf1062
        del buf1063
        del primals_581
        del primals_582
        buf1068 = reinterpret_tensor(buf1059, (8, 736, 7, 7), (36064, 1, 5152, 736), 0); del buf1059  # reuse
        buf1069 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_455, x_458], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_99.run(buf1060, buf1064, buf1065, primals_163, primals_164, buf1068, buf1069, 288512, grid=grid(288512), stream=stream0)
        del primals_164
        buf1070 = empty_strided((8, 736, 1, 1), (736, 1, 5888, 5888), device='cuda', dtype=torch.float32)
        buf1071 = reinterpret_tensor(buf1070, (8, 736, 1, 1), (736, 1, 736, 736), 0); del buf1070  # reuse
        # Source Nodes: [x_se_64], Original ATen: [aten.mean]
        triton_per_fused_mean_100.run(buf1071, buf1069, 5888, 49, grid=grid(5888), stream=stream0)
        # Source Nodes: [x_se_65], Original ATen: [aten.convolution]
        buf1072 = extern_kernels.convolution(buf1071, primals_323, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1072, (8, 48, 1, 1), (48, 1, 1, 1))
        buf1073 = reinterpret_tensor(buf1072, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf1072  # reuse
        buf1074 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_65, x_se_66], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_101.run(buf1073, primals_324, buf1074, 384, grid=grid(384), stream=stream0)
        del primals_324
        # Source Nodes: [x_se_67], Original ATen: [aten.convolution]
        buf1075 = extern_kernels.convolution(buf1074, primals_325, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1075, (8, 736, 1, 1), (736, 1, 1, 1))
        buf1076 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.float32)
        buf1144 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____5___se_gate, x_se_67], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_105.run(buf1075, primals_326, buf1076, buf1144, 5888, grid=grid(5888), stream=stream0)
        del buf1075
        del primals_326
        buf1077 = empty_strided((8, 736, 7, 7), (36064, 1, 5152, 736), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_459], Original ATen: [aten.mul]
        triton_poi_fused_mul_103.run(buf1069, buf1076, buf1077, 288512, grid=grid(288512), stream=stream0)
        # Source Nodes: [x_460], Original ATen: [aten.convolution]
        buf1078 = extern_kernels.convolution(buf1077, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1078, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf1079 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_460], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_92.run(buf1078, buf1079, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf1080 = reinterpret_tensor(buf1065, (1, 184, 1, 1, 4), (736, 1, 736, 736, 184), 0); del buf1065  # reuse
        buf1081 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf1082 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_461], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_93.run(buf1079, buf1080, buf1081, buf1082, 736, 98, grid=grid(736), stream=stream0)
        buf1083 = buf1044; del buf1044  # reuse
        buf1084 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf1086 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_461], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_94.run(buf1080, buf1081, buf1082, primals_584, primals_585, buf1083, buf1084, buf1086, primals_584, primals_585, 184, 4, grid=grid(184), stream=stream0)
        del buf1080
        del buf1081
        del buf1082
        del primals_584
        del primals_585
        buf1087 = reinterpret_tensor(buf1078, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf1078  # reuse
        # Source Nodes: [shortcut_28, x_461], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_104.run(buf1079, buf1083, buf1084, primals_165, primals_166, buf1047, buf1087, 72128, grid=grid(72128), stream=stream0)
        del buf1084
        del primals_166
        # Source Nodes: [x_466], Original ATen: [aten.convolution]
        buf1088 = extern_kernels.convolution(buf1087, primals_328, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1088, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf1089 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_466], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf1088, buf1089, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf1090 = empty_strided((1, 1104, 1, 1, 4), (4416, 1, 4416, 4416, 1104), device='cuda', dtype=torch.float32)
        buf1091 = empty_strided((1, 1104, 1, 1, 4), (4416, 1, 4416, 4416, 1104), device='cuda', dtype=torch.float32)
        buf1092 = empty_strided((1, 1104, 1, 1, 4), (4416, 1, 4416, 4416, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_467], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_107.run(buf1089, buf1090, buf1091, buf1092, 4416, 98, grid=grid(4416), stream=stream0)
        buf1093 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf1094 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf1096 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_467], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_108.run(buf1090, buf1091, buf1092, primals_587, primals_588, buf1093, buf1094, buf1096, primals_587, primals_588, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_587
        del primals_588
        buf1097 = reinterpret_tensor(buf1088, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf1088  # reuse
        buf1098 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_467, x_470], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_109.run(buf1089, buf1093, buf1094, primals_167, primals_168, buf1097, buf1098, 432768, grid=grid(432768), stream=stream0)
        del primals_168
        # Source Nodes: [x_471], Original ATen: [aten.convolution]
        buf1099 = extern_kernels.convolution(buf1098, primals_329, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf1099, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf1100 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_471], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_106.run(buf1099, buf1100, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf1101 = buf1092; del buf1092  # reuse
        buf1102 = buf1091; del buf1091  # reuse
        buf1103 = buf1090; del buf1090  # reuse
        # Source Nodes: [x_472], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_107.run(buf1100, buf1101, buf1102, buf1103, 4416, 98, grid=grid(4416), stream=stream0)
        buf1104 = buf1094; del buf1094  # reuse
        buf1105 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf1107 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_472], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_108.run(buf1101, buf1102, buf1103, primals_590, primals_591, buf1104, buf1105, buf1107, primals_590, primals_591, 1104, 4, grid=grid(1104), stream=stream0)
        del buf1101
        del buf1102
        del buf1103
        del primals_590
        del primals_591
        buf1108 = reinterpret_tensor(buf1099, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf1099  # reuse
        buf1109 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_472, x_475], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_109.run(buf1100, buf1104, buf1105, primals_169, primals_170, buf1108, buf1109, 432768, grid=grid(432768), stream=stream0)
        del buf1105
        del primals_170
        buf1110 = empty_strided((8, 1104, 1, 1), (1104, 1, 8832, 8832), device='cuda', dtype=torch.float32)
        buf1111 = reinterpret_tensor(buf1110, (8, 1104, 1, 1), (1104, 1, 1104, 1104), 0); del buf1110  # reuse
        # Source Nodes: [x_se_68], Original ATen: [aten.mean]
        triton_per_fused_mean_110.run(buf1111, buf1109, 8832, 49, grid=grid(8832), stream=stream0)
        # Source Nodes: [x_se_69], Original ATen: [aten.convolution]
        buf1112 = extern_kernels.convolution(buf1111, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1112, (8, 48, 1, 1), (48, 1, 1, 1))
        buf1113 = reinterpret_tensor(buf1112, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf1112  # reuse
        buf1114 = empty_strided((8, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_69, x_se_70], Original ATen: [aten.convolution, aten.hardswish]
        triton_poi_fused_convolution_hardswish_101.run(buf1113, primals_331, buf1114, 384, grid=grid(384), stream=stream0)
        del primals_331
        # Source Nodes: [x_se_71], Original ATen: [aten.convolution]
        buf1115 = extern_kernels.convolution(buf1114, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1115, (8, 1104, 1, 1), (1104, 1, 1, 1))
        buf1116 = empty_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf1143 = empty_strided((8, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____6___se_gate, x_se_71], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_hardsigmoid_backward_111.run(buf1115, primals_333, buf1116, buf1143, 8832, grid=grid(8832), stream=stream0)
        del buf1115
        del primals_333
        buf1117 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_476], Original ATen: [aten.mul]
        triton_poi_fused_mul_112.run(buf1109, buf1116, buf1117, 432768, grid=grid(432768), stream=stream0)
        # Source Nodes: [x_477], Original ATen: [aten.convolution]
        buf1118 = extern_kernels.convolution(buf1117, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1118, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf1119 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_477], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_113.run(buf1118, buf1119, 1792, 49, grid=grid(1792, 49), stream=stream0)
        buf1120 = empty_strided((1, 224, 1, 1, 4), (896, 1, 896, 896, 224), device='cuda', dtype=torch.float32)
        buf1121 = empty_strided((1, 224, 1, 1, 4), (896, 1, 896, 896, 224), device='cuda', dtype=torch.float32)
        buf1122 = empty_strided((1, 224, 1, 1, 4), (896, 1, 896, 896, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_114.run(buf1119, buf1120, buf1121, buf1122, 896, 98, grid=grid(896), stream=stream0)
        buf1123 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf1124 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf1126 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_115.run(buf1120, buf1121, buf1122, primals_593, primals_594, buf1123, buf1124, buf1126, primals_593, primals_594, 224, 4, grid=grid(224), stream=stream0)
        del buf1120
        del buf1121
        del buf1122
        del primals_593
        del primals_594
        buf1127 = reinterpret_tensor(buf1118, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf1118  # reuse
        # Source Nodes: [x_478], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_116.run(buf1119, buf1123, buf1124, primals_171, primals_172, buf1127, 87808, grid=grid(87808), stream=stream0)
        del buf1124
        del primals_172
        # Source Nodes: [x_482], Original ATen: [aten.convolution]
        buf1128 = extern_kernels.convolution(buf1127, primals_335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1128, (8, 1344, 7, 7), (65856, 49, 7, 1))
        buf1129 = empty_strided((8, 1344, 7, 7), (65856, 1, 9408, 1344), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_482], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_117.run(buf1128, buf1129, 10752, 49, grid=grid(10752, 49), stream=stream0)
        buf1130 = empty_strided((1, 1344, 1, 1, 4), (5376, 1, 5376, 5376, 1344), device='cuda', dtype=torch.float32)
        buf1131 = empty_strided((1, 1344, 1, 1, 4), (5376, 1, 5376, 5376, 1344), device='cuda', dtype=torch.float32)
        buf1132 = empty_strided((1, 1344, 1, 1, 4), (5376, 1, 5376, 5376, 1344), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_483], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_118.run(buf1129, buf1130, buf1131, buf1132, 5376, 98, grid=grid(5376), stream=stream0)
        buf1133 = empty_strided((1, 1344, 1, 1), (1344, 1, 1344, 1344), device='cuda', dtype=torch.float32)
        buf1134 = empty_strided((1, 1344, 1, 1), (1344, 1, 1344, 1344), device='cuda', dtype=torch.float32)
        buf1136 = empty((1344, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_483], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_119.run(buf1130, buf1131, buf1132, primals_596, primals_597, buf1133, buf1134, buf1136, primals_596, primals_597, 1344, 4, grid=grid(1344), stream=stream0)
        del buf1130
        del buf1131
        del buf1132
        del primals_596
        del primals_597
        buf1137 = reinterpret_tensor(buf1128, (8, 1344, 7, 7), (65856, 1, 9408, 1344), 0); del buf1128  # reuse
        # Source Nodes: [x_483], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_120.run(buf1129, buf1133, buf1134, primals_173, primals_174, buf1137, 526848, grid=grid(526848), stream=stream0)
        del buf1134
        del primals_174
        buf1138 = empty_strided((8, 1344, 1, 1), (1344, 1, 10752, 10752), device='cuda', dtype=torch.float32)
        buf1139 = reinterpret_tensor(buf1138, (8, 1344, 1, 1), (1344, 1, 1344, 1344), 0); del buf1138  # reuse
        # Source Nodes: [x_488, x_489], Original ATen: [aten.hardswish, aten.mean]
        triton_per_fused_hardswish_mean_121.run(buf1139, buf1137, 10752, 49, grid=grid(10752), stream=stream0)
        # Source Nodes: [x_492], Original ATen: [aten.convolution]
        buf1140 = extern_kernels.convolution(buf1139, primals_336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1140, (8, 1984, 1, 1), (1984, 1, 1, 1))
        buf1141 = empty((8, 1984), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred, x_493], Original ATen: [aten.hardswish, aten.view]
        triton_poi_fused_hardswish_view_122.run(buf1140, buf1141, 15872, grid=grid(15872), stream=stream0)
        buf1142 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf1141, reinterpret_tensor(primals_175, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf1142)
        del primals_176
        buf1145 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_123.run(buf1035, primals_319, buf1145, 5888, grid=grid(5888), stream=stream0)
        del buf1035
        del primals_319
        buf1146 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_123.run(buf995, primals_312, buf1146, 5888, grid=grid(5888), stream=stream0)
        del buf995
        del primals_312
        buf1147 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_123.run(buf955, primals_305, buf1147, 5888, grid=grid(5888), stream=stream0)
        del buf955
        del primals_305
        buf1148 = empty_strided((8, 736, 1, 1), (736, 1, 736, 736), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_123.run(buf915, primals_298, buf1148, 5888, grid=grid(5888), stream=stream0)
        del buf915
        del primals_298
        buf1149 = empty_strided((8, 720, 1, 1), (720, 1, 720, 720), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_124.run(buf875, primals_291, buf1149, 5760, grid=grid(5760), stream=stream0)
        del buf875
        del primals_291
        buf1150 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_125.run(buf835, primals_284, buf1150, 2880, grid=grid(2880), stream=stream0)
        del buf835
        del primals_284
        buf1151 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_125.run(buf794, primals_277, buf1151, 2880, grid=grid(2880), stream=stream0)
        del buf794
        del primals_277
        buf1152 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_125.run(buf753, primals_270, buf1152, 2880, grid=grid(2880), stream=stream0)
        del buf753
        del primals_270
        buf1153 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_125.run(buf712, primals_263, buf1153, 2880, grid=grid(2880), stream=stream0)
        del buf712
        del primals_263
        buf1154 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_125.run(buf671, primals_256, buf1154, 2880, grid=grid(2880), stream=stream0)
        del buf671
        del primals_256
        buf1155 = empty_strided((8, 360, 1, 1), (360, 1, 360, 360), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_125.run(buf630, primals_249, buf1155, 2880, grid=grid(2880), stream=stream0)
        del buf630
        del primals_249
        buf1156 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_126.run(buf429, primals_227, buf1156, 960, grid=grid(960), stream=stream0)
        del buf429
        del primals_227
        buf1157 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_126.run(buf388, primals_220, buf1157, 960, grid=grid(960), stream=stream0)
        del buf388
        del primals_220
        buf1158 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_126.run(buf347, primals_213, buf1158, 960, grid=grid(960), stream=stream0)
        del buf347
        del primals_213
        buf1159 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_126.run(buf306, primals_206, buf1159, 960, grid=grid(960), stream=stream0)
        del buf306
        del primals_206
        buf1160 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_126.run(buf265, primals_199, buf1160, 960, grid=grid(960), stream=stream0)
        del buf265
        del primals_199
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_337, primals_337, 1, grid=grid(1), stream=stream0)
        del primals_337
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_340, primals_340, 1, grid=grid(1), stream=stream0)
        del primals_340
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_343, primals_343, 1, grid=grid(1), stream=stream0)
        del primals_343
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_346, primals_346, 1, grid=grid(1), stream=stream0)
        del primals_346
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_349, primals_349, 1, grid=grid(1), stream=stream0)
        del primals_349
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_352, primals_352, 1, grid=grid(1), stream=stream0)
        del primals_352
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_355, primals_355, 1, grid=grid(1), stream=stream0)
        del primals_355
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_358, primals_358, 1, grid=grid(1), stream=stream0)
        del primals_358
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_361, primals_361, 1, grid=grid(1), stream=stream0)
        del primals_361
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_364, primals_364, 1, grid=grid(1), stream=stream0)
        del primals_364
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_367, primals_367, 1, grid=grid(1), stream=stream0)
        del primals_367
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_370, primals_370, 1, grid=grid(1), stream=stream0)
        del primals_370
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_373, primals_373, 1, grid=grid(1), stream=stream0)
        del primals_373
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_376, primals_376, 1, grid=grid(1), stream=stream0)
        del primals_376
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_379, primals_379, 1, grid=grid(1), stream=stream0)
        del primals_379
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_382, primals_382, 1, grid=grid(1), stream=stream0)
        del primals_382
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_385, primals_385, 1, grid=grid(1), stream=stream0)
        del primals_385
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_388, primals_388, 1, grid=grid(1), stream=stream0)
        del primals_388
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_391, primals_391, 1, grid=grid(1), stream=stream0)
        del primals_391
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_394, primals_394, 1, grid=grid(1), stream=stream0)
        del primals_394
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_397, primals_397, 1, grid=grid(1), stream=stream0)
        del primals_397
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_400, primals_400, 1, grid=grid(1), stream=stream0)
        del primals_400
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_403, primals_403, 1, grid=grid(1), stream=stream0)
        del primals_403
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_406, primals_406, 1, grid=grid(1), stream=stream0)
        del primals_406
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_409, primals_409, 1, grid=grid(1), stream=stream0)
        del primals_409
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_412, primals_412, 1, grid=grid(1), stream=stream0)
        del primals_412
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_415, primals_415, 1, grid=grid(1), stream=stream0)
        del primals_415
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_418, primals_418, 1, grid=grid(1), stream=stream0)
        del primals_418
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_421, primals_421, 1, grid=grid(1), stream=stream0)
        del primals_421
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_424, primals_424, 1, grid=grid(1), stream=stream0)
        del primals_424
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_427, primals_427, 1, grid=grid(1), stream=stream0)
        del primals_427
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_430, primals_430, 1, grid=grid(1), stream=stream0)
        del primals_430
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_433, primals_433, 1, grid=grid(1), stream=stream0)
        del primals_433
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_436, primals_436, 1, grid=grid(1), stream=stream0)
        del primals_436
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_439, primals_439, 1, grid=grid(1), stream=stream0)
        del primals_439
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_442, primals_442, 1, grid=grid(1), stream=stream0)
        del primals_442
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_445, primals_445, 1, grid=grid(1), stream=stream0)
        del primals_445
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_448, primals_448, 1, grid=grid(1), stream=stream0)
        del primals_448
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_451, primals_451, 1, grid=grid(1), stream=stream0)
        del primals_451
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_454, primals_454, 1, grid=grid(1), stream=stream0)
        del primals_454
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_457, primals_457, 1, grid=grid(1), stream=stream0)
        del primals_457
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_460, primals_460, 1, grid=grid(1), stream=stream0)
        del primals_460
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_463, primals_463, 1, grid=grid(1), stream=stream0)
        del primals_463
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_466, primals_466, 1, grid=grid(1), stream=stream0)
        del primals_466
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_469, primals_469, 1, grid=grid(1), stream=stream0)
        del primals_469
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_472, primals_472, 1, grid=grid(1), stream=stream0)
        del primals_472
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_475, primals_475, 1, grid=grid(1), stream=stream0)
        del primals_475
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_478, primals_478, 1, grid=grid(1), stream=stream0)
        del primals_478
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_481, primals_481, 1, grid=grid(1), stream=stream0)
        del primals_481
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_484, primals_484, 1, grid=grid(1), stream=stream0)
        del primals_484
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_487, primals_487, 1, grid=grid(1), stream=stream0)
        del primals_487
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_490, primals_490, 1, grid=grid(1), stream=stream0)
        del primals_490
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_493, primals_493, 1, grid=grid(1), stream=stream0)
        del primals_493
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_496, primals_496, 1, grid=grid(1), stream=stream0)
        del primals_496
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_499, primals_499, 1, grid=grid(1), stream=stream0)
        del primals_499
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_502, primals_502, 1, grid=grid(1), stream=stream0)
        del primals_502
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_505, primals_505, 1, grid=grid(1), stream=stream0)
        del primals_505
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_508, primals_508, 1, grid=grid(1), stream=stream0)
        del primals_508
        # Source Nodes: [add__58], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_511, primals_511, 1, grid=grid(1), stream=stream0)
        del primals_511
        # Source Nodes: [add__59], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_514, primals_514, 1, grid=grid(1), stream=stream0)
        del primals_514
        # Source Nodes: [add__60], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_517, primals_517, 1, grid=grid(1), stream=stream0)
        del primals_517
        # Source Nodes: [add__61], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_520, primals_520, 1, grid=grid(1), stream=stream0)
        del primals_520
        # Source Nodes: [add__62], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_523, primals_523, 1, grid=grid(1), stream=stream0)
        del primals_523
        # Source Nodes: [add__63], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_526, primals_526, 1, grid=grid(1), stream=stream0)
        del primals_526
        # Source Nodes: [add__64], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_529, primals_529, 1, grid=grid(1), stream=stream0)
        del primals_529
        # Source Nodes: [add__65], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_532, primals_532, 1, grid=grid(1), stream=stream0)
        del primals_532
        # Source Nodes: [add__66], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_535, primals_535, 1, grid=grid(1), stream=stream0)
        del primals_535
        # Source Nodes: [add__67], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_538, primals_538, 1, grid=grid(1), stream=stream0)
        del primals_538
        # Source Nodes: [add__68], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_541, primals_541, 1, grid=grid(1), stream=stream0)
        del primals_541
        # Source Nodes: [add__69], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_544, primals_544, 1, grid=grid(1), stream=stream0)
        del primals_544
        # Source Nodes: [add__70], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_547, primals_547, 1, grid=grid(1), stream=stream0)
        del primals_547
        # Source Nodes: [add__71], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_550, primals_550, 1, grid=grid(1), stream=stream0)
        del primals_550
        # Source Nodes: [add__72], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_553, primals_553, 1, grid=grid(1), stream=stream0)
        del primals_553
        # Source Nodes: [add__73], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_556, primals_556, 1, grid=grid(1), stream=stream0)
        del primals_556
        # Source Nodes: [add__74], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_559, primals_559, 1, grid=grid(1), stream=stream0)
        del primals_559
        # Source Nodes: [add__75], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_562, primals_562, 1, grid=grid(1), stream=stream0)
        del primals_562
        # Source Nodes: [add__76], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_565, primals_565, 1, grid=grid(1), stream=stream0)
        del primals_565
        # Source Nodes: [add__77], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_568, primals_568, 1, grid=grid(1), stream=stream0)
        del primals_568
        # Source Nodes: [add__78], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_571, primals_571, 1, grid=grid(1), stream=stream0)
        del primals_571
        # Source Nodes: [add__79], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_574, primals_574, 1, grid=grid(1), stream=stream0)
        del primals_574
        # Source Nodes: [add__80], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_577, primals_577, 1, grid=grid(1), stream=stream0)
        del primals_577
        # Source Nodes: [add__81], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_580, primals_580, 1, grid=grid(1), stream=stream0)
        del primals_580
        # Source Nodes: [add__82], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_583, primals_583, 1, grid=grid(1), stream=stream0)
        del primals_583
        # Source Nodes: [add__83], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_586, primals_586, 1, grid=grid(1), stream=stream0)
        del primals_586
        # Source Nodes: [add__84], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_589, primals_589, 1, grid=grid(1), stream=stream0)
        del primals_589
        # Source Nodes: [add__85], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_592, primals_592, 1, grid=grid(1), stream=stream0)
        del primals_592
        # Source Nodes: [add__86], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(primals_595, primals_595, 1, grid=grid(1), stream=stream0)
        del primals_595
        return (buf1142, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, buf0, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_198, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_264, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_274, primals_276, primals_278, primals_279, primals_280, primals_281, primals_283, primals_285, primals_286, primals_287, primals_288, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_299, primals_300, primals_301, primals_302, primals_304, primals_306, primals_307, primals_308, primals_309, primals_311, primals_313, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_323, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, buf1, buf3, buf13, buf14, buf15, buf17, buf27, buf28, buf29, buf31, buf41, buf42, buf44, buf54, buf55, buf56, buf58, buf68, buf69, buf71, buf81, buf82, buf83, buf85, buf95, buf96, buf97, buf99, buf109, buf110, buf112, buf122, buf123, buf124, buf126, buf136, buf137, buf138, buf140, buf150, buf151, buf153, buf163, buf164, buf165, buf167, buf177, buf178, buf179, buf181, buf191, buf192, buf194, buf204, buf205, buf206, buf208, buf218, buf219, buf220, buf222, buf232, buf233, buf235, buf245, buf246, buf247, buf249, buf256, buf257, buf258, buf261, buf263, buf264, buf266, buf267, buf269, buf276, buf277, buf279, buf286, buf287, buf288, buf290, buf297, buf298, buf299, buf302, buf304, buf305, buf307, buf308, buf310, buf317, buf318, buf320, buf327, buf328, buf329, buf331, buf338, buf339, buf340, buf343, buf345, buf346, buf348, buf349, buf351, buf358, buf359, buf361, buf368, buf369, buf370, buf372, buf379, buf380, buf381, buf384, buf386, buf387, buf389, buf390, buf392, buf399, buf400, buf402, buf409, buf410, buf411, buf413, buf420, buf421, buf422, buf425, buf427, buf428, buf430, buf431, buf433, buf440, buf441, buf443, buf450, buf451, buf452, buf454, buf461, buf462, buf463, buf465, buf472, buf473, buf475, buf482, buf483, buf484, buf486, buf493, buf494, buf495, buf497, buf504, buf505, buf507, buf514, buf515, buf516, buf518, buf525, buf526, buf527, buf529, buf536, buf537, buf539, buf546, buf547, buf548, buf550, buf557, buf558, buf559, buf561, buf568, buf569, buf571, buf578, buf579, buf580, buf582, buf589, buf590, buf591, buf593, buf600, buf601, buf603, buf610, buf611, buf612, buf614, buf621, buf622, buf623, buf626, buf628, buf629, buf631, buf632, buf634, buf641, buf642, buf644, buf651, buf652, buf653, buf655, buf662, buf663, buf664, buf667, buf669, buf670, buf672, buf673, buf675, buf682, buf683, buf685, buf692, buf693, buf694, buf696, buf703, buf704, buf705, buf708, buf710, buf711, buf713, buf714, buf716, buf723, buf724, buf726, buf733, buf734, buf735, buf737, buf744, buf745, buf746, buf749, buf751, buf752, buf754, buf755, buf757, buf764, buf765, buf767, buf774, buf775, buf776, buf778, buf785, buf786, buf787, buf790, buf792, buf793, buf795, buf796, buf798, buf805, buf806, buf808, buf815, buf816, buf817, buf819, buf826, buf827, buf828, buf831, buf833, buf834, buf836, buf837, buf839, buf846, buf847, buf849, buf856, buf857, buf858, buf860, buf867, buf868, buf869, buf871, buf873, buf874, buf876, buf877, buf879, buf886, buf887, buf889, buf896, buf897, buf898, buf900, buf907, buf908, buf909, buf911, buf913, buf914, buf916, buf917, buf919, buf926, buf927, buf929, buf936, buf937, buf938, buf940, buf947, buf948, buf949, buf951, buf953, buf954, buf956, buf957, buf959, buf966, buf967, buf969, buf976, buf977, buf978, buf980, buf987, buf988, buf989, buf991, buf993, buf994, buf996, buf997, buf999, buf1006, buf1007, buf1009, buf1016, buf1017, buf1018, buf1020, buf1027, buf1028, buf1029, buf1031, buf1033, buf1034, buf1036, buf1037, buf1039, buf1046, buf1047, buf1049, buf1056, buf1057, buf1058, buf1060, buf1067, buf1068, buf1069, buf1071, buf1073, buf1074, buf1076, buf1077, buf1079, buf1086, buf1087, buf1089, buf1096, buf1097, buf1098, buf1100, buf1107, buf1108, buf1109, buf1111, buf1113, buf1114, buf1116, buf1117, buf1119, buf1126, buf1127, buf1129, buf1136, buf1137, buf1139, buf1140, buf1141, reinterpret_tensor(primals_175, (1000, 1984), (1984, 1), 0), reinterpret_tensor(buf1133, (1, 1344, 1, 1), (1344, 1, 1, 1), 0), reinterpret_tensor(buf1123, (1, 224, 1, 1), (224, 1, 1, 1), 0), buf1143, reinterpret_tensor(buf1104, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf1093, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf1083, (1, 184, 1, 1), (184, 1, 1, 1), 0), buf1144, reinterpret_tensor(buf1064, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf1053, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf1043, (1, 184, 1, 1), (184, 1, 1, 1), 0), buf1145, reinterpret_tensor(buf1024, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf1013, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf1003, (1, 184, 1, 1), (184, 1, 1, 1), 0), buf1146, reinterpret_tensor(buf984, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf973, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf963, (1, 184, 1, 1), (184, 1, 1, 1), 0), buf1147, reinterpret_tensor(buf944, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf933, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf923, (1, 184, 1, 1), (184, 1, 1, 1), 0), buf1148, reinterpret_tensor(buf904, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf893, (1, 736, 1, 1), (736, 1, 1, 1), 0), reinterpret_tensor(buf883, (1, 184, 1, 1), (184, 1, 1, 1), 0), buf1149, reinterpret_tensor(buf864, (1, 720, 1, 1), (720, 1, 1, 1), 0), reinterpret_tensor(buf853, (1, 720, 1, 1), (720, 1, 1, 1), 0), reinterpret_tensor(buf843, (1, 120, 1, 1), (120, 1, 1, 1), 0), buf1150, reinterpret_tensor(buf823, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf812, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf802, (1, 120, 1, 1), (120, 1, 1, 1), 0), buf1151, reinterpret_tensor(buf782, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf771, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf761, (1, 120, 1, 1), (120, 1, 1, 1), 0), buf1152, reinterpret_tensor(buf741, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf730, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf720, (1, 120, 1, 1), (120, 1, 1, 1), 0), buf1153, reinterpret_tensor(buf700, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf689, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf679, (1, 120, 1, 1), (120, 1, 1, 1), 0), buf1154, reinterpret_tensor(buf659, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf648, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf638, (1, 120, 1, 1), (120, 1, 1, 1), 0), buf1155, reinterpret_tensor(buf618, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf607, (1, 360, 1, 1), (360, 1, 1, 1), 0), reinterpret_tensor(buf597, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf586, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf575, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf565, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf554, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf543, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf533, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf522, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf511, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf501, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf490, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf479, (1, 216, 1, 1), (216, 1, 1, 1), 0), reinterpret_tensor(buf469, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf458, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf447, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf437, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf1156, reinterpret_tensor(buf417, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf396, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf1157, reinterpret_tensor(buf376, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf365, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf355, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf1158, reinterpret_tensor(buf335, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf324, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf314, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf1159, reinterpret_tensor(buf294, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf283, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf273, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf1160, reinterpret_tensor(buf253, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf242, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf215, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf201, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf188, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf174, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf160, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf133, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf119, (1, 48, 1, 1), (48, 1, 1, 1), 0), reinterpret_tensor(buf106, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf92, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf78, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf65, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf51, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf38, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((48, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((8, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((120, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((16, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((120, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((200, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((200, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((72, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((216, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((72, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((360, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((360, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((24, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((360, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((360, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((360, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((32, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((360, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((120, 360, 1, 1), (360, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((720, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((720, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((720, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((184, 720, 1, 1), (720, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((736, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((736, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((48, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((736, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((184, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((48, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1104, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((224, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1984, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_338 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_341 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_344 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_347 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_350 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_353 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_356 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_359 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_362 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_365 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_368 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_371 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_374 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_377 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_380 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_383 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_386 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_389 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_392 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_395 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_398 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_401 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_404 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_407 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_410 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_413 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_416 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_419 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_422 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_425 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_428 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_431 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_434 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_437 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_440 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_443 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_446 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_449 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_452 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_455 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_458 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_461 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_464 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_467 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_470 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_473 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_476 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_479 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_482 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_485 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_488 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_491 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_494 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_497 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_500 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_503 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_506 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_509 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_512 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_515 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_518 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_521 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_524 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_527 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((360, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_530 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_533 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_536 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_539 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_542 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_545 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_548 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_551 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_554 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_557 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_560 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_563 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_566 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_569 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_572 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_575 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_578 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_581 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_584 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_587 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_590 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_593 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_596 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetv3_b', benchmark_compiled_module)
