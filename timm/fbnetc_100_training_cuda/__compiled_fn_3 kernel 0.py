
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


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mwjn57ehkvljl3pr2uwudzucplfpcidzsmhk5uydh2isrf4nsf.py
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


# kernel path: /tmp/torchinductor_youkaichao/r6/cr6yhrepecaxz6pxpz7srhgwf2hexe5n3p4h52jbbwwv7oiso3um.py
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cnisw4a3mggqrjdxdijaipefi7cm7f2wnmvruxrzysbqbhg2nxe4.py
# Source Nodes: [shortcut_1, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_1 => add_20
# x_17 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctlthoga2oaxvdfpyeegaupkaxen35ubnophwk3yutlcnlduozm.py
# Source Nodes: [x_22], Original ATen: [aten.convolution]
# x_22 => convolution_4
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ya/cyazoad6ouv6pabrcagsac7dbppt2ykjaudtw6cm5tudah3og5b6.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gn/cgnkuaqdv3wstgpatcycg34njebbx6vownngfunh3mxtsc35snxi.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mt4n6axn4n2fmljhmm6vtgl53bzvqxul3qqwuyu5r2kllkatf6.py
# Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
# x_23 => add_22, add_23, add_24, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzwiyvcczubanwatrqqpzf3h3lyuqtx6tg3ju3g32moqpnlma5h.py
# Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_23 => add_22, add_25, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_26 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2qw6skyf5ffrwbyp3k24ndybsslxersezrsntpjsrui64cnvmj.py
# Source Nodes: [x_27], Original ATen: [aten.convolution]
# x_27 => convolution_5
triton_poi_fused_convolution_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ft/cftivzmx6rrolfa6brhigwky3d7pxcxfuy4p7cfer2fos2tz64pc.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cncx6jawp37nhme3bvq23767ctjaipmxecyw4hrz3u7jgydjcnv4.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvb5soz2cspxgdbsokdxkdrot75nkglyyj2hux4blvuqmmqfkjt.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
# x_28 => add_27, add_28, add_29, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ea/ceabudhocpaizhjpufphpqrgyljnkktndl56i6qb7d3ak72mlxhl.py
# Source Nodes: [x_28, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_28 => add_27, add_30, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x_31 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_17', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvegdhuap23u3u6e55apqogd22sy6qq5p64vx3tsmzrtoriebkhm.py
# Source Nodes: [x_33], Original ATen: [aten.convolution]
# x_33 => convolution_6
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
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => var_mean_6
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
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => var_mean_6
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


# kernel path: /tmp/torchinductor_youkaichao/2u/c2uqel5ay6qghqfoorh65sbswjpnridm43tvsvy5nyi3woi2dbxh.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => add_32, add_33, add_34, mul_43, mul_44, mul_45, mul_46, mul_47, rsqrt_6, squeeze_19, var_mean_6
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


# kernel path: /tmp/torchinductor_youkaichao/6e/c6e2jhqecby7l3ltypytgkwxdye474fpo5zx4d4lswv3ck4fzc6k.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
# x_34 => add_32, add_35, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5bu2x2pqssl4uxpytaevmnzz77p2ybgiwhu3775caeeitpfjak.py
# Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_39 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# x_42 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_23', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tx5lur5xa7gagoe26rywybcm2qbf7f73gzfiyto5txgmq5zj3u.py
# Source Nodes: [shortcut_3, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_3 => add_51
# x_50 => add_47, add_50, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_add_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_24', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ms/cmscrce4qfkptqxx4ux7urasd5i66a7xfwvbzo76ufyceijq7ytn.py
# Source Nodes: [x_72], Original ATen: [aten.convolution]
# x_72 => convolution_13
triton_poi_fused_convolution_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnx57deqtwa3rxky3gzi3qg2xv73yih2fgjxirrca7aa3tjgnea.py
# Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
# x_73 => var_mean_13
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


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjdxgtwm6yw6lpbggfd4rdsqbzcibrelggnewcqj63hhxzqgl7v.py
# Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
# x_73 => var_mean_13
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdrzy63t6yfd6lkxonpchs2umxwxh7pzjwiyggsocaa62qawm6o.py
# Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
# x_73 => add_69, add_70, add_71, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_per_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vqexjipa7kronlbiey3yifi6qswl5vdxus7chw3qaaq6an4tlk.py
# Source Nodes: [x_73, x_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_73 => add_69, add_72, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# x_76 => relu_9
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caebovzg3bv45dveoomu7xrala72dmum7s5ziy3ypzhvhjwjayt7.py
# Source Nodes: [x_77], Original ATen: [aten.convolution]
# x_77 => convolution_14
triton_poi_fused_convolution_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5m/c5m6f3ehoxhigrjke6ciulutf5t755adnhtyfsid2ijoqqjtklh7.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
# x_78 => var_mean_14
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/oc/cocy5xxymmionl3lpsudxf7gbmolcbbz56ut5uv75ifn7qme63ah.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
# x_78 => add_74, add_75, add_76, mul_100, mul_101, mul_102, mul_103, mul_99, rsqrt_14, squeeze_43, var_mean_14
triton_per_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/rv/crvxcbqdqbn6av2q2ywfha4okvdzvf2hdgtoloouj7eoeqxohm3m.py
# Source Nodes: [x_78, x_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_78 => add_74, add_77, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# x_81 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_33', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chulhyjmyemakyla4sjvt3vcfoetfpsetag7jj34kghhcqe477yc.py
# Source Nodes: [x_83], Original ATen: [aten.convolution]
# x_83 => convolution_15
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ku/cku4q7wijr7rr3jcdudhcqqmtmlmvzdpov2mcrx4r25du7kftp7a.py
# Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
# x_84 => var_mean_15
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5ooondo3g552hoauigum76dibzsl732qbkrgugmfh2sbcofju7.py
# Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
# x_84 => add_79, add_80, add_81, mul_106, mul_107, mul_108, mul_109, mul_110, rsqrt_15, squeeze_46, var_mean_15
triton_per_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ad/cadogcbgyauzf3fjtcpe5f32encqsf4ndes4daxi7ul5czchricq.py
# Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
# x_84 => add_79, add_82, mul_105, mul_111, rsqrt_15, sub_15, var_mean_15
triton_poi_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []},
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ud/cudu264mkptjbqkpaahukydiyxi3qszedbpuqemogerk4k4g62xw.py
# Source Nodes: [x_88], Original ATen: [aten.convolution]
# x_88 => convolution_16
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cus2b4ldjatrryunwqnllwkrdieyouaxksi4isiplib3n6ld6bgi.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
# x_89 => var_mean_16
triton_red_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_youkaichao/sg/csga7yhqk7rujafjvqhtt2qpp55qchwufklhf3g4h7r4girk3qwg.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
# x_89 => add_84, add_85, add_86, mul_113, mul_114, mul_115, mul_116, mul_117, rsqrt_16, squeeze_49, var_mean_16
triton_per_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3gvz7quabfajma63a6jrz2u636eehnpaul4l66ts7rpuyitcul.py
# Source Nodes: [x_89, x_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_89 => add_84, add_87, mul_112, mul_118, rsqrt_16, sub_16, var_mean_16
# x_92 => relu_11
triton_poi_fused__native_batch_norm_legit_functional_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
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
    tmp4 = 6272.0
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


# kernel path: /tmp/torchinductor_youkaichao/ay/cayv7q7mm6a5hkmdhjgchzd3hvdoq4xwcikmavtdrbjm6mi6sbzl.py
# Source Nodes: [shortcut_6, x_100], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_6 => add_98
# x_100 => add_94, add_97, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
triton_poi_fused__native_batch_norm_legit_functional_add_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_42', 'mutated_arg_names': []},
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfr2bdj6egx2m74tdovmq7eklirntrvxx5kokm4fnlwdhopvpib.py
# Source Nodes: [x_105], Original ATen: [aten.convolution]
# x_105 => convolution_19
triton_poi_fused_convolution_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ct/cctk4bajtg5yesktuuf645qino2ltd7ugiz7v7nuk2t7mbcdfuux.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => var_mean_19
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


# kernel path: /tmp/torchinductor_youkaichao/er/cerrwafaxayyqwbrk3yyzlebnugycjsqb3al4arndms3dhxp4azf.py
# Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
# x_106 => add_100, add_101, add_102, mul_134, mul_135, mul_136, mul_137, mul_138, rsqrt_19, squeeze_58, var_mean_19
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


# kernel path: /tmp/torchinductor_youkaichao/2i/c2il5wwuqgc2dzv5chgy5yw5kdxzcfagqs4fh4lmylbidfusf5pj.py
# Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_106 => add_100, add_103, mul_133, mul_139, rsqrt_19, sub_19, var_mean_19
# x_109 => relu_13
triton_poi_fused__native_batch_norm_legit_functional_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xgovmc7yibms57olkolhuvzupwiwlm42gzh3vcujojltmez3g2.py
# Source Nodes: [x_144], Original ATen: [aten.convolution]
# x_144 => convolution_26
triton_poi_fused_convolution_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_47', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/o7/co7g5hgczvkut4duzrtfdrh5u2vtyvvpcr5vvtumkfst2phqaalb.py
# Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
# x_145 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/at/catlr4vmc2on5pcfmdot46bt6iutsutujuhv5o3m2nabdprojucc.py
# Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
# x_145 => add_137, add_138, add_139, mul_183, mul_184, mul_185, mul_186, mul_187, rsqrt_26, squeeze_79, var_mean_26
triton_per_fused__native_batch_norm_legit_functional_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_49', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/db/cdb5q3q4632qz3fcnd5mdzn43s5vlec2wk52teqgsc3qcx4pbgqj.py
# Source Nodes: [x_145, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_145 => add_137, add_140, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
# x_148 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_relu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs46qplwfu3bktxea47ahigw6lnnwy5qxqlzmf2wlmay4miffehg.py
# Source Nodes: [x_150], Original ATen: [aten.convolution]
# x_150 => convolution_27
triton_poi_fused_convolution_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrshcp4tpfhjueuw2enli7bbdbiqueao6uu6ni3kgwg4vix6ues.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => var_mean_27
triton_red_fused__native_batch_norm_legit_functional_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_52', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3ukrznuyckznynaop37xv7hlvhv6hx7kuq7bjhkn3eqcocevmkv.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => add_142, add_143, add_144, mul_190, mul_191, mul_192, mul_193, mul_194, rsqrt_27, squeeze_82, var_mean_27
triton_per_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqhumlivrzivqqyjc4xxjdn332swopem4zmti2ak7biwb5ijdpd.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
# x_151 => add_142, add_145, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
triton_poi_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': []},
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwsvw6miy6dcctmgxzzxpdeomhfr7fl6knrdolp7fciqcpfowqiu.py
# Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_10 => add_161
# x_167 => add_157, add_160, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
triton_poi_fused__native_batch_norm_legit_functional_add_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_55', 'mutated_arg_names': []},
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2ylltpxomrkbzxgovmicntpuccrhwhcecihthwrsptk2qbvbdd.py
# Source Nodes: [x_172], Original ATen: [aten.convolution]
# x_172 => convolution_31
triton_poi_fused_convolution_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_56', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgn2k7tnoyjx7b7u4sytdfzl3rkou4dp4paivxgayc6qlwpdix3.py
# Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
# x_173 => var_mean_31
triton_red_fused__native_batch_norm_legit_functional_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_57', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2c/c2cvygxwxqkb5hcfaol7mfzi2wi55t64g7vzhipvymrurjrqfj5c.py
# Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
# x_173 => add_163, add_164, add_165, mul_218, mul_219, mul_220, mul_221, mul_222, rsqrt_31, squeeze_94, var_mean_31
triton_per_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/qd/cqd2o6dlgv2z7w7aepbfsbibu72by3vni2pojuvxugfixcsq6r5x.py
# Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_173 => add_163, add_166, mul_217, mul_223, rsqrt_31, sub_31, var_mean_31
# x_176 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_relu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4q7z3i3zg2gg4z3hghmsrvi5fimi6tmh26tkthss2vajrki72l.py
# Source Nodes: [x_217], Original ATen: [aten.convolution]
# x_217 => convolution_39
triton_poi_fused_convolution_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qs/cqsdifhjyrlf43ifhxfuxwhsj6r3nu25khbxxncxynkz5v3wv5ew.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
# x_218 => var_mean_39
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/em/ceml5l5hr4k6nf4mbfjnry7xu2bljtfzwbgeuik4ghunpqfr3piz.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
# x_218 => add_205, add_206, add_207, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
triton_per_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5bks34i774dk7tcn7wjnbwc62tjksmcthltisqjjp2ozozmchy.py
# Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
# x_218 => add_205, add_208, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
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


# kernel path: /tmp/torchinductor_youkaichao/de/cderpbyu7w2vdqwkbamwi4lxw5s6wcyrvz33pnz2ipf36rv3zzcq.py
# Source Nodes: [x_222], Original ATen: [aten.convolution]
# x_222 => convolution_40
triton_poi_fused_convolution_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hg/chgnuhybzwjuajhmqsnfctwvxwfea7rkj4ta73rccbj53flt7pxd.py
# Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
# x_223 => var_mean_40
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtx5eczexwaghult6innxs67zffneyfwnkpxvszl6f4xyt2zd35.py
# Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
# x_223 => add_210, add_211, add_212, mul_281, mul_282, mul_283, mul_284, mul_285, rsqrt_40, squeeze_121, var_mean_40
triton_per_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/cj/ccjwypoycf26z4jcj424nbx7rjakxsahcpmj2dshk7zstvpq54uf.py
# Source Nodes: [x_223, x_226], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_223 => add_210, add_213, mul_280, mul_286, rsqrt_40, sub_40, var_mean_40
# x_226 => relu_27
triton_poi_fused__native_batch_norm_legit_functional_relu_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_67', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgxandbx3fzhtxdog6jknqpnjw2dxcx3y7oeyk7dxu3jx7p5srri.py
# Source Nodes: [shortcut_14, x_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_14 => add_224
# x_234 => add_220, add_223, mul_294, mul_300, rsqrt_42, sub_42, var_mean_42
triton_poi_fused__native_batch_norm_legit_functional_add_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_68', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6kpbqgmblk5k56dzbwefjtpsoec34vaagv3n2iuceyo4cwitgk.py
# Source Nodes: [x_256], Original ATen: [aten.convolution]
# x_256 => convolution_46
triton_poi_fused_convolution_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chmrxbswfvas25kizlrjkher7dz4vwuaehzrpt6wo2vpdvqn7wy4.py
# Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
# x_257 => var_mean_46
triton_red_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_70', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyvcvyzji4boy3uvwnihvfz2a54voc2adxvizmhxaqcnbxqalxq.py
# Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
# x_257 => add_242, add_243, add_244, mul_323, mul_324, mul_325, mul_326, mul_327, rsqrt_46, squeeze_139, var_mean_46
triton_per_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/si/csiimsjwnuwvuf33ujccl6dc4resuhno74yxhwy5wrjx3smldtbq.py
# Source Nodes: [x_257, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_257 => add_242, add_245, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
# x_260 => relu_31
triton_poi_fused__native_batch_norm_legit_functional_relu_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_72', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2yo6hlj7ybwpv5eywz6mb4fcjpou7wg7hvbo6rqxrvkreakbldj.py
# Source Nodes: [x_278], Original ATen: [aten.convolution]
# x_278 => convolution_50
triton_poi_fused_convolution_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_73', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ng/cngv3cz7yinkkh2z6c4mc3wtf3dlbblys34gdad2ansmcvkggarc.py
# Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
# x_279 => var_mean_50
triton_red_fused__native_batch_norm_legit_functional_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_74', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ul/culbk5rtq7wme5722yfrnz36qgtgeejmv5vr6a4kyh3hh2a5muq6.py
# Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
# x_279 => add_263, add_264, add_265, mul_351, mul_352, mul_353, mul_354, mul_355, rsqrt_50, squeeze_151, var_mean_50
triton_per_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/vs/cvss4mu3pcrcm5ewhgjzwnkw4c7hggasyl6jgqb6etbx3flcixzu.py
# Source Nodes: [x_279, x_282], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_279 => add_263, add_266, mul_350, mul_356, rsqrt_50, sub_50, var_mean_50
# x_282 => relu_34
triton_poi_fused__native_batch_norm_legit_functional_relu_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_76', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozkquvj65b3glh7ocp5cwqj7bsp44pxb573o4fxypjpycbmofln.py
# Source Nodes: [x_284], Original ATen: [aten.convolution]
# x_284 => convolution_51
triton_poi_fused_convolution_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_77', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ki/ckic5667vs4iqmhry3f6cwrqumqm5cutwnxpvodm6bpellugxtfx.py
# Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
# x_285 => var_mean_51
triton_red_fused__native_batch_norm_legit_functional_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_78', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/at/catjujgcr55cc3lrg4halijk6v2qeamwrzbq7oyogxz2daqwamfu.py
# Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
# x_285 => add_268, add_269, add_270, mul_358, mul_359, mul_360, mul_361, mul_362, rsqrt_51, squeeze_154, var_mean_51
triton_per_fused__native_batch_norm_legit_functional_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_79', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/j4/cj47zi5aqdslc7o52xwwdayumzsraqb6ketiomt3cocukgcg4iz4.py
# Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
# x_285 => add_268, add_271, mul_357, mul_363, rsqrt_51, sub_51, var_mean_51
triton_poi_fused__native_batch_norm_legit_functional_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_80', 'mutated_arg_names': []},
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kb/ckb4ohmj5tu6nuajdtpsvhvkf6tiedgebouvchjgjcewstd773qx.py
# Source Nodes: [x_289], Original ATen: [aten.convolution]
# x_289 => convolution_52
triton_poi_fused_convolution_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_81', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zcdjjkqfsepvx54uuvdzwvxj4fhv3s3w4nel5noknrrd2cuvgn.py
# Source Nodes: [x_290], Original ATen: [aten._native_batch_norm_legit_functional]
# x_290 => var_mean_52
triton_red_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpnegw3ef67kfzby3yh27szzuv745yjfrmt2ukp2td7qgsbjk7t.py
# Source Nodes: [x_290], Original ATen: [aten._native_batch_norm_legit_functional]
# x_290 => add_273, add_274, add_275, mul_365, mul_366, mul_367, mul_368, mul_369, rsqrt_52, squeeze_157, var_mean_52
triton_per_fused__native_batch_norm_legit_functional_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_83', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/gf/cgf5lp7csrovc4ehaiosn6zytgh4mnjotz3w3dextai3vyu7pu42.py
# Source Nodes: [x_290, x_293], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_290 => add_273, add_276, mul_364, mul_370, rsqrt_52, sub_52, var_mean_52
# x_293 => relu_35
triton_poi_fused__native_batch_norm_legit_functional_relu_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cer5remczezidofjffuulwemr5nrjskkso5ik5l3fb7oywknjr73.py
# Source Nodes: [shortcut_18, x_301], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# shortcut_18 => add_287
# x_301 => add_283, add_286, mul_378, mul_384, rsqrt_54, sub_54, var_mean_54
triton_poi_fused__native_batch_norm_legit_functional_add_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_85', 'mutated_arg_names': []},
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5n/c5noepk4w2y4tzorvpe6gfigxlnedbvesifutu6m2tpgxvxunejw.py
# Source Nodes: [x_351], Original ATen: [aten.convolution]
# x_351 => convolution_63
triton_poi_fused_convolution_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2816
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 352
    y1 = (yindex // 352)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (352*x2) + (17248*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp27knmh7s3zaai4rtaq5vypyzmzcmpd3pnms7mf4byanqjrjoto.py
# Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_functional]
# x_352 => var_mean_63
triton_red_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1408
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 352
    x1 = (xindex // 352)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (352*r2) + (34496*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7t5ojdpeko5lgc7h3lbqp3hslxlmq3gnmmtdqtqgglq7awodhw.py
# Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_functional]
# x_352 => add_331, add_332, add_333, mul_442, mul_443, mul_444, mul_445, mul_446, rsqrt_63, squeeze_190, var_mean_63
triton_per_fused__native_batch_norm_legit_functional_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_88', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 352
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (352*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (352*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (352*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4vepcznhttrm5w3kgwmcxshd5wlnis5ch6falzl47ww3dzbegl.py
# Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_functional]
# x_352 => add_331, add_334, mul_441, mul_447, rsqrt_63, sub_63, var_mean_63
triton_poi_fused__native_batch_norm_legit_functional_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 137984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 352
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3g73hoabqacdutr4ob5jqqehxkia3lzdskmil5gqssamm2xszs.py
# Source Nodes: [x_357], Original ATen: [aten.convolution]
# x_357 => convolution_64
triton_poi_fused_convolution_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15872
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1984
    y1 = (yindex // 1984)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1984*x2) + (97216*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvs5ajfo3es54ksguom7pphusp3ln4hrzdfmijrp2vnpquioafzc.py
# Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_functional]
# x_358 => var_mean_64
triton_red_fused__native_batch_norm_legit_functional_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7936
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1984
    x1 = (xindex // 1984)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1984*r2) + (194432*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bfle4fc2leycsebkh7sjudszavvonak6nadkftlrcy5kajbopk.py
# Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_functional]
# x_358 => add_336, add_337, add_338, mul_449, mul_450, mul_451, mul_452, mul_453, rsqrt_64, squeeze_193, var_mean_64
triton_per_fused__native_batch_norm_legit_functional_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_92', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1984
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1984*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1984*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1984*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciy7zs7tt2hxm3hsc7nm6ak62aohqrs4kqnjzrdds2fm7zwe5s34.py
# Source Nodes: [x_358, x_362], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_358 => add_336, add_339, mul_448, mul_454, rsqrt_64, sub_64, var_mean_64
# x_362 => relu_43
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 777728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1984
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbe7oa6u34zf7otkzsotzroeibdg2jdwk7dn7ohy2vczcwnnu2c.py
# Source Nodes: [x_363, x_365], Original ATen: [aten.mean, aten.view]
# x_363 => mean
# x_365 => view
triton_per_fused_mean_view_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_94', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 15872
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1984
    x1 = (xindex // 1984)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1984*r2) + (97216*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgfsmzjqlyp6m7y7bbsstaiuygsulaksvwbwy74bttcirwtqxgf3.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_95', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (96, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (24, ), (1, ))
    assert_size_stride(primals_15, (24, ), (1, ))
    assert_size_stride(primals_16, (24, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_20, (24, ), (1, ))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_22, (24, ), (1, ))
    assert_size_stride(primals_23, (24, ), (1, ))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, ), (1, ))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_27, (144, ), (1, ))
    assert_size_stride(primals_28, (144, ), (1, ))
    assert_size_stride(primals_29, (144, ), (1, ))
    assert_size_stride(primals_30, (144, ), (1, ))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (32, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (32, ), (1, ))
    assert_size_stride(primals_38, (32, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (32, ), (1, ))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (32, ), (1, ))
    assert_size_stride(primals_50, (32, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (192, ), (1, ))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (384, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (112, ), (1, ))
    assert_size_stride(primals_80, (112, ), (1, ))
    assert_size_stride(primals_81, (672, ), (1, ))
    assert_size_stride(primals_82, (672, ), (1, ))
    assert_size_stride(primals_83, (672, ), (1, ))
    assert_size_stride(primals_84, (672, ), (1, ))
    assert_size_stride(primals_85, (112, ), (1, ))
    assert_size_stride(primals_86, (112, ), (1, ))
    assert_size_stride(primals_87, (672, ), (1, ))
    assert_size_stride(primals_88, (672, ), (1, ))
    assert_size_stride(primals_89, (672, ), (1, ))
    assert_size_stride(primals_90, (672, ), (1, ))
    assert_size_stride(primals_91, (112, ), (1, ))
    assert_size_stride(primals_92, (112, ), (1, ))
    assert_size_stride(primals_93, (336, ), (1, ))
    assert_size_stride(primals_94, (336, ), (1, ))
    assert_size_stride(primals_95, (336, ), (1, ))
    assert_size_stride(primals_96, (336, ), (1, ))
    assert_size_stride(primals_97, (112, ), (1, ))
    assert_size_stride(primals_98, (112, ), (1, ))
    assert_size_stride(primals_99, (672, ), (1, ))
    assert_size_stride(primals_100, (672, ), (1, ))
    assert_size_stride(primals_101, (672, ), (1, ))
    assert_size_stride(primals_102, (672, ), (1, ))
    assert_size_stride(primals_103, (184, ), (1, ))
    assert_size_stride(primals_104, (184, ), (1, ))
    assert_size_stride(primals_105, (1104, ), (1, ))
    assert_size_stride(primals_106, (1104, ), (1, ))
    assert_size_stride(primals_107, (1104, ), (1, ))
    assert_size_stride(primals_108, (1104, ), (1, ))
    assert_size_stride(primals_109, (184, ), (1, ))
    assert_size_stride(primals_110, (184, ), (1, ))
    assert_size_stride(primals_111, (1104, ), (1, ))
    assert_size_stride(primals_112, (1104, ), (1, ))
    assert_size_stride(primals_113, (1104, ), (1, ))
    assert_size_stride(primals_114, (1104, ), (1, ))
    assert_size_stride(primals_115, (184, ), (1, ))
    assert_size_stride(primals_116, (184, ), (1, ))
    assert_size_stride(primals_117, (1104, ), (1, ))
    assert_size_stride(primals_118, (1104, ), (1, ))
    assert_size_stride(primals_119, (1104, ), (1, ))
    assert_size_stride(primals_120, (1104, ), (1, ))
    assert_size_stride(primals_121, (184, ), (1, ))
    assert_size_stride(primals_122, (184, ), (1, ))
    assert_size_stride(primals_123, (1104, ), (1, ))
    assert_size_stride(primals_124, (1104, ), (1, ))
    assert_size_stride(primals_125, (1104, ), (1, ))
    assert_size_stride(primals_126, (1104, ), (1, ))
    assert_size_stride(primals_127, (352, ), (1, ))
    assert_size_stride(primals_128, (352, ), (1, ))
    assert_size_stride(primals_129, (1984, ), (1, ))
    assert_size_stride(primals_130, (1984, ), (1, ))
    assert_size_stride(primals_131, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_132, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_133, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_135, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_136, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_138, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_139, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_140, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_141, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_142, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_144, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_145, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (32, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_147, (96, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_148, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_149, (32, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_150, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_151, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_152, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_153, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_154, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_156, (192, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_157, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_158, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_159, (192, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_160, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_161, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_162, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_163, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_164, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_165, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_166, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_167, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_168, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_169, (384, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_170, (112, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_171, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_172, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_173, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_174, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_175, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_176, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_177, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_178, (336, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_179, (112, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_180, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_181, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_182, (184, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_183, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_184, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_186, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_187, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_188, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_189, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_190, (1104, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_191, (184, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_192, (1104, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_193, (1104, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_194, (352, 1104, 1, 1), (1104, 1, 1, 1))
    assert_size_stride(primals_195, (1984, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_196, (1000, 1984), (1984, 1))
    assert_size_stride(primals_197, (1000, ), (1, ))
    assert_size_stride(primals_198, (), ())
    assert_size_stride(primals_199, (16, ), (1, ))
    assert_size_stride(primals_200, (16, ), (1, ))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (16, ), (1, ))
    assert_size_stride(primals_203, (16, ), (1, ))
    assert_size_stride(primals_204, (), ())
    assert_size_stride(primals_205, (16, ), (1, ))
    assert_size_stride(primals_206, (16, ), (1, ))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (16, ), (1, ))
    assert_size_stride(primals_209, (16, ), (1, ))
    assert_size_stride(primals_210, (), ())
    assert_size_stride(primals_211, (96, ), (1, ))
    assert_size_stride(primals_212, (96, ), (1, ))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (96, ), (1, ))
    assert_size_stride(primals_215, (96, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (24, ), (1, ))
    assert_size_stride(primals_218, (24, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (24, ), (1, ))
    assert_size_stride(primals_221, (24, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (24, ), (1, ))
    assert_size_stride(primals_224, (24, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (24, ), (1, ))
    assert_size_stride(primals_227, (24, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (24, ), (1, ))
    assert_size_stride(primals_230, (24, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (24, ), (1, ))
    assert_size_stride(primals_233, (24, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (24, ), (1, ))
    assert_size_stride(primals_236, (24, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (144, ), (1, ))
    assert_size_stride(primals_239, (144, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (144, ), (1, ))
    assert_size_stride(primals_242, (144, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (32, ), (1, ))
    assert_size_stride(primals_245, (32, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (96, ), (1, ))
    assert_size_stride(primals_248, (96, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (96, ), (1, ))
    assert_size_stride(primals_251, (96, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (32, ), (1, ))
    assert_size_stride(primals_254, (32, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (192, ), (1, ))
    assert_size_stride(primals_257, (192, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (192, ), (1, ))
    assert_size_stride(primals_260, (192, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (192, ), (1, ))
    assert_size_stride(primals_266, (192, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (192, ), (1, ))
    assert_size_stride(primals_269, (192, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (32, ), (1, ))
    assert_size_stride(primals_272, (32, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (192, ), (1, ))
    assert_size_stride(primals_275, (192, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (192, ), (1, ))
    assert_size_stride(primals_278, (192, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (64, ), (1, ))
    assert_size_stride(primals_281, (64, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (192, ), (1, ))
    assert_size_stride(primals_284, (192, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (192, ), (1, ))
    assert_size_stride(primals_287, (192, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (384, ), (1, ))
    assert_size_stride(primals_293, (384, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (384, ), (1, ))
    assert_size_stride(primals_296, (384, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (64, ), (1, ))
    assert_size_stride(primals_299, (64, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (384, ), (1, ))
    assert_size_stride(primals_302, (384, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (384, ), (1, ))
    assert_size_stride(primals_305, (384, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (384, ), (1, ))
    assert_size_stride(primals_311, (384, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (384, ), (1, ))
    assert_size_stride(primals_314, (384, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (112, ), (1, ))
    assert_size_stride(primals_317, (112, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (672, ), (1, ))
    assert_size_stride(primals_320, (672, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (672, ), (1, ))
    assert_size_stride(primals_323, (672, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (112, ), (1, ))
    assert_size_stride(primals_326, (112, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (672, ), (1, ))
    assert_size_stride(primals_329, (672, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (672, ), (1, ))
    assert_size_stride(primals_332, (672, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (112, ), (1, ))
    assert_size_stride(primals_335, (112, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (336, ), (1, ))
    assert_size_stride(primals_338, (336, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (336, ), (1, ))
    assert_size_stride(primals_341, (336, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (112, ), (1, ))
    assert_size_stride(primals_344, (112, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (672, ), (1, ))
    assert_size_stride(primals_347, (672, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (672, ), (1, ))
    assert_size_stride(primals_350, (672, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (184, ), (1, ))
    assert_size_stride(primals_353, (184, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (1104, ), (1, ))
    assert_size_stride(primals_356, (1104, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (1104, ), (1, ))
    assert_size_stride(primals_359, (1104, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (184, ), (1, ))
    assert_size_stride(primals_362, (184, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (1104, ), (1, ))
    assert_size_stride(primals_365, (1104, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (1104, ), (1, ))
    assert_size_stride(primals_368, (1104, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (184, ), (1, ))
    assert_size_stride(primals_371, (184, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (1104, ), (1, ))
    assert_size_stride(primals_374, (1104, ), (1, ))
    assert_size_stride(primals_375, (), ())
    assert_size_stride(primals_376, (1104, ), (1, ))
    assert_size_stride(primals_377, (1104, ), (1, ))
    assert_size_stride(primals_378, (), ())
    assert_size_stride(primals_379, (184, ), (1, ))
    assert_size_stride(primals_380, (184, ), (1, ))
    assert_size_stride(primals_381, (), ())
    assert_size_stride(primals_382, (1104, ), (1, ))
    assert_size_stride(primals_383, (1104, ), (1, ))
    assert_size_stride(primals_384, (), ())
    assert_size_stride(primals_385, (1104, ), (1, ))
    assert_size_stride(primals_386, (1104, ), (1, ))
    assert_size_stride(primals_387, (), ())
    assert_size_stride(primals_388, (352, ), (1, ))
    assert_size_stride(primals_389, (352, ), (1, ))
    assert_size_stride(primals_390, (), ())
    assert_size_stride(primals_391, (1984, ), (1, ))
    assert_size_stride(primals_392, (1984, ), (1, ))
    assert_size_stride(primals_393, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_131, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_131
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_393, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_393
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
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_199, primals_200, buf10, buf11, buf13, primals_199, primals_200, 16, 7, grid=grid(16), stream=stream0)
        del primals_199
        del primals_200
        buf14 = reinterpret_tensor(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf2  # reuse
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf3, buf10, buf11, primals_1, primals_2, buf14, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf16 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf15, buf16, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf17 = buf6; del buf6  # reuse
        buf18 = buf5; del buf5  # reuse
        buf19 = buf4; del buf4  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf16, buf17, buf18, buf19, 12544, 128, grid=grid(12544), stream=stream0)
        buf20 = buf9; del buf9  # reuse
        buf21 = buf8; del buf8  # reuse
        buf22 = buf7; del buf7  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf17, buf18, buf19, buf20, buf21, buf22, 112, 112, grid=grid(112), stream=stream0)
        buf23 = buf11; del buf11  # reuse
        buf24 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf26 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf20, buf21, buf22, primals_202, primals_203, buf23, buf24, buf26, primals_202, primals_203, 16, 7, grid=grid(16), stream=stream0)
        del primals_202
        del primals_203
        buf27 = reinterpret_tensor(buf15, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf15  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf16, buf23, buf24, primals_3, primals_4, buf27, 1605632, grid=grid(1605632), stream=stream0)
        del primals_4
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf28, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf29 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf28, buf29, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf30 = buf19; del buf19  # reuse
        buf31 = buf18; del buf18  # reuse
        buf32 = buf17; del buf17  # reuse
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf29, buf30, buf31, buf32, 12544, 128, grid=grid(12544), stream=stream0)
        buf33 = buf22; del buf22  # reuse
        buf34 = buf21; del buf21  # reuse
        buf35 = buf20; del buf20  # reuse
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf30, buf31, buf32, buf33, buf34, buf35, 112, 112, grid=grid(112), stream=stream0)
        buf36 = buf24; del buf24  # reuse
        buf37 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf39 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf33, buf34, buf35, primals_205, primals_206, buf36, buf37, buf39, primals_205, primals_206, 16, 7, grid=grid(16), stream=stream0)
        del primals_205
        del primals_206
        buf40 = reinterpret_tensor(buf28, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf28  # reuse
        # Source Nodes: [x_11, x_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf29, buf36, buf37, primals_5, primals_6, buf40, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf42 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf41, buf42, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf43 = buf32; del buf32  # reuse
        buf44 = buf31; del buf31  # reuse
        buf45 = buf30; del buf30  # reuse
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf42, buf43, buf44, buf45, 12544, 128, grid=grid(12544), stream=stream0)
        buf46 = buf35; del buf35  # reuse
        buf47 = buf34; del buf34  # reuse
        buf48 = buf33; del buf33  # reuse
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf43, buf44, buf45, buf46, buf47, buf48, 112, 112, grid=grid(112), stream=stream0)
        del buf43
        del buf44
        del buf45
        buf49 = buf37; del buf37  # reuse
        buf50 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf52 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf46, buf47, buf48, primals_208, primals_209, buf49, buf50, buf52, primals_208, primals_209, 16, 7, grid=grid(16), stream=stream0)
        del primals_208
        del primals_209
        buf53 = reinterpret_tensor(buf41, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf41  # reuse
        # Source Nodes: [shortcut_1, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_7.run(buf42, buf49, buf50, primals_7, primals_8, buf14, buf53, 1605632, grid=grid(1605632), stream=stream0)
        del buf50
        del primals_8
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        buf55 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf54, buf55, 768, 12544, grid=grid(768, 12544), stream=stream0)
        buf56 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        buf58 = empty_strided((1, 96, 1, 1, 784), (75264, 1, 75264, 75264, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf55, buf56, buf57, buf58, 75264, 128, grid=grid(75264), stream=stream0)
        buf59 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((1, 96, 1, 1, 7), (672, 1, 672, 672, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf56, buf57, buf58, buf59, buf60, buf61, 672, 112, grid=grid(672), stream=stream0)
        del buf56
        del buf57
        del buf58
        buf62 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf65 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf59, buf60, buf61, primals_211, primals_212, buf62, buf63, buf65, primals_211, primals_212, 96, 7, grid=grid(96), stream=stream0)
        del primals_211
        del primals_212
        buf66 = reinterpret_tensor(buf54, (8, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf54  # reuse
        # Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf55, buf62, buf63, primals_9, primals_10, buf66, 9633792, grid=grid(9633792), stream=stream0)
        del primals_10
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_136, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf67, (8, 96, 56, 56), (301056, 3136, 56, 1))
        buf68 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(buf67, buf68, 768, 3136, grid=grid(768, 3136), stream=stream0)
        buf69 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        buf71 = empty_strided((1, 96, 1, 1, 196), (18816, 1, 18816, 18816, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf68, buf69, buf70, buf71, 18816, 128, grid=grid(18816), stream=stream0)
        buf72 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((1, 96, 1, 1, 2), (192, 1, 192, 192, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf69, buf70, buf71, buf72, buf73, buf74, 192, 98, grid=grid(192), stream=stream0)
        del buf69
        del buf70
        del buf71
        buf75 = buf63; del buf63  # reuse
        buf76 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf78 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf72, buf73, buf74, primals_214, primals_215, buf75, buf76, buf78, primals_214, primals_215, 96, 2, grid=grid(96), stream=stream0)
        del primals_214
        del primals_215
        buf79 = reinterpret_tensor(buf67, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf67  # reuse
        # Source Nodes: [x_28, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_17.run(buf68, buf75, buf76, primals_11, primals_12, buf79, 2408448, grid=grid(2408448), stream=stream0)
        del primals_12
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf81 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf80, buf81, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf82 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf83 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        buf84 = empty_strided((1, 24, 1, 1, 196), (4704, 1, 4704, 4704, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf81, buf82, buf83, buf84, 4704, 128, grid=grid(4704), stream=stream0)
        buf85 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf86 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((1, 24, 1, 1, 2), (48, 1, 48, 48, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf82, buf83, buf84, buf85, buf86, buf87, 48, 98, grid=grid(48), stream=stream0)
        buf88 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf89 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf91 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf85, buf86, buf87, primals_217, primals_218, buf88, buf89, buf91, primals_217, primals_218, 24, 2, grid=grid(24), stream=stream0)
        del primals_217
        del primals_218
        buf92 = reinterpret_tensor(buf80, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf80  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_22.run(buf81, buf88, buf89, primals_13, primals_14, buf92, 602112, grid=grid(602112), stream=stream0)
        del primals_14
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf94 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf93, buf94, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf95 = buf84; del buf84  # reuse
        buf96 = buf83; del buf83  # reuse
        buf97 = buf82; del buf82  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf94, buf95, buf96, buf97, 4704, 128, grid=grid(4704), stream=stream0)
        buf98 = buf87; del buf87  # reuse
        buf99 = buf86; del buf86  # reuse
        buf100 = buf85; del buf85  # reuse
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf95, buf96, buf97, buf98, buf99, buf100, 48, 98, grid=grid(48), stream=stream0)
        buf101 = buf89; del buf89  # reuse
        buf102 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf104 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf98, buf99, buf100, primals_220, primals_221, buf101, buf102, buf104, primals_220, primals_221, 24, 2, grid=grid(24), stream=stream0)
        del primals_220
        del primals_221
        buf105 = reinterpret_tensor(buf93, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf93  # reuse
        # Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf94, buf101, buf102, primals_15, primals_16, buf105, 602112, grid=grid(602112), stream=stream0)
        del primals_16
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf106, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf107 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf106, buf107, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf108 = buf97; del buf97  # reuse
        buf109 = buf96; del buf96  # reuse
        buf110 = buf95; del buf95  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf107, buf108, buf109, buf110, 4704, 128, grid=grid(4704), stream=stream0)
        buf111 = buf99; del buf99  # reuse
        buf112 = buf98; del buf98  # reuse
        buf113 = buf100; del buf100  # reuse
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf108, buf109, buf110, buf111, buf112, buf113, 48, 98, grid=grid(48), stream=stream0)
        buf114 = buf102; del buf102  # reuse
        buf115 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf117 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf111, buf112, buf113, primals_223, primals_224, buf114, buf115, buf117, primals_223, primals_224, 24, 2, grid=grid(24), stream=stream0)
        del primals_223
        del primals_224
        buf118 = reinterpret_tensor(buf106, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf106  # reuse
        # Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf107, buf114, buf115, primals_17, primals_18, buf118, 602112, grid=grid(602112), stream=stream0)
        del primals_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf120 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf119, buf120, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf121 = buf110; del buf110  # reuse
        buf122 = buf109; del buf109  # reuse
        buf123 = buf108; del buf108  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf120, buf121, buf122, buf123, 4704, 128, grid=grid(4704), stream=stream0)
        buf124 = buf113; del buf113  # reuse
        buf125 = buf112; del buf112  # reuse
        buf126 = buf111; del buf111  # reuse
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf121, buf122, buf123, buf124, buf125, buf126, 48, 98, grid=grid(48), stream=stream0)
        buf127 = buf115; del buf115  # reuse
        buf128 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf130 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf124, buf125, buf126, primals_226, primals_227, buf127, buf128, buf130, primals_226, primals_227, 24, 2, grid=grid(24), stream=stream0)
        del primals_226
        del primals_227
        buf131 = reinterpret_tensor(buf119, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf119  # reuse
        # Source Nodes: [shortcut_3, x_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_24.run(buf120, buf127, buf128, primals_19, primals_20, buf92, buf131, 602112, grid=grid(602112), stream=stream0)
        del primals_20
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf133 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf132, buf133, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf134 = buf123; del buf123  # reuse
        buf135 = buf122; del buf122  # reuse
        buf136 = buf121; del buf121  # reuse
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf133, buf134, buf135, buf136, 4704, 128, grid=grid(4704), stream=stream0)
        buf137 = buf126; del buf126  # reuse
        buf138 = buf125; del buf125  # reuse
        buf139 = buf124; del buf124  # reuse
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf134, buf135, buf136, buf137, buf138, buf139, 48, 98, grid=grid(48), stream=stream0)
        buf140 = buf128; del buf128  # reuse
        buf141 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf143 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf137, buf138, buf139, primals_229, primals_230, buf140, buf141, buf143, primals_229, primals_230, 24, 2, grid=grid(24), stream=stream0)
        del primals_229
        del primals_230
        buf144 = reinterpret_tensor(buf132, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf132  # reuse
        # Source Nodes: [x_56, x_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf133, buf140, buf141, primals_21, primals_22, buf144, 602112, grid=grid(602112), stream=stream0)
        del primals_22
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf145, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf146 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf145, buf146, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf147 = buf136; del buf136  # reuse
        buf148 = buf135; del buf135  # reuse
        buf149 = buf134; del buf134  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf146, buf147, buf148, buf149, 4704, 128, grid=grid(4704), stream=stream0)
        buf150 = buf139; del buf139  # reuse
        buf151 = buf138; del buf138  # reuse
        buf152 = buf137; del buf137  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf147, buf148, buf149, buf150, buf151, buf152, 48, 98, grid=grid(48), stream=stream0)
        buf153 = buf141; del buf141  # reuse
        buf154 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf156 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf150, buf151, buf152, primals_232, primals_233, buf153, buf154, buf156, primals_232, primals_233, 24, 2, grid=grid(24), stream=stream0)
        del primals_232
        del primals_233
        buf157 = reinterpret_tensor(buf145, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf145  # reuse
        # Source Nodes: [x_61, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf146, buf153, buf154, primals_23, primals_24, buf157, 602112, grid=grid(602112), stream=stream0)
        del primals_24
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf159 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf158, buf159, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf160 = buf149; del buf149  # reuse
        buf161 = buf148; del buf148  # reuse
        buf162 = buf147; del buf147  # reuse
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf159, buf160, buf161, buf162, 4704, 128, grid=grid(4704), stream=stream0)
        buf163 = buf152; del buf152  # reuse
        buf164 = buf151; del buf151  # reuse
        buf165 = buf150; del buf150  # reuse
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf160, buf161, buf162, buf163, buf164, buf165, 48, 98, grid=grid(48), stream=stream0)
        buf166 = buf154; del buf154  # reuse
        buf167 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf169 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf163, buf164, buf165, primals_235, primals_236, buf166, buf167, buf169, primals_235, primals_236, 24, 2, grid=grid(24), stream=stream0)
        del buf163
        del buf164
        del buf165
        del primals_235
        del primals_236
        buf170 = reinterpret_tensor(buf158, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf158  # reuse
        # Source Nodes: [shortcut_4, x_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_24.run(buf159, buf166, buf167, primals_25, primals_26, buf131, buf170, 602112, grid=grid(602112), stream=stream0)
        del buf167
        del primals_26
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 144, 56, 56), (451584, 3136, 56, 1))
        buf172 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf171, buf172, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        buf173 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        buf175 = empty_strided((1, 144, 1, 1, 196), (28224, 1, 28224, 28224, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf172, buf173, buf174, buf175, 28224, 128, grid=grid(28224), stream=stream0)
        buf176 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf177 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((1, 144, 1, 1, 2), (288, 1, 288, 288, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf173, buf174, buf175, buf176, buf177, buf178, 288, 98, grid=grid(288), stream=stream0)
        del buf173
        del buf174
        del buf175
        buf179 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf182 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_28.run(buf176, buf177, buf178, primals_238, primals_239, buf179, buf180, buf182, primals_238, primals_239, 144, 2, grid=grid(144), stream=stream0)
        del buf176
        del buf177
        del buf178
        del primals_238
        del primals_239
        buf183 = reinterpret_tensor(buf171, (8, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf171  # reuse
        # Source Nodes: [x_73, x_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_29.run(buf172, buf179, buf180, primals_27, primals_28, buf183, 3612672, grid=grid(3612672), stream=stream0)
        del primals_28
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_145, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf184, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf185 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf184, buf185, 1152, 784, grid=grid(1152, 784), stream=stream0)
        buf186 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        buf187 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((1, 144, 1, 1, 49), (7056, 1, 7056, 7056, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf185, buf186, buf187, buf188, 7056, 128, grid=grid(7056), stream=stream0)
        buf189 = buf180; del buf180  # reuse
        buf190 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf192 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_32.run(buf186, buf187, buf188, primals_241, primals_242, buf189, buf190, buf192, primals_241, primals_242, 144, 49, grid=grid(144), stream=stream0)
        del buf186
        del buf187
        del buf188
        del primals_241
        del primals_242
        buf193 = reinterpret_tensor(buf184, (8, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf184  # reuse
        # Source Nodes: [x_78, x_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_33.run(buf185, buf189, buf190, primals_29, primals_30, buf193, 903168, grid=grid(903168), stream=stream0)
        del buf190
        del primals_30
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf195 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf194, buf195, 256, 784, grid=grid(256, 784), stream=stream0)
        buf196 = empty_strided((1, 32, 1, 1, 49), (1568, 1, 1568, 1568, 32), device='cuda', dtype=torch.float32)
        buf197 = empty_strided((1, 32, 1, 1, 49), (1568, 1, 1568, 1568, 32), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1, 32, 1, 1, 49), (1568, 1, 1568, 1568, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf195, buf196, buf197, buf198, 1568, 128, grid=grid(1568), stream=stream0)
        buf199 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf200 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf202 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf196, buf197, buf198, primals_244, primals_245, buf199, buf200, buf202, primals_244, primals_245, 32, 49, grid=grid(32), stream=stream0)
        del primals_244
        del primals_245
        buf203 = reinterpret_tensor(buf194, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf194  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_37.run(buf195, buf199, buf200, primals_31, primals_32, buf203, 200704, grid=grid(200704), stream=stream0)
        del primals_32
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 96, 28, 28), (75264, 784, 28, 1))
        buf205 = empty_strided((8, 96, 28, 28), (75264, 1, 2688, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf204, buf205, 768, 784, grid=grid(768, 784), stream=stream0)
        buf206 = reinterpret_tensor(buf162, (1, 96, 1, 1, 49), (4704, 1, 4704, 4704, 96), 0); del buf162  # reuse
        buf207 = reinterpret_tensor(buf161, (1, 96, 1, 1, 49), (4704, 1, 4704, 4704, 96), 0); del buf161  # reuse
        buf208 = reinterpret_tensor(buf160, (1, 96, 1, 1, 49), (4704, 1, 4704, 4704, 96), 0); del buf160  # reuse
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf205, buf206, buf207, buf208, 4704, 128, grid=grid(4704), stream=stream0)
        buf209 = buf76; del buf76  # reuse
        buf210 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf212 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_40.run(buf206, buf207, buf208, primals_247, primals_248, buf209, buf210, buf212, primals_247, primals_248, 96, 49, grid=grid(96), stream=stream0)
        del primals_247
        del primals_248
        buf213 = reinterpret_tensor(buf204, (8, 96, 28, 28), (75264, 1, 2688, 96), 0); del buf204  # reuse
        # Source Nodes: [x_89, x_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_41.run(buf205, buf209, buf210, primals_33, primals_34, buf213, 602112, grid=grid(602112), stream=stream0)
        del primals_34
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_148, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf214, (8, 96, 28, 28), (75264, 784, 28, 1))
        buf215 = empty_strided((8, 96, 28, 28), (75264, 1, 2688, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf214, buf215, 768, 784, grid=grid(768, 784), stream=stream0)
        buf216 = buf208; del buf208  # reuse
        buf217 = buf207; del buf207  # reuse
        buf218 = buf206; del buf206  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_39.run(buf215, buf216, buf217, buf218, 4704, 128, grid=grid(4704), stream=stream0)
        buf219 = buf210; del buf210  # reuse
        buf220 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf222 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_40.run(buf216, buf217, buf218, primals_250, primals_251, buf219, buf220, buf222, primals_250, primals_251, 96, 49, grid=grid(96), stream=stream0)
        del buf216
        del buf217
        del buf218
        del primals_250
        del primals_251
        buf223 = reinterpret_tensor(buf214, (8, 96, 28, 28), (75264, 1, 2688, 96), 0); del buf214  # reuse
        # Source Nodes: [x_94, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_41.run(buf215, buf219, buf220, primals_35, primals_36, buf223, 602112, grid=grid(602112), stream=stream0)
        del buf220
        del primals_36
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf225 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf224, buf225, 256, 784, grid=grid(256, 784), stream=stream0)
        buf226 = buf198; del buf198  # reuse
        buf227 = buf197; del buf197  # reuse
        buf228 = buf196; del buf196  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf225, buf226, buf227, buf228, 1568, 128, grid=grid(1568), stream=stream0)
        buf229 = buf200; del buf200  # reuse
        buf230 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf232 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf226, buf227, buf228, primals_253, primals_254, buf229, buf230, buf232, primals_253, primals_254, 32, 49, grid=grid(32), stream=stream0)
        del primals_253
        del primals_254
        buf233 = reinterpret_tensor(buf224, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf224  # reuse
        # Source Nodes: [shortcut_6, x_100], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_42.run(buf225, buf229, buf230, primals_37, primals_38, buf203, buf233, 200704, grid=grid(200704), stream=stream0)
        del primals_38
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf235 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf234, buf235, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf236 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        buf237 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((1, 192, 1, 1, 49), (9408, 1, 9408, 9408, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf235, buf236, buf237, buf238, 9408, 128, grid=grid(9408), stream=stream0)
        buf239 = reinterpret_tensor(buf74, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf74  # reuse
        buf240 = reinterpret_tensor(buf73, (1, 192, 1, 1), (192, 1, 192, 192), 0); del buf73  # reuse
        buf242 = reinterpret_tensor(buf72, (192, ), (1, ), 0); del buf72  # reuse
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf236, buf237, buf238, primals_256, primals_257, buf239, buf240, buf242, primals_256, primals_257, 192, 49, grid=grid(192), stream=stream0)
        del primals_256
        del primals_257
        buf243 = reinterpret_tensor(buf234, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf234  # reuse
        # Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf235, buf239, buf240, primals_39, primals_40, buf243, 1204224, grid=grid(1204224), stream=stream0)
        del primals_40
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_151, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf244, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf245 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf244, buf245, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf246 = buf238; del buf238  # reuse
        buf247 = buf237; del buf237  # reuse
        buf248 = buf236; del buf236  # reuse
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf245, buf246, buf247, buf248, 9408, 128, grid=grid(9408), stream=stream0)
        buf249 = buf240; del buf240  # reuse
        buf250 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf252 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf246, buf247, buf248, primals_259, primals_260, buf249, buf250, buf252, primals_259, primals_260, 192, 49, grid=grid(192), stream=stream0)
        del primals_259
        del primals_260
        buf253 = reinterpret_tensor(buf244, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf244  # reuse
        # Source Nodes: [x_111, x_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf245, buf249, buf250, primals_41, primals_42, buf253, 1204224, grid=grid(1204224), stream=stream0)
        del primals_42
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf255 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf254, buf255, 256, 784, grid=grid(256, 784), stream=stream0)
        buf256 = buf228; del buf228  # reuse
        buf257 = buf227; del buf227  # reuse
        buf258 = buf226; del buf226  # reuse
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf255, buf256, buf257, buf258, 1568, 128, grid=grid(1568), stream=stream0)
        buf259 = buf230; del buf230  # reuse
        buf260 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf262 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf256, buf257, buf258, primals_262, primals_263, buf259, buf260, buf262, primals_262, primals_263, 32, 49, grid=grid(32), stream=stream0)
        del primals_262
        del primals_263
        buf263 = reinterpret_tensor(buf254, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf254  # reuse
        # Source Nodes: [shortcut_7, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_42.run(buf255, buf259, buf260, primals_43, primals_44, buf233, buf263, 200704, grid=grid(200704), stream=stream0)
        del primals_44
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf265 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf264, buf265, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf266 = buf248; del buf248  # reuse
        buf267 = buf247; del buf247  # reuse
        buf268 = buf246; del buf246  # reuse
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf265, buf266, buf267, buf268, 9408, 128, grid=grid(9408), stream=stream0)
        buf269 = buf250; del buf250  # reuse
        buf270 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf272 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf266, buf267, buf268, primals_265, primals_266, buf269, buf270, buf272, primals_265, primals_266, 192, 49, grid=grid(192), stream=stream0)
        del primals_265
        del primals_266
        buf273 = reinterpret_tensor(buf264, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf264  # reuse
        # Source Nodes: [x_123, x_126], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf265, buf269, buf270, primals_45, primals_46, buf273, 1204224, grid=grid(1204224), stream=stream0)
        del primals_46
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf274, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf275 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf274, buf275, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf276 = buf268; del buf268  # reuse
        buf277 = buf267; del buf267  # reuse
        buf278 = buf266; del buf266  # reuse
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf275, buf276, buf277, buf278, 9408, 128, grid=grid(9408), stream=stream0)
        buf279 = buf270; del buf270  # reuse
        buf280 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf282 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf276, buf277, buf278, primals_268, primals_269, buf279, buf280, buf282, primals_268, primals_269, 192, 49, grid=grid(192), stream=stream0)
        del primals_268
        del primals_269
        buf283 = reinterpret_tensor(buf274, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf274  # reuse
        # Source Nodes: [x_128, x_131], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf275, buf279, buf280, primals_47, primals_48, buf283, 1204224, grid=grid(1204224), stream=stream0)
        del primals_48
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (8, 32, 28, 28), (25088, 784, 28, 1))
        buf285 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf284, buf285, 256, 784, grid=grid(256, 784), stream=stream0)
        buf286 = buf258; del buf258  # reuse
        buf287 = buf257; del buf257  # reuse
        buf288 = buf256; del buf256  # reuse
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf285, buf286, buf287, buf288, 1568, 128, grid=grid(1568), stream=stream0)
        buf289 = buf260; del buf260  # reuse
        buf290 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf292 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_36.run(buf286, buf287, buf288, primals_271, primals_272, buf289, buf290, buf292, primals_271, primals_272, 32, 49, grid=grid(32), stream=stream0)
        del buf286
        del buf287
        del buf288
        del primals_271
        del primals_272
        buf293 = reinterpret_tensor(buf284, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf284  # reuse
        # Source Nodes: [shortcut_8, x_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_42.run(buf285, buf289, buf290, primals_49, primals_50, buf263, buf293, 200704, grid=grid(200704), stream=stream0)
        del buf290
        del primals_50
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf295 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(buf294, buf295, 1536, 784, grid=grid(1536, 784), stream=stream0)
        buf296 = buf278; del buf278  # reuse
        buf297 = buf277; del buf277  # reuse
        buf298 = buf276; del buf276  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf295, buf296, buf297, buf298, 9408, 128, grid=grid(9408), stream=stream0)
        buf299 = buf280; del buf280  # reuse
        buf300 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf302 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf296, buf297, buf298, primals_274, primals_275, buf299, buf300, buf302, primals_274, primals_275, 192, 49, grid=grid(192), stream=stream0)
        del buf296
        del buf297
        del buf298
        del primals_274
        del primals_275
        buf303 = reinterpret_tensor(buf294, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf294  # reuse
        # Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf295, buf299, buf300, primals_51, primals_52, buf303, 1204224, grid=grid(1204224), stream=stream0)
        del primals_52
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_157, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf304, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf305 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf304, buf305, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf306 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        buf308 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf305, buf306, buf307, buf308, 2496, 121, grid=grid(2496), stream=stream0)
        buf309 = buf300; del buf300  # reuse
        buf310 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf312 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf306, buf307, buf308, primals_277, primals_278, buf309, buf310, buf312, primals_277, primals_278, 192, 13, grid=grid(192), stream=stream0)
        del primals_277
        del primals_278
        buf313 = reinterpret_tensor(buf304, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf304  # reuse
        # Source Nodes: [x_145, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_50.run(buf305, buf309, buf310, primals_53, primals_54, buf313, 301056, grid=grid(301056), stream=stream0)
        del primals_54
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf315 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf314, buf315, 512, 196, grid=grid(512, 196), stream=stream0)
        buf316 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        buf317 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        buf318 = empty_strided((1, 64, 1, 1, 13), (832, 1, 832, 832, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_52.run(buf315, buf316, buf317, buf318, 832, 121, grid=grid(832), stream=stream0)
        buf319 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf320 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf322 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_53.run(buf316, buf317, buf318, primals_280, primals_281, buf319, buf320, buf322, primals_280, primals_281, 64, 13, grid=grid(64), stream=stream0)
        del primals_280
        del primals_281
        buf323 = reinterpret_tensor(buf314, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf314  # reuse
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_54.run(buf315, buf319, buf320, primals_55, primals_56, buf323, 100352, grid=grid(100352), stream=stream0)
        del primals_56
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf325 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf324, buf325, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf326 = buf308; del buf308  # reuse
        buf327 = buf307; del buf307  # reuse
        buf328 = buf306; del buf306  # reuse
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf325, buf326, buf327, buf328, 2496, 121, grid=grid(2496), stream=stream0)
        buf329 = buf310; del buf310  # reuse
        buf330 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf332 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf326, buf327, buf328, primals_283, primals_284, buf329, buf330, buf332, primals_283, primals_284, 192, 13, grid=grid(192), stream=stream0)
        del primals_283
        del primals_284
        buf333 = reinterpret_tensor(buf324, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf324  # reuse
        # Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_50.run(buf325, buf329, buf330, primals_57, primals_58, buf333, 301056, grid=grid(301056), stream=stream0)
        del primals_58
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_160, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf334, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf335 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_47.run(buf334, buf335, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf336 = buf328; del buf328  # reuse
        buf337 = buf327; del buf327  # reuse
        buf338 = buf326; del buf326  # reuse
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf335, buf336, buf337, buf338, 2496, 121, grid=grid(2496), stream=stream0)
        buf339 = buf330; del buf330  # reuse
        buf340 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf342 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_49.run(buf336, buf337, buf338, primals_286, primals_287, buf339, buf340, buf342, primals_286, primals_287, 192, 13, grid=grid(192), stream=stream0)
        del buf336
        del buf337
        del buf338
        del primals_286
        del primals_287
        buf343 = reinterpret_tensor(buf334, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf334  # reuse
        # Source Nodes: [x_161, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_50.run(buf335, buf339, buf340, primals_59, primals_60, buf343, 301056, grid=grid(301056), stream=stream0)
        del buf340
        del primals_60
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf345 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf344, buf345, 512, 196, grid=grid(512, 196), stream=stream0)
        buf346 = buf318; del buf318  # reuse
        buf347 = buf317; del buf317  # reuse
        buf348 = buf316; del buf316  # reuse
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_52.run(buf345, buf346, buf347, buf348, 832, 121, grid=grid(832), stream=stream0)
        buf349 = buf320; del buf320  # reuse
        buf350 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf352 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_53.run(buf346, buf347, buf348, primals_289, primals_290, buf349, buf350, buf352, primals_289, primals_290, 64, 13, grid=grid(64), stream=stream0)
        del primals_289
        del primals_290
        buf353 = reinterpret_tensor(buf344, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf344  # reuse
        # Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_55.run(buf345, buf349, buf350, primals_61, primals_62, buf323, buf353, 100352, grid=grid(100352), stream=stream0)
        del primals_62
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf355 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf354, buf355, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf356 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        buf357 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        buf358 = empty_strided((1, 384, 1, 1, 13), (4992, 1, 4992, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf355, buf356, buf357, buf358, 4992, 121, grid=grid(4992), stream=stream0)
        buf359 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf360 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf362 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf356, buf357, buf358, primals_292, primals_293, buf359, buf360, buf362, primals_292, primals_293, 384, 13, grid=grid(384), stream=stream0)
        del primals_292
        del primals_293
        buf363 = reinterpret_tensor(buf354, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf354  # reuse
        # Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf355, buf359, buf360, primals_63, primals_64, buf363, 602112, grid=grid(602112), stream=stream0)
        del primals_64
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_163, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf364, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf365 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf364, buf365, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf366 = buf358; del buf358  # reuse
        buf367 = buf357; del buf357  # reuse
        buf368 = buf356; del buf356  # reuse
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf365, buf366, buf367, buf368, 4992, 121, grid=grid(4992), stream=stream0)
        buf369 = buf360; del buf360  # reuse
        buf370 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf372 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf366, buf367, buf368, primals_295, primals_296, buf369, buf370, buf372, primals_295, primals_296, 384, 13, grid=grid(384), stream=stream0)
        del primals_295
        del primals_296
        buf373 = reinterpret_tensor(buf364, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf364  # reuse
        # Source Nodes: [x_178, x_181], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf365, buf369, buf370, primals_65, primals_66, buf373, 602112, grid=grid(602112), stream=stream0)
        del primals_66
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf375 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf374, buf375, 512, 196, grid=grid(512, 196), stream=stream0)
        buf376 = buf348; del buf348  # reuse
        buf377 = buf347; del buf347  # reuse
        buf378 = buf346; del buf346  # reuse
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_52.run(buf375, buf376, buf377, buf378, 832, 121, grid=grid(832), stream=stream0)
        buf379 = buf350; del buf350  # reuse
        buf380 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf382 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_53.run(buf376, buf377, buf378, primals_298, primals_299, buf379, buf380, buf382, primals_298, primals_299, 64, 13, grid=grid(64), stream=stream0)
        del primals_298
        del primals_299
        buf383 = reinterpret_tensor(buf374, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf374  # reuse
        # Source Nodes: [shortcut_11, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_55.run(buf375, buf379, buf380, primals_67, primals_68, buf353, buf383, 100352, grid=grid(100352), stream=stream0)
        del primals_68
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf383, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf385 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf384, buf385, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf386 = buf368; del buf368  # reuse
        buf387 = buf367; del buf367  # reuse
        buf388 = buf366; del buf366  # reuse
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf385, buf386, buf387, buf388, 4992, 121, grid=grid(4992), stream=stream0)
        buf389 = buf370; del buf370  # reuse
        buf390 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf392 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf386, buf387, buf388, primals_301, primals_302, buf389, buf390, buf392, primals_301, primals_302, 384, 13, grid=grid(384), stream=stream0)
        del primals_301
        del primals_302
        buf393 = reinterpret_tensor(buf384, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf384  # reuse
        # Source Nodes: [x_190, x_193], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf385, buf389, buf390, primals_69, primals_70, buf393, 602112, grid=grid(602112), stream=stream0)
        del primals_70
        # Source Nodes: [x_194], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_166, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf394, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf395 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf394, buf395, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf396 = buf388; del buf388  # reuse
        buf397 = buf387; del buf387  # reuse
        buf398 = buf386; del buf386  # reuse
        # Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf395, buf396, buf397, buf398, 4992, 121, grid=grid(4992), stream=stream0)
        buf399 = buf390; del buf390  # reuse
        buf400 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf402 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf396, buf397, buf398, primals_304, primals_305, buf399, buf400, buf402, primals_304, primals_305, 384, 13, grid=grid(384), stream=stream0)
        del primals_304
        del primals_305
        buf403 = reinterpret_tensor(buf394, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf394  # reuse
        # Source Nodes: [x_195, x_198], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf395, buf399, buf400, primals_71, primals_72, buf403, 602112, grid=grid(602112), stream=stream0)
        del primals_72
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 64, 14, 14), (12544, 196, 14, 1))
        buf405 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf404, buf405, 512, 196, grid=grid(512, 196), stream=stream0)
        buf406 = buf378; del buf378  # reuse
        buf407 = buf377; del buf377  # reuse
        buf408 = buf376; del buf376  # reuse
        # Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_52.run(buf405, buf406, buf407, buf408, 832, 121, grid=grid(832), stream=stream0)
        buf409 = buf380; del buf380  # reuse
        buf410 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf412 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_53.run(buf406, buf407, buf408, primals_307, primals_308, buf409, buf410, buf412, primals_307, primals_308, 64, 13, grid=grid(64), stream=stream0)
        del buf406
        del buf407
        del buf408
        del primals_307
        del primals_308
        buf413 = reinterpret_tensor(buf404, (8, 64, 14, 14), (12544, 1, 896, 64), 0); del buf404  # reuse
        # Source Nodes: [shortcut_12, x_201], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_55.run(buf405, buf409, buf410, primals_73, primals_74, buf383, buf413, 100352, grid=grid(100352), stream=stream0)
        del buf410
        del primals_74
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf415 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf414, buf415, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf416 = buf398; del buf398  # reuse
        buf417 = buf397; del buf397  # reuse
        buf418 = buf396; del buf396  # reuse
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf415, buf416, buf417, buf418, 4992, 121, grid=grid(4992), stream=stream0)
        buf419 = buf400; del buf400  # reuse
        buf420 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf422 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf416, buf417, buf418, primals_310, primals_311, buf419, buf420, buf422, primals_310, primals_311, 384, 13, grid=grid(384), stream=stream0)
        del primals_310
        del primals_311
        buf423 = reinterpret_tensor(buf414, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf414  # reuse
        # Source Nodes: [x_207, x_210], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf415, buf419, buf420, primals_75, primals_76, buf423, 602112, grid=grid(602112), stream=stream0)
        del primals_76
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_169, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf424, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf425 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_56.run(buf424, buf425, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf426 = buf418; del buf418  # reuse
        buf427 = buf417; del buf417  # reuse
        buf428 = buf416; del buf416  # reuse
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_57.run(buf425, buf426, buf427, buf428, 4992, 121, grid=grid(4992), stream=stream0)
        buf429 = buf420; del buf420  # reuse
        buf430 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf432 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_58.run(buf426, buf427, buf428, primals_313, primals_314, buf429, buf430, buf432, primals_313, primals_314, 384, 13, grid=grid(384), stream=stream0)
        del buf426
        del buf427
        del buf428
        del primals_313
        del primals_314
        buf433 = reinterpret_tensor(buf424, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf424  # reuse
        # Source Nodes: [x_212, x_215], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf425, buf429, buf430, primals_77, primals_78, buf433, 602112, grid=grid(602112), stream=stream0)
        del buf430
        del primals_78
        # Source Nodes: [x_217], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf435 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf434, buf435, 896, 196, grid=grid(896, 196), stream=stream0)
        buf436 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf437 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        buf438 = empty_strided((1, 112, 1, 1, 13), (1456, 1, 1456, 1456, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf435, buf436, buf437, buf438, 1456, 121, grid=grid(1456), stream=stream0)
        buf439 = reinterpret_tensor(buf48, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf48  # reuse
        buf440 = reinterpret_tensor(buf47, (1, 112, 1, 1), (112, 1, 112, 112), 0); del buf47  # reuse
        buf442 = reinterpret_tensor(buf46, (112, ), (1, ), 0); del buf46  # reuse
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf436, buf437, buf438, primals_316, primals_317, buf439, buf440, buf442, primals_316, primals_317, 112, 13, grid=grid(112), stream=stream0)
        del primals_316
        del primals_317
        buf443 = reinterpret_tensor(buf434, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf434  # reuse
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_63.run(buf435, buf439, buf440, primals_79, primals_80, buf443, 175616, grid=grid(175616), stream=stream0)
        del primals_80
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(buf443, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf445 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf444, buf445, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf446 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        buf447 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        buf448 = empty_strided((1, 672, 1, 1, 13), (8736, 1, 8736, 8736, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf445, buf446, buf447, buf448, 8736, 121, grid=grid(8736), stream=stream0)
        buf449 = reinterpret_tensor(buf61, (1, 672, 1, 1), (672, 1, 672, 672), 0); del buf61  # reuse
        buf450 = reinterpret_tensor(buf60, (1, 672, 1, 1), (672, 1, 672, 672), 0); del buf60  # reuse
        buf452 = reinterpret_tensor(buf59, (672, ), (1, ), 0); del buf59  # reuse
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf446, buf447, buf448, primals_319, primals_320, buf449, buf450, buf452, primals_319, primals_320, 672, 13, grid=grid(672), stream=stream0)
        del primals_319
        del primals_320
        buf453 = reinterpret_tensor(buf444, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf444  # reuse
        # Source Nodes: [x_223, x_226], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_67.run(buf445, buf449, buf450, primals_81, primals_82, buf453, 1053696, grid=grid(1053696), stream=stream0)
        del primals_82
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, primals_172, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf454, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf455 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf454, buf455, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf456 = buf448; del buf448  # reuse
        buf457 = buf447; del buf447  # reuse
        buf458 = buf446; del buf446  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf455, buf456, buf457, buf458, 8736, 121, grid=grid(8736), stream=stream0)
        buf459 = buf450; del buf450  # reuse
        buf460 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf462 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf456, buf457, buf458, primals_322, primals_323, buf459, buf460, buf462, primals_322, primals_323, 672, 13, grid=grid(672), stream=stream0)
        del primals_322
        del primals_323
        buf463 = reinterpret_tensor(buf454, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf454  # reuse
        # Source Nodes: [x_228, x_231], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_67.run(buf455, buf459, buf460, primals_83, primals_84, buf463, 1053696, grid=grid(1053696), stream=stream0)
        del primals_84
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf465 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf464, buf465, 896, 196, grid=grid(896, 196), stream=stream0)
        buf466 = buf438; del buf438  # reuse
        buf467 = buf437; del buf437  # reuse
        buf468 = buf436; del buf436  # reuse
        # Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf465, buf466, buf467, buf468, 1456, 121, grid=grid(1456), stream=stream0)
        buf469 = buf440; del buf440  # reuse
        buf470 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf472 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf466, buf467, buf468, primals_325, primals_326, buf469, buf470, buf472, primals_325, primals_326, 112, 13, grid=grid(112), stream=stream0)
        del primals_325
        del primals_326
        buf473 = reinterpret_tensor(buf464, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf464  # reuse
        # Source Nodes: [shortcut_14, x_234], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_68.run(buf465, buf469, buf470, primals_85, primals_86, buf443, buf473, 175616, grid=grid(175616), stream=stream0)
        del primals_86
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf475 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf474, buf475, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf476 = buf458; del buf458  # reuse
        buf477 = buf457; del buf457  # reuse
        buf478 = buf456; del buf456  # reuse
        # Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf475, buf476, buf477, buf478, 8736, 121, grid=grid(8736), stream=stream0)
        buf479 = buf460; del buf460  # reuse
        buf480 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf482 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_240], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf476, buf477, buf478, primals_328, primals_329, buf479, buf480, buf482, primals_328, primals_329, 672, 13, grid=grid(672), stream=stream0)
        del primals_328
        del primals_329
        buf483 = reinterpret_tensor(buf474, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf474  # reuse
        # Source Nodes: [x_240, x_243], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_67.run(buf475, buf479, buf480, primals_87, primals_88, buf483, 1053696, grid=grid(1053696), stream=stream0)
        del primals_88
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, primals_175, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf484, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf485 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf484, buf485, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf486 = buf478; del buf478  # reuse
        buf487 = buf477; del buf477  # reuse
        buf488 = buf476; del buf476  # reuse
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf485, buf486, buf487, buf488, 8736, 121, grid=grid(8736), stream=stream0)
        buf489 = buf480; del buf480  # reuse
        buf490 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf492 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf486, buf487, buf488, primals_331, primals_332, buf489, buf490, buf492, primals_331, primals_332, 672, 13, grid=grid(672), stream=stream0)
        del primals_331
        del primals_332
        buf493 = reinterpret_tensor(buf484, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf484  # reuse
        # Source Nodes: [x_245, x_248], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_67.run(buf485, buf489, buf490, primals_89, primals_90, buf493, 1053696, grid=grid(1053696), stream=stream0)
        del primals_90
        # Source Nodes: [x_250], Original ATen: [aten.convolution]
        buf494 = extern_kernels.convolution(buf493, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf494, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf495 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf494, buf495, 896, 196, grid=grid(896, 196), stream=stream0)
        buf496 = buf468; del buf468  # reuse
        buf497 = buf467; del buf467  # reuse
        buf498 = buf466; del buf466  # reuse
        # Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf495, buf496, buf497, buf498, 1456, 121, grid=grid(1456), stream=stream0)
        buf499 = buf470; del buf470  # reuse
        buf500 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf502 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf496, buf497, buf498, primals_334, primals_335, buf499, buf500, buf502, primals_334, primals_335, 112, 13, grid=grid(112), stream=stream0)
        del primals_334
        del primals_335
        buf503 = reinterpret_tensor(buf494, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf494  # reuse
        # Source Nodes: [shortcut_15, x_251], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_68.run(buf495, buf499, buf500, primals_91, primals_92, buf473, buf503, 175616, grid=grid(175616), stream=stream0)
        del primals_92
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf504 = extern_kernels.convolution(buf503, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf504, (8, 336, 14, 14), (65856, 196, 14, 1))
        buf505 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf504, buf505, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf506 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf507 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        buf508 = empty_strided((1, 336, 1, 1, 13), (4368, 1, 4368, 4368, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf505, buf506, buf507, buf508, 4368, 121, grid=grid(4368), stream=stream0)
        buf509 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf510 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf512 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf506, buf507, buf508, primals_337, primals_338, buf509, buf510, buf512, primals_337, primals_338, 336, 13, grid=grid(336), stream=stream0)
        del primals_337
        del primals_338
        buf513 = reinterpret_tensor(buf504, (8, 336, 14, 14), (65856, 1, 4704, 336), 0); del buf504  # reuse
        # Source Nodes: [x_257, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_72.run(buf505, buf509, buf510, primals_93, primals_94, buf513, 526848, grid=grid(526848), stream=stream0)
        del primals_94
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf513, primals_178, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf514, (8, 336, 14, 14), (65856, 196, 14, 1))
        buf515 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_69.run(buf514, buf515, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf516 = buf508; del buf508  # reuse
        buf517 = buf507; del buf507  # reuse
        buf518 = buf506; del buf506  # reuse
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf515, buf516, buf517, buf518, 4368, 121, grid=grid(4368), stream=stream0)
        buf519 = buf510; del buf510  # reuse
        buf520 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf522 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf516, buf517, buf518, primals_340, primals_341, buf519, buf520, buf522, primals_340, primals_341, 336, 13, grid=grid(336), stream=stream0)
        del buf516
        del buf517
        del buf518
        del primals_340
        del primals_341
        buf523 = reinterpret_tensor(buf514, (8, 336, 14, 14), (65856, 1, 4704, 336), 0); del buf514  # reuse
        # Source Nodes: [x_262, x_265], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_72.run(buf515, buf519, buf520, primals_95, primals_96, buf523, 526848, grid=grid(526848), stream=stream0)
        del buf520
        del primals_96
        # Source Nodes: [x_267], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (8, 112, 14, 14), (21952, 196, 14, 1))
        buf525 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_60.run(buf524, buf525, 896, 196, grid=grid(896, 196), stream=stream0)
        buf526 = buf498; del buf498  # reuse
        buf527 = buf497; del buf497  # reuse
        buf528 = buf496; del buf496  # reuse
        # Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf525, buf526, buf527, buf528, 1456, 121, grid=grid(1456), stream=stream0)
        buf529 = buf500; del buf500  # reuse
        buf530 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cuda', dtype=torch.float32)
        buf532 = empty((112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_268], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf526, buf527, buf528, primals_343, primals_344, buf529, buf530, buf532, primals_343, primals_344, 112, 13, grid=grid(112), stream=stream0)
        del buf526
        del buf527
        del buf528
        del primals_343
        del primals_344
        buf533 = reinterpret_tensor(buf524, (8, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf524  # reuse
        # Source Nodes: [shortcut_16, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_68.run(buf525, buf529, buf530, primals_97, primals_98, buf503, buf533, 175616, grid=grid(175616), stream=stream0)
        del buf530
        del primals_98
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 672, 14, 14), (131712, 196, 14, 1))
        buf535 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf534, buf535, 5376, 196, grid=grid(5376, 196), stream=stream0)
        buf536 = buf488; del buf488  # reuse
        buf537 = buf487; del buf487  # reuse
        buf538 = buf486; del buf486  # reuse
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_65.run(buf535, buf536, buf537, buf538, 8736, 121, grid=grid(8736), stream=stream0)
        buf539 = buf490; del buf490  # reuse
        buf540 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf542 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_66.run(buf536, buf537, buf538, primals_346, primals_347, buf539, buf540, buf542, primals_346, primals_347, 672, 13, grid=grid(672), stream=stream0)
        del buf536
        del buf537
        del buf538
        del primals_346
        del primals_347
        buf543 = reinterpret_tensor(buf534, (8, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf534  # reuse
        # Source Nodes: [x_274, x_277], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_67.run(buf535, buf539, buf540, primals_99, primals_100, buf543, 1053696, grid=grid(1053696), stream=stream0)
        del primals_100
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_181, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf544, (8, 672, 7, 7), (32928, 49, 7, 1))
        buf545 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_73.run(buf544, buf545, 5376, 49, grid=grid(5376, 49), stream=stream0)
        buf546 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf547 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        buf548 = empty_strided((1, 672, 1, 1, 4), (2688, 1, 2688, 2688, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_74.run(buf545, buf546, buf547, buf548, 2688, 98, grid=grid(2688), stream=stream0)
        buf549 = buf540; del buf540  # reuse
        buf550 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cuda', dtype=torch.float32)
        buf552 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_75.run(buf546, buf547, buf548, primals_349, primals_350, buf549, buf550, buf552, primals_349, primals_350, 672, 4, grid=grid(672), stream=stream0)
        del buf546
        del buf547
        del buf548
        del primals_349
        del primals_350
        buf553 = reinterpret_tensor(buf544, (8, 672, 7, 7), (32928, 1, 4704, 672), 0); del buf544  # reuse
        # Source Nodes: [x_279, x_282], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_76.run(buf545, buf549, buf550, primals_101, primals_102, buf553, 263424, grid=grid(263424), stream=stream0)
        del buf550
        del primals_102
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf555 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf554, buf555, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf556 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf557 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        buf558 = empty_strided((1, 184, 1, 1, 4), (736, 1, 736, 736, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf555, buf556, buf557, buf558, 736, 98, grid=grid(736), stream=stream0)
        buf559 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf560 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf562 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_79.run(buf556, buf557, buf558, primals_352, primals_353, buf559, buf560, buf562, primals_352, primals_353, 184, 4, grid=grid(184), stream=stream0)
        del primals_352
        del primals_353
        buf563 = reinterpret_tensor(buf554, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf554  # reuse
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_80.run(buf555, buf559, buf560, primals_103, primals_104, buf563, 72128, grid=grid(72128), stream=stream0)
        del primals_104
        # Source Nodes: [x_289], Original ATen: [aten.convolution]
        buf564 = extern_kernels.convolution(buf563, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf565 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf564, buf565, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf566 = empty_strided((1, 1104, 1, 1, 4), (4416, 1, 4416, 4416, 1104), device='cuda', dtype=torch.float32)
        buf567 = empty_strided((1, 1104, 1, 1, 4), (4416, 1, 4416, 4416, 1104), device='cuda', dtype=torch.float32)
        buf568 = empty_strided((1, 1104, 1, 1, 4), (4416, 1, 4416, 4416, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf565, buf566, buf567, buf568, 4416, 98, grid=grid(4416), stream=stream0)
        buf569 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf570 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf572 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf566, buf567, buf568, primals_355, primals_356, buf569, buf570, buf572, primals_355, primals_356, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_355
        del primals_356
        buf573 = reinterpret_tensor(buf564, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf564  # reuse
        # Source Nodes: [x_290, x_293], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf565, buf569, buf570, primals_105, primals_106, buf573, 432768, grid=grid(432768), stream=stream0)
        del primals_106
        # Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_184, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf574, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf575 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf574, buf575, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf576 = buf568; del buf568  # reuse
        buf577 = buf567; del buf567  # reuse
        buf578 = buf566; del buf566  # reuse
        # Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf575, buf576, buf577, buf578, 4416, 98, grid=grid(4416), stream=stream0)
        buf579 = buf570; del buf570  # reuse
        buf580 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf582 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf576, buf577, buf578, primals_358, primals_359, buf579, buf580, buf582, primals_358, primals_359, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_358
        del primals_359
        buf583 = reinterpret_tensor(buf574, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf574  # reuse
        # Source Nodes: [x_295, x_298], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf575, buf579, buf580, primals_107, primals_108, buf583, 432768, grid=grid(432768), stream=stream0)
        del primals_108
        # Source Nodes: [x_300], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf585 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf584, buf585, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf586 = buf558; del buf558  # reuse
        buf587 = buf557; del buf557  # reuse
        buf588 = buf556; del buf556  # reuse
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf585, buf586, buf587, buf588, 736, 98, grid=grid(736), stream=stream0)
        buf589 = buf560; del buf560  # reuse
        buf590 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf592 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_79.run(buf586, buf587, buf588, primals_361, primals_362, buf589, buf590, buf592, primals_361, primals_362, 184, 4, grid=grid(184), stream=stream0)
        del primals_361
        del primals_362
        buf593 = reinterpret_tensor(buf584, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf584  # reuse
        # Source Nodes: [shortcut_18, x_301], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_85.run(buf585, buf589, buf590, primals_109, primals_110, buf563, buf593, 72128, grid=grid(72128), stream=stream0)
        del primals_110
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf594 = extern_kernels.convolution(buf593, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf594, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf595 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf594, buf595, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf596 = buf578; del buf578  # reuse
        buf597 = buf577; del buf577  # reuse
        buf598 = buf576; del buf576  # reuse
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf595, buf596, buf597, buf598, 4416, 98, grid=grid(4416), stream=stream0)
        buf599 = buf580; del buf580  # reuse
        buf600 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf602 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf596, buf597, buf598, primals_364, primals_365, buf599, buf600, buf602, primals_364, primals_365, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_364
        del primals_365
        buf603 = reinterpret_tensor(buf594, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf594  # reuse
        # Source Nodes: [x_307, x_310], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf595, buf599, buf600, primals_111, primals_112, buf603, 432768, grid=grid(432768), stream=stream0)
        del primals_112
        # Source Nodes: [x_311], Original ATen: [aten.convolution]
        buf604 = extern_kernels.convolution(buf603, primals_187, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf604, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf605 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_311], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf604, buf605, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf606 = buf598; del buf598  # reuse
        buf607 = buf597; del buf597  # reuse
        buf608 = buf596; del buf596  # reuse
        # Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf605, buf606, buf607, buf608, 4416, 98, grid=grid(4416), stream=stream0)
        buf609 = buf600; del buf600  # reuse
        buf610 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf612 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf606, buf607, buf608, primals_367, primals_368, buf609, buf610, buf612, primals_367, primals_368, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_367
        del primals_368
        buf613 = reinterpret_tensor(buf604, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf604  # reuse
        # Source Nodes: [x_312, x_315], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf605, buf609, buf610, primals_113, primals_114, buf613, 432768, grid=grid(432768), stream=stream0)
        del primals_114
        # Source Nodes: [x_317], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf615 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_317], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf614, buf615, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf616 = buf588; del buf588  # reuse
        buf617 = buf587; del buf587  # reuse
        buf618 = buf586; del buf586  # reuse
        # Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf615, buf616, buf617, buf618, 736, 98, grid=grid(736), stream=stream0)
        buf619 = buf590; del buf590  # reuse
        buf620 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf622 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_318], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_79.run(buf616, buf617, buf618, primals_370, primals_371, buf619, buf620, buf622, primals_370, primals_371, 184, 4, grid=grid(184), stream=stream0)
        del primals_370
        del primals_371
        buf623 = reinterpret_tensor(buf614, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf614  # reuse
        # Source Nodes: [shortcut_19, x_318], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_85.run(buf615, buf619, buf620, primals_115, primals_116, buf593, buf623, 72128, grid=grid(72128), stream=stream0)
        del primals_116
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf623, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf625 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf624, buf625, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf626 = buf608; del buf608  # reuse
        buf627 = buf607; del buf607  # reuse
        buf628 = buf606; del buf606  # reuse
        # Source Nodes: [x_324], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf625, buf626, buf627, buf628, 4416, 98, grid=grid(4416), stream=stream0)
        buf629 = buf610; del buf610  # reuse
        buf630 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf632 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf626, buf627, buf628, primals_373, primals_374, buf629, buf630, buf632, primals_373, primals_374, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_373
        del primals_374
        buf633 = reinterpret_tensor(buf624, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf624  # reuse
        # Source Nodes: [x_324, x_327], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf625, buf629, buf630, primals_117, primals_118, buf633, 432768, grid=grid(432768), stream=stream0)
        del primals_118
        # Source Nodes: [x_328], Original ATen: [aten.convolution]
        buf634 = extern_kernels.convolution(buf633, primals_190, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf634, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf635 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_328], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf634, buf635, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf636 = buf628; del buf628  # reuse
        buf637 = buf627; del buf627  # reuse
        buf638 = buf626; del buf626  # reuse
        # Source Nodes: [x_329], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf635, buf636, buf637, buf638, 4416, 98, grid=grid(4416), stream=stream0)
        buf639 = buf630; del buf630  # reuse
        buf640 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf642 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf636, buf637, buf638, primals_376, primals_377, buf639, buf640, buf642, primals_376, primals_377, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_376
        del primals_377
        buf643 = reinterpret_tensor(buf634, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf634  # reuse
        # Source Nodes: [x_329, x_332], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf635, buf639, buf640, primals_119, primals_120, buf643, 432768, grid=grid(432768), stream=stream0)
        del primals_120
        # Source Nodes: [x_334], Original ATen: [aten.convolution]
        buf644 = extern_kernels.convolution(buf643, primals_191, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (8, 184, 7, 7), (9016, 49, 7, 1))
        buf645 = empty_strided((8, 184, 7, 7), (9016, 1, 1288, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_334], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_77.run(buf644, buf645, 1472, 49, grid=grid(1472, 49), stream=stream0)
        buf646 = buf618; del buf618  # reuse
        buf647 = buf617; del buf617  # reuse
        buf648 = buf616; del buf616  # reuse
        # Source Nodes: [x_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_78.run(buf645, buf646, buf647, buf648, 736, 98, grid=grid(736), stream=stream0)
        buf649 = buf620; del buf620  # reuse
        buf650 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cuda', dtype=torch.float32)
        buf652 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_79.run(buf646, buf647, buf648, primals_379, primals_380, buf649, buf650, buf652, primals_379, primals_380, 184, 4, grid=grid(184), stream=stream0)
        del buf646
        del buf647
        del buf648
        del primals_379
        del primals_380
        buf653 = reinterpret_tensor(buf644, (8, 184, 7, 7), (9016, 1, 1288, 184), 0); del buf644  # reuse
        # Source Nodes: [shortcut_20, x_335], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_85.run(buf645, buf649, buf650, primals_121, primals_122, buf623, buf653, 72128, grid=grid(72128), stream=stream0)
        del buf650
        del primals_122
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf655 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf654, buf655, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf656 = buf638; del buf638  # reuse
        buf657 = buf637; del buf637  # reuse
        buf658 = buf636; del buf636  # reuse
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf655, buf656, buf657, buf658, 4416, 98, grid=grid(4416), stream=stream0)
        buf659 = buf640; del buf640  # reuse
        buf660 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf662 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf656, buf657, buf658, primals_382, primals_383, buf659, buf660, buf662, primals_382, primals_383, 1104, 4, grid=grid(1104), stream=stream0)
        del primals_382
        del primals_383
        buf663 = reinterpret_tensor(buf654, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf654  # reuse
        # Source Nodes: [x_341, x_344], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf655, buf659, buf660, primals_123, primals_124, buf663, 432768, grid=grid(432768), stream=stream0)
        del primals_124
        # Source Nodes: [x_345], Original ATen: [aten.convolution]
        buf664 = extern_kernels.convolution(buf663, primals_193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1104, bias=None)
        assert_size_stride(buf664, (8, 1104, 7, 7), (54096, 49, 7, 1))
        buf665 = empty_strided((8, 1104, 7, 7), (54096, 1, 7728, 1104), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_345], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf664, buf665, 8832, 49, grid=grid(8832, 49), stream=stream0)
        buf666 = buf658; del buf658  # reuse
        buf667 = buf657; del buf657  # reuse
        buf668 = buf656; del buf656  # reuse
        # Source Nodes: [x_346], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_82.run(buf665, buf666, buf667, buf668, 4416, 98, grid=grid(4416), stream=stream0)
        buf669 = buf660; del buf660  # reuse
        buf670 = empty_strided((1, 1104, 1, 1), (1104, 1, 1104, 1104), device='cuda', dtype=torch.float32)
        buf672 = empty((1104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_346], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_83.run(buf666, buf667, buf668, primals_385, primals_386, buf669, buf670, buf672, primals_385, primals_386, 1104, 4, grid=grid(1104), stream=stream0)
        del buf666
        del buf667
        del buf668
        del primals_385
        del primals_386
        buf673 = reinterpret_tensor(buf664, (8, 1104, 7, 7), (54096, 1, 7728, 1104), 0); del buf664  # reuse
        # Source Nodes: [x_346, x_349], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_84.run(buf665, buf669, buf670, primals_125, primals_126, buf673, 432768, grid=grid(432768), stream=stream0)
        del buf670
        del primals_126
        # Source Nodes: [x_351], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (8, 352, 7, 7), (17248, 49, 7, 1))
        buf675 = empty_strided((8, 352, 7, 7), (17248, 1, 2464, 352), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_351], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_86.run(buf674, buf675, 2816, 49, grid=grid(2816, 49), stream=stream0)
        buf676 = empty_strided((1, 352, 1, 1, 4), (1408, 1, 1408, 1408, 352), device='cuda', dtype=torch.float32)
        buf677 = empty_strided((1, 352, 1, 1, 4), (1408, 1, 1408, 1408, 352), device='cuda', dtype=torch.float32)
        buf678 = empty_strided((1, 352, 1, 1, 4), (1408, 1, 1408, 1408, 352), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_87.run(buf675, buf676, buf677, buf678, 1408, 98, grid=grid(1408), stream=stream0)
        buf679 = empty_strided((1, 352, 1, 1), (352, 1, 352, 352), device='cuda', dtype=torch.float32)
        buf680 = empty_strided((1, 352, 1, 1), (352, 1, 352, 352), device='cuda', dtype=torch.float32)
        buf682 = empty((352, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_88.run(buf676, buf677, buf678, primals_388, primals_389, buf679, buf680, buf682, primals_388, primals_389, 352, 4, grid=grid(352), stream=stream0)
        del buf676
        del buf677
        del buf678
        del primals_388
        del primals_389
        buf683 = reinterpret_tensor(buf674, (8, 352, 7, 7), (17248, 1, 2464, 352), 0); del buf674  # reuse
        # Source Nodes: [x_352], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_89.run(buf675, buf679, buf680, primals_127, primals_128, buf683, 137984, grid=grid(137984), stream=stream0)
        del buf680
        del primals_128
        # Source Nodes: [x_357], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf684, (8, 1984, 7, 7), (97216, 49, 7, 1))
        buf685 = empty_strided((8, 1984, 7, 7), (97216, 1, 13888, 1984), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_357], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_90.run(buf684, buf685, 15872, 49, grid=grid(15872, 49), stream=stream0)
        buf686 = empty_strided((1, 1984, 1, 1, 4), (7936, 1, 7936, 7936, 1984), device='cuda', dtype=torch.float32)
        buf687 = empty_strided((1, 1984, 1, 1, 4), (7936, 1, 7936, 7936, 1984), device='cuda', dtype=torch.float32)
        buf688 = empty_strided((1, 1984, 1, 1, 4), (7936, 1, 7936, 7936, 1984), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_91.run(buf685, buf686, buf687, buf688, 7936, 98, grid=grid(7936), stream=stream0)
        buf689 = empty_strided((1, 1984, 1, 1), (1984, 1, 1984, 1984), device='cuda', dtype=torch.float32)
        buf690 = empty_strided((1, 1984, 1, 1), (1984, 1, 1984, 1984), device='cuda', dtype=torch.float32)
        buf692 = empty((1984, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_92.run(buf686, buf687, buf688, primals_391, primals_392, buf689, buf690, buf692, primals_391, primals_392, 1984, 4, grid=grid(1984), stream=stream0)
        del buf686
        del buf687
        del buf688
        del primals_391
        del primals_392
        buf693 = reinterpret_tensor(buf684, (8, 1984, 7, 7), (97216, 1, 13888, 1984), 0); del buf684  # reuse
        buf697 = empty_strided((8, 1984, 7, 7), (97216, 1, 13888, 1984), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_358, x_362], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_93.run(buf685, buf689, buf690, primals_129, primals_130, buf693, buf697, 777728, grid=grid(777728), stream=stream0)
        del buf690
        del primals_130
        buf694 = empty_strided((8, 1984, 1, 1), (1984, 1, 15872, 15872), device='cuda', dtype=torch.float32)
        buf695 = reinterpret_tensor(buf694, (8, 1984), (1984, 1), 0); del buf694  # reuse
        # Source Nodes: [x_363, x_365], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_94.run(buf695, buf693, 15872, 49, grid=grid(15872), stream=stream0)
        del buf693
        buf696 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_197, buf695, reinterpret_tensor(primals_196, (1984, 1000), (1, 1984), 0), alpha=1, beta=1, out=buf696)
        del primals_197
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_198, primals_198, 1, grid=grid(1), stream=stream0)
        del primals_198
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_201, primals_201, 1, grid=grid(1), stream=stream0)
        del primals_201
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_204, primals_204, 1, grid=grid(1), stream=stream0)
        del primals_204
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_207, primals_207, 1, grid=grid(1), stream=stream0)
        del primals_207
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_210, primals_210, 1, grid=grid(1), stream=stream0)
        del primals_210
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_213, primals_213, 1, grid=grid(1), stream=stream0)
        del primals_213
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_216, primals_216, 1, grid=grid(1), stream=stream0)
        del primals_216
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_219, primals_219, 1, grid=grid(1), stream=stream0)
        del primals_219
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_222, primals_222, 1, grid=grid(1), stream=stream0)
        del primals_222
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_345, primals_345, 1, grid=grid(1), stream=stream0)
        del primals_345
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_348, primals_348, 1, grid=grid(1), stream=stream0)
        del primals_348
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_351, primals_351, 1, grid=grid(1), stream=stream0)
        del primals_351
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_354, primals_354, 1, grid=grid(1), stream=stream0)
        del primals_354
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_357, primals_357, 1, grid=grid(1), stream=stream0)
        del primals_357
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_360, primals_360, 1, grid=grid(1), stream=stream0)
        del primals_360
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_363, primals_363, 1, grid=grid(1), stream=stream0)
        del primals_363
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_366, primals_366, 1, grid=grid(1), stream=stream0)
        del primals_366
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_369, primals_369, 1, grid=grid(1), stream=stream0)
        del primals_369
        # Source Nodes: [add__58], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_372, primals_372, 1, grid=grid(1), stream=stream0)
        del primals_372
        # Source Nodes: [add__59], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_375, primals_375, 1, grid=grid(1), stream=stream0)
        del primals_375
        # Source Nodes: [add__60], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_378, primals_378, 1, grid=grid(1), stream=stream0)
        del primals_378
        # Source Nodes: [add__61], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_381, primals_381, 1, grid=grid(1), stream=stream0)
        del primals_381
        # Source Nodes: [add__62], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_384, primals_384, 1, grid=grid(1), stream=stream0)
        del primals_384
        # Source Nodes: [add__63], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_387, primals_387, 1, grid=grid(1), stream=stream0)
        del primals_387
        # Source Nodes: [add__64], Original ATen: [aten.add]
        triton_poi_fused_add_95.run(primals_390, primals_390, 1, grid=grid(1), stream=stream0)
        del primals_390
        return (buf696, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, buf0, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, buf1, buf3, buf13, buf14, buf16, buf26, buf27, buf29, buf39, buf40, buf42, buf52, buf53, buf55, buf65, buf66, buf68, buf78, buf79, buf81, buf91, buf92, buf94, buf104, buf105, buf107, buf117, buf118, buf120, buf130, buf131, buf133, buf143, buf144, buf146, buf156, buf157, buf159, buf169, buf170, buf172, buf182, buf183, buf185, buf192, buf193, buf195, buf202, buf203, buf205, buf212, buf213, buf215, buf222, buf223, buf225, buf232, buf233, buf235, buf242, buf243, buf245, buf252, buf253, buf255, buf262, buf263, buf265, buf272, buf273, buf275, buf282, buf283, buf285, buf292, buf293, buf295, buf302, buf303, buf305, buf312, buf313, buf315, buf322, buf323, buf325, buf332, buf333, buf335, buf342, buf343, buf345, buf352, buf353, buf355, buf362, buf363, buf365, buf372, buf373, buf375, buf382, buf383, buf385, buf392, buf393, buf395, buf402, buf403, buf405, buf412, buf413, buf415, buf422, buf423, buf425, buf432, buf433, buf435, buf442, buf443, buf445, buf452, buf453, buf455, buf462, buf463, buf465, buf472, buf473, buf475, buf482, buf483, buf485, buf492, buf493, buf495, buf502, buf503, buf505, buf512, buf513, buf515, buf522, buf523, buf525, buf532, buf533, buf535, buf542, buf543, buf545, buf552, buf553, buf555, buf562, buf563, buf565, buf572, buf573, buf575, buf582, buf583, buf585, buf592, buf593, buf595, buf602, buf603, buf605, buf612, buf613, buf615, buf622, buf623, buf625, buf632, buf633, buf635, buf642, buf643, buf645, buf652, buf653, buf655, buf662, buf663, buf665, buf672, buf673, buf675, buf682, buf683, buf685, buf692, buf695, reinterpret_tensor(primals_196, (1000, 1984), (1984, 1), 0), buf697, reinterpret_tensor(buf689, (1, 1984, 1, 1), (1984, 1, 1, 1), 0), reinterpret_tensor(buf679, (1, 352, 1, 1), (352, 1, 1, 1), 0), reinterpret_tensor(buf669, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf659, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf649, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf639, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf629, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf619, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf609, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf599, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf589, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf579, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf569, (1, 1104, 1, 1), (1104, 1, 1, 1), 0), reinterpret_tensor(buf559, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf549, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf539, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf529, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf519, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf509, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf499, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf489, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf479, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf469, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf459, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf449, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf439, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf429, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf419, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf409, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf399, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf389, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf379, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf369, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf359, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf349, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf339, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf329, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf319, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf309, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf299, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf289, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf279, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf269, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf259, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf249, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf239, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf209, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf199, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf189, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf179, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf166, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf140, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf127, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf114, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf101, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf88, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf62, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf49, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf36, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf23, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


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
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((32, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((96, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((32, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((192, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((192, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((112, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((336, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((112, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((184, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1104, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((184, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1104, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1104, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((352, 1104, 1, 1), (1104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1984, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1000, 1984), (1984, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_199 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_205 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_319 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_325 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_328 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_349 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_352 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_355 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_358 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_364 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_367 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_370 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_373 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_376 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_379 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_382 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_385 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_388 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_391 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('fbnetc_100', benchmark_compiled_module)
