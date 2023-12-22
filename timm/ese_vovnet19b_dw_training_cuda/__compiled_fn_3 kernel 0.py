
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


# kernel path: /tmp/torchinductor_youkaichao/ss/cssi6obvbluhzm5ehlld5fw4iqwah2dltb5glr27sm4g7urwdaoj.py
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
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/im/cimfpbefz2kjojcnerl2nyubo2kfdawh3k336vxued2icsevxpvs.py
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
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwlsav6owf46tzj5egbyn2nbpk6fgaynigrff7avp6ppiyrhee4.py
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/i7/ci72evga2n6fjpox45biouafsufa4unu7ru3ruy557uudzxfqs3t.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45cmpnkwou6my4yxqexungfuzrzenynr6hohyhsc5laufccchz6.py
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
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7x25vrpyq6exnm3pebiophihcdjlqfre64v223i4k54hry7j4d.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_4 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxkuw4fqyjelpichrmzelki32brslhz4ml5n62r3scva4dvulfs.py
# Source Nodes: [x_11], Original ATen: [aten.convolution]
# x_11 => convolution_3
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/np/cnpzz446fhehbaml7gw5y44hq2kivdfwewku6hvmxzptwm6he26p.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
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


# kernel path: /tmp/torchinductor_youkaichao/yc/cycaltwxfrudnraokf6evu4mjtrgfa4susriscpjk4qulzpw42ml.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbggjkqg2wzntgdxkjadjnwl5v6tf7fihyxacoiytkd2rtlim4s.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckx4emo5d4d5rouqaqst26bnznkvr2npnslutvisjgrveou3jrh7.py
# Source Nodes: [cat_7, x_13, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_7 => cat
# x_13 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_17 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11', 'mutated_arg_names': []},
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
    x1 = (xindex // 64)
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
    tl.store(out_ptr1 + (x0 + (448*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4wey6kyzmi5bdn6e4yuk764me23vmptfodwng5yu74w5rsedha.py
# Source Nodes: [x_18], Original ATen: [aten.convolution]
# x_18 => convolution_5
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (401408*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedteqgfjnwscpm3djdheal6m3tg2pqywba2c6d46bfmde72x4kz.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lp/clp57s7zu6xmcrrltqyzrfhojs4viyqlejbyhoydz6ohd2c45rm4.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjn6arcauwzhck5os2vdp22bhbxnytqbjkepkezp5rfs5snwgah.py
# Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
# x_19 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/it/citvfju6sd5vjzhqxqqq7gcg4rey7dofy4pmotzpgwbolktk3mws.py
# Source Nodes: [x_19, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_19 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_23 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
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


# kernel path: /tmp/torchinductor_youkaichao/ok/cokyqnqx5qgrnnaspymnkdoiedi74j5bjc42cdfvyg4da7ksaizf.py
# Source Nodes: [cat_7, x_26, x_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_7 => cat
# x_26 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_29 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
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
    tl.store(out_ptr1 + (x0 + (448*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cnadq6w4vaw5cplx7btpkdo4augkafbyh5u3kooskmfyywn54ots.py
# Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_38 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# x_41 => relu_6
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
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
    tl.store(out_ptr0 + (x0 + (448*x1)), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cm/ccmrur4mnjcjngl72gyh2pqvxem33ningtpo3uprwpsfzp7hbdvi.py
# Source Nodes: [x_44], Original ATen: [aten.convolution]
# x_44 => convolution_12
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (802816*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvw7rpaqfkroqmtjvqloa7jf62s5onkocgpm2uehzpyuqszbbsa.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5xjnqk4f2ecpfjniy4kaxbdch3toh4dgqqizq7j47nmsdnxdlj.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => var_mean_7
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (256*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (256*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (256*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p3/cp3wqk3sdlkjq7kmkm4aigubpd7eialjjs27omzgbmua35rlzlqv.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => add_36, add_37, add_38, mul_50, mul_51, mul_52, mul_53, mul_54, rsqrt_7, squeeze_22, var_mean_7
triton_per_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ph/cphq4632fjrvnbqhnippg6axiesddtcqizbszhufzaltrej4w4ew.py
# Source Nodes: [x_45, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_45 => add_36, add_39, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# x_49 => relu_7
triton_poi_fused__native_batch_norm_legit_functional_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
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


# kernel path: /tmp/torchinductor_youkaichao/fu/cfueyouvddhe5jdlesdpkn72s2kzqz7ws4h6erextqbixkfokks7.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_red_fused_mean_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256) % 25
    x0 = xindex % 256
    x2 = (xindex // 6400)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r3 + (126*x1)) % 3136)) + (802816*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6kesoa4rtu3rsrtgzcl2yeruabjbxpmjrg7vqibom63mtiyrx7.py
# Source Nodes: [x_se], Original ATen: [aten.mean]
# x_se => mean
triton_per_fused_mean_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (6400*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zj/czjihnwscswvsrueo6kc4xgce2jtoecmzm2g2ccsl7pjh4n265mq.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_gate => add_40, clamp_max, clamp_min, div
# x_se_1 => convolution_13
triton_poi_fused_convolution_hardsigmoid_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gz/cgz32zrijhrlzkgdkyq3maqlpxwolsl2cuhi42puwz3u6pdlofy6.py
# Source Nodes: [x_51], Original ATen: [aten.mul]
# x_51 => mul_56
triton_poi_fused_mul_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 802816)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pp/cpp3ulkdwuzdunuwvsfbbob2dqnrz2ijpte45ol3ru4my4szzsh4.py
# Source Nodes: [x_52], Original ATen: [aten.max_pool2d_with_indices]
# x_52 => getitem_16, getitem_17
triton_poi_fused_max_pool2d_with_indices_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7168) % 28
    x1 = (xindex // 256) % 28
    x0 = xindex % 256
    x5 = (xindex // 7168)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (512*x1) + (28672*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x1)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (28672*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (512 + x0 + (512*x1) + (28672*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x2)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (14336 + x0 + (512*x1) + (28672*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (14592 + x0 + (512*x1) + (28672*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (14848 + x0 + (512*x1) + (28672*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (28672 + x0 + (512*x1) + (28672*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (28928 + x0 + (512*x1) + (28672*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (29184 + x0 + (512*x1) + (28672*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = 1 + (2*x1) + (112*x2)
    tmp72 = (2*x1) + (112*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = 2 + (2*x1) + (112*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = 56 + (2*x1) + (112*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = 57 + (2*x1) + (112*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 58 + (2*x1) + (112*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 112 + (2*x1) + (112*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 113 + (2*x1) + (112*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 114 + (2*x1) + (112*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7j5s4epr5sszchwt6rqutbmmxhbga4l7yddwlcqqcccww2yxvrb.py
# Source Nodes: [x_53], Original ATen: [aten.convolution]
# x_53 => convolution_14
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (125440*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4xgioqaycn33gh3om7g25nrcrhxui7vihdcsn2ceda5psgmscs.py
# Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
# x_54 => var_mean_8
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (160*r2) + (20480*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vd/cvdtljgzij3gcqafyps44cig52sgwhgf5yzhrqll4ki7ism7arki.py
# Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
# x_54 => add_42, add_43, add_44, mul_58, mul_59, mul_60, mul_61, mul_62, rsqrt_8, squeeze_25, var_mean_8
triton_per_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nxr2cpkkjgdhrwxmeftwun6rcrn6pffed7yjjd4x4oqk3hejsc.py
# Source Nodes: [x_54, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_54 => add_42, add_45, mul_57, mul_63, rsqrt_8, sub_8, var_mean_8
# x_58 => relu_8
triton_poi_fused__native_batch_norm_legit_functional_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
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


# kernel path: /tmp/torchinductor_youkaichao/o6/co6pyudbfwwfx74rems7bybk23jogzt7jirk6nwo7w4yj5o23irh.py
# Source Nodes: [cat_6, x_61, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_6 => cat_1
# x_61 => add_47, add_50, mul_64, mul_70, rsqrt_9, sub_9, var_mean_9
# x_64 => relu_9
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    x1 = (xindex // 160)
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
    tl.store(out_ptr1 + (x0 + (736*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijach5l7rzh24l7efeitfaffivibxfgelnz5a2bdbcjbpy52xof.py
# Source Nodes: [x_73, x_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_73 => add_57, add_60, mul_78, mul_84, rsqrt_11, sub_11, var_mean_11
# x_76 => relu_11
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 160
    x1 = (xindex // 160)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (736*x1)), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xz/cxztdcwacpirzagadvlahpp764w76lp3sl75rilsipbhobrjqlek.py
# Source Nodes: [cat_6], Original ATen: [aten.cat]
# cat_6 => cat_1
triton_poi_fused_cat_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (736*x2) + (577024*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhgnbtn3cluyay7ws54wtwjpchxdiae3om3v6vsuag5craxq6ay.py
# Source Nodes: [x_79], Original ATen: [aten.convolution]
# x_79 => convolution_21
triton_poi_fused_convolution_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (401408*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coehieykmyhgqf5prjebrf3ifeb2kbdef6lk6wquotssjs7kgp6v.py
# Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
# x_80 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hb/chbx3326lsy7b3cfqxkhxl3lemnvuc4kws6jbh3dlobhkowuhoro.py
# Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
# x_80 => add_62, add_63, add_64, mul_86, mul_87, mul_88, mul_89, mul_90, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ev/cevgzffeqncnofkdkwwskxwjhjijwhhiks2iki7othy2xktiq47o.py
# Source Nodes: [x_80, x_84], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_80 => add_62, add_65, mul_85, mul_91, rsqrt_12, sub_12, var_mean_12
# x_84 => relu_12
triton_poi_fused__native_batch_norm_legit_functional_relu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
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


# kernel path: /tmp/torchinductor_youkaichao/7k/c7kvq3cjd72b3enitd6u2c27buiwyylwxdzgo2yxfad4xgj3hc57.py
# Source Nodes: [x_se_2], Original ATen: [aten.mean]
# x_se_2 => mean_1
triton_red_fused_mean_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28672
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (57344*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aa/caa6lrdmpd2s2isvovrgw72xesol6ek3qdvf5svp7gv6vl4vpqgk.py
# Source Nodes: [x_se_2], Original ATen: [aten.mean]
# x_se_2 => mean_1
triton_per_fused_mean_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_41', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (3584*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7pmwm7zltdew35bvf7fbgc6rlahnrpzuhncmneokdoweryu5jc.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___stages___1___blocks___0___attn_gate => add_66, clamp_max_1, clamp_min_1, div_1
# x_se_3 => convolution_22
triton_poi_fused_convolution_hardsigmoid_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdlp3iexzp4uh6xilgvmmv7u5pfhd2372mbjq2n2e7semtwo6qx.py
# Source Nodes: [x_86], Original ATen: [aten.mul]
# x_86 => mul_92
triton_poi_fused_mul_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqphvzyqvh4oqazarddzxwvljfclg6b5y4rcfca2nnqakgkwa5w.py
# Source Nodes: [x_87], Original ATen: [aten.max_pool2d_with_indices]
# x_87 => getitem_28, getitem_29
triton_poi_fused_max_pool2d_with_indices_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7168) % 14
    x1 = (xindex // 512) % 14
    x0 = xindex % 512
    x5 = (xindex // 7168)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (1024*x1) + (28672*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x1)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (28672*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (1024 + x0 + (1024*x1) + (28672*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x2)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (14336 + x0 + (1024*x1) + (28672*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (14848 + x0 + (1024*x1) + (28672*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (15360 + x0 + (1024*x1) + (28672*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (28672 + x0 + (1024*x1) + (28672*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (29184 + x0 + (1024*x1) + (28672*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (29696 + x0 + (1024*x1) + (28672*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = 1 + (2*x1) + (56*x2)
    tmp72 = (2*x1) + (56*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = 2 + (2*x1) + (56*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = 28 + (2*x1) + (56*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = 29 + (2*x1) + (56*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 30 + (2*x1) + (56*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 56 + (2*x1) + (56*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 57 + (2*x1) + (56*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 58 + (2*x1) + (56*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfs346lsnjmgws4lawm5ubv6khcfxijr5kx3j422vo7ctysroppm.py
# Source Nodes: [x_88], Original ATen: [aten.convolution]
# x_88 => convolution_23
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pp/cpp6yzn67ygrwvtfqsbsnsh2gioboew5la2onjny3rie6a5zb34o.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
# x_89 => var_mean_13
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/e4/ce46it6nxoqwnn22t4qgdlpu6p22hllorlbuxs2o4fslwzwh7ebi.py
# Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
# x_89 => add_68, add_69, add_70, mul_94, mul_95, mul_96, mul_97, mul_98, rsqrt_13, squeeze_40, var_mean_13
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5qg7pk476w3ubcpeigs3gwh64kl4b7nkhzigy7ikzwq7fmf4th.py
# Source Nodes: [x_89, x_93], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_89 => add_68, add_71, mul_93, mul_99, rsqrt_13, sub_13, var_mean_13
# x_93 => relu_13
triton_poi_fused__native_batch_norm_legit_functional_relu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_48', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ng/cnguwen7bzwqrzsi25rmoxr3bqendheyvabd5c2r5ekpc5orcj6s.py
# Source Nodes: [cat_5, x_96, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_5 => cat_2
# x_96 => add_73, add_76, mul_100, mul_106, rsqrt_14, sub_14, var_mean_14
# x_99 => relu_14
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    x1 = (xindex // 192)
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
    tl.store(out_ptr1 + (x0 + (1088*x1)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfykup7jk6cy3rkg6fjet5cwdsokmtfzaggsi4iia35wlzqspadj.py
# Source Nodes: [x_108, x_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_108 => add_83, add_86, mul_114, mul_120, rsqrt_16, sub_16, var_mean_16
# x_111 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    x1 = (xindex // 192)
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x0 + (1088*x1)), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4sty3cv3fcfnbxnzec2zvvnyy2fc5smgpyjalqrrqzmvqautbe4.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat_2
triton_poi_fused_cat_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1088*x2) + (213248*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/cslxi3t7j66acjtydpax74zqezaauj2ml4fpiox7qhq6jfvgucya.py
# Source Nodes: [x_114], Original ATen: [aten.convolution]
# x_114 => convolution_30
triton_poi_fused_convolution_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2kpv2uz6xhiufnyrnpf5e77g2qj4tuz7bbjfryokxd3uqrvjn6.py
# Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
# x_115 => var_mean_17
triton_red_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
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
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/d3/cd33lcnizg7rwry65cqj2w2y6cudkumyizxfwocu3lg5sewst4ry.py
# Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
# x_115 => add_88, add_89, add_90, mul_122, mul_123, mul_124, mul_125, mul_126, rsqrt_17, squeeze_52, var_mean_17
triton_per_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsqe73b4v4acsmuljnepylqzhf7u2brlrmsxeczsyhlvb5j34yu.py
# Source Nodes: [x_115, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_115 => add_88, add_91, mul_121, mul_127, rsqrt_17, sub_17, var_mean_17
# x_119 => relu_17
triton_poi_fused__native_batch_norm_legit_functional_relu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrvdweuo6afbyha4nrjo6rudfq3dodxboytieqauyjvepiie56x.py
# Source Nodes: [x_se_4], Original ATen: [aten.mean]
# x_se_4 => mean_2
triton_red_fused_mean_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (75264*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ne/cnekpasdjev36tcp4etwby7yo7kzi35jhzwuzy4pfpsidtzc4w3o.py
# Source Nodes: [x_se_4], Original ATen: [aten.mean]
# x_se_4 => mean_2
triton_per_fused_mean_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_57', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (1536*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7chkydg6nhqmvmzyflpqoko2k2x67dlzo4hegza2lvuhy3z5o6.py
# Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_gate, x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid]
# getattr_getattr_l__mod___stages___2___blocks___0___attn_gate => add_92, clamp_max_2, clamp_min_2, div_2
# x_se_5 => convolution_31
triton_poi_fused_convolution_hardsigmoid_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tl.store(out_ptr0 + (x2), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caebj4wmpefsbgqsdesmrcv36ufdqdtedbeupqh7v2fwnb6lbg4k.py
# Source Nodes: [x_121], Original ATen: [aten.mul]
# x_121 => mul_128
triton_poi_fused_mul_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4m/c4m7qiookztf2gbbr3fkrgweiai57tcfpeudvlij25677bwl77ow.py
# Source Nodes: [x_122], Original ATen: [aten.max_pool2d_with_indices]
# x_122 => getitem_40, getitem_41
triton_poi_fused_max_pool2d_with_indices_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5376) % 7
    x1 = (xindex // 768) % 7
    x0 = xindex % 768
    x5 = (xindex // 5376)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (1536*x1) + (21504*x5)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x1)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (768 + x0 + (1536*x1) + (21504*x5)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x1)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (1536 + x0 + (1536*x1) + (21504*x5)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x2)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (10752 + x0 + (1536*x1) + (21504*x5)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (11520 + x0 + (1536*x1) + (21504*x5)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (12288 + x0 + (1536*x1) + (21504*x5)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x2)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (21504 + x0 + (1536*x1) + (21504*x5)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (22272 + x0 + (1536*x1) + (21504*x5)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (23040 + x0 + (1536*x1) + (21504*x5)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = 1 + (2*x1) + (28*x2)
    tmp72 = (2*x1) + (28*x2)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = 2 + (2*x1) + (28*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = 14 + (2*x1) + (28*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = 15 + (2*x1) + (28*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 16 + (2*x1) + (28*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 28 + (2*x1) + (28*x2)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 29 + (2*x1) + (28*x2)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 30 + (2*x1) + (28*x2)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x6), tmp69, None)
    tl.store(out_ptr1 + (x6), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbqjpywu3ho4vmldxjihlugmyz3acwvpnwrcphnpxaijcxy3m2k.py
# Source Nodes: [x_123], Original ATen: [aten.convolution]
# x_123 => convolution_32
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/26/c26ytigv4zcpyooueghs34psivjs7wagizjomj64kmjym7kat24r.py
# Source Nodes: [x_124], Original ATen: [aten._native_batch_norm_legit_functional]
# x_124 => var_mean_18
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/de/cdeyh74fenxtocqq6jcnl6kyqcblm62r4driq4wwy53pseciidrq.py
# Source Nodes: [x_124], Original ATen: [aten._native_batch_norm_legit_functional]
# x_124 => add_94, add_95, add_96, mul_130, mul_131, mul_132, mul_133, mul_134, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_63', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/sm/csme4vz34uroyqepqatfvvdq2n5lidqsoe6vkvlljtvgamcy3h3h.py
# Source Nodes: [x_124, x_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_124 => add_94, add_97, mul_129, mul_135, rsqrt_18, sub_18, var_mean_18
# x_128 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_relu_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_64', 'mutated_arg_names': []},
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tplgukzzgdwchvzpx3zrjkztgqflhrmm5k5sp4ge72hnj5z3dw.py
# Source Nodes: [cat_4, x_131, x_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_4 => cat_3
# x_131 => add_102, add_99, mul_136, mul_142, rsqrt_19, sub_19, var_mean_19
# x_134 => relu_19
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
    x1 = (xindex // 224)
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
    tl.store(out_ptr1 + (x0 + (1440*x1)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oo/cooils2cohcvlz53mk43i255qzlhyxgcowp6q7iwdgidjk3b6ymu.py
# Source Nodes: [x_143, x_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# x_143 => add_109, add_112, mul_150, mul_156, rsqrt_21, sub_21, var_mean_21
# x_146 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
    x1 = (xindex // 224)
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
    tl.store(out_ptr0 + (x0 + (1440*x1)), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepjqpezphjogygmlxec3uewrkxq4q5uml22ida7akpdf4rqhbhy.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_3
triton_poi_fused_cat_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1440*x2) + (70560*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cyd4vrprubzd7p2tickas3qrp7nehiwzas5wdcm3xucg7qeqzs5g.py
# Source Nodes: [x_149], Original ATen: [aten.convolution]
# x_149 => convolution_39
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
    ynumel = 8192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crznthwlgg7ijizimlu7nftwlgmqykhrytoo66qub3jvegcqnrwy.py
# Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
# x_150 => var_mean_22
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
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/cc/cccoqnvbdj2oqa3tlkvervgbqty3hozgl2iki55dzkuvwkrdxyqh.py
# Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
# x_150 => add_114, add_115, add_116, mul_158, mul_159, mul_160, mul_161, mul_162, rsqrt_22, squeeze_67, var_mean_22
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
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/lo/closyrfnyirtu6tbulne7bqd6nbo7f4zty76vbtqzi44asvj5tte.py
# Source Nodes: [x_150, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_150 => add_114, add_117, mul_157, mul_163, rsqrt_22, sub_22, var_mean_22
# x_154 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cceotfjpxtymyml44g6owjlhu6r25hng2zhkmrawmse57f4nrw6n.py
# Source Nodes: [x_se_6], Original ATen: [aten.mean]
# x_se_6 => mean_3
triton_per_fused_mean_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/car6kbwe2j4grb4ol7orp3lr44kepsnkdrwsnaqhnnotyseczuo4.py
# Source Nodes: [getattr_getattr_l__mod___stages___3___blocks___0___attn_gate, x_157, x_158, x_160, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward, aten.mean, aten.mul, aten.view]
# getattr_getattr_l__mod___stages___3___blocks___0___attn_gate => add_118, clamp_max_3, clamp_min_3, div_3
# x_157 => mul_164
# x_158 => mean_4
# x_160 => view
# x_se_7 => convolution_40
triton_per_fused_convolution_hardsigmoid_hardsigmoid_backward_mean_mul_view_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_hardsigmoid_hardsigmoid_backward_mean_mul_view_73', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x2 = xindex
    x0 = xindex % 1024
    r3 = rindex
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x0 + (1024*r3) + (50176*x1)), rmask, other=0.0)
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
    tmp15 = tmp14 * tmp9
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.store(out_ptr0 + (x2), tmp9, None)
    tl.store(out_ptr1 + (x2), tmp13, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/collk4p7l6hajowgpuxmsc3unenymtac4hlsbv4tc4kfi5ngdqcc.py
# Source Nodes: [x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_5 => convolution_31
triton_poi_fused_convolution_hardsigmoid_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/clytis6otm7wiufjxurk7eh2kkpf4jgscobjw7bxjoushyvq3ofc.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_3 => convolution_22
triton_poi_fused_convolution_hardsigmoid_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73jitlkkqe2zl3cjtcgxsjpdrmen5vqjghwxbwmd3gpi4smbrrl.py
# Source Nodes: [x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
# x_se_1 => convolution_13
triton_poi_fused_convolution_hardsigmoid_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_backward_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -3.0
    tmp4 = tmp2 > tmp3
    tmp5 = 3.0
    tmp6 = tmp2 < tmp5
    tmp7 = tmp4 & tmp6
    tl.store(out_ptr0 + (x2), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fdylsuaojmwmjimsjkv2ga5oruosq53bjcaoq7psyqg74txlck.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_77', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (160, ), (1, ))
    assert_size_stride(primals_18, (160, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (160, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (224, ), (1, ))
    assert_size_stride(primals_38, (224, ), (1, ))
    assert_size_stride(primals_39, (224, ), (1, ))
    assert_size_stride(primals_40, (224, ), (1, ))
    assert_size_stride(primals_41, (224, ), (1, ))
    assert_size_stride(primals_42, (224, ), (1, ))
    assert_size_stride(primals_43, (224, ), (1, ))
    assert_size_stride(primals_44, (224, ), (1, ))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_46, (1024, ), (1, ))
    assert_size_stride(primals_47, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_48, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_50, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_52, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_53, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_54, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_55, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_56, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_57, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_58, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_59, (256, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_60, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (160, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_64, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_65, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_66, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_67, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_69, (512, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_70, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (192, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_73, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_75, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_77, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_78, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_79, (768, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_80, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (224, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_83, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_84, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_85, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_86, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_87, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_88, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_89, (1024, 1440, 1, 1), (1440, 1, 1, 1))
    assert_size_stride(primals_90, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_91, (1024, ), (1, ))
    assert_size_stride(primals_92, (1000, 1024), (1024, 1))
    assert_size_stride(primals_93, (1000, ), (1, ))
    assert_size_stride(primals_94, (), ())
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (), ())
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (), ())
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (64, ), (1, ))
    assert_size_stride(primals_103, (), ())
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (), ())
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (), ())
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (), ())
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (), ())
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (), ())
    assert_size_stride(primals_119, (160, ), (1, ))
    assert_size_stride(primals_120, (160, ), (1, ))
    assert_size_stride(primals_121, (), ())
    assert_size_stride(primals_122, (160, ), (1, ))
    assert_size_stride(primals_123, (160, ), (1, ))
    assert_size_stride(primals_124, (), ())
    assert_size_stride(primals_125, (160, ), (1, ))
    assert_size_stride(primals_126, (160, ), (1, ))
    assert_size_stride(primals_127, (), ())
    assert_size_stride(primals_128, (160, ), (1, ))
    assert_size_stride(primals_129, (160, ), (1, ))
    assert_size_stride(primals_130, (), ())
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (), ())
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (), ())
    assert_size_stride(primals_137, (192, ), (1, ))
    assert_size_stride(primals_138, (192, ), (1, ))
    assert_size_stride(primals_139, (), ())
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_142, (), ())
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, ), (1, ))
    assert_size_stride(primals_145, (), ())
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (), ())
    assert_size_stride(primals_149, (224, ), (1, ))
    assert_size_stride(primals_150, (224, ), (1, ))
    assert_size_stride(primals_151, (), ())
    assert_size_stride(primals_152, (224, ), (1, ))
    assert_size_stride(primals_153, (224, ), (1, ))
    assert_size_stride(primals_154, (), ())
    assert_size_stride(primals_155, (224, ), (1, ))
    assert_size_stride(primals_156, (224, ), (1, ))
    assert_size_stride(primals_157, (), ())
    assert_size_stride(primals_158, (224, ), (1, ))
    assert_size_stride(primals_159, (224, ), (1, ))
    assert_size_stride(primals_160, (), ())
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_47, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_47
        buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_163, buf1, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del primals_163
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf3 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf4 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf3, buf4, buf5, buf6, 50176, 128, grid=grid(50176), stream=stream0)
        buf7 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf4, buf5, buf6, buf7, buf8, buf9, 448, 112, grid=grid(448), stream=stream0)
        buf10 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf13 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf7, buf8, buf9, primals_95, primals_96, buf10, buf11, buf13, primals_95, primals_96, 64, 7, grid=grid(64), stream=stream0)
        del primals_95
        del primals_96
        buf14 = reinterpret_tensor(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf2  # reuse
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf3, buf10, buf11, primals_1, primals_2, buf14, 6422528, grid=grid(6422528), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf15, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf16 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf15, buf16, 512, 12544, grid=grid(512, 12544), stream=stream0)
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf18 = reinterpret_tensor(buf15, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf15  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf17, buf18, 512, 12544, grid=grid(512, 12544), stream=stream0)
        buf19 = buf6; del buf6  # reuse
        buf20 = buf5; del buf5  # reuse
        buf21 = buf4; del buf4  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf18, buf19, buf20, buf21, 50176, 128, grid=grid(50176), stream=stream0)
        buf22 = buf9; del buf9  # reuse
        buf23 = buf8; del buf8  # reuse
        buf24 = buf7; del buf7  # reuse
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf19, buf20, buf21, buf22, buf23, buf24, 448, 112, grid=grid(448), stream=stream0)
        buf25 = buf11; del buf11  # reuse
        buf26 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf28 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf22, buf23, buf24, primals_98, primals_99, buf25, buf26, buf28, primals_98, primals_99, 64, 7, grid=grid(64), stream=stream0)
        del buf22
        del buf23
        del buf24
        del primals_98
        del primals_99
        buf29 = reinterpret_tensor(buf17, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf17  # reuse
        # Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf18, buf25, buf26, primals_3, primals_4, buf29, 6422528, grid=grid(6422528), stream=stream0)
        del primals_4
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_50, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf30, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf31 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf30, buf31, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf33 = reinterpret_tensor(buf30, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf30  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(buf32, buf33, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf34 = empty_strided((1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 64, 1, 1, 196), (12544, 1, 12544, 12544, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf33, buf34, buf35, buf36, 12544, 128, grid=grid(12544), stream=stream0)
        buf37 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf38 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((1, 64, 1, 1, 2), (128, 1, 128, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf34, buf35, buf36, buf37, buf38, buf39, 128, 98, grid=grid(128), stream=stream0)
        del buf34
        del buf35
        del buf36
        buf40 = buf26; del buf26  # reuse
        buf41 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf43 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf37, buf38, buf39, primals_101, primals_102, buf40, buf41, buf43, primals_101, primals_102, 64, 2, grid=grid(64), stream=stream0)
        del primals_101
        del primals_102
        buf44 = reinterpret_tensor(buf32, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf32  # reuse
        buf106 = empty_strided((8, 448, 56, 56), (1404928, 1, 25088, 448), device='cuda', dtype=torch.float32)
        buf103 = reinterpret_tensor(buf106, (8, 64, 56, 56), (1404928, 1, 25088, 448), 0)  # alias
        # Source Nodes: [cat_7, x_13, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11.run(buf33, buf40, buf41, primals_5, primals_6, buf44, buf103, 1605632, grid=grid(1605632), stream=stream0)
        del buf41
        del primals_6
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf46 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf45, buf46, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf47 = empty_strided((1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), device='cuda', dtype=torch.float32)
        buf48 = empty_strided((1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((1, 128, 1, 1, 196), (25088, 1, 25088, 25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf46, buf47, buf48, buf49, 25088, 128, grid=grid(25088), stream=stream0)
        buf50 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((1, 128, 1, 1, 2), (256, 1, 256, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf47, buf48, buf49, buf50, buf51, buf52, 256, 98, grid=grid(256), stream=stream0)
        buf53 = reinterpret_tensor(buf39, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf39  # reuse
        buf54 = reinterpret_tensor(buf38, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf38  # reuse
        buf56 = reinterpret_tensor(buf37, (128, ), (1, ), 0); del buf37  # reuse
        # Source Nodes: [x_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf50, buf51, buf52, primals_104, primals_105, buf53, buf54, buf56, primals_104, primals_105, 128, 2, grid=grid(128), stream=stream0)
        del primals_104
        del primals_105
        buf57 = reinterpret_tensor(buf45, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf45  # reuse
        # Source Nodes: [x_19, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf46, buf53, buf54, primals_7, primals_8, buf57, 3211264, grid=grid(3211264), stream=stream0)
        del primals_8
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf58, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf59 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf58, buf59, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf61 = reinterpret_tensor(buf58, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf58  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf60, buf61, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf62 = buf49; del buf49  # reuse
        buf63 = buf48; del buf48  # reuse
        buf64 = buf47; del buf47  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf61, buf62, buf63, buf64, 25088, 128, grid=grid(25088), stream=stream0)
        buf65 = buf52; del buf52  # reuse
        buf66 = buf51; del buf51  # reuse
        buf67 = buf50; del buf50  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf62, buf63, buf64, buf65, buf66, buf67, 256, 98, grid=grid(256), stream=stream0)
        buf68 = buf54; del buf54  # reuse
        buf69 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf71 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf65, buf66, buf67, primals_107, primals_108, buf68, buf69, buf71, primals_107, primals_108, 128, 2, grid=grid(128), stream=stream0)
        del primals_107
        del primals_108
        buf72 = reinterpret_tensor(buf60, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf60  # reuse
        buf104 = reinterpret_tensor(buf106, (8, 128, 56, 56), (1404928, 1, 25088, 448), 64)  # alias
        # Source Nodes: [cat_7, x_26, x_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_17.run(buf61, buf68, buf69, primals_9, primals_10, buf72, buf104, 3211264, grid=grid(3211264), stream=stream0)
        del primals_10
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_55, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf73, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf74 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf73, buf74, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf76 = reinterpret_tensor(buf73, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf73  # reuse
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf75, buf76, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf77 = buf64; del buf64  # reuse
        buf78 = buf63; del buf63  # reuse
        buf79 = buf62; del buf62  # reuse
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf76, buf77, buf78, buf79, 25088, 128, grid=grid(25088), stream=stream0)
        buf80 = buf67; del buf67  # reuse
        buf81 = buf66; del buf66  # reuse
        buf82 = buf65; del buf65  # reuse
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf77, buf78, buf79, buf80, buf81, buf82, 256, 98, grid=grid(256), stream=stream0)
        buf83 = buf69; del buf69  # reuse
        buf84 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf86 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf80, buf81, buf82, primals_110, primals_111, buf83, buf84, buf86, primals_110, primals_111, 128, 2, grid=grid(128), stream=stream0)
        del primals_110
        del primals_111
        buf87 = reinterpret_tensor(buf75, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf75  # reuse
        buf105 = reinterpret_tensor(buf106, (8, 128, 56, 56), (1404928, 1, 25088, 448), 192)  # alias
        # Source Nodes: [cat_7, x_32, x_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_17.run(buf76, buf83, buf84, primals_11, primals_12, buf87, buf105, 3211264, grid=grid(3211264), stream=stream0)
        del primals_12
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf88, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf89 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf88, buf89, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf91 = reinterpret_tensor(buf88, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf88  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf90, buf91, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        buf92 = buf79; del buf79  # reuse
        buf93 = buf78; del buf78  # reuse
        buf94 = buf77; del buf77  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf91, buf92, buf93, buf94, 25088, 128, grid=grid(25088), stream=stream0)
        buf95 = buf82; del buf82  # reuse
        buf96 = buf81; del buf81  # reuse
        buf97 = buf80; del buf80  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf92, buf93, buf94, buf95, buf96, buf97, 256, 98, grid=grid(256), stream=stream0)
        buf98 = buf84; del buf84  # reuse
        buf99 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf101 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf95, buf96, buf97, primals_113, primals_114, buf98, buf99, buf101, primals_113, primals_114, 128, 2, grid=grid(128), stream=stream0)
        del primals_113
        del primals_114
        buf102 = reinterpret_tensor(buf106, (8, 128, 56, 56), (1404928, 1, 25088, 448), 320)  # alias
        buf338 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_18.run(buf91, buf98, buf99, primals_13, primals_14, buf102, buf338, 3211264, grid=grid(3211264), stream=stream0)
        del buf99
        del primals_14
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_59, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf108 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf107, buf108, 2048, 3136, grid=grid(2048, 3136), stream=stream0)
        buf109 = reinterpret_tensor(buf21, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf21  # reuse
        buf110 = reinterpret_tensor(buf20, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf20  # reuse
        buf111 = reinterpret_tensor(buf19, (1, 256, 1, 1, 196), (50176, 1, 50176, 50176, 256), 0); del buf19  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf108, buf109, buf110, buf111, 50176, 128, grid=grid(50176), stream=stream0)
        buf112 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf113 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        buf114 = empty_strided((1, 256, 1, 1, 2), (512, 1, 512, 512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf109, buf110, buf111, buf112, buf113, buf114, 512, 98, grid=grid(512), stream=stream0)
        del buf109
        del buf110
        del buf111
        buf115 = reinterpret_tensor(buf97, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf97  # reuse
        buf116 = reinterpret_tensor(buf96, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf96  # reuse
        buf118 = reinterpret_tensor(buf95, (256, ), (1, ), 0); del buf95  # reuse
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_22.run(buf112, buf113, buf114, primals_116, primals_117, buf115, buf116, buf118, primals_116, primals_117, 256, 2, grid=grid(256), stream=stream0)
        del primals_116
        del primals_117
        buf119 = reinterpret_tensor(buf107, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf107  # reuse
        # Source Nodes: [x_45, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf108, buf115, buf116, primals_15, primals_16, buf119, 6422528, grid=grid(6422528), stream=stream0)
        del buf116
        del primals_16
        buf120 = empty_strided((8, 256, 1, 1, 25), (6400, 1, 51200, 51200, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_red_fused_mean_24.run(buf119, buf120, 51200, 126, grid=grid(51200), stream=stream0)
        buf121 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf122 = reinterpret_tensor(buf121, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf121  # reuse
        # Source Nodes: [x_se], Original ATen: [aten.mean]
        triton_per_fused_mean_25.run(buf122, buf120, 2048, 25, grid=grid(2048), stream=stream0)
        del buf120
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 256, 1, 1), (256, 1, 1, 1))
        buf124 = empty_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_gate, x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_26.run(buf123, primals_61, buf124, 2048, grid=grid(2048), stream=stream0)
        buf125 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51], Original ATen: [aten.mul]
        triton_poi_fused_mul_27.run(buf119, buf124, buf125, 6422528, grid=grid(6422528), stream=stream0)
        buf126 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        buf127 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_52], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_28.run(buf125, buf126, buf127, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf126, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf129 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf128, buf129, 1280, 784, grid=grid(1280, 784), stream=stream0)
        buf130 = empty_strided((1, 160, 1, 1, 49), (7840, 1, 7840, 7840, 160), device='cuda', dtype=torch.float32)
        buf131 = empty_strided((1, 160, 1, 1, 49), (7840, 1, 7840, 7840, 160), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((1, 160, 1, 1, 49), (7840, 1, 7840, 7840, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf129, buf130, buf131, buf132, 7840, 128, grid=grid(7840), stream=stream0)
        buf133 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf136 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf130, buf131, buf132, primals_119, primals_120, buf133, buf134, buf136, primals_119, primals_120, 160, 49, grid=grid(160), stream=stream0)
        del primals_119
        del primals_120
        buf137 = reinterpret_tensor(buf128, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf128  # reuse
        # Source Nodes: [x_54, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf129, buf133, buf134, primals_17, primals_18, buf137, 1003520, grid=grid(1003520), stream=stream0)
        del primals_18
        # Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf138, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf139 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf138, buf139, 1280, 784, grid=grid(1280, 784), stream=stream0)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf141 = reinterpret_tensor(buf138, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf138  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf140, buf141, 1280, 784, grid=grid(1280, 784), stream=stream0)
        buf142 = buf132; del buf132  # reuse
        buf143 = buf131; del buf131  # reuse
        buf144 = buf130; del buf130  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf141, buf142, buf143, buf144, 7840, 128, grid=grid(7840), stream=stream0)
        buf145 = buf134; del buf134  # reuse
        buf146 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf148 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf142, buf143, buf144, primals_122, primals_123, buf145, buf146, buf148, primals_122, primals_123, 160, 49, grid=grid(160), stream=stream0)
        del primals_122
        del primals_123
        buf149 = reinterpret_tensor(buf140, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf140  # reuse
        buf177 = empty_strided((8, 736, 28, 28), (577024, 1, 20608, 736), device='cuda', dtype=torch.float32)
        buf175 = reinterpret_tensor(buf177, (8, 160, 28, 28), (577024, 1, 20608, 736), 256)  # alias
        # Source Nodes: [cat_6, x_61, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_33.run(buf141, buf145, buf146, primals_19, primals_20, buf149, buf175, 1003520, grid=grid(1003520), stream=stream0)
        del primals_20
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf150, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf151 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf150, buf151, 1280, 784, grid=grid(1280, 784), stream=stream0)
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf153 = reinterpret_tensor(buf150, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf150  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf152, buf153, 1280, 784, grid=grid(1280, 784), stream=stream0)
        buf154 = buf144; del buf144  # reuse
        buf155 = buf143; del buf143  # reuse
        buf156 = buf142; del buf142  # reuse
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf153, buf154, buf155, buf156, 7840, 128, grid=grid(7840), stream=stream0)
        buf157 = buf146; del buf146  # reuse
        buf158 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf160 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf154, buf155, buf156, primals_125, primals_126, buf157, buf158, buf160, primals_125, primals_126, 160, 49, grid=grid(160), stream=stream0)
        del primals_125
        del primals_126
        buf161 = reinterpret_tensor(buf152, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf152  # reuse
        buf176 = reinterpret_tensor(buf177, (8, 160, 28, 28), (577024, 1, 20608, 736), 416)  # alias
        # Source Nodes: [cat_6, x_67, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_33.run(buf153, buf157, buf158, primals_21, primals_22, buf161, buf176, 1003520, grid=grid(1003520), stream=stream0)
        del primals_22
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf162, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf163 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf162, buf163, 1280, 784, grid=grid(1280, 784), stream=stream0)
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 160, 28, 28), (125440, 784, 28, 1))
        buf165 = reinterpret_tensor(buf162, (8, 160, 28, 28), (125440, 1, 4480, 160), 0); del buf162  # reuse
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf164, buf165, 1280, 784, grid=grid(1280, 784), stream=stream0)
        del buf164
        buf166 = buf156; del buf156  # reuse
        buf167 = buf155; del buf155  # reuse
        buf168 = buf154; del buf154  # reuse
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf165, buf166, buf167, buf168, 7840, 128, grid=grid(7840), stream=stream0)
        buf169 = buf158; del buf158  # reuse
        buf170 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf172 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_31.run(buf166, buf167, buf168, primals_128, primals_129, buf169, buf170, buf172, primals_128, primals_129, 160, 49, grid=grid(160), stream=stream0)
        del buf166
        del buf167
        del buf168
        del primals_128
        del primals_129
        buf173 = reinterpret_tensor(buf177, (8, 160, 28, 28), (577024, 1, 20608, 736), 576)  # alias
        buf336 = empty_strided((8, 160, 28, 28), (125440, 1, 4480, 160), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_73, x_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_34.run(buf165, buf169, buf170, primals_23, primals_24, buf173, buf336, 1003520, grid=grid(1003520), stream=stream0)
        del buf170
        del primals_24
        buf174 = reinterpret_tensor(buf177, (8, 256, 28, 28), (577024, 1, 20608, 736), 0)  # alias
        # Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf126, buf174, 2048, 784, grid=grid(2048, 784), stream=stream0)
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf179 = reinterpret_tensor(buf90, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf90  # reuse
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_36.run(buf178, buf179, 4096, 784, grid=grid(4096, 784), stream=stream0)
        buf180 = reinterpret_tensor(buf94, (1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), 0); del buf94  # reuse
        buf181 = reinterpret_tensor(buf93, (1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), 0); del buf93  # reuse
        buf182 = reinterpret_tensor(buf92, (1, 512, 1, 1, 49), (25088, 1, 25088, 25088, 512), 0); del buf92  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf179, buf180, buf181, buf182, 25088, 128, grid=grid(25088), stream=stream0)
        buf183 = reinterpret_tensor(buf114, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf114  # reuse
        buf184 = reinterpret_tensor(buf113, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf113  # reuse
        buf186 = reinterpret_tensor(buf112, (512, ), (1, ), 0); del buf112  # reuse
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf180, buf181, buf182, primals_131, primals_132, buf183, buf184, buf186, primals_131, primals_132, 512, 49, grid=grid(512), stream=stream0)
        del buf180
        del buf181
        del buf182
        del primals_131
        del primals_132
        buf187 = reinterpret_tensor(buf178, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf178  # reuse
        # Source Nodes: [x_80, x_84], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_39.run(buf179, buf183, buf184, primals_25, primals_26, buf187, 3211264, grid=grid(3211264), stream=stream0)
        del buf184
        del primals_26
        buf188 = empty_strided((8, 512, 1, 1, 7), (3584, 1, 28672, 28672, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_2], Original ATen: [aten.mean]
        triton_red_fused_mean_40.run(buf187, buf188, 28672, 112, grid=grid(28672), stream=stream0)
        buf189 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf190 = reinterpret_tensor(buf189, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf189  # reuse
        # Source Nodes: [x_se_2], Original ATen: [aten.mean]
        triton_per_fused_mean_41.run(buf190, buf188, 4096, 7, grid=grid(4096), stream=stream0)
        del buf188
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 512, 1, 1), (512, 1, 1, 1))
        buf192 = empty_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_gate, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_42.run(buf191, primals_71, buf192, 4096, grid=grid(4096), stream=stream0)
        buf193 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.mul]
        triton_poi_fused_mul_43.run(buf187, buf192, buf193, 3211264, grid=grid(3211264), stream=stream0)
        buf194 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        buf195 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_87], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_44.run(buf193, buf194, buf195, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf194, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf197 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf196, buf197, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf198 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        buf199 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        buf200 = empty_strided((1, 192, 1, 1, 13), (2496, 1, 2496, 2496, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf197, buf198, buf199, buf200, 2496, 121, grid=grid(2496), stream=stream0)
        buf201 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf202 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf204 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf198, buf199, buf200, primals_134, primals_135, buf201, buf202, buf204, primals_134, primals_135, 192, 13, grid=grid(192), stream=stream0)
        del primals_134
        del primals_135
        buf205 = reinterpret_tensor(buf196, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf196  # reuse
        # Source Nodes: [x_89, x_93], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf197, buf201, buf202, primals_27, primals_28, buf205, 301056, grid=grid(301056), stream=stream0)
        del primals_28
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf206, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf207 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf206, buf207, 1536, 196, grid=grid(1536, 196), stream=stream0)
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf209 = reinterpret_tensor(buf206, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf206  # reuse
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf208, buf209, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf210 = buf200; del buf200  # reuse
        buf211 = buf199; del buf199  # reuse
        buf212 = buf198; del buf198  # reuse
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf209, buf210, buf211, buf212, 2496, 121, grid=grid(2496), stream=stream0)
        buf213 = buf202; del buf202  # reuse
        buf214 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf216 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf210, buf211, buf212, primals_137, primals_138, buf213, buf214, buf216, primals_137, primals_138, 192, 13, grid=grid(192), stream=stream0)
        del primals_137
        del primals_138
        buf217 = reinterpret_tensor(buf208, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf208  # reuse
        buf245 = empty_strided((8, 1088, 14, 14), (213248, 1, 15232, 1088), device='cuda', dtype=torch.float32)
        buf243 = reinterpret_tensor(buf245, (8, 192, 14, 14), (213248, 1, 15232, 1088), 512)  # alias
        # Source Nodes: [cat_5, x_96, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_49.run(buf209, buf213, buf214, primals_29, primals_30, buf217, buf243, 301056, grid=grid(301056), stream=stream0)
        del primals_30
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf218, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf219 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf218, buf219, 1536, 196, grid=grid(1536, 196), stream=stream0)
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf221 = reinterpret_tensor(buf218, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf218  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf220, buf221, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf222 = buf212; del buf212  # reuse
        buf223 = buf211; del buf211  # reuse
        buf224 = buf210; del buf210  # reuse
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf221, buf222, buf223, buf224, 2496, 121, grid=grid(2496), stream=stream0)
        buf225 = buf214; del buf214  # reuse
        buf226 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf228 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf222, buf223, buf224, primals_140, primals_141, buf225, buf226, buf228, primals_140, primals_141, 192, 13, grid=grid(192), stream=stream0)
        del primals_140
        del primals_141
        buf229 = reinterpret_tensor(buf220, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf220  # reuse
        buf244 = reinterpret_tensor(buf245, (8, 192, 14, 14), (213248, 1, 15232, 1088), 704)  # alias
        # Source Nodes: [cat_5, x_102, x_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_49.run(buf221, buf225, buf226, primals_31, primals_32, buf229, buf244, 301056, grid=grid(301056), stream=stream0)
        del primals_32
        # Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf230, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf231 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf230, buf231, 1536, 196, grid=grid(1536, 196), stream=stream0)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 192, 14, 14), (37632, 196, 14, 1))
        buf233 = reinterpret_tensor(buf230, (8, 192, 14, 14), (37632, 1, 2688, 192), 0); del buf230  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf232, buf233, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf234 = buf224; del buf224  # reuse
        buf235 = buf223; del buf223  # reuse
        buf236 = buf222; del buf222  # reuse
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf233, buf234, buf235, buf236, 2496, 121, grid=grid(2496), stream=stream0)
        buf237 = buf226; del buf226  # reuse
        buf238 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf240 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf234, buf235, buf236, primals_143, primals_144, buf237, buf238, buf240, primals_143, primals_144, 192, 13, grid=grid(192), stream=stream0)
        del buf234
        del buf235
        del buf236
        del primals_143
        del primals_144
        buf241 = reinterpret_tensor(buf245, (8, 192, 14, 14), (213248, 1, 15232, 1088), 896)  # alias
        buf334 = empty_strided((8, 192, 14, 14), (37632, 1, 2688, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_108, x_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_50.run(buf233, buf237, buf238, primals_33, primals_34, buf241, buf334, 301056, grid=grid(301056), stream=stream0)
        del buf238
        del primals_34
        buf242 = reinterpret_tensor(buf245, (8, 512, 14, 14), (213248, 1, 15232, 1088), 0)  # alias
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf194, buf242, 4096, 196, grid=grid(4096, 196), stream=stream0)
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 768, 14, 14), (150528, 196, 14, 1))
        buf247 = empty_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_52.run(buf246, buf247, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf248 = empty_strided((1, 768, 1, 1, 13), (9984, 1, 9984, 9984, 768), device='cuda', dtype=torch.float32)
        buf249 = empty_strided((1, 768, 1, 1, 13), (9984, 1, 9984, 9984, 768), device='cuda', dtype=torch.float32)
        buf250 = empty_strided((1, 768, 1, 1, 13), (9984, 1, 9984, 9984, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf247, buf248, buf249, buf250, 9984, 121, grid=grid(9984), stream=stream0)
        buf251 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf252 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf254 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_54.run(buf248, buf249, buf250, primals_146, primals_147, buf251, buf252, buf254, primals_146, primals_147, 768, 13, grid=grid(768), stream=stream0)
        del buf248
        del buf249
        del buf250
        del primals_146
        del primals_147
        buf255 = reinterpret_tensor(buf246, (8, 768, 14, 14), (150528, 1, 10752, 768), 0); del buf246  # reuse
        # Source Nodes: [x_115, x_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf247, buf251, buf252, primals_35, primals_36, buf255, 1204224, grid=grid(1204224), stream=stream0)
        del buf252
        del primals_36
        buf256 = empty_strided((8, 768, 1, 1, 2), (1536, 1, 12288, 12288, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_red_fused_mean_56.run(buf255, buf256, 12288, 98, grid=grid(12288), stream=stream0)
        buf257 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf258 = reinterpret_tensor(buf257, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf257  # reuse
        # Source Nodes: [x_se_4], Original ATen: [aten.mean]
        triton_per_fused_mean_57.run(buf258, buf256, 6144, 2, grid=grid(6144), stream=stream0)
        del buf256
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 768, 1, 1), (768, 1, 1, 1))
        buf260 = empty_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_gate, x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid]
        triton_poi_fused_convolution_hardsigmoid_58.run(buf259, primals_81, buf260, 6144, grid=grid(6144), stream=stream0)
        buf261 = empty_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten.mul]
        triton_poi_fused_mul_59.run(buf255, buf260, buf261, 1204224, grid=grid(1204224), stream=stream0)
        buf262 = reinterpret_tensor(buf232, (8, 768, 7, 7), (37632, 1, 5376, 768), 0); del buf232  # reuse
        buf263 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_122], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_60.run(buf261, buf262, buf263, 301056, grid=grid(301056), stream=stream0)
        # Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf262, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf265 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf264, buf265, 1792, 49, grid=grid(1792, 49), stream=stream0)
        buf266 = empty_strided((1, 224, 1, 1, 4), (896, 1, 896, 896, 224), device='cuda', dtype=torch.float32)
        buf267 = empty_strided((1, 224, 1, 1, 4), (896, 1, 896, 896, 224), device='cuda', dtype=torch.float32)
        buf268 = empty_strided((1, 224, 1, 1, 4), (896, 1, 896, 896, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf265, buf266, buf267, buf268, 896, 98, grid=grid(896), stream=stream0)
        buf269 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf270 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf272 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf266, buf267, buf268, primals_149, primals_150, buf269, buf270, buf272, primals_149, primals_150, 224, 4, grid=grid(224), stream=stream0)
        del primals_149
        del primals_150
        buf273 = reinterpret_tensor(buf264, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf264  # reuse
        # Source Nodes: [x_124, x_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_64.run(buf265, buf269, buf270, primals_37, primals_38, buf273, 87808, grid=grid(87808), stream=stream0)
        del primals_38
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf274, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf275 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf274, buf275, 1792, 49, grid=grid(1792, 49), stream=stream0)
        # Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf277 = reinterpret_tensor(buf274, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf274  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf276, buf277, 1792, 49, grid=grid(1792, 49), stream=stream0)
        buf278 = buf268; del buf268  # reuse
        buf279 = buf267; del buf267  # reuse
        buf280 = buf266; del buf266  # reuse
        # Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf277, buf278, buf279, buf280, 896, 98, grid=grid(896), stream=stream0)
        buf281 = buf270; del buf270  # reuse
        buf282 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf284 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf278, buf279, buf280, primals_152, primals_153, buf281, buf282, buf284, primals_152, primals_153, 224, 4, grid=grid(224), stream=stream0)
        del primals_152
        del primals_153
        buf285 = reinterpret_tensor(buf276, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf276  # reuse
        buf313 = empty_strided((8, 1440, 7, 7), (70560, 1, 10080, 1440), device='cuda', dtype=torch.float32)
        buf311 = reinterpret_tensor(buf313, (8, 224, 7, 7), (70560, 1, 10080, 1440), 768)  # alias
        # Source Nodes: [cat_4, x_131, x_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_65.run(buf277, buf281, buf282, primals_39, primals_40, buf285, buf311, 87808, grid=grid(87808), stream=stream0)
        del primals_40
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf286, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf287 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf286, buf287, 1792, 49, grid=grid(1792, 49), stream=stream0)
        # Source Nodes: [x_136], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf289 = reinterpret_tensor(buf286, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf286  # reuse
        # Source Nodes: [x_136], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf288, buf289, 1792, 49, grid=grid(1792, 49), stream=stream0)
        buf290 = buf280; del buf280  # reuse
        buf291 = buf279; del buf279  # reuse
        buf292 = buf278; del buf278  # reuse
        # Source Nodes: [x_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf289, buf290, buf291, buf292, 896, 98, grid=grid(896), stream=stream0)
        buf293 = buf282; del buf282  # reuse
        buf294 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf296 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf290, buf291, buf292, primals_155, primals_156, buf293, buf294, buf296, primals_155, primals_156, 224, 4, grid=grid(224), stream=stream0)
        del primals_155
        del primals_156
        buf297 = reinterpret_tensor(buf288, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf288  # reuse
        buf312 = reinterpret_tensor(buf313, (8, 224, 7, 7), (70560, 1, 10080, 1440), 992)  # alias
        # Source Nodes: [cat_4, x_137, x_140], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_65.run(buf289, buf293, buf294, primals_41, primals_42, buf297, buf312, 87808, grid=grid(87808), stream=stream0)
        del primals_42
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf298, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf299 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf298, buf299, 1792, 49, grid=grid(1792, 49), stream=stream0)
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 224, 7, 7), (10976, 49, 7, 1))
        buf301 = reinterpret_tensor(buf298, (8, 224, 7, 7), (10976, 1, 1568, 224), 0); del buf298  # reuse
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf300, buf301, 1792, 49, grid=grid(1792, 49), stream=stream0)
        del buf300
        buf302 = buf292; del buf292  # reuse
        buf303 = buf291; del buf291  # reuse
        buf304 = buf290; del buf290  # reuse
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf301, buf302, buf303, buf304, 896, 98, grid=grid(896), stream=stream0)
        buf305 = buf294; del buf294  # reuse
        buf306 = empty_strided((1, 224, 1, 1), (224, 1, 224, 224), device='cuda', dtype=torch.float32)
        buf308 = empty((224, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_63.run(buf302, buf303, buf304, primals_158, primals_159, buf305, buf306, buf308, primals_158, primals_159, 224, 4, grid=grid(224), stream=stream0)
        del buf302
        del buf303
        del buf304
        del primals_158
        del primals_159
        buf309 = reinterpret_tensor(buf313, (8, 224, 7, 7), (70560, 1, 10080, 1440), 1216)  # alias
        buf332 = empty_strided((8, 224, 7, 7), (10976, 1, 1568, 224), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_143, x_146], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_66.run(buf301, buf305, buf306, primals_43, primals_44, buf309, buf332, 87808, grid=grid(87808), stream=stream0)
        del buf306
        del primals_44
        buf310 = reinterpret_tensor(buf313, (8, 768, 7, 7), (70560, 1, 10080, 1440), 0)  # alias
        # Source Nodes: [cat_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_67.run(buf262, buf310, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf315 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_68.run(buf314, buf315, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf316 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        buf317 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        buf318 = empty_strided((1, 1024, 1, 1, 4), (4096, 1, 4096, 4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_69.run(buf315, buf316, buf317, buf318, 4096, 98, grid=grid(4096), stream=stream0)
        buf319 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf320 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf322 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_70.run(buf316, buf317, buf318, primals_161, primals_162, buf319, buf320, buf322, primals_161, primals_162, 1024, 4, grid=grid(1024), stream=stream0)
        del buf316
        del buf317
        del buf318
        del primals_161
        del primals_162
        buf323 = reinterpret_tensor(buf314, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf314  # reuse
        # Source Nodes: [x_150, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_71.run(buf315, buf319, buf320, primals_45, primals_46, buf323, 401408, grid=grid(401408), stream=stream0)
        del buf320
        del primals_46
        buf324 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf325 = reinterpret_tensor(buf324, (8, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf324  # reuse
        # Source Nodes: [x_se_6], Original ATen: [aten.mean]
        triton_per_fused_mean_72.run(buf325, buf323, 8192, 49, grid=grid(8192), stream=stream0)
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 1024, 1, 1), (1024, 1, 1, 1))
        buf327 = empty_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf331 = empty_strided((8, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.bool)
        buf328 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf329 = reinterpret_tensor(buf328, (8, 1024), (1024, 1), 0); del buf328  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3___blocks___0___attn_gate, x_157, x_158, x_160, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardsigmoid_backward, aten.mean, aten.mul, aten.view]
        triton_per_fused_convolution_hardsigmoid_hardsigmoid_backward_mean_mul_view_73.run(buf329, buf326, primals_91, buf323, buf327, buf331, 8192, 49, grid=grid(8192), stream=stream0)
        del buf326
        del primals_91
        buf330 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_93, buf329, reinterpret_tensor(primals_92, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf330)
        del primals_93
        buf333 = empty_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_74.run(buf259, primals_81, buf333, 6144, grid=grid(6144), stream=stream0)
        del buf259
        del primals_81
        buf335 = empty_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_75.run(buf191, primals_71, buf335, 4096, grid=grid(4096), stream=stream0)
        del buf191
        del primals_71
        buf337 = empty_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution, aten.hardsigmoid_backward]
        triton_poi_fused_convolution_hardsigmoid_backward_76.run(buf123, primals_61, buf337, 2048, grid=grid(2048), stream=stream0)
        del buf123
        del primals_61
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_94, primals_94, 1, grid=grid(1), stream=stream0)
        del primals_94
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_97, primals_97, 1, grid=grid(1), stream=stream0)
        del primals_97
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_100, primals_100, 1, grid=grid(1), stream=stream0)
        del primals_100
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_103, primals_103, 1, grid=grid(1), stream=stream0)
        del primals_103
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_106, primals_106, 1, grid=grid(1), stream=stream0)
        del primals_106
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_109, primals_109, 1, grid=grid(1), stream=stream0)
        del primals_109
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_112, primals_112, 1, grid=grid(1), stream=stream0)
        del primals_112
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_115, primals_115, 1, grid=grid(1), stream=stream0)
        del primals_115
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_118, primals_118, 1, grid=grid(1), stream=stream0)
        del primals_118
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_121, primals_121, 1, grid=grid(1), stream=stream0)
        del primals_121
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_124, primals_124, 1, grid=grid(1), stream=stream0)
        del primals_124
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_127, primals_127, 1, grid=grid(1), stream=stream0)
        del primals_127
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_130, primals_130, 1, grid=grid(1), stream=stream0)
        del primals_130
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_133, primals_133, 1, grid=grid(1), stream=stream0)
        del primals_133
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_136, primals_136, 1, grid=grid(1), stream=stream0)
        del primals_136
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_139, primals_139, 1, grid=grid(1), stream=stream0)
        del primals_139
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_142, primals_142, 1, grid=grid(1), stream=stream0)
        del primals_142
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_145, primals_145, 1, grid=grid(1), stream=stream0)
        del primals_145
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_148, primals_148, 1, grid=grid(1), stream=stream0)
        del primals_148
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_151, primals_151, 1, grid=grid(1), stream=stream0)
        del primals_151
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_154, primals_154, 1, grid=grid(1), stream=stream0)
        del primals_154
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_157, primals_157, 1, grid=grid(1), stream=stream0)
        del primals_157
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(primals_160, primals_160, 1, grid=grid(1), stream=stream0)
        del primals_160
        return (buf330, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, buf0, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, buf1, buf3, buf13, buf14, buf16, buf18, buf28, buf29, buf31, buf33, buf43, buf44, buf46, buf56, buf57, buf59, buf61, buf71, buf72, buf74, buf76, buf86, buf87, buf89, buf91, buf101, buf106, buf108, buf118, buf119, buf122, buf124, buf125, buf126, buf127, buf129, buf136, buf137, buf139, buf141, buf148, buf149, buf151, buf153, buf160, buf161, buf163, buf165, buf172, buf177, buf179, buf186, buf187, buf190, buf192, buf193, buf194, buf195, buf197, buf204, buf205, buf207, buf209, buf216, buf217, buf219, buf221, buf228, buf229, buf231, buf233, buf240, buf245, buf247, buf254, buf255, buf258, buf260, buf261, buf262, buf263, buf265, buf272, buf273, buf275, buf277, buf284, buf285, buf287, buf289, buf296, buf297, buf299, buf301, buf308, buf313, buf315, buf322, buf323, buf325, buf327, buf329, reinterpret_tensor(primals_92, (1000, 1024), (1024, 1), 0), buf331, reinterpret_tensor(buf319, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf332, reinterpret_tensor(buf305, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf293, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf281, (1, 224, 1, 1), (224, 1, 1, 1), 0), reinterpret_tensor(buf269, (1, 224, 1, 1), (224, 1, 1, 1), 0), buf333, reinterpret_tensor(buf251, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf334, reinterpret_tensor(buf237, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf225, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf213, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf201, (1, 192, 1, 1), (192, 1, 1, 1), 0), buf335, reinterpret_tensor(buf183, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf336, reinterpret_tensor(buf169, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf157, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf145, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf133, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf337, reinterpret_tensor(buf115, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf338, reinterpret_tensor(buf98, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf83, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf68, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf53, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf40, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf25, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_119 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_122 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_125 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_128 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_137 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_140 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_143 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_149 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_152 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_155 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_158 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ese_vovnet19b_dw', benchmark_compiled_module)
