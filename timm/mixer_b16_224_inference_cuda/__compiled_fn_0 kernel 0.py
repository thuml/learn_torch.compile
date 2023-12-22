
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


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnrztk2rmko3dhpusxqdwpejdm6pm73v2hll3xrjsuhqd7kvkpg.py
# Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm1 => clone, var_mean
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3ytpllgumpktjg2udvyyddv43mojt5c7bwcqnkrf22voxl2cjq.py
# Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm1 => clone, var_mean
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmckljck7jwituilw5muari3mkglsjjvibbt7t6zwdv4hnur5orf.py
# Source Nodes: [x_4], Original ATen: [aten.clone]
# x_4 => clone_1
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 768
    x0 = xindex % 196
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnutdt672ezznzwqienemgtcuadvopztl3dep5reihmppy7nljxg.py
# Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.gelu]
# x_4 => add_2
# x_5 => add_3, erf, mul_2, mul_3, mul_4
triton_poi_fused_add_gelu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cerhehodookplcrwao6meuh7x2nj7tm4vg6q5uh2mkwaoenmplyk.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => clone_4, var_mean_1
# x_10 => add_4
triton_red_fused_add_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpcwqpkbgrwqkm6mxpyzmuuvevcowb53sxrryrfkdu5pawdv6m2.py
# Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm2 => add_5, add_6, clone_4, mul_5, mul_6, rsqrt_1, sub_1, var_mean_1
# x_10 => add_4
triton_poi_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 768.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsr5ewhibzglehqjyybmee3lxg5qb7zmk2n4aqeltwn62alhwpl.py
# Source Nodes: [x_12], Original ATen: [aten.gelu]
# x_12 => add_7, erf_1, mul_7, mul_8, mul_9
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/crolsylfoymetnctl67mjicg5objg3q34a6xh54473gugeitewnx.py
# Source Nodes: [x_10, x_17], Original ATen: [aten.add]
# x_10 => add_4
# x_17 => add_8
triton_poi_fused_add_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wk/cwke25j6rjqkbvsvchut7btitng63deuok4aofmtqmi7yfoicboq.py
# Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm1 => clone_7, var_mean_2
triton_red_fused_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 196
    x2 = (xindex // 1176)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (196*x0) + (1176*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (196*x0) + (1176*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (196*x0) + (1176*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvl6opf6p6337zbmwfrdbudjhjy52yqmgbgvo6vx4mfddnkihsoz.py
# Source Nodes: [x_18], Original ATen: [aten.clone]
# x_18 => clone_8
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 196
    x2 = (xindex // 150528)
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (196*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pl/cplme4ovgwvu7bvgbvilhu2fybetolgmood74pqnvb4dg6ubllkt.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm2 => clone_11, var_mean_3
# x_24 => add_13
triton_red_fused_add_native_layer_norm_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
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
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgx3kdmbgxmttror5hc727wcfa3de5pnrmp6jullpffdxjgy474f.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm2 => add_14, add_15, clone_11, mul_15, mul_16, rsqrt_3, sub_3, var_mean_3
# x_24 => add_13
triton_poi_fused_add_native_layer_norm_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvuusw3eihvdnsbgxqwaayn47s3kto75tqov5jawbpwhoxjsyjh.py
# Source Nodes: [x_24, x_31], Original ATen: [aten.add]
# x_24 => add_13
# x_31 => add_17
triton_poi_fused_add_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_out_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (196*x2) + (150528*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coa6d3j3qhstwsxobrna5nyn64qstcx5la5edaxkhgtcuxnlkg76.py
# Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___2___norm1 => clone_14, var_mean_4
triton_red_fused_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs32zwksbugzu3ravn6fpmjl43ocrcafxq3dpxmc46oi53jq32nc.py
# Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___2___norm2 => add_23, add_24, clone_18, mul_25, mul_26, rsqrt_5, sub_5, var_mean_5
# x_38 => add_22
triton_poi_fused_add_native_layer_norm_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y1 = (yindex // 768)
    y0 = yindex % 768
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (196*y1)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 768.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + (768*x2) + (150528*y1)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqizfmn7zpr27rv3irsjmw4mny2zds4w5a5g6fl6qcp563qdqxqe.py
# Source Nodes: [x_38, x_45], Original ATen: [aten.add]
# x_38 => add_22
# x_45 => add_26
triton_poi_fused_add_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dk/cdk7pngk75l3k2bdhea2372kqwwvqt6nh5nx6i6tp5lozsfqf7hx.py
# Source Nodes: [x_174, x_175], Original ATen: [aten.mean, aten.native_layer_norm]
# x_174 => add_108, add_109, mul_120, mul_121, rsqrt_24, sub_24, var_mean_24
# x_175 => mean
triton_per_fused_mean_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 768)
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (196*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (196*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 196.0
    tmp19 = tmp17 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (384, 196), (196, 1))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (196, 384), (384, 1))
    assert_size_stride(arg7_1, (196, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (3072, 768), (768, 1))
    assert_size_stride(arg11_1, (3072, ), (1, ))
    assert_size_stride(arg12_1, (768, 3072), (3072, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (384, 196), (196, 1))
    assert_size_stride(arg17_1, (384, ), (1, ))
    assert_size_stride(arg18_1, (196, 384), (384, 1))
    assert_size_stride(arg19_1, (196, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (3072, 768), (768, 1))
    assert_size_stride(arg23_1, (3072, ), (1, ))
    assert_size_stride(arg24_1, (768, 3072), (3072, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (384, 196), (196, 1))
    assert_size_stride(arg29_1, (384, ), (1, ))
    assert_size_stride(arg30_1, (196, 384), (384, 1))
    assert_size_stride(arg31_1, (196, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (3072, 768), (768, 1))
    assert_size_stride(arg35_1, (3072, ), (1, ))
    assert_size_stride(arg36_1, (768, 3072), (3072, 1))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (384, 196), (196, 1))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (196, 384), (384, 1))
    assert_size_stride(arg43_1, (196, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (3072, 768), (768, 1))
    assert_size_stride(arg47_1, (3072, ), (1, ))
    assert_size_stride(arg48_1, (768, 3072), (3072, 1))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (384, 196), (196, 1))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (196, 384), (384, 1))
    assert_size_stride(arg55_1, (196, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (3072, 768), (768, 1))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (768, 3072), (3072, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (384, 196), (196, 1))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (196, 384), (384, 1))
    assert_size_stride(arg67_1, (196, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (384, 196), (196, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (196, 384), (384, 1))
    assert_size_stride(arg79_1, (196, ), (1, ))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (3072, 768), (768, 1))
    assert_size_stride(arg83_1, (3072, ), (1, ))
    assert_size_stride(arg84_1, (768, 3072), (3072, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (384, 196), (196, 1))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (196, 384), (384, 1))
    assert_size_stride(arg91_1, (196, ), (1, ))
    assert_size_stride(arg92_1, (768, ), (1, ))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (384, 196), (196, 1))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (196, 384), (384, 1))
    assert_size_stride(arg103_1, (196, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (3072, 768), (768, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (768, 3072), (3072, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (384, 196), (196, 1))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (196, 384), (384, 1))
    assert_size_stride(arg115_1, (196, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (3072, 768), (768, 1))
    assert_size_stride(arg119_1, (3072, ), (1, ))
    assert_size_stride(arg120_1, (768, 3072), (3072, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (384, 196), (196, 1))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (196, 384), (384, 1))
    assert_size_stride(arg127_1, (196, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (384, 196), (196, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (196, 384), (384, 1))
    assert_size_stride(arg139_1, (196, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (1000, 768), (768, 1))
    assert_size_stride(arg149_1, (1000, ), (1, ))
    assert_size_stride(arg150_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg150_1, arg0_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        del arg0_1
        del arg150_1
        buf1 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg1_1, buf1, buf2, buf3, 9408, 128, grid=grid(9408), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 1568, 6, grid=grid(1568), stream=stream0)
        buf7 = empty((8, 768, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf0, arg1_1, buf4, buf5, arg2_1, arg3_1, buf7, 1204224, grid=grid(1204224), stream=stream0)
        del arg2_1
        del arg3_1
        buf8 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (6144, 196), (196, 1), 0), reinterpret_tensor(arg4_1, (196, 384), (1, 196), 0), out=buf8)
        del arg4_1
        buf9 = reinterpret_tensor(buf8, (8, 768, 384), (294912, 384, 1), 0); del buf8  # reuse
        # Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf9, arg5_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg5_1
        buf10 = reinterpret_tensor(buf7, (6144, 196), (196, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (6144, 384), (384, 1), 0), reinterpret_tensor(arg6_1, (384, 196), (1, 384), 0), out=buf10)
        del arg6_1
        buf11 = buf3; del buf3  # reuse
        buf12 = buf2; del buf2  # reuse
        buf13 = buf1; del buf1  # reuse
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_4.run(buf0, arg1_1, buf10, arg7_1, buf11, buf12, buf13, 9408, 128, grid=grid(9408), stream=stream0)
        buf14 = buf5; del buf5  # reuse
        buf15 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf11, buf12, buf13, buf14, buf15, 1568, 6, grid=grid(1568), stream=stream0)
        buf17 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm2, x_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_5.run(buf0, arg1_1, buf10, arg7_1, buf14, buf15, arg8_1, arg9_1, buf17, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg8_1
        del arg9_1
        buf18 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (1568, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 3072), (1, 768), 0), out=buf18)
        del arg10_1
        buf19 = reinterpret_tensor(buf18, (8, 196, 3072), (602112, 3072, 1), 0); del buf18  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf19, arg11_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg11_1
        buf20 = reinterpret_tensor(buf17, (1568, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg12_1, (3072, 768), (1, 3072), 0), out=buf20)
        del arg12_1
        buf21 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 1, 196), 0); del buf0  # reuse
        # Source Nodes: [x_10, x_17], Original ATen: [aten.add]
        triton_poi_fused_add_7.run(buf21, arg1_1, buf10, arg7_1, buf20, arg13_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg13_1
        del arg1_1
        del arg7_1
        buf22 = buf13; del buf13  # reuse
        buf23 = buf12; del buf12  # reuse
        buf24 = buf11; del buf11  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf21, buf22, buf23, buf24, 9408, 128, grid=grid(9408), stream=stream0)
        buf25 = buf15; del buf15  # reuse
        buf26 = buf14; del buf14  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf22, buf23, buf24, buf25, buf26, 1568, 6, grid=grid(1568), stream=stream0)
        buf28 = reinterpret_tensor(buf20, (8, 768, 196), (150528, 196, 1), 0); del buf20  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf21, buf25, buf26, arg14_1, arg15_1, buf28, 1204224, grid=grid(1204224), stream=stream0)
        del arg14_1
        del arg15_1
        buf29 = reinterpret_tensor(buf9, (6144, 384), (384, 1), 0); del buf9  # reuse
        # Source Nodes: [x_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (6144, 196), (196, 1), 0), reinterpret_tensor(arg16_1, (196, 384), (1, 196), 0), out=buf29)
        del arg16_1
        buf30 = reinterpret_tensor(buf29, (8, 768, 384), (294912, 384, 1), 0); del buf29  # reuse
        # Source Nodes: [x_18, x_19], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf30, arg17_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg17_1
        buf31 = reinterpret_tensor(buf28, (6144, 196), (196, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf30, (6144, 384), (384, 1), 0), reinterpret_tensor(arg18_1, (384, 196), (1, 384), 0), out=buf31)
        del arg18_1
        buf32 = buf24; del buf24  # reuse
        buf33 = buf23; del buf23  # reuse
        buf34 = buf22; del buf22  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf21, buf31, arg19_1, buf32, buf33, buf34, 9408, 128, grid=grid(9408), stream=stream0)
        buf35 = buf26; del buf26  # reuse
        buf36 = buf25; del buf25  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf32, buf33, buf34, buf35, buf36, 1568, 6, grid=grid(1568), stream=stream0)
        buf38 = reinterpret_tensor(buf10, (8, 196, 768), (150528, 768, 1), 0); del buf10  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf21, buf31, arg19_1, buf35, buf36, arg20_1, arg21_1, buf38, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg20_1
        del arg21_1
        buf39 = reinterpret_tensor(buf19, (1568, 3072), (3072, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf38, (1568, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 3072), (1, 768), 0), out=buf39)
        del arg22_1
        buf40 = reinterpret_tensor(buf39, (8, 196, 3072), (602112, 3072, 1), 0); del buf39  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf40, arg23_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg23_1
        buf41 = reinterpret_tensor(buf38, (1568, 768), (768, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg24_1, (3072, 768), (1, 3072), 0), out=buf41)
        del arg24_1
        buf42 = buf21; del buf21  # reuse
        # Source Nodes: [x_24, x_31], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf42, buf31, arg19_1, buf41, arg25_1, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg19_1
        del arg25_1
        buf43 = buf34; del buf34  # reuse
        buf44 = buf33; del buf33  # reuse
        buf45 = buf32; del buf32  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf42, buf43, buf44, buf45, 9408, 128, grid=grid(9408), stream=stream0)
        buf46 = buf36; del buf36  # reuse
        buf47 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf43, buf44, buf45, buf46, buf47, 1568, 6, grid=grid(1568), stream=stream0)
        buf49 = reinterpret_tensor(buf41, (8, 768, 196), (150528, 196, 1), 0); del buf41  # reuse
        # Source Nodes: [x_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf42, buf46, buf47, arg26_1, arg27_1, buf49, 1204224, grid=grid(1204224), stream=stream0)
        del arg26_1
        del arg27_1
        buf50 = reinterpret_tensor(buf30, (6144, 384), (384, 1), 0); del buf30  # reuse
        # Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (6144, 196), (196, 1), 0), reinterpret_tensor(arg28_1, (196, 384), (1, 196), 0), out=buf50)
        del arg28_1
        buf51 = reinterpret_tensor(buf50, (8, 768, 384), (294912, 384, 1), 0); del buf50  # reuse
        # Source Nodes: [x_32, x_33], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf51, arg29_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg29_1
        buf52 = reinterpret_tensor(buf49, (6144, 196), (196, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (6144, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 196), (1, 384), 0), out=buf52)
        del arg30_1
        buf53 = buf45; del buf45  # reuse
        buf54 = buf44; del buf44  # reuse
        buf55 = buf43; del buf43  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf42, buf52, arg31_1, buf53, buf54, buf55, 9408, 128, grid=grid(9408), stream=stream0)
        buf56 = buf47; del buf47  # reuse
        buf57 = buf46; del buf46  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf53, buf54, buf55, buf56, buf57, 1568, 6, grid=grid(1568), stream=stream0)
        buf59 = reinterpret_tensor(buf31, (8, 196, 768), (150528, 768, 1), 0); del buf31  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_14.run(buf42, buf52, arg31_1, buf56, buf57, arg32_1, arg33_1, buf59, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg32_1
        del arg33_1
        buf60 = reinterpret_tensor(buf40, (1568, 3072), (3072, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf59, (1568, 768), (768, 1), 0), reinterpret_tensor(arg34_1, (768, 3072), (1, 768), 0), out=buf60)
        del arg34_1
        buf61 = reinterpret_tensor(buf60, (8, 196, 3072), (602112, 3072, 1), 0); del buf60  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf61, arg35_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg35_1
        buf62 = reinterpret_tensor(buf59, (1568, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg36_1, (3072, 768), (1, 3072), 0), out=buf62)
        del arg36_1
        buf63 = buf42; del buf42  # reuse
        # Source Nodes: [x_38, x_45], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf63, buf52, arg31_1, buf62, arg37_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg31_1
        del arg37_1
        buf64 = buf55; del buf55  # reuse
        buf65 = buf54; del buf54  # reuse
        buf66 = buf53; del buf53  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf63, buf64, buf65, buf66, 9408, 128, grid=grid(9408), stream=stream0)
        buf67 = buf57; del buf57  # reuse
        buf68 = buf56; del buf56  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf64, buf65, buf66, buf67, buf68, 1568, 6, grid=grid(1568), stream=stream0)
        buf70 = reinterpret_tensor(buf62, (8, 768, 196), (150528, 196, 1), 0); del buf62  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf63, buf67, buf68, arg38_1, arg39_1, buf70, 1204224, grid=grid(1204224), stream=stream0)
        del arg38_1
        del arg39_1
        buf71 = reinterpret_tensor(buf51, (6144, 384), (384, 1), 0); del buf51  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (6144, 196), (196, 1), 0), reinterpret_tensor(arg40_1, (196, 384), (1, 196), 0), out=buf71)
        del arg40_1
        buf72 = reinterpret_tensor(buf71, (8, 768, 384), (294912, 384, 1), 0); del buf71  # reuse
        # Source Nodes: [x_46, x_47], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf72, arg41_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg41_1
        buf73 = reinterpret_tensor(buf70, (6144, 196), (196, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (6144, 384), (384, 1), 0), reinterpret_tensor(arg42_1, (384, 196), (1, 384), 0), out=buf73)
        del arg42_1
        buf74 = buf66; del buf66  # reuse
        buf75 = buf65; del buf65  # reuse
        buf76 = buf64; del buf64  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf63, buf73, arg43_1, buf74, buf75, buf76, 9408, 128, grid=grid(9408), stream=stream0)
        buf77 = buf68; del buf68  # reuse
        buf78 = buf67; del buf67  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf74, buf75, buf76, buf77, buf78, 1568, 6, grid=grid(1568), stream=stream0)
        buf80 = reinterpret_tensor(buf52, (8, 196, 768), (150528, 768, 1), 0); del buf52  # reuse
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf63, buf73, arg43_1, buf77, buf78, arg44_1, arg45_1, buf80, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg44_1
        del arg45_1
        buf81 = reinterpret_tensor(buf61, (1568, 3072), (3072, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (1568, 768), (768, 1), 0), reinterpret_tensor(arg46_1, (768, 3072), (1, 768), 0), out=buf81)
        del arg46_1
        buf82 = reinterpret_tensor(buf81, (8, 196, 3072), (602112, 3072, 1), 0); del buf81  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf82, arg47_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg47_1
        buf83 = reinterpret_tensor(buf80, (1568, 768), (768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf82, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg48_1, (3072, 768), (1, 3072), 0), out=buf83)
        del arg48_1
        buf84 = buf63; del buf63  # reuse
        # Source Nodes: [x_52, x_59], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf84, buf73, arg43_1, buf83, arg49_1, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg43_1
        del arg49_1
        buf85 = buf76; del buf76  # reuse
        buf86 = buf75; del buf75  # reuse
        buf87 = buf74; del buf74  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf84, buf85, buf86, buf87, 9408, 128, grid=grid(9408), stream=stream0)
        buf88 = buf78; del buf78  # reuse
        buf89 = buf77; del buf77  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf85, buf86, buf87, buf88, buf89, 1568, 6, grid=grid(1568), stream=stream0)
        buf91 = reinterpret_tensor(buf83, (8, 768, 196), (150528, 196, 1), 0); del buf83  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf84, buf88, buf89, arg50_1, arg51_1, buf91, 1204224, grid=grid(1204224), stream=stream0)
        del arg50_1
        del arg51_1
        buf92 = reinterpret_tensor(buf72, (6144, 384), (384, 1), 0); del buf72  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (6144, 196), (196, 1), 0), reinterpret_tensor(arg52_1, (196, 384), (1, 196), 0), out=buf92)
        del arg52_1
        buf93 = reinterpret_tensor(buf92, (8, 768, 384), (294912, 384, 1), 0); del buf92  # reuse
        # Source Nodes: [x_60, x_61], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf93, arg53_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg53_1
        buf94 = reinterpret_tensor(buf91, (6144, 196), (196, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (6144, 384), (384, 1), 0), reinterpret_tensor(arg54_1, (384, 196), (1, 384), 0), out=buf94)
        del arg54_1
        buf95 = buf87; del buf87  # reuse
        buf96 = buf86; del buf86  # reuse
        buf97 = buf85; del buf85  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf84, buf94, arg55_1, buf95, buf96, buf97, 9408, 128, grid=grid(9408), stream=stream0)
        buf98 = buf89; del buf89  # reuse
        buf99 = buf88; del buf88  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf95, buf96, buf97, buf98, buf99, 1568, 6, grid=grid(1568), stream=stream0)
        buf101 = reinterpret_tensor(buf73, (8, 196, 768), (150528, 768, 1), 0); del buf73  # reuse
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_14.run(buf84, buf94, arg55_1, buf98, buf99, arg56_1, arg57_1, buf101, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg56_1
        del arg57_1
        buf102 = reinterpret_tensor(buf82, (1568, 3072), (3072, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (1568, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 3072), (1, 768), 0), out=buf102)
        del arg58_1
        buf103 = reinterpret_tensor(buf102, (8, 196, 3072), (602112, 3072, 1), 0); del buf102  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf103, arg59_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg59_1
        buf104 = reinterpret_tensor(buf101, (1568, 768), (768, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg60_1, (3072, 768), (1, 3072), 0), out=buf104)
        del arg60_1
        buf105 = buf84; del buf84  # reuse
        # Source Nodes: [x_66, x_73], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf105, buf94, arg55_1, buf104, arg61_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg55_1
        del arg61_1
        buf106 = buf97; del buf97  # reuse
        buf107 = buf96; del buf96  # reuse
        buf108 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf105, buf106, buf107, buf108, 9408, 128, grid=grid(9408), stream=stream0)
        buf109 = buf99; del buf99  # reuse
        buf110 = buf98; del buf98  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf106, buf107, buf108, buf109, buf110, 1568, 6, grid=grid(1568), stream=stream0)
        buf112 = reinterpret_tensor(buf94, (8, 768, 196), (150528, 196, 1), 0); del buf94  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf105, buf109, buf110, arg62_1, arg63_1, buf112, 1204224, grid=grid(1204224), stream=stream0)
        del arg62_1
        del arg63_1
        buf113 = reinterpret_tensor(buf93, (6144, 384), (384, 1), 0); del buf93  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (6144, 196), (196, 1), 0), reinterpret_tensor(arg64_1, (196, 384), (1, 196), 0), out=buf113)
        del arg64_1
        buf114 = reinterpret_tensor(buf113, (8, 768, 384), (294912, 384, 1), 0); del buf113  # reuse
        # Source Nodes: [x_74, x_75], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf114, arg65_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg65_1
        buf115 = reinterpret_tensor(buf112, (6144, 196), (196, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (6144, 384), (384, 1), 0), reinterpret_tensor(arg66_1, (384, 196), (1, 384), 0), out=buf115)
        del arg66_1
        buf116 = buf108; del buf108  # reuse
        buf117 = buf107; del buf107  # reuse
        buf118 = buf106; del buf106  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf105, buf115, arg67_1, buf116, buf117, buf118, 9408, 128, grid=grid(9408), stream=stream0)
        buf119 = buf110; del buf110  # reuse
        buf120 = buf109; del buf109  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf116, buf117, buf118, buf119, buf120, 1568, 6, grid=grid(1568), stream=stream0)
        buf122 = reinterpret_tensor(buf104, (8, 196, 768), (150528, 768, 1), 0); del buf104  # reuse
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf105, buf115, arg67_1, buf119, buf120, arg68_1, arg69_1, buf122, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg68_1
        del arg69_1
        buf123 = reinterpret_tensor(buf103, (1568, 3072), (3072, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), out=buf123)
        del arg70_1
        buf124 = reinterpret_tensor(buf123, (8, 196, 3072), (602112, 3072, 1), 0); del buf123  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf124, arg71_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg71_1
        buf125 = reinterpret_tensor(buf122, (1568, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), out=buf125)
        del arg72_1
        buf126 = buf105; del buf105  # reuse
        # Source Nodes: [x_80, x_87], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf126, buf115, arg67_1, buf125, arg73_1, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg67_1
        del arg73_1
        buf127 = buf118; del buf118  # reuse
        buf128 = buf117; del buf117  # reuse
        buf129 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf126, buf127, buf128, buf129, 9408, 128, grid=grid(9408), stream=stream0)
        buf130 = buf120; del buf120  # reuse
        buf131 = buf119; del buf119  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf127, buf128, buf129, buf130, buf131, 1568, 6, grid=grid(1568), stream=stream0)
        buf133 = reinterpret_tensor(buf125, (8, 768, 196), (150528, 196, 1), 0); del buf125  # reuse
        # Source Nodes: [x_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf126, buf130, buf131, arg74_1, arg75_1, buf133, 1204224, grid=grid(1204224), stream=stream0)
        del arg74_1
        del arg75_1
        buf134 = reinterpret_tensor(buf114, (6144, 384), (384, 1), 0); del buf114  # reuse
        # Source Nodes: [x_88], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (6144, 196), (196, 1), 0), reinterpret_tensor(arg76_1, (196, 384), (1, 196), 0), out=buf134)
        del arg76_1
        buf135 = reinterpret_tensor(buf134, (8, 768, 384), (294912, 384, 1), 0); del buf134  # reuse
        # Source Nodes: [x_88, x_89], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf135, arg77_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg77_1
        buf136 = reinterpret_tensor(buf133, (6144, 196), (196, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (6144, 384), (384, 1), 0), reinterpret_tensor(arg78_1, (384, 196), (1, 384), 0), out=buf136)
        del arg78_1
        buf137 = buf129; del buf129  # reuse
        buf138 = buf128; del buf128  # reuse
        buf139 = buf127; del buf127  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf126, buf136, arg79_1, buf137, buf138, buf139, 9408, 128, grid=grid(9408), stream=stream0)
        buf140 = buf131; del buf131  # reuse
        buf141 = buf130; del buf130  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf137, buf138, buf139, buf140, buf141, 1568, 6, grid=grid(1568), stream=stream0)
        buf143 = reinterpret_tensor(buf115, (8, 196, 768), (150528, 768, 1), 0); del buf115  # reuse
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_14.run(buf126, buf136, arg79_1, buf140, buf141, arg80_1, arg81_1, buf143, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg80_1
        del arg81_1
        buf144 = reinterpret_tensor(buf124, (1568, 3072), (3072, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (1568, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 3072), (1, 768), 0), out=buf144)
        del arg82_1
        buf145 = reinterpret_tensor(buf144, (8, 196, 3072), (602112, 3072, 1), 0); del buf144  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf145, arg83_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg83_1
        buf146 = reinterpret_tensor(buf143, (1568, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg84_1, (3072, 768), (1, 3072), 0), out=buf146)
        del arg84_1
        buf147 = buf126; del buf126  # reuse
        # Source Nodes: [x_101, x_94], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf147, buf136, arg79_1, buf146, arg85_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg79_1
        del arg85_1
        buf148 = buf139; del buf139  # reuse
        buf149 = buf138; del buf138  # reuse
        buf150 = buf137; del buf137  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf147, buf148, buf149, buf150, 9408, 128, grid=grid(9408), stream=stream0)
        buf151 = buf141; del buf141  # reuse
        buf152 = buf140; del buf140  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf148, buf149, buf150, buf151, buf152, 1568, 6, grid=grid(1568), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (8, 768, 196), (150528, 196, 1), 0); del buf146  # reuse
        # Source Nodes: [x_102], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf147, buf151, buf152, arg86_1, arg87_1, buf154, 1204224, grid=grid(1204224), stream=stream0)
        del arg86_1
        del arg87_1
        buf155 = reinterpret_tensor(buf135, (6144, 384), (384, 1), 0); del buf135  # reuse
        # Source Nodes: [x_102], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (6144, 196), (196, 1), 0), reinterpret_tensor(arg88_1, (196, 384), (1, 196), 0), out=buf155)
        del arg88_1
        buf156 = reinterpret_tensor(buf155, (8, 768, 384), (294912, 384, 1), 0); del buf155  # reuse
        # Source Nodes: [x_102, x_103], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf156, arg89_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg89_1
        buf157 = reinterpret_tensor(buf154, (6144, 196), (196, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (6144, 384), (384, 1), 0), reinterpret_tensor(arg90_1, (384, 196), (1, 384), 0), out=buf157)
        del arg90_1
        buf158 = buf150; del buf150  # reuse
        buf159 = buf149; del buf149  # reuse
        buf160 = buf148; del buf148  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf147, buf157, arg91_1, buf158, buf159, buf160, 9408, 128, grid=grid(9408), stream=stream0)
        buf161 = buf152; del buf152  # reuse
        buf162 = buf151; del buf151  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf158, buf159, buf160, buf161, buf162, 1568, 6, grid=grid(1568), stream=stream0)
        buf164 = reinterpret_tensor(buf136, (8, 196, 768), (150528, 768, 1), 0); del buf136  # reuse
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_108], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf147, buf157, arg91_1, buf161, buf162, arg92_1, arg93_1, buf164, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg92_1
        del arg93_1
        buf165 = reinterpret_tensor(buf145, (1568, 3072), (3072, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1568, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), out=buf165)
        del arg94_1
        buf166 = reinterpret_tensor(buf165, (8, 196, 3072), (602112, 3072, 1), 0); del buf165  # reuse
        # Source Nodes: [x_110], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf166, arg95_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg95_1
        buf167 = reinterpret_tensor(buf164, (1568, 768), (768, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), out=buf167)
        del arg96_1
        buf168 = buf147; del buf147  # reuse
        # Source Nodes: [x_108, x_115], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf168, buf157, arg91_1, buf167, arg97_1, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg91_1
        del arg97_1
        buf169 = buf160; del buf160  # reuse
        buf170 = buf159; del buf159  # reuse
        buf171 = buf158; del buf158  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf168, buf169, buf170, buf171, 9408, 128, grid=grid(9408), stream=stream0)
        buf172 = buf162; del buf162  # reuse
        buf173 = buf161; del buf161  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf169, buf170, buf171, buf172, buf173, 1568, 6, grid=grid(1568), stream=stream0)
        buf175 = reinterpret_tensor(buf167, (8, 768, 196), (150528, 196, 1), 0); del buf167  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf168, buf172, buf173, arg98_1, arg99_1, buf175, 1204224, grid=grid(1204224), stream=stream0)
        del arg98_1
        del arg99_1
        buf176 = reinterpret_tensor(buf156, (6144, 384), (384, 1), 0); del buf156  # reuse
        # Source Nodes: [x_116], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (6144, 196), (196, 1), 0), reinterpret_tensor(arg100_1, (196, 384), (1, 196), 0), out=buf176)
        del arg100_1
        buf177 = reinterpret_tensor(buf176, (8, 768, 384), (294912, 384, 1), 0); del buf176  # reuse
        # Source Nodes: [x_116, x_117], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf177, arg101_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg101_1
        buf178 = reinterpret_tensor(buf175, (6144, 196), (196, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (6144, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 196), (1, 384), 0), out=buf178)
        del arg102_1
        buf179 = buf171; del buf171  # reuse
        buf180 = buf170; del buf170  # reuse
        buf181 = buf169; del buf169  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf168, buf178, arg103_1, buf179, buf180, buf181, 9408, 128, grid=grid(9408), stream=stream0)
        buf182 = buf173; del buf173  # reuse
        buf183 = buf172; del buf172  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf179, buf180, buf181, buf182, buf183, 1568, 6, grid=grid(1568), stream=stream0)
        buf185 = reinterpret_tensor(buf157, (8, 196, 768), (150528, 768, 1), 0); del buf157  # reuse
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_14.run(buf168, buf178, arg103_1, buf182, buf183, arg104_1, arg105_1, buf185, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg104_1
        del arg105_1
        buf186 = reinterpret_tensor(buf166, (1568, 3072), (3072, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1568, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 3072), (1, 768), 0), out=buf186)
        del arg106_1
        buf187 = reinterpret_tensor(buf186, (8, 196, 3072), (602112, 3072, 1), 0); del buf186  # reuse
        # Source Nodes: [x_124], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf187, arg107_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg107_1
        buf188 = reinterpret_tensor(buf185, (1568, 768), (768, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg108_1, (3072, 768), (1, 3072), 0), out=buf188)
        del arg108_1
        buf189 = buf168; del buf168  # reuse
        # Source Nodes: [x_122, x_129], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf189, buf178, arg103_1, buf188, arg109_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg103_1
        del arg109_1
        buf190 = buf181; del buf181  # reuse
        buf191 = buf180; del buf180  # reuse
        buf192 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf189, buf190, buf191, buf192, 9408, 128, grid=grid(9408), stream=stream0)
        buf193 = buf183; del buf183  # reuse
        buf194 = buf182; del buf182  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf190, buf191, buf192, buf193, buf194, 1568, 6, grid=grid(1568), stream=stream0)
        buf196 = reinterpret_tensor(buf188, (8, 768, 196), (150528, 196, 1), 0); del buf188  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf189, buf193, buf194, arg110_1, arg111_1, buf196, 1204224, grid=grid(1204224), stream=stream0)
        del arg110_1
        del arg111_1
        buf197 = reinterpret_tensor(buf177, (6144, 384), (384, 1), 0); del buf177  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (6144, 196), (196, 1), 0), reinterpret_tensor(arg112_1, (196, 384), (1, 196), 0), out=buf197)
        del arg112_1
        buf198 = reinterpret_tensor(buf197, (8, 768, 384), (294912, 384, 1), 0); del buf197  # reuse
        # Source Nodes: [x_130, x_131], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf198, arg113_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg113_1
        buf199 = reinterpret_tensor(buf196, (6144, 196), (196, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (6144, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 196), (1, 384), 0), out=buf199)
        del arg114_1
        buf200 = buf192; del buf192  # reuse
        buf201 = buf191; del buf191  # reuse
        buf202 = buf190; del buf190  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf189, buf199, arg115_1, buf200, buf201, buf202, 9408, 128, grid=grid(9408), stream=stream0)
        buf203 = buf194; del buf194  # reuse
        buf204 = buf193; del buf193  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf200, buf201, buf202, buf203, buf204, 1568, 6, grid=grid(1568), stream=stream0)
        buf206 = reinterpret_tensor(buf178, (8, 196, 768), (150528, 768, 1), 0); del buf178  # reuse
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_136], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf189, buf199, arg115_1, buf203, buf204, arg116_1, arg117_1, buf206, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg116_1
        del arg117_1
        buf207 = reinterpret_tensor(buf187, (1568, 3072), (3072, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1568, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 3072), (1, 768), 0), out=buf207)
        del arg118_1
        buf208 = reinterpret_tensor(buf207, (8, 196, 3072), (602112, 3072, 1), 0); del buf207  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf208, arg119_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg119_1
        buf209 = reinterpret_tensor(buf206, (1568, 768), (768, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg120_1, (3072, 768), (1, 3072), 0), out=buf209)
        del arg120_1
        buf210 = buf189; del buf189  # reuse
        # Source Nodes: [x_136, x_143], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf210, buf199, arg115_1, buf209, arg121_1, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg115_1
        del arg121_1
        buf211 = buf202; del buf202  # reuse
        buf212 = buf201; del buf201  # reuse
        buf213 = buf200; del buf200  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf210, buf211, buf212, buf213, 9408, 128, grid=grid(9408), stream=stream0)
        buf214 = buf204; del buf204  # reuse
        buf215 = buf203; del buf203  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf211, buf212, buf213, buf214, buf215, 1568, 6, grid=grid(1568), stream=stream0)
        buf217 = reinterpret_tensor(buf209, (8, 768, 196), (150528, 196, 1), 0); del buf209  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf210, buf214, buf215, arg122_1, arg123_1, buf217, 1204224, grid=grid(1204224), stream=stream0)
        del arg122_1
        del arg123_1
        buf218 = reinterpret_tensor(buf198, (6144, 384), (384, 1), 0); del buf198  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (6144, 196), (196, 1), 0), reinterpret_tensor(arg124_1, (196, 384), (1, 196), 0), out=buf218)
        del arg124_1
        buf219 = reinterpret_tensor(buf218, (8, 768, 384), (294912, 384, 1), 0); del buf218  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf219, arg125_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg125_1
        buf220 = reinterpret_tensor(buf217, (6144, 196), (196, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (6144, 384), (384, 1), 0), reinterpret_tensor(arg126_1, (384, 196), (1, 384), 0), out=buf220)
        del arg126_1
        buf221 = buf213; del buf213  # reuse
        buf222 = buf212; del buf212  # reuse
        buf223 = buf211; del buf211  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf210, buf220, arg127_1, buf221, buf222, buf223, 9408, 128, grid=grid(9408), stream=stream0)
        buf224 = buf215; del buf215  # reuse
        buf225 = buf214; del buf214  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf221, buf222, buf223, buf224, buf225, 1568, 6, grid=grid(1568), stream=stream0)
        buf227 = reinterpret_tensor(buf199, (8, 196, 768), (150528, 768, 1), 0); del buf199  # reuse
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_14.run(buf210, buf220, arg127_1, buf224, buf225, arg128_1, arg129_1, buf227, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg128_1
        del arg129_1
        buf228 = reinterpret_tensor(buf208, (1568, 3072), (3072, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf227, (1568, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), out=buf228)
        del arg130_1
        buf229 = reinterpret_tensor(buf228, (8, 196, 3072), (602112, 3072, 1), 0); del buf228  # reuse
        # Source Nodes: [x_152], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf229, arg131_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg131_1
        buf230 = reinterpret_tensor(buf227, (1568, 768), (768, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), out=buf230)
        del arg132_1
        buf231 = buf210; del buf210  # reuse
        # Source Nodes: [x_150, x_157], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf231, buf220, arg127_1, buf230, arg133_1, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del arg127_1
        del arg133_1
        buf232 = buf223; del buf223  # reuse
        buf233 = buf222; del buf222  # reuse
        buf234 = buf221; del buf221  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf231, buf232, buf233, buf234, 9408, 128, grid=grid(9408), stream=stream0)
        buf235 = buf225; del buf225  # reuse
        buf236 = buf224; del buf224  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf232, buf233, buf234, buf235, buf236, 1568, 6, grid=grid(1568), stream=stream0)
        buf238 = reinterpret_tensor(buf230, (8, 768, 196), (150528, 196, 1), 0); del buf230  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf231, buf235, buf236, arg134_1, arg135_1, buf238, 1204224, grid=grid(1204224), stream=stream0)
        del arg134_1
        del arg135_1
        buf239 = reinterpret_tensor(buf219, (6144, 384), (384, 1), 0); del buf219  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (6144, 196), (196, 1), 0), reinterpret_tensor(arg136_1, (196, 384), (1, 196), 0), out=buf239)
        del arg136_1
        buf240 = reinterpret_tensor(buf239, (8, 768, 384), (294912, 384, 1), 0); del buf239  # reuse
        # Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu]
        triton_poi_fused_add_gelu_3.run(buf240, arg137_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg137_1
        buf241 = reinterpret_tensor(buf238, (6144, 196), (196, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (6144, 384), (384, 1), 0), reinterpret_tensor(arg138_1, (384, 196), (1, 384), 0), out=buf241)
        del arg138_1
        del buf240
        buf242 = buf234; del buf234  # reuse
        buf243 = buf233; del buf233  # reuse
        buf244 = buf232; del buf232  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_10.run(buf231, buf241, arg139_1, buf242, buf243, buf244, 9408, 128, grid=grid(9408), stream=stream0)
        buf245 = buf236; del buf236  # reuse
        buf246 = buf235; del buf235  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf242, buf243, buf244, buf245, buf246, 1568, 6, grid=grid(1568), stream=stream0)
        buf248 = reinterpret_tensor(buf220, (8, 196, 768), (150528, 768, 1), 0); del buf220  # reuse
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_11.run(buf231, buf241, arg139_1, buf245, buf246, arg140_1, arg141_1, buf248, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg140_1
        del arg141_1
        buf249 = reinterpret_tensor(buf229, (1568, 3072), (3072, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (1568, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), out=buf249)
        del arg142_1
        buf250 = reinterpret_tensor(buf249, (8, 196, 3072), (602112, 3072, 1), 0); del buf249  # reuse
        # Source Nodes: [x_166], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf250, arg143_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg143_1
        buf251 = reinterpret_tensor(buf248, (1568, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), out=buf251)
        del arg144_1
        del buf250
        buf252 = buf231; del buf231  # reuse
        # Source Nodes: [x_164, x_172, x_174], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_12.run(buf252, buf241, arg139_1, buf251, arg145_1, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del arg139_1
        del arg145_1
        del buf241
        del buf251
        buf253 = buf244; del buf244  # reuse
        buf254 = buf243; del buf243  # reuse
        buf255 = buf242; del buf242  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf252, buf253, buf254, buf255, 9408, 128, grid=grid(9408), stream=stream0)
        buf256 = buf246; del buf246  # reuse
        buf257 = buf245; del buf245  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_1.run(buf253, buf254, buf255, buf256, buf257, 1568, 6, grid=grid(1568), stream=stream0)
        del buf253
        del buf254
        del buf255
        buf259 = empty((8, 768), device='cuda', dtype=torch.float32)
        buf260 = buf259; del buf259  # reuse
        # Source Nodes: [x_174, x_175], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_16.run(buf260, buf252, buf256, buf257, arg146_1, arg147_1, 6144, 196, grid=grid(6144), stream=stream0)
        del arg146_1
        del arg147_1
        del buf252
        del buf256
        del buf257
        buf261 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174, x_175, x_177], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
        extern_kernels.addmm(arg149_1, buf260, reinterpret_tensor(arg148_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf261)
        del arg148_1
        del arg149_1
        return (buf261, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
