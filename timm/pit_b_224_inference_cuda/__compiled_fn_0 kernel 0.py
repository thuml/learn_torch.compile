
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


# kernel path: /tmp/torchinductor_youkaichao/zh/czhiolbbz4l3wxdtyd6tglndssb3a7xaih2p7bwbnueesaxw5vlo.py
# Source Nodes: [cat_5, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_5 => cat
# getattr_l__mod___transformers_0_blocks___0___norm1 => var_mean
triton_red_fused_cat_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 962
    x1 = (xindex // 962) % 2
    x4 = (xindex // 962)
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 962, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((961*r3) + (123008*x4) + (((-1) + x0) % 961)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r3 + (128*x1)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr3 + ((961*r3) + (123008*x1) + (((-1) + x0) % 961)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp8, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp7, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp20, xmask)
    tl.store(out_ptr1 + (x5), tmp21, xmask)
    tl.store(out_ptr2 + (x5), tmp22, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7v7x5npy2qor64o4jghaltgxf6a2th6b2vrsgjqy2pbokjfq4p.py
# Source Nodes: [cat_5, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_5 => cat
# getattr_l__mod___transformers_0_blocks___0___norm1 => var_mean
triton_per_fused_cat_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7696
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 962
    x1 = (xindex // 962)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (962*r2) + (1924*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (962*r2) + (1924*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (962*r2) + (1924*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yeutjts4v3je5fvsuaxgmszq64zlhjabmr4o5ofkxceeoxwwku.py
# Source Nodes: [cat_5, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_5 => cat
# getattr_l__mod___transformers_0_blocks___0___norm1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_cat_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 962
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y3 = yindex
    y1 = (yindex // 256)
    tmp19 = tl.load(in_ptr4 + (x2 + (962*y1)), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2 + (962*y1)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 962, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((961*y3) + (((-1) + x2) % 961)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((961*y0) + (((-1) + x2) % 961)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 256.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (y0 + (256*x2) + (246272*y1)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrewgitrxiqp7gbxqumbt4jxficlsevs2hmnrbdorf2stusl56o.py
# Source Nodes: [cat_5, x_10], Original ATen: [aten.add, aten.cat]
# cat_5 => cat
# x_10 => add_3
triton_poi_fused_add_cat_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 962
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y3 = yindex
    y1 = (yindex // 256)
    tmp19 = tl.load(in_ptr4 + (y0 + (256*x2) + (246272*y1)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 962, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((961*y3) + (((-1) + x2) % 961)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((961*y0) + (((-1) + x2) % 961)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tl.store(out_ptr0 + (x2 + (962*y3)), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/canp2gjw2tg2u2zppesb2rlbusvcwleekik7ewxfxys43un4ex3c.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___0___norm2 => var_mean_1
triton_red_fused_native_layer_norm_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 962
    x2 = (xindex // 1924)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (962*r3) + (123136*x0) + (246272*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (962*x0) + (1924*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (962*x0) + (1924*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (962*x0) + (1924*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3e5rzeoez25ji6yqxcxxqvonvokoboskzpryxpzoqmb3csafco.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___0___norm2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_poi_fused_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7696
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 962
    y1 = (yindex // 962)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (962*x2) + (246272*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmld7hmwirxknvpfiu7u6ljhyhm5mio34af2otd3nmfw2oug2ex.py
# Source Nodes: [x_12], Original ATen: [aten.gelu]
# x_12 => add_6, erf, mul_4, mul_5, mul_6
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
    xnumel = 7880704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tuzuoaqrjoursnbcvtu5bwvw5ctbj4y2ofo3ssbom46av3qo57.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm1, x_17], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___1___norm1 => var_mean_2
# x_17 => add_7
triton_red_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15392
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 962
    x2 = (xindex // 1924)
    x4 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (962*r3) + (123136*x0) + (246272*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6h4upd6r5o2snhrcatbmjvj3ji2cuzpvdc3tjcot674wkxf5drf.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm1, x_17], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___1___norm1 => var_mean_2
# x_17 => add_7
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7696
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymxwwj575x62pv7ipwjwdvwxtsw5vpubwqptyyysefqujp5is3n.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm1, x_17], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___1___norm1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
# x_17 => add_7
triton_poi_fused_add_native_layer_norm_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7696
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 962
    y1 = (yindex // 962)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (962*x2) + (246272*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 256.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl73eb4qgt3vpp76roi4wg6gpbzjtlmn22nlta5c6exvi7uhjxqy.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm2, x_17, x_22], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___1___norm2 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
# x_17 => add_7
# x_22 => add_10
triton_per_fused_add_native_layer_norm_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 7696
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 962
    x1 = (xindex // 962)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (962*r2) + (246272*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 256, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 256.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/ckstjvafmo7bkvhuuhxgbx4isixklx4nxhjzu2jtzi2bbzzjjeai.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___2___norm1, x_29], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___2___norm1 => add_15, add_16, mul_14, mul_15, rsqrt_4, sub_4, var_mean_4
# x_29 => add_14
triton_per_fused_add_native_layer_norm_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 7696
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 256.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckm2q6mdhn3w7poxlj6wiwrcltjmx7nct4pjdquflmkc7xldjerq.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___2___norm2, x_29, x_34], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_0_blocks___2___norm2 => add_18, add_19, mul_16, mul_17, rsqrt_5, sub_5, var_mean_5
# x_29 => add_14
# x_34 => add_17
triton_per_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 7696
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 256, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 256.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwidbfpnjjufznfncexarnmlgxp5cluvznjyaljtkqp6pu3fotkc.py
# Source Nodes: [x_42], Original ATen: [aten.add]
# x_42 => add_21
triton_poi_fused_add_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1970176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct34vwwf5aagga3aannunhf46k5nsh2k2tfxuxi4gqi4fod7itl3.py
# Source Nodes: [x_47], Original ATen: [aten.convolution]
# x_47 => convolution_1
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 961
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (256*x2) + (246272*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (961*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfobdifmgczaypgxvdv57kedaeymgk42oamvnum4ucnvadlgyily.py
# Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_4 => cat_1
# getattr_l__mod___transformers_1_blocks___0___norm1 => var_mean_6
triton_red_fused_cat_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8224
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 257
    x4 = (xindex // 257)
    x1 = (xindex // 257) % 4
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 257, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr2 + ((256*r3) + (32768*x4) + (((-1) + x0) % 256)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r3 + (128*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp10, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp9, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp20, xmask)
    tl.store(out_ptr1 + (x5), tmp21, xmask)
    tl.store(out_ptr2 + (x5), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpizf42wq2avjfyq6s3iignggdaibq66zrgpvogxmfj5ognlm63s.py
# Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_4 => cat_1
# getattr_l__mod___transformers_1_blocks___0___norm1 => var_mean_6
triton_per_fused_cat_native_layer_norm_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2056
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 257
    x1 = (xindex // 257)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (257*r2) + (1028*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (257*r2) + (1028*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (257*r2) + (1028*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ym6dgjjotrnwfm3nuqdzfwcjjjskvkhbuxoew3flrzx4u2f74m.py
# Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_4 => cat_1
# getattr_l__mod___transformers_1_blocks___0___norm1 => add_23, add_24, mul_21, mul_22, rsqrt_6, sub_6, var_mean_6
triton_poi_fused_cat_native_layer_norm_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1052672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 257
    x0 = xindex % 512
    x2 = (xindex // 131584)
    x3 = (xindex // 512)
    x4 = xindex
    tmp19 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (512*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((256*x0) + (131072*x2) + (((-1) + x1) % 256)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxtfbl3jjhifpg3kv74wsuz377zcatls2wjzmd33kbiejsbx3hq.py
# Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm2, x_55], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_4 => cat_1
# getattr_l__mod___transformers_1_blocks___0___norm2 => add_26, add_27, mul_23, mul_24, rsqrt_7, sub_7, var_mean_7
# x_55 => add_25
triton_per_fused_add_cat_native_layer_norm_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 257
    r2 = rindex
    x1 = (xindex // 257)
    x3 = xindex
    tmp19 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((256*r2) + (131072*x1) + (((-1) + x0) % 256)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 512, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 512.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgdm5uqdf63537t3odm2uhadrahnhmheey633krk3kojl56z3ra.py
# Source Nodes: [x_57], Original ATen: [aten.gelu]
# x_57 => add_28, erf_3, mul_25, mul_26, mul_27
triton_poi_fused_gelu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4210688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
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


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbz3ufwbxye5b4qd2ak6oo3p4wpkvukp2bm4ju4jpsysi5klhh5.py
# Source Nodes: [getattr_l__mod___transformers_1_blocks___1___norm1, x_62], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_1_blocks___1___norm1 => add_30, add_31, mul_28, mul_29, rsqrt_8, sub_8, var_mean_8
# x_62 => add_29
triton_per_fused_add_native_layer_norm_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 512.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/76/c766b4kyqb3hqwozj5stcwei74ineyvu76przndsenmgou5j2ywd.py
# Source Nodes: [getattr_l__mod___transformers_1_blocks___1___norm2, x_62, x_67], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_1_blocks___1___norm2 => add_33, add_34, mul_30, mul_31, rsqrt_9, sub_9, var_mean_9
# x_62 => add_29
# x_67 => add_32
triton_per_fused_add_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgozs4amqw3hq2dz7ejp3h2b6iavz43za4hcyx3mvkarj736jmpz.py
# Source Nodes: [x_123], Original ATen: [aten.add]
# x_123 => add_64
triton_poi_fused_add_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1052672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crntvmqvoycvw75foueygfsgmnn3gmthq7wupyufyqmydx6gtjeg.py
# Source Nodes: [x_128], Original ATen: [aten.convolution]
# x_128 => convolution_2
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (512*x2) + (131584*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhjoa76hygxsx73gzat5uqnktbjqcjwd64z764oyekzluhdsfss.py
# Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_3 => cat_2
# getattr_l__mod___transformers_2_blocks___0___norm1 => var_mean_18
triton_red_fused_cat_native_layer_norm_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 65
    x4 = (xindex // 65)
    x1 = (xindex // 65) % 8
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 65, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr2 + ((64*r3) + (8192*x4) + (((-1) + x0) % 64)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r3 + (128*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp10, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp9, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp20, xmask)
    tl.store(out_ptr1 + (x5), tmp21, xmask)
    tl.store(out_ptr2 + (x5), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c566opz27lbkumrkplszc2iipt6npffxkczp4437qh2w7sty2g2n.py
# Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_3 => cat_2
# getattr_l__mod___transformers_2_blocks___0___norm1 => var_mean_18
triton_per_fused_cat_native_layer_norm_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 520
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 65
    x1 = (xindex // 65)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (65*r2) + (520*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (65*r2) + (520*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (65*r2) + (520*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/25/c25stgaw3jo5dcoi64ojiex6czpsyw2dzonfqs2mjjyyn6pyrvdc.py
# Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_3 => cat_2
# getattr_l__mod___transformers_2_blocks___0___norm1 => add_66, add_67, mul_63, mul_64, rsqrt_18, sub_18, var_mean_18
triton_poi_fused_cat_native_layer_norm_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 532480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024) % 65
    x0 = xindex % 1024
    x2 = (xindex // 66560)
    x3 = (xindex // 1024)
    x4 = xindex
    tmp19 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (1024*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 65, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((64*x0) + (65536*x2) + (((-1) + x1) % 64)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), tmp10, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 1024.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/costgp7vsofhyqzy565oewi7c7sbmgrfobnqegsyc7fvk2qm6f2e.py
# Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm2, x_136], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_3 => cat_2
# getattr_l__mod___transformers_2_blocks___0___norm2 => add_69, add_70, mul_65, mul_66, rsqrt_19, sub_19, var_mean_19
# x_136 => add_68
triton_per_fused_add_cat_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 65
    r2 = rindex
    x1 = (xindex // 65)
    x3 = xindex
    tmp19 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 65, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((64*r2) + (65536*x1) + (((-1) + x0) % 64)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 1024, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 1024.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/br/cbrpslisurui7c2jvzvayhqfuvtexouthdm3r3vgrr5hoizskiid.py
# Source Nodes: [x_138], Original ATen: [aten.gelu]
# x_138 => add_71, erf_9, mul_67, mul_68, mul_69
triton_poi_fused_gelu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2129920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/eo/ceoghvhktsjrnqobieou5b7c7nixwxuojp6bdi3suxv75nbv6g2x.py
# Source Nodes: [getattr_l__mod___transformers_2_blocks___1___norm1, x_143], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_2_blocks___1___norm1 => add_73, add_74, mul_70, mul_71, rsqrt_20, sub_20, var_mean_20
# x_143 => add_72
triton_per_fused_add_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckuggh2fym457zh5xp3exscmba4xcdqxanhzyemuvwga2brut6fq.py
# Source Nodes: [getattr_l__mod___transformers_2_blocks___1___norm2, x_143, x_148], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___transformers_2_blocks___1___norm2 => add_76, add_77, mul_72, mul_73, rsqrt_21, sub_21, var_mean_21
# x_143 => add_72
# x_148 => add_75
triton_per_fused_add_native_layer_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvew76qe5znf5jqh23dajwlsnqhoxaf5a4brmtx7y7gqyfjgisuj.py
# Source Nodes: [x_184], Original ATen: [aten.native_layer_norm]
# x_184 => add_94, add_95, clone_40, mul_91, mul_92, rsqrt_26, sub_26, var_mean_26
triton_per_fused_native_layer_norm_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (66560*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (66560*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 256, 31, 31), (246016, 961, 31, 1))
    assert_size_stride(arg1_1, (1, 1, 256), (256, 256, 1))
    assert_size_stride(arg2_1, (256, 3, 14, 14), (588, 196, 14, 1))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg5_1, (256, ), (1, ))
    assert_size_stride(arg6_1, (768, 256), (256, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (256, 256), (256, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (1024, 256), (256, 1))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (256, 1024), (1024, 1))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (768, 256), (256, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (256, 256), (256, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (1024, 256), (256, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (256, 1024), (1024, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (768, 256), (256, 1))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (256, 256), (256, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (1024, 256), (256, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (256, 1024), (1024, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256), (256, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (1536, 512), (512, 1))
    assert_size_stride(arg47_1, (1536, ), (1, ))
    assert_size_stride(arg48_1, (512, 512), (512, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (2048, 512), (512, 1))
    assert_size_stride(arg53_1, (2048, ), (1, ))
    assert_size_stride(arg54_1, (512, 2048), (2048, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (1536, 512), (512, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (2048, 512), (512, 1))
    assert_size_stride(arg65_1, (2048, ), (1, ))
    assert_size_stride(arg66_1, (512, 2048), (2048, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (1536, 512), (512, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (2048, 512), (512, 1))
    assert_size_stride(arg77_1, (2048, ), (1, ))
    assert_size_stride(arg78_1, (512, 2048), (2048, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (1536, 512), (512, 1))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (512, 512), (512, 1))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (2048, 512), (512, 1))
    assert_size_stride(arg89_1, (2048, ), (1, ))
    assert_size_stride(arg90_1, (512, 2048), (2048, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (1536, 512), (512, 1))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (2048, 512), (512, 1))
    assert_size_stride(arg101_1, (2048, ), (1, ))
    assert_size_stride(arg102_1, (512, 2048), (2048, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (1536, 512), (512, 1))
    assert_size_stride(arg107_1, (1536, ), (1, ))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (2048, 512), (512, 1))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (512, 2048), (2048, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, 512), (512, 1))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg123_1, (3072, ), (1, ))
    assert_size_stride(arg124_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg129_1, (4096, ), (1, ))
    assert_size_stride(arg130_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg135_1, (3072, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg141_1, (4096, ), (1, ))
    assert_size_stride(arg142_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg147_1, (3072, ), (1, ))
    assert_size_stride(arg148_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg153_1, (4096, ), (1, ))
    assert_size_stride(arg154_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg159_1, (3072, ), (1, ))
    assert_size_stride(arg160_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg165_1, (4096, ), (1, ))
    assert_size_stride(arg166_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1000, ), (1, ))
    assert_size_stride(arg172_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg172_1, arg2_1, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 256, 31, 31), (246016, 961, 31, 1))
        del arg172_1
        del arg2_1
        buf1 = empty_strided((8, 962, 1, 2), (1924, 1, 15392, 962), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 962, 1, 2), (1924, 1, 15392, 962), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 962, 1, 2), (1924, 1, 15392, 962), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_cat_native_layer_norm_0.run(arg1_1, buf0, arg3_1, arg0_1, buf1, buf2, buf3, 15392, 128, grid=grid(15392), stream=stream0)
        buf4 = empty_strided((8, 962, 1), (962, 1, 7696), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 962, 1), (962, 1, 7696), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_1.run(buf1, buf2, buf3, buf4, buf5, 7696, 2, grid=grid(7696), stream=stream0)
        buf7 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5, getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_2.run(arg1_1, buf0, arg3_1, arg0_1, buf4, buf5, arg4_1, arg5_1, buf7, 2048, 962, grid=grid(2048, 962), stream=stream0)
        del arg4_1
        del arg5_1
        buf8 = empty((7696, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg7_1, reinterpret_tensor(buf7, (7696, 256), (256, 1), 0), reinterpret_tensor(arg6_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf8)
        del arg6_1
        del arg7_1
        # Source Nodes: [x_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf9 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf8, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf8, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf8, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, False)
        buf10 = buf9[0]
        del buf9
        buf14 = reinterpret_tensor(buf7, (7696, 256), (256, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf10, (7696, 256), (256, 1), 0), reinterpret_tensor(arg8_1, (256, 256), (1, 256), 0), out=buf14)
        del arg8_1
        buf15 = reinterpret_tensor(buf10, (8, 962, 256), (246272, 1, 962), 0); del buf10  # reuse
        # Source Nodes: [cat_5, x_10], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_3.run(arg1_1, buf0, arg3_1, arg0_1, buf14, arg9_1, buf15, 2048, 962, grid=grid(2048, 962), stream=stream0)
        del arg0_1
        del arg1_1
        del arg3_1
        del arg9_1
        buf16 = buf3; del buf3  # reuse
        buf17 = buf2; del buf2  # reuse
        buf18 = buf1; del buf1  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_4.run(buf15, buf16, buf17, buf18, 15392, 128, grid=grid(15392), stream=stream0)
        buf19 = buf5; del buf5  # reuse
        buf20 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_1.run(buf16, buf17, buf18, buf19, buf20, 7696, 2, grid=grid(7696), stream=stream0)
        buf22 = reinterpret_tensor(buf14, (8, 962, 256), (246272, 256, 1), 0); del buf14  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_5.run(buf15, buf19, buf20, arg10_1, arg11_1, buf22, 7696, 256, grid=grid(7696, 256), stream=stream0)
        del arg10_1
        del arg11_1
        buf23 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (7696, 256), (256, 1), 0), reinterpret_tensor(arg12_1, (256, 1024), (1, 256), 0), out=buf23)
        del arg12_1
        buf24 = reinterpret_tensor(buf23, (8, 962, 1024), (985088, 1024, 1), 0); del buf23  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf24, arg13_1, 7880704, grid=grid(7880704), stream=stream0)
        del arg13_1
        buf25 = reinterpret_tensor(buf22, (7696, 256), (256, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg14_1, (1024, 256), (1, 1024), 0), out=buf25)
        del arg14_1
        buf26 = reinterpret_tensor(buf18, (8, 962, 1, 2), (1924, 2, 15392, 1), 0); del buf18  # reuse
        buf27 = reinterpret_tensor(buf17, (8, 962, 1, 2), (1924, 2, 15392, 1), 0); del buf17  # reuse
        buf28 = reinterpret_tensor(buf16, (8, 962, 1, 2), (1924, 2, 15392, 1), 0); del buf16  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm1, x_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf15, buf25, arg15_1, buf26, buf27, buf28, 15392, 128, grid=grid(15392), stream=stream0)
        buf29 = buf20; del buf20  # reuse
        buf30 = buf19; del buf19  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm1, x_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf26, buf27, buf28, buf29, buf30, 7696, 2, grid=grid(7696), stream=stream0)
        del buf26
        del buf27
        del buf28
        buf32 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm1, x_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_9.run(buf15, buf25, arg15_1, buf29, buf30, arg16_1, arg17_1, buf32, 7696, 256, grid=grid(7696, 256), stream=stream0)
        del arg16_1
        del arg17_1
        del buf29
        del buf30
        buf33 = buf8; del buf8  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg19_1, reinterpret_tensor(buf32, (7696, 256), (256, 1), 0), reinterpret_tensor(arg18_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf33)
        del arg18_1
        del arg19_1
        # Source Nodes: [x_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf34 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf33, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf33, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf33, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, False)
        buf35 = buf34[0]
        del buf34
        buf39 = reinterpret_tensor(buf32, (7696, 256), (256, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (7696, 256), (256, 1), 0), reinterpret_tensor(arg20_1, (256, 256), (1, 256), 0), out=buf39)
        del arg20_1
        buf40 = reinterpret_tensor(buf25, (8, 962, 256), (246272, 256, 1), 0); del buf25  # reuse
        buf44 = reinterpret_tensor(buf35, (8, 962, 256), (246272, 256, 1), 0); del buf35  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___norm2, x_17, x_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf40, buf15, arg15_1, buf39, arg21_1, arg22_1, arg23_1, buf44, 7696, 256, grid=grid(7696), stream=stream0)
        del arg15_1
        del arg21_1
        del arg22_1
        del arg23_1
        del buf15
        buf45 = reinterpret_tensor(buf24, (7696, 1024), (1024, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (7696, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 1024), (1, 256), 0), out=buf45)
        del arg24_1
        buf46 = reinterpret_tensor(buf45, (8, 962, 1024), (985088, 1024, 1), 0); del buf45  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf46, arg25_1, 7880704, grid=grid(7880704), stream=stream0)
        del arg25_1
        buf47 = reinterpret_tensor(buf44, (7696, 256), (256, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 256), (1, 1024), 0), out=buf47)
        del arg26_1
        buf51 = reinterpret_tensor(buf39, (8, 962, 256), (246272, 256, 1), 0); del buf39  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___norm1, x_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_11.run(buf40, buf47, arg27_1, arg28_1, arg29_1, buf51, 7696, 256, grid=grid(7696), stream=stream0)
        del arg28_1
        del arg29_1
        buf52 = buf33; del buf33  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg31_1, reinterpret_tensor(buf51, (7696, 256), (256, 1), 0), reinterpret_tensor(arg30_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf52)
        del arg30_1
        del arg31_1
        # Source Nodes: [x_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf53 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf52, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf52, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf52, (8, 4, 962, 64), (738816, 64, 768, 1), 512), None, False)
        del buf52
        buf54 = buf53[0]
        del buf53
        buf58 = reinterpret_tensor(buf51, (7696, 256), (256, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (7696, 256), (256, 1), 0), reinterpret_tensor(arg32_1, (256, 256), (1, 256), 0), out=buf58)
        del arg32_1
        buf59 = reinterpret_tensor(buf58, (8, 962, 256), (246272, 256, 1), 0); del buf58  # reuse
        buf63 = reinterpret_tensor(buf54, (8, 962, 256), (246272, 256, 1), 0); del buf54  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___norm2, x_29, x_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf59, buf40, buf47, arg27_1, arg33_1, arg34_1, arg35_1, buf63, 7696, 256, grid=grid(7696), stream=stream0)
        del arg27_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf40
        del buf47
        buf64 = reinterpret_tensor(buf46, (7696, 1024), (1024, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (7696, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 1024), (1, 256), 0), out=buf64)
        del arg36_1
        buf65 = reinterpret_tensor(buf64, (8, 962, 1024), (985088, 1024, 1), 0); del buf64  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf65, arg37_1, 7880704, grid=grid(7880704), stream=stream0)
        del arg37_1
        buf66 = reinterpret_tensor(buf63, (7696, 256), (256, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 256), (1, 1024), 0), out=buf66)
        del arg38_1
        del buf65
        buf67 = reinterpret_tensor(buf66, (8, 962, 256), (246272, 256, 1), 0); del buf66  # reuse
        # Source Nodes: [x_42], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf67, buf59, arg39_1, 1970176, grid=grid(1970176), stream=stream0)
        del arg39_1
        del buf59
        buf68 = empty((8, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_tokens_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (8, 256), (246272, 1), 0), reinterpret_tensor(arg42_1, (256, 512), (1, 256), 0), out=buf68)
        del arg42_1
        buf69 = buf0; del buf0  # reuse
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf67, buf69, 2048, 961, grid=grid(2048, 961), stream=stream0)
        del buf67
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg40_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf70, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg40_1
        del buf69
        buf71 = empty_strided((8, 257, 1, 4), (1028, 1, 8224, 257), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((8, 257, 1, 4), (1028, 1, 8224, 257), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((8, 257, 1, 4), (1028, 1, 8224, 257), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_red_fused_cat_native_layer_norm_15.run(buf68, arg43_1, buf70, arg41_1, buf71, buf72, buf73, 8224, 128, grid=grid(8224), stream=stream0)
        buf74 = empty_strided((8, 257, 1), (257, 1, 2056), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((8, 257, 1), (257, 1, 2056), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf71, buf72, buf73, buf74, buf75, 2056, 4, grid=grid(2056), stream=stream0)
        del buf71
        del buf72
        del buf73
        buf77 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_17.run(buf68, arg43_1, buf70, arg41_1, buf74, buf75, arg44_1, arg45_1, buf77, 1052672, grid=grid(1052672), stream=stream0)
        del arg44_1
        del arg45_1
        del buf74
        del buf75
        buf78 = empty((2056, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg47_1, reinterpret_tensor(buf77, (2056, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf78)
        del arg46_1
        del arg47_1
        # Source Nodes: [x_51], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf79 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf78, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf78, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf78, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf80 = buf79[0]
        del buf79
        buf84 = reinterpret_tensor(buf77, (2056, 512), (512, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf80, (2056, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), out=buf84)
        del arg48_1
        buf85 = reinterpret_tensor(buf84, (8, 257, 512), (131584, 512, 1), 0); del buf84  # reuse
        buf89 = reinterpret_tensor(buf80, (8, 257, 512), (131584, 512, 1), 0); del buf80  # reuse
        # Source Nodes: [cat_4, getattr_l__mod___transformers_1_blocks___0___norm2, x_55], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_18.run(buf85, buf68, arg43_1, buf70, arg41_1, arg49_1, arg50_1, arg51_1, buf89, 2056, 512, grid=grid(2056), stream=stream0)
        del arg41_1
        del arg43_1
        del arg49_1
        del arg50_1
        del arg51_1
        del buf68
        buf90 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf89, (2056, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 2048), (1, 512), 0), out=buf90)
        del arg52_1
        buf91 = reinterpret_tensor(buf90, (8, 257, 2048), (526336, 2048, 1), 0); del buf90  # reuse
        # Source Nodes: [x_57], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_19.run(buf91, arg53_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg53_1
        buf92 = reinterpret_tensor(buf89, (2056, 512), (512, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg54_1, (2048, 512), (1, 2048), 0), out=buf92)
        del arg54_1
        buf96 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___norm1, x_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf85, buf92, arg55_1, arg56_1, arg57_1, buf96, 2056, 512, grid=grid(2056), stream=stream0)
        del arg56_1
        del arg57_1
        buf97 = buf78; del buf78  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf96, (2056, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf97)
        del arg58_1
        del arg59_1
        # Source Nodes: [x_63], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf98 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf97, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf97, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf97, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf99 = buf98[0]
        del buf98
        buf103 = reinterpret_tensor(buf96, (2056, 512), (512, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (2056, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), out=buf103)
        del arg60_1
        buf104 = reinterpret_tensor(buf103, (8, 257, 512), (131584, 512, 1), 0); del buf103  # reuse
        buf108 = reinterpret_tensor(buf99, (8, 257, 512), (131584, 512, 1), 0); del buf99  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___norm2, x_62, x_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf104, buf85, buf92, arg55_1, arg61_1, arg62_1, arg63_1, buf108, 2056, 512, grid=grid(2056), stream=stream0)
        del arg55_1
        del arg61_1
        del arg62_1
        del arg63_1
        del buf85
        buf109 = reinterpret_tensor(buf91, (2056, 2048), (2048, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (2056, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 2048), (1, 512), 0), out=buf109)
        del arg64_1
        buf110 = reinterpret_tensor(buf109, (8, 257, 2048), (526336, 2048, 1), 0); del buf109  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_19.run(buf110, arg65_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg65_1
        buf111 = reinterpret_tensor(buf108, (2056, 512), (512, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg66_1, (2048, 512), (1, 2048), 0), out=buf111)
        del arg66_1
        buf115 = reinterpret_tensor(buf92, (8, 257, 512), (131584, 512, 1), 0); del buf92  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___norm1, x_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf104, buf111, arg67_1, arg68_1, arg69_1, buf115, 2056, 512, grid=grid(2056), stream=stream0)
        del arg68_1
        del arg69_1
        buf116 = buf97; del buf97  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg71_1, reinterpret_tensor(buf115, (2056, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf116)
        del arg70_1
        del arg71_1
        # Source Nodes: [x_75], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf117 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf116, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf118 = buf117[0]
        del buf117
        buf122 = reinterpret_tensor(buf115, (2056, 512), (512, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (2056, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), out=buf122)
        del arg72_1
        buf123 = reinterpret_tensor(buf122, (8, 257, 512), (131584, 512, 1), 0); del buf122  # reuse
        buf127 = reinterpret_tensor(buf118, (8, 257, 512), (131584, 512, 1), 0); del buf118  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___norm2, x_74, x_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf123, buf104, buf111, arg67_1, arg73_1, arg74_1, arg75_1, buf127, 2056, 512, grid=grid(2056), stream=stream0)
        del arg67_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf104
        buf128 = reinterpret_tensor(buf110, (2056, 2048), (2048, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (2056, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 2048), (1, 512), 0), out=buf128)
        del arg76_1
        buf129 = reinterpret_tensor(buf128, (8, 257, 2048), (526336, 2048, 1), 0); del buf128  # reuse
        # Source Nodes: [x_81], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_19.run(buf129, arg77_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg77_1
        buf130 = reinterpret_tensor(buf127, (2056, 512), (512, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg78_1, (2048, 512), (1, 2048), 0), out=buf130)
        del arg78_1
        buf134 = reinterpret_tensor(buf111, (8, 257, 512), (131584, 512, 1), 0); del buf111  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___norm1, x_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf123, buf130, arg79_1, arg80_1, arg81_1, buf134, 2056, 512, grid=grid(2056), stream=stream0)
        del arg80_1
        del arg81_1
        buf135 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg83_1, reinterpret_tensor(buf134, (2056, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf135)
        del arg82_1
        del arg83_1
        # Source Nodes: [x_87], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf136 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf135, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf135, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf135, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf137 = buf136[0]
        del buf136
        buf141 = reinterpret_tensor(buf134, (2056, 512), (512, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (2056, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), out=buf141)
        del arg84_1
        buf142 = reinterpret_tensor(buf141, (8, 257, 512), (131584, 512, 1), 0); del buf141  # reuse
        buf146 = reinterpret_tensor(buf137, (8, 257, 512), (131584, 512, 1), 0); del buf137  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___norm2, x_86, x_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf142, buf123, buf130, arg79_1, arg85_1, arg86_1, arg87_1, buf146, 2056, 512, grid=grid(2056), stream=stream0)
        del arg79_1
        del arg85_1
        del arg86_1
        del arg87_1
        del buf123
        buf147 = reinterpret_tensor(buf129, (2056, 2048), (2048, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (2056, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 2048), (1, 512), 0), out=buf147)
        del arg88_1
        buf148 = reinterpret_tensor(buf147, (8, 257, 2048), (526336, 2048, 1), 0); del buf147  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_19.run(buf148, arg89_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg89_1
        buf149 = reinterpret_tensor(buf146, (2056, 512), (512, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), out=buf149)
        del arg90_1
        buf153 = reinterpret_tensor(buf130, (8, 257, 512), (131584, 512, 1), 0); del buf130  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___norm1, x_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf142, buf149, arg91_1, arg92_1, arg93_1, buf153, 2056, 512, grid=grid(2056), stream=stream0)
        del arg92_1
        del arg93_1
        buf154 = buf135; del buf135  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf153, (2056, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf154)
        del arg94_1
        del arg95_1
        # Source Nodes: [x_99], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf155 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf154, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf154, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf154, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        buf156 = buf155[0]
        del buf155
        buf160 = reinterpret_tensor(buf153, (2056, 512), (512, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (2056, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), out=buf160)
        del arg96_1
        buf161 = reinterpret_tensor(buf160, (8, 257, 512), (131584, 512, 1), 0); del buf160  # reuse
        buf165 = reinterpret_tensor(buf156, (8, 257, 512), (131584, 512, 1), 0); del buf156  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___norm2, x_103, x_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf161, buf142, buf149, arg91_1, arg97_1, arg98_1, arg99_1, buf165, 2056, 512, grid=grid(2056), stream=stream0)
        del arg91_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf142
        buf166 = reinterpret_tensor(buf148, (2056, 2048), (2048, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (2056, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 2048), (1, 512), 0), out=buf166)
        del arg100_1
        buf167 = reinterpret_tensor(buf166, (8, 257, 2048), (526336, 2048, 1), 0); del buf166  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_19.run(buf167, arg101_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg101_1
        buf168 = reinterpret_tensor(buf165, (2056, 512), (512, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg102_1, (2048, 512), (1, 2048), 0), out=buf168)
        del arg102_1
        buf172 = reinterpret_tensor(buf149, (8, 257, 512), (131584, 512, 1), 0); del buf149  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___norm1, x_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf161, buf168, arg103_1, arg104_1, arg105_1, buf172, 2056, 512, grid=grid(2056), stream=stream0)
        del arg104_1
        del arg105_1
        buf173 = buf154; del buf154  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf172, (2056, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf173)
        del arg106_1
        del arg107_1
        # Source Nodes: [x_111], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf174 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf173, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf173, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf173, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), None, False)
        del buf173
        buf175 = buf174[0]
        del buf174
        buf179 = reinterpret_tensor(buf172, (2056, 512), (512, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (2056, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), out=buf179)
        del arg108_1
        buf180 = reinterpret_tensor(buf179, (8, 257, 512), (131584, 512, 1), 0); del buf179  # reuse
        buf184 = reinterpret_tensor(buf175, (8, 257, 512), (131584, 512, 1), 0); del buf175  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___norm2, x_110, x_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_21.run(buf180, buf161, buf168, arg103_1, arg109_1, arg110_1, arg111_1, buf184, 2056, 512, grid=grid(2056), stream=stream0)
        del arg103_1
        del arg109_1
        del arg110_1
        del arg111_1
        del buf161
        del buf168
        buf185 = reinterpret_tensor(buf167, (2056, 2048), (2048, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (2056, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 2048), (1, 512), 0), out=buf185)
        del arg112_1
        buf186 = reinterpret_tensor(buf185, (8, 257, 2048), (526336, 2048, 1), 0); del buf185  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_19.run(buf186, arg113_1, 4210688, grid=grid(4210688), stream=stream0)
        del arg113_1
        buf187 = reinterpret_tensor(buf184, (2056, 512), (512, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg114_1, (2048, 512), (1, 2048), 0), out=buf187)
        del arg114_1
        del buf186
        buf188 = reinterpret_tensor(buf187, (8, 257, 512), (131584, 512, 1), 0); del buf187  # reuse
        # Source Nodes: [x_123], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf188, buf180, arg115_1, 1052672, grid=grid(1052672), stream=stream0)
        del arg115_1
        del buf180
        buf189 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_tokens_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (8, 512), (131584, 1), 0), reinterpret_tensor(arg118_1, (512, 1024), (1, 512), 0), out=buf189)
        del arg118_1
        buf190 = buf70; del buf70  # reuse
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf188, buf190, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del buf188
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg116_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf191, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg116_1
        del buf190
        buf192 = empty_strided((8, 65, 1, 8), (520, 1, 4160, 65), device='cuda', dtype=torch.float32)
        buf193 = empty_strided((8, 65, 1, 8), (520, 1, 4160, 65), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((8, 65, 1, 8), (520, 1, 4160, 65), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_red_fused_cat_native_layer_norm_24.run(buf189, arg119_1, buf191, arg117_1, buf192, buf193, buf194, 4160, 128, grid=grid(4160), stream=stream0)
        buf195 = empty_strided((8, 65, 1), (65, 1, 520), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((8, 65, 1), (65, 1, 520), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_25.run(buf192, buf193, buf194, buf195, buf196, 520, 8, grid=grid(520), stream=stream0)
        del buf192
        del buf193
        del buf194
        buf198 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_26.run(buf189, arg119_1, buf191, arg117_1, buf195, buf196, arg120_1, arg121_1, buf198, 532480, grid=grid(532480), stream=stream0)
        del arg120_1
        del arg121_1
        del buf195
        del buf196
        buf199 = empty((520, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg123_1, reinterpret_tensor(buf198, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf199)
        del arg122_1
        del arg123_1
        # Source Nodes: [x_132], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf200 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf199, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf199, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf199, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        buf201 = buf200[0]
        del buf200
        buf205 = reinterpret_tensor(buf198, (520, 1024), (1024, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), out=buf205)
        del arg124_1
        buf206 = reinterpret_tensor(buf205, (8, 65, 1024), (66560, 1024, 1), 0); del buf205  # reuse
        buf210 = reinterpret_tensor(buf201, (8, 65, 1024), (66560, 1024, 1), 0); del buf201  # reuse
        # Source Nodes: [cat_3, getattr_l__mod___transformers_2_blocks___0___norm2, x_136], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_27.run(buf206, buf189, arg119_1, buf191, arg117_1, arg125_1, arg126_1, arg127_1, buf210, 520, 1024, grid=grid(520), stream=stream0)
        del arg117_1
        del arg119_1
        del arg125_1
        del arg126_1
        del arg127_1
        del buf191
        buf211 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf210, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 4096), (1, 1024), 0), out=buf211)
        del arg128_1
        buf212 = reinterpret_tensor(buf211, (8, 65, 4096), (266240, 4096, 1), 0); del buf211  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf212, arg129_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg129_1
        buf213 = reinterpret_tensor(buf210, (520, 1024), (1024, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg130_1, (4096, 1024), (1, 4096), 0), out=buf213)
        del arg130_1
        buf217 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___norm1, x_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_29.run(buf206, buf213, arg131_1, arg132_1, arg133_1, buf217, 520, 1024, grid=grid(520), stream=stream0)
        del arg132_1
        del arg133_1
        buf218 = buf199; del buf199  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg135_1, reinterpret_tensor(buf217, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf218)
        del arg134_1
        del arg135_1
        # Source Nodes: [x_144], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf219 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf218, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf218, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf218, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        buf220 = buf219[0]
        del buf219
        buf224 = reinterpret_tensor(buf217, (520, 1024), (1024, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), out=buf224)
        del arg136_1
        buf225 = reinterpret_tensor(buf224, (8, 65, 1024), (66560, 1024, 1), 0); del buf224  # reuse
        buf229 = reinterpret_tensor(buf220, (8, 65, 1024), (66560, 1024, 1), 0); del buf220  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___norm2, x_143, x_148], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_30.run(buf225, buf206, buf213, arg131_1, arg137_1, arg138_1, arg139_1, buf229, 520, 1024, grid=grid(520), stream=stream0)
        del arg131_1
        del arg137_1
        del arg138_1
        del arg139_1
        del buf206
        buf230 = reinterpret_tensor(buf212, (520, 4096), (4096, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 4096), (1, 1024), 0), out=buf230)
        del arg140_1
        buf231 = reinterpret_tensor(buf230, (8, 65, 4096), (266240, 4096, 1), 0); del buf230  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf231, arg141_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg141_1
        buf232 = reinterpret_tensor(buf229, (520, 1024), (1024, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg142_1, (4096, 1024), (1, 4096), 0), out=buf232)
        del arg142_1
        buf236 = reinterpret_tensor(buf213, (8, 65, 1024), (66560, 1024, 1), 0); del buf213  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___norm1, x_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_29.run(buf225, buf232, arg143_1, arg144_1, arg145_1, buf236, 520, 1024, grid=grid(520), stream=stream0)
        del arg144_1
        del arg145_1
        buf237 = buf218; del buf218  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf236, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg146_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf237)
        del arg146_1
        del arg147_1
        # Source Nodes: [x_156], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf238 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf237, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf237, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf237, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        buf239 = buf238[0]
        del buf238
        buf243 = reinterpret_tensor(buf236, (520, 1024), (1024, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg148_1, (1024, 1024), (1, 1024), 0), out=buf243)
        del arg148_1
        buf244 = reinterpret_tensor(buf243, (8, 65, 1024), (66560, 1024, 1), 0); del buf243  # reuse
        buf248 = reinterpret_tensor(buf239, (8, 65, 1024), (66560, 1024, 1), 0); del buf239  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___norm2, x_155, x_160], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_30.run(buf244, buf225, buf232, arg143_1, arg149_1, arg150_1, arg151_1, buf248, 520, 1024, grid=grid(520), stream=stream0)
        del arg143_1
        del arg149_1
        del arg150_1
        del arg151_1
        del buf225
        buf249 = reinterpret_tensor(buf231, (520, 4096), (4096, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 4096), (1, 1024), 0), out=buf249)
        del arg152_1
        buf250 = reinterpret_tensor(buf249, (8, 65, 4096), (266240, 4096, 1), 0); del buf249  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf250, arg153_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg153_1
        buf251 = reinterpret_tensor(buf248, (520, 1024), (1024, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg154_1, (4096, 1024), (1, 4096), 0), out=buf251)
        del arg154_1
        buf255 = reinterpret_tensor(buf232, (8, 65, 1024), (66560, 1024, 1), 0); del buf232  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___norm1, x_167], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_29.run(buf244, buf251, arg155_1, arg156_1, arg157_1, buf255, 520, 1024, grid=grid(520), stream=stream0)
        del arg156_1
        del arg157_1
        buf256 = buf237; del buf237  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg159_1, reinterpret_tensor(buf255, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg158_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf256)
        del arg158_1
        del arg159_1
        # Source Nodes: [x_168], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf257 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf256, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf256, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf256, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), None, False)
        del buf256
        buf258 = buf257[0]
        del buf257
        buf262 = reinterpret_tensor(buf255, (520, 1024), (1024, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 1024), (1, 1024), 0), out=buf262)
        del arg160_1
        buf263 = reinterpret_tensor(buf262, (8, 65, 1024), (66560, 1024, 1), 0); del buf262  # reuse
        buf267 = reinterpret_tensor(buf258, (8, 65, 1024), (66560, 1024, 1), 0); del buf258  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___norm2, x_167, x_172], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_30.run(buf263, buf244, buf251, arg155_1, arg161_1, arg162_1, arg163_1, buf267, 520, 1024, grid=grid(520), stream=stream0)
        del arg155_1
        del arg161_1
        del arg162_1
        del arg163_1
        del buf244
        del buf251
        buf268 = reinterpret_tensor(buf250, (520, 4096), (4096, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg164_1, (1024, 4096), (1, 1024), 0), out=buf268)
        del arg164_1
        buf269 = reinterpret_tensor(buf268, (8, 65, 4096), (266240, 4096, 1), 0); del buf268  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_28.run(buf269, arg165_1, 2129920, grid=grid(2129920), stream=stream0)
        del arg165_1
        buf270 = reinterpret_tensor(buf267, (520, 1024), (1024, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg166_1, (4096, 1024), (1, 4096), 0), out=buf270)
        del arg166_1
        del buf269
        buf274 = reinterpret_tensor(buf189, (8, 1, 1024), (1024, 1024, 1), 0); del buf189  # reuse
        # Source Nodes: [x_184], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_31.run(buf263, buf270, arg167_1, arg168_1, arg169_1, buf274, 8, 1024, grid=grid(8), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del buf263
        del buf270
        buf275 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf274, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf275)
        del arg170_1
        del arg171_1
        return (buf275, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 256, 31, 31), (246016, 961, 31, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, 3, 14, 14), (588, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
