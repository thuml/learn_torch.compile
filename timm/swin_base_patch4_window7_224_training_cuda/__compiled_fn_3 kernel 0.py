
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


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpivv3keeosqracgearty27ie6ihs5ypxzdlijg6xn4zj7rxmuz.py
# Source Nodes: [shifted_x, x_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# shifted_x => add_2, mul_2, rsqrt_1, sub_1, var_mean_1
# x_4 => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight,
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp16, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp31_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp31_mean_next, tmp31_m2_next, tmp31_weight_next = triton_helpers.welford_reduce(
            tmp30, tmp31_mean, tmp31_m2, tmp31_weight,
        )
        tmp31_mean = tl.where(rmask & xmask, tmp31_mean_next, tmp31_mean)
        tmp31_m2 = tl.where(rmask & xmask, tmp31_m2_next, tmp31_m2)
        tmp31_weight = tl.where(rmask & xmask, tmp31_weight_next, tmp31_weight)
    tmp31_tmp, tmp32_tmp, tmp33_tmp = triton_helpers.welford(
        tmp31_mean, tmp31_m2, tmp31_weight, 1
    )
    tmp31 = tmp31_tmp[:, None]
    tmp32 = tmp32_tmp[:, None]
    tmp33 = tmp33_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp34 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38 - tmp22
        tmp40 = 128.0
        tmp41 = tmp32 / tmp40
        tmp42 = 1e-05
        tmp43 = tmp41 + tmp42
        tmp44 = tl.math.rsqrt(tmp43)
        tmp45 = tmp39 * tmp44
        tl.store(out_ptr5 + (r2 + (128*x3)), tmp45, rmask & xmask)
    tmp46 = 128.0
    tmp47 = tmp32 / tmp46
    tmp48 = 1e-05
    tmp49 = tmp47 + tmp48
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp50 / tmp46
    tmp52 = tmp5 / tmp46
    tmp53 = tmp52 + tmp48
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp54 / tmp46
    tl.store(out_ptr6 + (x3), tmp51, xmask)
    tl.store(out_ptr7 + (x3), tmp55, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5kf6h5vumibgyjtpx3bog3afie4tiknwue5pexxdrpuwus7fsr.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv => view_3
triton_poi_fused_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((x1 % 49) % 7)) + (896*((x1 // 49) % 8)) + (7168*((x1 % 49) // 7)) + (50176*(x1 // 392))), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chy7wilb54hgtxm4udhtjgynn76ldaa7mkaicw2b6hqeulbeakcl.py
# Source Nodes: [attn, q_1], Original ATen: [aten.clone, aten.mul]
# attn => clone_2
# q_1 => mul_4
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (384*x1) + (18816*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjgystxi6syfc3cdourgmlitkt6r4ui7s65kcwziwwcyzrh6ugx2.py
# Source Nodes: [attn], Original ATen: [aten.clone]
# attn => clone_3
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (384*x2) + (18816*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy42tdhay4py3ozjvujrt37y6pr3cb2nnz3gfpiybizy5bb234xt.py
# Source Nodes: [attn_1, attn_2, attn_3], Original ATen: [aten._softmax, aten.add, aten.clone]
# attn_1 => add_4
# attn_2 => amax, div, exp, sub_2, sum_1
# attn_3 => clone_5
triton_per_fused__softmax_add_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~rmask, "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (4*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask)
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5ebosdf6lku3wtfn3goc44s5neqiqd4lyntkf25gwnazjg24rq.py
# Source Nodes: [x_7], Original ATen: [aten.clone]
# x_7 => clone_6
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (384*x1) + (18816*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uqwp7ildxlc3nwap75oburqu3xgcfssi76olqdjdvumzypx5ym.py
# Source Nodes: [x_9], Original ATen: [aten.view]
# x_9 => view_15
triton_poi_fused_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 49)) + (1568*(x0 // 32)) + (6272*(x1 // 49)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/co/ccozxa223apz5rzkjeneoigxldodt3f5ljomkuqrrdiocngjio5r.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___norm2, x_14, x_16, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___0___blocks___0___norm2 => add_6, add_7, mul_5, mul_6, rsqrt_2, sub_3, var_mean_2
# x_14 => add_5
# x_16 => view_21
# x_4 => add_1, mul_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r3 + (128*(x0 % 7)) + (896*(x1 % 7)) + (6272*(x0 // 7)) + (50176*(x1 // 7)) + (401408*x2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr0 + (r3 + (128*x4)), tmp8, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (128*x4)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (r3 + (128*x4)), tmp35, rmask & xmask)
    tl.store(out_ptr5 + (x4), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5p5523ershrtcvhfnxeezoq6g5sxno556dqzowmuljljrc4rx3.py
# Source Nodes: [x_17, x_20], Original ATen: [aten.gelu, aten.view]
# x_17 => add_8, erf, mul_7, mul_8, mul_9
# x_20 => view_23
triton_poi_fused_gelu_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6whzdfvtci7es5izcs7dtn4mrogedljbim4uome6jd2onhravhm.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_l__mod___layers___0___blocks___1___norm1 => add_10, mul_10, rsqrt_3, sub_4, var_mean_3
triton_per_fused_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp28 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6i5s3nwx3pmwkaty4luhfpbhzbgp25vuputvrfflsucuxscdtt3.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv => view_29
triton_poi_fused_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((3 + (7*((x1 // 49) % 8)) + ((x1 % 49) % 7)) % 56)) + (7168*((3 + (7*((x1 // 392) % 8)) + ((x1 % 49) // 7)) % 56)) + (401408*(x1 // 3136))), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/cray37dmnbf3b6u46jvdhyt5tihn6ewgkq77324tjrshxwtl4zd3.py
# Source Nodes: [attn_8, attn_9], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_8 => amax_1, div_1, exp_1, sub_5, sum_2
# attn_9 => clone_16
triton_per_fused__softmax_clone_detach_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    x2 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 64))), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~rmask, "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (4*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask)
    tl.store(out_ptr4 + (r3 + (49*x4)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yn/cyn22zwyhos3ahpwwxrdat46ziukkalyajojgdeewln6it2o77w5.py
# Source Nodes: [random_tensor], Original ATen: [aten.bernoulli]
# random_tensor => bernoulli
triton_poi_fused_bernoulli_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bernoulli_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = float("nan")
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2r5lqi6jgil6a4cyifz4rcnbxejyg3p2pok2tjrzvjeu24qcekt.py
# Source Nodes: [div_, getattr_getattr_l__mod___layers___0___blocks___1___norm2, mul_2, x_31, x_32, x_34], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.roll, aten.view]
# div_ => div_2
# getattr_getattr_l__mod___layers___0___blocks___1___norm2 => add_15, add_16, mul_14, mul_15, rsqrt_4, sub_6, var_mean_4
# mul_2 => mul_13
# x_31 => roll_1
# x_32 => add_14
# x_34 => view_49
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_roll_view_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_roll_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r3 + (128*(((53 + x0) % 56) % 7)) + (896*(((53 + x1) % 56) % 7)) + (6272*(((53 + x0) % 56) // 7)) + (50176*(((53 + x1) % 56) // 7)) + (401408*x2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = 0.9956521736457944
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp4 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp34 / tmp30
    tl.store(out_ptr0 + (r3 + (128*x4)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (128*x4)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r3 + (128*x4)), tmp39, rmask & xmask)
    tl.store(out_ptr5 + (x4), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/35/c35azsphjyok34oftjdof7tz7kp5esam4xldt5rdttlflic6gplc.py
# Source Nodes: [x_44, x_46], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_44 => add_19, add_20, mul_20, mul_21, rsqrt_5, sub_7, var_mean_5
# x_46 => view_56
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x4 = (xindex // 28)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*(r3 // 256)) + (256*x0) + (7168*((((2*x0) + (56*((r3 // 128) % 2)) + (112*x1) + (r3 // 256)) // 56) % 56)) + (401408*x2) + (r3 % 128)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((128*(r3 // 256)) + (256*x0) + (7168*((r3 // 128) % 2)) + (14336*x4) + (r3 % 128)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r3 % 128), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9956521736457944
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r3 + (512*x5)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (512*x5)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x5), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpslvqxqqsorsn7vijidnfw2pde3ihjeznsmk7ixnemqwkzbxt7m.py
# Source Nodes: [shifted_x_8], Original ATen: [aten.native_layer_norm]
# shifted_x_8 => add_21, rsqrt_6, var_mean_6
triton_per_fused_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 6272
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 256.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nj/cnj53avbali5zge4u44flfosxezh2xinxfim6gtuxhu3kupjjsjj.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv => view_61
triton_poi_fused_view_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((x1 % 49) % 7)) + (1792*((x1 // 49) % 4)) + (7168*((x1 % 49) // 7)) + (50176*(x1 // 196))), None)
    tmp1 = tl.load(in_ptr1 + ((7*((x1 // 49) % 4)) + (28*((x1 % 49) // 7)) + (196*(x1 // 196)) + ((x1 % 49) % 7)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((7*((x1 // 49) % 4)) + (28*((x1 % 49) // 7)) + (196*(x1 // 196)) + ((x1 % 49) % 7)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23xnrtyt6slhqvk2x3w2dnkpnr6ula2xkpmp77joxqs5ea2d7ix.py
# Source Nodes: [attn_10, q_5], Original ATen: [aten.clone, aten.mul]
# attn_10 => clone_25
# q_5 => mul_24
triton_poi_fused_clone_mul_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (37632*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/car6r4e3cl22yusr35wdvl2qudibfv2qi6a7yhogynjusufuyayt.py
# Source Nodes: [attn_10], Original ATen: [aten.clone]
# attn_10 => clone_26
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lxk2peu36hig6npjykpwthjxers5664snloqionnajqxejmqqg.py
# Source Nodes: [attn_11, attn_12, attn_13], Original ATen: [aten._softmax, aten.add, aten.clone]
# attn_11 => add_23
# attn_12 => amax_2, div_4, exp_2, sub_9, sum_3
# attn_13 => clone_28
triton_per_fused__softmax_add_clone_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (8*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyy2ihc7aibetj6e26i2hfzojt755vo6vxqjhgv3dgdo2xo3hmhn.py
# Source Nodes: [x_48], Original ATen: [aten.clone]
# x_48 => clone_29
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (37632*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ax/caxyzembdpj5xaakc5a246lnbmdko6qnuvniz3np67i5iljx7gpv.py
# Source Nodes: [x_50], Original ATen: [aten.view]
# x_50 => view_73
triton_poi_fused_view_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 49)) + (1568*(x0 // 32)) + (12544*(x1 // 49)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucqjfpcivm6tqhohgomslxtrah7kiobwv2igvfj66eisr6eip5y.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___norm2, x_57], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___1___blocks___0___norm2 => add_25, add_26, mul_26, mul_27, rsqrt_7, sub_10, var_mean_7
# x_57 => view_79
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*((x0 // 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(x0 // 196)) + (200704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9913043472915888
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhrjp3bbpceijpqlaieg7qinogooh2o4ziyolg2milub7mpoplc.py
# Source Nodes: [x_58, x_61], Original ATen: [aten.gelu, aten.view]
# x_58 => add_27, erf_2, mul_28, mul_29, mul_30
# x_61 => view_81
triton_poi_fused_gelu_view_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqfvliblpnwdxfhyzgkew246ivgucrwg25zyzryxptoxclysmik.py
# Source Nodes: [div__3, getattr_getattr_l__mod___layers___1___blocks___1___norm1, mul_6, x_63], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__3 => div_6
# getattr_getattr_l__mod___layers___1___blocks___1___norm1 => add_29, mul_32, rsqrt_8, sub_11, var_mean_8
# mul_6 => mul_31
# x_63 => add_28
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*((x0 // 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(x0 // 196)) + (200704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9913043472915888
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 256, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 256.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgoa7ntdxoqu6wlnq3falzbu4yvr66qzygyx5iqcj4tk3b53sle4.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv => view_87
triton_poi_fused_view_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((3 + (7*((x1 // 49) % 4)) + ((x1 % 49) % 7)) % 28)) + (7168*((3 + (7*((x1 // 196) % 4)) + ((x1 % 49) // 7)) % 28)) + (200704*(x1 // 784))), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fnb5nha2axqhwumukcv64ystpayfxi6nxypdciuglmaawjuqpg.py
# Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_18 => amax_3, div_7, exp_3, sub_12, sum_4
# attn_19 => clone_39
triton_per_fused__softmax_clone_detach_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    x2 = (xindex // 392)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 16))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (8*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask & xmask)
    tl.store(out_ptr4 + (r3 + (49*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/ckltbilbluui2uznnhankynho7kdjt7pwcaek747bbsg62hr2ml5.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm2, x_75], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___1___blocks___1___norm2 => add_34, add_35, mul_36, mul_37, rsqrt_9, sub_13, var_mean_9
# x_75 => view_107
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*(((25 + (x0 % 28)) % 28) % 7)) + (1792*(((25 + (x0 // 28)) % 28) % 7)) + (12544*(((25 + (x0 % 28)) % 28) // 7)) + (50176*(((25 + (x0 // 28)) % 28) // 7)) + (200704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9869565209373832
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jm2y6lyle5mna6cvkjrr3g5vcmoxpsi5q3m4ickwt2dgp3yf2x.py
# Source Nodes: [x_84], Original ATen: [aten.clone]
# x_84 => clone_46
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(((25 + x1) % 28) % 7)) + (1792*(((25 + x2) % 28) % 7)) + (12544*(((25 + x1) % 28) // 7)) + (50176*(((25 + x2) % 28) // 7)) + (200704*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x4), None)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9869565209373832
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tl.store(in_out_ptr0 + (x4), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3s6mbrytjxhz6nhbdkefxbs23anozjylq4g2645zox74ijfd6zs.py
# Source Nodes: [x_85, x_87], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_85 => add_38, add_39, mul_42, mul_43, rsqrt_10, sub_14, var_mean_10
# x_87 => view_114
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(r2 // 512)) + (512*x0) + (7168*((r2 // 256) % 2)) + (14336*x1) + (r2 % 256)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 1024.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp27, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvek2pigh4m4fem2nnfdwwaftizcg4hvvopf5nm2q4352qgs62c.py
# Source Nodes: [shifted_x_16], Original ATen: [aten.native_layer_norm]
# shifted_x_16 => add_40, rsqrt_11, var_mean_11
triton_per_fused_native_layer_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1568
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crngpti2rym3hs4eonbefbgex5m22ngowt6wyopdpulxh5kqk5fb.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv => view_119
triton_poi_fused_view_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((x1 % 49) % 7)) + (3584*((x1 // 49) % 2)) + (7168*((x1 % 49) // 7)) + (50176*(x1 // 98))), None)
    tmp1 = tl.load(in_ptr1 + ((7*((x1 // 49) % 2)) + (14*((x1 % 49) // 7)) + (98*(x1 // 98)) + ((x1 % 49) % 7)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((7*((x1 // 49) % 2)) + (14*((x1 % 49) // 7)) + (98*(x1 // 98)) + ((x1 % 49) % 7)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hs56lcsviuub5svqqiqujg2sc5dx5dedujxgeexwh5dsiiyip7.py
# Source Nodes: [attn_20, q_9], Original ATen: [aten.clone, aten.mul]
# attn_20 => clone_48
# q_9 => mul_46
triton_poi_fused_clone_mul_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1536*x1) + (75264*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cec2lapvro5wb65l2tncdmarkwnkovy2x5kpxokvay7zknf3gc4s.py
# Source Nodes: [attn_20], Original ATen: [aten.clone]
# attn_20 => clone_49
triton_poi_fused_clone_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkkd7nsritngzrp3mvjnddm7f464psuxv47jnz7m5jpmc4znxhu.py
# Source Nodes: [attn_21, attn_22, attn_23], Original ATen: [aten._softmax, aten.add, aten.clone]
# attn_21 => add_42
# attn_22 => amax_4, div_10, exp_4, sub_16, sum_5
# attn_23 => clone_51
triton_per_fused__softmax_add_clone_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (16*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5d5dfsl5eb5w7ym7c25qipwyzcmxbu7cf5xnyt7c2h3teippop.py
# Source Nodes: [x_89], Original ATen: [aten.clone]
# x_89 => clone_52
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (32*x2) + (1536*x1) + (75264*x3)), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4j25lm5rcydadui5xjmhtcqjhvlf3gq2vxdioweyoyzv2nfyn3u.py
# Source Nodes: [x_91], Original ATen: [aten.view]
# x_91 => view_131
triton_poi_fused_view_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 49)) + (1568*(x0 // 32)) + (25088*(x1 // 49)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cipekcv3w36ap37ul6yghodtv7p3oeaane3t77ht3ekwvzputwi3.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___norm2, x_98], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___0___norm2 => add_44, add_45, mul_48, mul_49, rsqrt_12, sub_17, var_mean_12
# x_98 => view_137
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9826086945831776
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnh6pkppup5avxuc2xex2yl7xwf6q2ziiuvvjzumqb5i64g6pem3.py
# Source Nodes: [x_102, x_99], Original ATen: [aten.gelu, aten.view]
# x_102 => view_139
# x_99 => add_46, erf_4, mul_50, mul_51, mul_52
triton_poi_fused_gelu_view_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5cpstsygep4ptc24u7kjz3hs5gssvjcu34dppbvv4eq6c7f4du.py
# Source Nodes: [div__7, getattr_getattr_l__mod___layers___2___blocks___1___norm1, mul_12, x_104], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__7 => div_12
# getattr_getattr_l__mod___layers___2___blocks___1___norm1 => add_48, mul_54, rsqrt_13, sub_18, var_mean_13
# mul_12 => mul_53
# x_104 => add_47
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9826086945831776
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63ig22eorjlkppvwisvrb4segdnflmh5fsqslyqy5aelqkmtdns.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv => view_145
triton_poi_fused_view_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + (7*((x1 // 49) % 2)) + ((x1 % 49) % 7)) % 14)) + (7168*((3 + (7*((x1 // 98) % 2)) + ((x1 % 49) // 7)) % 14)) + (100352*(x1 // 196))), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cuagpecuplveavuueulfx7sp2bndznu74aykcldqt6r3n567c33t.py
# Source Nodes: [attn_28, attn_29], Original ATen: [aten._softmax, aten.clone, aten.detach]
# attn_28 => amax_5, div_13, exp_5, sub_19, sum_6
# attn_29 => clone_62
triton_per_fused__softmax_clone_detach_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 16
    x2 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 4))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (16*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask & xmask)
    tl.store(out_ptr4 + (r3 + (49*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co65y7u6xflhhb234eeudxwvdpvttluriswppzmq476scbidouku.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___norm2, x_116], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___1___norm2 => add_53, add_54, mul_58, mul_59, rsqrt_14, sub_20, var_mean_14
# x_116 => view_165
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9782608672976494
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch56yhurbhce4apmxgrjrmosvmlimoct227x26dmwxaky4uwn4f3.py
# Source Nodes: [div__9, mul_15, shifted_x_24, x_122], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__9 => div_15
# mul_15 => mul_63
# shifted_x_24 => add_57, mul_64, rsqrt_15, sub_21, var_mean_15
# x_122 => add_56
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_43', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9782608672976494
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yo/cyoixczhkfr3ctmmeppjnkkcpbdwik6r3qrjrqbecqmx5ovutmlz.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv], Original ATen: [aten.view]
# getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv => view_173
triton_poi_fused_view_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((x1 % 49) % 7)) + (3584*((x1 // 49) % 2)) + (7168*((x1 % 49) // 7)) + (50176*(x1 // 98))), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmimvs63jtvgglgr2bxagrwn7dp56q4zqhkb7bwtq7camnebv2sn.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___2___norm2, x_134], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___2___norm2 => add_61, add_62, mul_68, mul_69, rsqrt_16, sub_23, var_mean_16
# x_134 => view_191
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9739130418747663
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxhsvevdtyrmmzoqjzlqpm5e5e4p3clw2d3fwt2vieqkk2hxdjn.py
# Source Nodes: [div__11, getattr_getattr_l__mod___layers___2___blocks___3___norm1, mul_18, x_140], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__11 => div_18
# getattr_getattr_l__mod___layers___2___blocks___3___norm1 => add_65, mul_74, rsqrt_17, sub_24, var_mean_17
# mul_18 => mul_73
# x_140 => add_64
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9739130418747663
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2zuxx3x2buwjduzxxvp3udl2aajzcmga7gctf56dcsgfsoverr.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___3___norm2, x_152], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___3___norm2 => add_70, add_71, mul_78, mul_79, rsqrt_18, sub_26, var_mean_18
# x_152 => view_219
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9695652164518833
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cav4p5olis4twygqswi7jwswfyy47wpthtwjxqu3qjkpnxjp33yw.py
# Source Nodes: [div__13, mul_21, shifted_x_32, x_158], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__13 => div_21
# mul_21 => mul_83
# shifted_x_32 => add_74, mul_84, rsqrt_19, sub_27, var_mean_19
# x_158 => add_73
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9695652164518833
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhioemdprspwc3tza5bawv2nt5fa2yhd3fmbcf6gupfqtfab7ex.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___4___norm2, x_170], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___4___norm2 => add_78, add_79, mul_88, mul_89, rsqrt_20, sub_29, var_mean_20
# x_170 => view_245
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9652173891663551
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cduks2jlefnqyzxgsawsrfyyosljl6efd6zdx2rgewzvets2f5yk.py
# Source Nodes: [div__15, getattr_getattr_l__mod___layers___2___blocks___5___norm1, mul_24, x_176], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__15 => div_24
# getattr_getattr_l__mod___layers___2___blocks___5___norm1 => add_82, mul_94, rsqrt_21, sub_30, var_mean_21
# mul_24 => mul_93
# x_176 => add_81
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9652173891663551
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jq/cjqvcxvnejoe3ccrwxpv2y672qdhxjdon65y6upf25mgaoozs3jy.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___5___norm2, x_188], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___5___norm2 => add_87, add_88, mul_98, mul_99, rsqrt_22, sub_32, var_mean_22
# x_188 => view_273
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.960869561880827
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhayhv3df2gtwnwp4vxbqp23y7w6nbhidwhf6t3zs2uulu7efo2.py
# Source Nodes: [div__17, mul_27, shifted_x_40, x_194], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__17 => div_27
# mul_27 => mul_103
# shifted_x_40 => add_91, mul_104, rsqrt_23, sub_33, var_mean_23
# x_194 => add_90
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.960869561880827
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvntgezzqwtc74cxovv5pmqagvdh6mxf5kha72eckwpgxrmowaq.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___6___norm2, x_206], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___6___norm2 => add_95, add_96, mul_108, mul_109, rsqrt_24, sub_35, var_mean_24
# x_206 => view_299
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9565217345952988
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zr/czrpjyocit5nwb4fe4o5fxyzxvrpenjgezl73tqao565fjyce7ef.py
# Source Nodes: [div__19, getattr_getattr_l__mod___layers___2___blocks___7___norm1, mul_30, x_212], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__19 => div_30
# getattr_getattr_l__mod___layers___2___blocks___7___norm1 => add_99, mul_114, rsqrt_25, sub_36, var_mean_25
# mul_30 => mul_113
# x_212 => add_98
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9565217345952988
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4lm53a225dnnbvunfcir26zm54kptp4z5txjqlttuhj4bix5eh.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___7___norm2, x_224], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___7___norm2 => add_104, add_105, mul_118, mul_119, rsqrt_26, sub_38, var_mean_26
# x_224 => view_327
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9521739110350609
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqd34bdmy6lhkx2taawxuavywavbezajybiubdvwewbwecjb7d2.py
# Source Nodes: [div__21, mul_33, shifted_x_48, x_230], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__21 => div_33
# mul_33 => mul_123
# shifted_x_48 => add_108, mul_124, rsqrt_27, sub_39, var_mean_27
# x_230 => add_107
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9521739110350609
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/cso5aux3sueya4di3ergiwsk3r4wrigjzscalzlsx5strgl7dger.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___8___norm2, x_242], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___8___norm2 => add_112, add_113, mul_128, mul_129, rsqrt_28, sub_41, var_mean_28
# x_242 => view_353
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.947826087474823
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyq63ifci25uqwm4troq7ohp6keykcmhfrtthwhdezzsvxbmw4tq.py
# Source Nodes: [div__23, getattr_getattr_l__mod___layers___2___blocks___9___norm1, mul_36, x_248], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__23 => div_36
# getattr_getattr_l__mod___layers___2___blocks___9___norm1 => add_116, mul_134, rsqrt_29, sub_42, var_mean_29
# mul_36 => mul_133
# x_248 => add_115
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.947826087474823
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czggknyk4oaqwkeyr7bcc6tr3rlr66lpx5kha5eslbvdqwnutu6f.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___9___norm2, x_260], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___9___norm2 => add_121, add_122, mul_138, mul_139, rsqrt_30, sub_44, var_mean_30
# x_260 => view_381
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9434782639145851
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2kzah4djgugbf56ze2sb5iadvlppwbmydjc7x6odxbaosc633i.py
# Source Nodes: [div__25, mul_39, shifted_x_56, x_266], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__25 => div_39
# mul_39 => mul_143
# shifted_x_56 => add_125, mul_144, rsqrt_31, sub_45, var_mean_31
# x_266 => add_124
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_60', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9434782639145851
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmj457ja355ilv6prc5yfv5gfymtwvt44ug3ygmcvfutd5w3d6b.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___10___norm2, x_278], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___10___norm2 => add_129, add_130, mul_148, mul_149, rsqrt_32, sub_47, var_mean_32
# x_278 => view_407
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9391304366290569
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cydyigq6yarhd5qxhzefwygnfddvda6o5auycty4wumsk54tklrt.py
# Source Nodes: [div__27, getattr_getattr_l__mod___layers___2___blocks___11___norm1, mul_42, x_284], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__27 => div_42
# getattr_getattr_l__mod___layers___2___blocks___11___norm1 => add_133, mul_154, rsqrt_33, sub_48, var_mean_33
# mul_42 => mul_153
# x_284 => add_132
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9391304366290569
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5df5y2hkla3bu2tyq5dbicajgxkv5zssow2zcgfitx4w75wczo.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___11___norm2, x_296], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___11___norm2 => add_138, add_139, mul_158, mul_159, rsqrt_34, sub_50, var_mean_34
# x_296 => view_435
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9347826093435287
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3jofz7st5q7tv3tzl72klwn4lq4yorw64zlsbnv3fjaqjhurhk.py
# Source Nodes: [div__29, mul_45, shifted_x_64, x_302], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__29 => div_45
# mul_45 => mul_163
# shifted_x_64 => add_142, mul_164, rsqrt_35, sub_51, var_mean_35
# x_302 => add_141
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_64', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9347826093435287
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrrtkcy6qpmmdz5fyzoebazqdyucq4oz2brpetr3xr6lizjvrvx.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___12___norm2, x_314], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___12___norm2 => add_146, add_147, mul_168, mul_169, rsqrt_36, sub_53, var_mean_36
# x_314 => view_461
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9304347857832909
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvfp4xzf75254cfm35ddmthdqh67n2tb7flds2jgvj2yj7zq3av.py
# Source Nodes: [div__31, getattr_getattr_l__mod___layers___2___blocks___13___norm1, mul_48, x_320], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__31 => div_48
# getattr_getattr_l__mod___layers___2___blocks___13___norm1 => add_150, mul_174, rsqrt_37, sub_54, var_mean_37
# mul_48 => mul_173
# x_320 => add_149
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_66', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9304347857832909
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjom4qgd2foe3a4zd5q6d6ddqllxobfdlal7iq3gxruyfyvpluky.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___13___norm2, x_332], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___13___norm2 => add_155, add_156, mul_178, mul_179, rsqrt_38, sub_56, var_mean_38
# x_332 => view_489
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9260869547724724
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbsd52ozcjgiveykulghosiwqqzz5gexw3tfmjb36vrch5qglax.py
# Source Nodes: [div__33, mul_51, shifted_x_72, x_338], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__33 => div_51
# mul_51 => mul_183
# shifted_x_72 => add_159, mul_184, rsqrt_39, sub_57, var_mean_39
# x_338 => add_158
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_68', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9260869547724724
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutfoeqnr2zj332h6ajeosid737dwguochtw5djitmaajir4qfoe.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___14___norm2, x_350], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___14___norm2 => add_163, add_164, mul_188, mul_189, rsqrt_40, sub_59, var_mean_40
# x_350 => view_515
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9217391312122345
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6iinievklirvlms6e7elzp3ypugaxrdqqutkdvnppyh2loaxdu.py
# Source Nodes: [div__35, getattr_getattr_l__mod___layers___2___blocks___15___norm1, mul_54, x_356], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__35 => div_54
# getattr_getattr_l__mod___layers___2___blocks___15___norm1 => add_167, mul_194, rsqrt_41, sub_60, var_mean_41
# mul_54 => mul_193
# x_356 => add_166
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_70', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9217391312122345
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ta/ctaka7p2hgdfjrvlwp3rxtmhie7w4qkzxqhvipywbqdd7cark6n2.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___15___norm2, x_368], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___15___norm2 => add_172, add_173, mul_198, mul_199, rsqrt_42, sub_62, var_mean_42
# x_368 => view_543
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.917391300201416
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7a/c7a5crdxcozmzva36rwxo5c6exfqwjhsrhyag7qzvxkbc25moj36.py
# Source Nodes: [div__37, mul_57, shifted_x_80, x_374], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__37 => div_57
# mul_57 => mul_203
# shifted_x_80 => add_176, mul_204, rsqrt_43, sub_63, var_mean_43
# x_374 => add_175
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.917391300201416
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepo3tpiuc6ip5kuqimtqognrh6hr5kynqnipxsmdmjdvoe2s5zh.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___16___norm2, x_386], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___16___norm2 => add_180, add_181, mul_208, mul_209, rsqrt_44, sub_65, var_mean_44
# x_386 => view_569
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9130434766411781
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yn/cynn6rxzusznnihkjxk7qgl54o2avvxoact26laxkbpgg5ehwa7i.py
# Source Nodes: [div__39, getattr_getattr_l__mod___layers___2___blocks___17___norm1, mul_60, x_392], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__39 => div_60
# getattr_getattr_l__mod___layers___2___blocks___17___norm1 => add_184, mul_214, rsqrt_45, sub_66, var_mean_45
# mul_60 => mul_213
# x_392 => add_183
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_74', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9130434766411781
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 512, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 512.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxrqkb5ppvq6ozqw4bzkesb5njyk7b55lt5ieujyj42hwmts3w6.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___17___norm2, x_404], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___2___blocks___17___norm2 => add_189, add_190, mul_218, mul_219, rsqrt_46, sub_68, var_mean_46
# x_404 => view_597
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9086956530809402
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cnivlh7aid4miid4w5fioux5rrpakfkdkedpmytrtik7lb6bhmyd.py
# Source Nodes: [x_413], Original ATen: [aten.clone]
# x_413 => clone_245
triton_poi_fused_clone_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_76', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 14
    x2 = (xindex // 7168) % 14
    x3 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*(((11 + x1) % 14) % 7)) + (3584*(((11 + x2) % 14) % 7)) + (25088*(((11 + x1) % 14) // 7)) + (50176*(((11 + x2) % 14) // 7)) + (100352*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x4), None)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9086956530809402
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tl.store(in_out_ptr0 + (x4), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coioi5sepra5vasp32bc6n6u62honeolx6vkpbz5ihsivm7pax73.py
# Source Nodes: [x_414, x_416], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_414 => add_193, add_194, mul_224, mul_225, rsqrt_47, sub_69, var_mean_47
# x_416 => view_604
triton_red_fused_native_layer_norm_native_layer_norm_backward_view_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_view_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((512*(r2 // 1024)) + (1024*x0) + (7168*((r2 // 512) % 2)) + (14336*x1) + (r2 % 512)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + ((512*(r2 // 1024)) + (1024*x0) + (7168*((r2 // 512) % 2)) + (14336*x1) + (r2 % 512)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 2048.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp12, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp16, rmask & xmask)
    tmp17 = 2048.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp21 / tmp17
    tl.store(out_ptr4 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qhw4l4bxxer32ns4vujmf7k64w7kiku7tmgzor6rod7ycwjjuo.py
# Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv, shifted_x_88], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv => view_609
# shifted_x_88 => add_195, rsqrt_48, var_mean_48
triton_per_fused_native_layer_norm_view_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_view_78', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 392
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
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 1024.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqr5zx7wzdu5s2bmv37a2a6dvcna3lvlzgfpy5plexn6ywul5flt.py
# Source Nodes: [attn_110, q_45], Original ATen: [aten.clone, aten.mul]
# attn_110 => clone_246
# q_45 => mul_228
triton_poi_fused_clone_mul_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (3072*x1) + (150528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/claznzylyk3nymqk4szidcwdznuvvkgu2mu7gfo4olug7d37btq6.py
# Source Nodes: [attn_110], Original ATen: [aten.clone]
# attn_110 => clone_247
triton_poi_fused_clone_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (1024 + y0 + (3072*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1024 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46ojbjktcfpll4ts3dxhoakapgmzfo3u4tstrhsshmpvwkmv6tb.py
# Source Nodes: [attn_111, attn_112, attn_113], Original ATen: [aten._softmax, aten.add, aten.clone]
# attn_111 => add_197
# attn_112 => amax_22, div_64, exp_22, sub_71, sum_23
# attn_113 => clone_249
triton_per_fused__softmax_add_clone_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_clone_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 32
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (32*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czq6lb7klmfgoijeqtx3fy3awmxq2vh544izrziyyi33gqkcehb7.py
# Source Nodes: [x_418], Original ATen: [aten.clone]
# x_418 => clone_250
triton_poi_fused_clone_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (32*x2) + (3072*x1) + (150528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (2048 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5recj67rwsqnatczl2htwygf4jwjwmf3allogbpqskwctn7jdu.py
# Source Nodes: [x_420], Original ATen: [aten.view]
# x_420 => view_621
triton_poi_fused_view_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 49)) + (1568*(x0 // 32)) + (50176*(x1 // 49)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjbgmj6ihkfrv7vcrro6fbug7fvm7alsxvoitzxkxdknmuk5auh.py
# Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___norm2, x_427], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___3___blocks___0___norm2 => add_199, add_200, mul_230, mul_231, rsqrt_49, sub_72, var_mean_49
# x_427 => view_627
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9043478220701218
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5z5uhbvuhia77agoio5bnldb362wju6gxnpvaqftfm4ji7nzbn.py
# Source Nodes: [x_428, x_431], Original ATen: [aten.gelu, aten.view]
# x_428 => add_201, erf_22, mul_232, mul_233, mul_234
# x_431 => view_629
triton_poi_fused_gelu_view_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxqc42avogsibk5yaeb6lknwip2peclmdzxmlh4cfbdlr4fznyb.py
# Source Nodes: [div__43, getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv, mul_66, shifted_x_92, x_433], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__43 => div_66
# getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv => view_635
# mul_66 => mul_235
# shifted_x_92 => add_203, mul_236, rsqrt_50, sub_73, var_mean_50
# x_433 => add_202
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_86', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9043478220701218
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 1024, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 1024.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp42, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp43, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4ezxd7tfk77kopdsp4jj54pktazycz4dtbp6nkruxezpytrl23i.py
# Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___1___norm2, x_445], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___layers___3___blocks___1___norm2 => add_207, add_208, mul_240, mul_241, rsqrt_51, sub_75, var_mean_51
# x_445 => view_653
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_87 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8999999985098839
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (1024*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqu2i26hganzbhygodivya5snmyl5oqjfqtugxua6xtz5afqfns.py
# Source Nodes: [div__45, mul_69, x_451, x_456], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# div__45 => div_69
# mul_69 => mul_245
# x_451 => add_210
# x_456 => add_211, mul_246, rsqrt_52, sub_76, var_mean_52
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_88', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (1024*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8999999985098839
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 1024, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp15 - tmp25
    tmp33 = 1024.0
    tmp34 = tmp31 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = tl.math.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp39 = tmp37 / tmp33
    tl.store(in_out_ptr0 + (r2 + (1024*x3)), tmp15, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp39, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5ly5tphopzx5syr24apmdlmy3kigisrh3sbohl4lz5jv25hznmg.py
# Source Nodes: [x_456, x_457], Original ATen: [aten.mean, aten.native_layer_norm]
# x_456 => add_212, mul_247
# x_457 => mean
triton_per_fused_mean_native_layer_norm_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_89', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365 = args
    args.clear()
    assert_size_stride(primals_1, (169, 4), (4, 1))
    assert_size_stride(primals_2, (169, 4), (4, 1))
    assert_size_stride(primals_3, (169, 8), (8, 1))
    assert_size_stride(primals_4, (169, 8), (8, 1))
    assert_size_stride(primals_5, (169, 16), (16, 1))
    assert_size_stride(primals_6, (169, 16), (16, 1))
    assert_size_stride(primals_7, (169, 16), (16, 1))
    assert_size_stride(primals_8, (169, 16), (16, 1))
    assert_size_stride(primals_9, (169, 16), (16, 1))
    assert_size_stride(primals_10, (169, 16), (16, 1))
    assert_size_stride(primals_11, (169, 16), (16, 1))
    assert_size_stride(primals_12, (169, 16), (16, 1))
    assert_size_stride(primals_13, (169, 16), (16, 1))
    assert_size_stride(primals_14, (169, 16), (16, 1))
    assert_size_stride(primals_15, (169, 16), (16, 1))
    assert_size_stride(primals_16, (169, 16), (16, 1))
    assert_size_stride(primals_17, (169, 16), (16, 1))
    assert_size_stride(primals_18, (169, 16), (16, 1))
    assert_size_stride(primals_19, (169, 16), (16, 1))
    assert_size_stride(primals_20, (169, 16), (16, 1))
    assert_size_stride(primals_21, (169, 16), (16, 1))
    assert_size_stride(primals_22, (169, 16), (16, 1))
    assert_size_stride(primals_23, (169, 32), (32, 1))
    assert_size_stride(primals_24, (169, 32), (32, 1))
    assert_size_stride(primals_25, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (384, 128), (128, 1))
    assert_size_stride(primals_32, (384, ), (1, ))
    assert_size_stride(primals_33, (128, 128), (128, 1))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (512, 128), (128, 1))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (128, 512), (512, 1))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, ), (1, ))
    assert_size_stride(primals_43, (384, 128), (128, 1))
    assert_size_stride(primals_44, (384, ), (1, ))
    assert_size_stride(primals_45, (128, 128), (128, 1))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (512, 128), (128, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (128, 512), (512, 1))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (256, 512), (512, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (768, 256), (256, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (256, 256), (256, 1))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (1024, 256), (256, 1))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_66, (256, 1024), (1024, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (768, 256), (256, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (256, 256), (256, 1))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (1024, 256), (256, 1))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (256, 1024), (1024, 1))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (512, 1024), (1024, 1))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (1536, 512), (512, 1))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_87, (512, 512), (512, 1))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (2048, 512), (512, 1))
    assert_size_stride(primals_92, (2048, ), (1, ))
    assert_size_stride(primals_93, (512, 2048), (2048, 1))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (1536, 512), (512, 1))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_99, (512, 512), (512, 1))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (2048, 512), (512, 1))
    assert_size_stride(primals_104, (2048, ), (1, ))
    assert_size_stride(primals_105, (512, 2048), (2048, 1))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (1536, 512), (512, 1))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_111, (512, 512), (512, 1))
    assert_size_stride(primals_112, (512, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (2048, 512), (512, 1))
    assert_size_stride(primals_116, (2048, ), (1, ))
    assert_size_stride(primals_117, (512, 2048), (2048, 1))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (1536, 512), (512, 1))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_123, (512, 512), (512, 1))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (2048, 512), (512, 1))
    assert_size_stride(primals_128, (2048, ), (1, ))
    assert_size_stride(primals_129, (512, 2048), (2048, 1))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (1536, 512), (512, 1))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_135, (512, 512), (512, 1))
    assert_size_stride(primals_136, (512, ), (1, ))
    assert_size_stride(primals_137, (512, ), (1, ))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (2048, 512), (512, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (512, 2048), (2048, 1))
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (1536, 512), (512, 1))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (512, 512), (512, 1))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (2048, 512), (512, 1))
    assert_size_stride(primals_152, (2048, ), (1, ))
    assert_size_stride(primals_153, (512, 2048), (2048, 1))
    assert_size_stride(primals_154, (512, ), (1, ))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (1536, 512), (512, 1))
    assert_size_stride(primals_158, (1536, ), (1, ))
    assert_size_stride(primals_159, (512, 512), (512, 1))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, ), (1, ))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (2048, 512), (512, 1))
    assert_size_stride(primals_164, (2048, ), (1, ))
    assert_size_stride(primals_165, (512, 2048), (2048, 1))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (1536, 512), (512, 1))
    assert_size_stride(primals_170, (1536, ), (1, ))
    assert_size_stride(primals_171, (512, 512), (512, 1))
    assert_size_stride(primals_172, (512, ), (1, ))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (2048, 512), (512, 1))
    assert_size_stride(primals_176, (2048, ), (1, ))
    assert_size_stride(primals_177, (512, 2048), (2048, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (1536, 512), (512, 1))
    assert_size_stride(primals_182, (1536, ), (1, ))
    assert_size_stride(primals_183, (512, 512), (512, 1))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (2048, 512), (512, 1))
    assert_size_stride(primals_188, (2048, ), (1, ))
    assert_size_stride(primals_189, (512, 2048), (2048, 1))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (1536, 512), (512, 1))
    assert_size_stride(primals_194, (1536, ), (1, ))
    assert_size_stride(primals_195, (512, 512), (512, 1))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (512, ), (1, ))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (2048, 512), (512, 1))
    assert_size_stride(primals_200, (2048, ), (1, ))
    assert_size_stride(primals_201, (512, 2048), (2048, 1))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (1536, 512), (512, 1))
    assert_size_stride(primals_206, (1536, ), (1, ))
    assert_size_stride(primals_207, (512, 512), (512, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (2048, 512), (512, 1))
    assert_size_stride(primals_212, (2048, ), (1, ))
    assert_size_stride(primals_213, (512, 2048), (2048, 1))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (1536, 512), (512, 1))
    assert_size_stride(primals_218, (1536, ), (1, ))
    assert_size_stride(primals_219, (512, 512), (512, 1))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, ), (1, ))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (2048, 512), (512, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_225, (512, 2048), (2048, 1))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (1536, 512), (512, 1))
    assert_size_stride(primals_230, (1536, ), (1, ))
    assert_size_stride(primals_231, (512, 512), (512, 1))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (2048, 512), (512, 1))
    assert_size_stride(primals_236, (2048, ), (1, ))
    assert_size_stride(primals_237, (512, 2048), (2048, 1))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (1536, 512), (512, 1))
    assert_size_stride(primals_242, (1536, ), (1, ))
    assert_size_stride(primals_243, (512, 512), (512, 1))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (2048, 512), (512, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_249, (512, 2048), (2048, 1))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (512, ), (1, ))
    assert_size_stride(primals_252, (512, ), (1, ))
    assert_size_stride(primals_253, (1536, 512), (512, 1))
    assert_size_stride(primals_254, (1536, ), (1, ))
    assert_size_stride(primals_255, (512, 512), (512, 1))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, ), (1, ))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (2048, 512), (512, 1))
    assert_size_stride(primals_260, (2048, ), (1, ))
    assert_size_stride(primals_261, (512, 2048), (2048, 1))
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (1536, 512), (512, 1))
    assert_size_stride(primals_266, (1536, ), (1, ))
    assert_size_stride(primals_267, (512, 512), (512, 1))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (2048, 512), (512, 1))
    assert_size_stride(primals_272, (2048, ), (1, ))
    assert_size_stride(primals_273, (512, 2048), (2048, 1))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (1536, 512), (512, 1))
    assert_size_stride(primals_278, (1536, ), (1, ))
    assert_size_stride(primals_279, (512, 512), (512, 1))
    assert_size_stride(primals_280, (512, ), (1, ))
    assert_size_stride(primals_281, (512, ), (1, ))
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (2048, 512), (512, 1))
    assert_size_stride(primals_284, (2048, ), (1, ))
    assert_size_stride(primals_285, (512, 2048), (2048, 1))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (1536, 512), (512, 1))
    assert_size_stride(primals_290, (1536, ), (1, ))
    assert_size_stride(primals_291, (512, 512), (512, 1))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (2048, 512), (512, 1))
    assert_size_stride(primals_296, (2048, ), (1, ))
    assert_size_stride(primals_297, (512, 2048), (2048, 1))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (2048, ), (1, ))
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (1024, 2048), (2048, 1))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (3072, 1024), (1024, 1))
    assert_size_stride(primals_305, (3072, ), (1, ))
    assert_size_stride(primals_306, (1024, 1024), (1024, 1))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_309, (1024, ), (1, ))
    assert_size_stride(primals_310, (4096, 1024), (1024, 1))
    assert_size_stride(primals_311, (4096, ), (1, ))
    assert_size_stride(primals_312, (1024, 4096), (4096, 1))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (3072, 1024), (1024, 1))
    assert_size_stride(primals_317, (3072, ), (1, ))
    assert_size_stride(primals_318, (1024, 1024), (1024, 1))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (4096, 1024), (1024, 1))
    assert_size_stride(primals_323, (4096, ), (1, ))
    assert_size_stride(primals_324, (1024, 4096), (4096, 1))
    assert_size_stride(primals_325, (1024, ), (1, ))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (1000, 1024), (1024, 1))
    assert_size_stride(primals_329, (1000, ), (1, ))
    assert_size_stride(primals_330, (49, 49), (49, 1))
    assert_size_stride(primals_331, (64, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_332, (49, 49), (49, 1))
    assert_size_stride(primals_333, (49, 49), (49, 1))
    assert_size_stride(primals_334, (16, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_335, (49, 49), (49, 1))
    assert_size_stride(primals_336, (49, 49), (49, 1))
    assert_size_stride(primals_337, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_338, (49, 49), (49, 1))
    assert_size_stride(primals_339, (49, 49), (49, 1))
    assert_size_stride(primals_340, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_341, (49, 49), (49, 1))
    assert_size_stride(primals_342, (49, 49), (49, 1))
    assert_size_stride(primals_343, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_344, (49, 49), (49, 1))
    assert_size_stride(primals_345, (49, 49), (49, 1))
    assert_size_stride(primals_346, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_347, (49, 49), (49, 1))
    assert_size_stride(primals_348, (49, 49), (49, 1))
    assert_size_stride(primals_349, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_350, (49, 49), (49, 1))
    assert_size_stride(primals_351, (49, 49), (49, 1))
    assert_size_stride(primals_352, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_353, (49, 49), (49, 1))
    assert_size_stride(primals_354, (49, 49), (49, 1))
    assert_size_stride(primals_355, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_356, (49, 49), (49, 1))
    assert_size_stride(primals_357, (49, 49), (49, 1))
    assert_size_stride(primals_358, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_359, (49, 49), (49, 1))
    assert_size_stride(primals_360, (49, 49), (49, 1))
    assert_size_stride(primals_361, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(primals_362, (49, 49), (49, 1))
    assert_size_stride(primals_363, (49, 49), (49, 1))
    assert_size_stride(primals_364, (49, 49), (49, 1))
    assert_size_stride(primals_365, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_365, primals_25, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf4 = empty((8, 56, 56, 128), device='cuda', dtype=torch.float32)
        buf8 = empty((8, 56, 56, 128), device='cuda', dtype=torch.float32)
        buf853 = empty((8, 56, 56, 1), device='cuda', dtype=torch.float32)
        buf854 = empty((8, 56, 56, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [shifted_x, x_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_0.run(buf0, primals_26, primals_27, primals_28, buf4, buf8, buf853, buf854, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_26
        buf9 = reinterpret_tensor(buf0, (25088, 128), (128, 1), 0); del buf0  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf8, primals_29, primals_30, buf9, 3211264, grid=grid(3211264), stream=stream0)
        del primals_30
        buf10 = empty((25088, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf9, reinterpret_tensor(primals_31, (128, 384), (1, 128), 0), out=buf10)
        buf11 = empty((512, 4, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, q_1], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_2.run(buf10, primals_32, buf11, 3211264, grid=grid(3211264), stream=stream0)
        buf12 = empty((512, 4, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf10, primals_32, buf12, 65536, 49, grid=grid(65536, 49), stream=stream0)
        buf13 = empty((2048, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf12, (2048, 32, 49), (1568, 49, 1), 0), out=buf13)
        buf16 = empty((512, 4, 49, 49), device='cuda', dtype=torch.float32)
        buf18 = empty((512, 4, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_1, attn_2, attn_3], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_4.run(buf13, primals_330, primals_1, buf16, buf18, 100352, 49, grid=grid(100352), stream=stream0)
        del primals_1
        buf17 = empty((512, 4, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf10, primals_32, buf17, 3211264, grid=grid(3211264), stream=stream0)
        del primals_32
        buf19 = empty((2048, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (2048, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf17, (2048, 49, 32), (1568, 32, 1), 0), out=buf19)
        buf20 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf19, buf20, 3211264, grid=grid(3211264), stream=stream0)
        buf21 = reinterpret_tensor(buf19, (25088, 128), (128, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_33, (128, 128), (1, 128), 0), out=buf21)
        buf22 = empty((8, 56, 56, 128), device='cuda', dtype=torch.float32)
        buf26 = empty((8, 3136, 128), device='cuda', dtype=torch.float32)
        buf27 = empty((25088, 128), device='cuda', dtype=torch.float32)
        buf852 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___norm2, x_14, x_16, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_7.run(buf4, primals_27, primals_28, buf21, primals_34, primals_35, primals_36, buf22, buf26, buf27, buf852, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_28
        del primals_34
        del primals_36
        buf28 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_38, buf27, reinterpret_tensor(primals_37, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf28)
        del primals_38
        buf29 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf28, buf29, 12845056, grid=grid(12845056), stream=stream0)
        buf30 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf29, reinterpret_tensor(primals_39, (512, 128), (1, 512), 0), out=buf30)
        buf34 = empty((8, 56, 56, 128), device='cuda', dtype=torch.float32)
        buf851 = empty((8, 56, 56, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_9.run(buf22, buf30, primals_40, buf34, buf851, 25088, 128, grid=grid(25088), stream=stream0)
        buf35 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf34, primals_41, primals_42, buf35, 3211264, grid=grid(3211264), stream=stream0)
        del primals_42
        buf36 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf35, reinterpret_tensor(primals_43, (128, 384), (1, 128), 0), out=buf36)
        buf37 = empty((512, 4, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4, q_3], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_2.run(buf36, primals_44, buf37, 3211264, grid=grid(3211264), stream=stream0)
        buf38 = empty((512, 4, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf36, primals_44, buf38, 65536, 49, grid=grid(65536, 49), stream=stream0)
        buf39 = buf13; del buf13  # reuse
        # Source Nodes: [attn_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf38, (2048, 32, 49), (1568, 49, 1), 0), out=buf39)
        buf43 = empty((512, 4, 49, 49), device='cuda', dtype=torch.float32)
        buf850 = empty((512, 4, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_8, attn_9], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_11.run(buf39, primals_332, primals_2, primals_331, buf43, buf850, 100352, 49, grid=grid(100352), stream=stream0)
        del buf39
        del primals_2
        del primals_331
        buf44 = empty((512, 4, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf36, primals_44, buf44, 3211264, grid=grid(3211264), stream=stream0)
        del buf36
        del primals_44
        buf45 = empty((2048, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (2048, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf44, (2048, 49, 32), (1568, 32, 1), 0), out=buf45)
        buf46 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.view]
        triton_poi_fused_view_6.run(buf45, buf46, 3211264, grid=grid(3211264), stream=stream0)
        buf47 = reinterpret_tensor(buf45, (25088, 128), (128, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf46, reinterpret_tensor(primals_45, (128, 128), (1, 128), 0), out=buf47)
        buf49 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf49, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf49, 0.9956521736457944)
        buf52 = empty((8, 56, 56, 128), device='cuda', dtype=torch.float32)
        buf56 = empty((8, 3136, 128), device='cuda', dtype=torch.float32)
        buf57 = empty((25088, 128), device='cuda', dtype=torch.float32)
        buf849 = empty((8, 3136, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div_, getattr_getattr_l__mod___layers___0___blocks___1___norm2, mul_2, x_31, x_32, x_34], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.roll, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_roll_view_13.run(buf22, buf30, primals_40, buf47, primals_46, buf49, primals_47, primals_48, buf52, buf56, buf57, buf849, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_40
        del primals_46
        del primals_48
        buf58 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_50, buf57, reinterpret_tensor(primals_49, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf58)
        del primals_50
        buf59 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35, x_38], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_8.run(buf58, buf59, 12845056, grid=grid(12845056), stream=stream0)
        buf60 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf59, reinterpret_tensor(primals_51, (512, 128), (1, 512), 0), out=buf60)
        buf62 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_1], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf62, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf62, 0.9956521736457944)
        buf68 = reinterpret_tensor(buf30, (8, 28, 28, 512), (401408, 14336, 512, 1), 0); del buf30  # reuse
        buf69 = reinterpret_tensor(buf22, (6272, 512), (512, 1), 0); del buf22  # reuse
        buf848 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_46], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_14.run(buf52, buf60, primals_52, buf62, primals_53, primals_54, buf68, buf69, buf848, 6272, 512, grid=grid(6272), stream=stream0)
        del primals_52
        del primals_54
        buf70 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(buf69, reinterpret_tensor(primals_55, (512, 256), (1, 512), 0), out=buf70)
        buf71 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf74 = reinterpret_tensor(buf72, (8, 28, 28, 1), (784, 28, 1, 1), 0); del buf72  # reuse
        # Source Nodes: [shifted_x_8], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf74, buf70, buf71, 6272, 256, grid=grid(6272), stream=stream0)
        buf75 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf70, buf71, buf74, primals_56, primals_57, buf75, 1605632, grid=grid(1605632), stream=stream0)
        del primals_57
        buf76 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf75, reinterpret_tensor(primals_58, (256, 768), (1, 256), 0), out=buf76)
        buf77 = empty((128, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, q_5], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_17.run(buf76, primals_59, buf77, 1605632, grid=grid(1605632), stream=stream0)
        buf78 = empty((128, 8, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf76, primals_59, buf78, 32768, 49, grid=grid(32768, 49), stream=stream0)
        buf79 = empty((1024, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf78, (1024, 32, 49), (1568, 49, 1), 0), out=buf79)
        buf82 = empty((128, 8, 49, 49), device='cuda', dtype=torch.float32)
        buf84 = empty((128, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_11, attn_12, attn_13], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_19.run(buf79, primals_333, primals_3, buf82, buf84, 50176, 49, grid=grid(50176), stream=stream0)
        del primals_3
        buf83 = empty((128, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf76, primals_59, buf83, 1605632, grid=grid(1605632), stream=stream0)
        del primals_59
        buf85 = empty((1024, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (1024, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf83, (1024, 49, 32), (1568, 32, 1), 0), out=buf85)
        buf86 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.view]
        triton_poi_fused_view_21.run(buf85, buf86, 1605632, grid=grid(1605632), stream=stream0)
        buf87 = reinterpret_tensor(buf85, (6272, 256), (256, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf86, reinterpret_tensor(primals_60, (256, 256), (1, 256), 0), out=buf87)
        buf88 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_2], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf88, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf88, 0.9913043472915888)
        buf94 = empty((8, 784, 256), device='cuda', dtype=torch.float32)
        buf95 = empty((6272, 256), device='cuda', dtype=torch.float32)
        buf847 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___norm2, x_57], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_22.run(buf70, buf87, primals_61, buf88, primals_62, primals_63, buf94, buf95, buf847, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_63
        buf96 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, buf95, reinterpret_tensor(primals_64, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf96)
        del primals_65
        buf97 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58, x_61], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_23.run(buf96, buf97, 6422528, grid=grid(6422528), stream=stream0)
        buf98 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf97, reinterpret_tensor(primals_66, (1024, 256), (1, 1024), 0), out=buf98)
        buf99 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_3], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf99, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf99, 0.9913043472915888)
        buf102 = reinterpret_tensor(buf98, (8, 784, 256), (200704, 256, 1), 0); del buf98  # reuse
        buf106 = empty((8, 28, 28, 256), device='cuda', dtype=torch.float32)
        buf846 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__3, getattr_getattr_l__mod___layers___1___blocks___1___norm1, mul_6, x_63], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_24.run(buf102, buf70, buf87, primals_61, buf88, primals_67, buf99, buf106, buf846, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_61
        del primals_67
        buf107 = buf87; del buf87  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_25.run(buf106, primals_68, primals_69, buf107, 1605632, grid=grid(1605632), stream=stream0)
        del primals_69
        buf108 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf107, reinterpret_tensor(primals_70, (256, 768), (1, 256), 0), out=buf108)
        buf109 = empty((128, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_14, q_7], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_17.run(buf108, primals_71, buf109, 1605632, grid=grid(1605632), stream=stream0)
        buf110 = empty((128, 8, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf108, primals_71, buf110, 32768, 49, grid=grid(32768, 49), stream=stream0)
        buf111 = buf79; del buf79  # reuse
        # Source Nodes: [attn_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf110, (1024, 32, 49), (1568, 49, 1), 0), out=buf111)
        buf115 = empty((128, 8, 49, 49), device='cuda', dtype=torch.float32)
        buf845 = empty((128, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_18, attn_19], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_26.run(buf111, primals_335, primals_4, primals_334, buf115, buf845, 50176, 49, grid=grid(50176), stream=stream0)
        del buf111
        del primals_334
        del primals_4
        buf116 = empty((128, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf108, primals_71, buf116, 1605632, grid=grid(1605632), stream=stream0)
        del buf108
        del primals_71
        buf117 = empty((1024, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (1024, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf116, (1024, 49, 32), (1568, 32, 1), 0), out=buf117)
        buf118 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.view]
        triton_poi_fused_view_21.run(buf117, buf118, 1605632, grid=grid(1605632), stream=stream0)
        buf119 = reinterpret_tensor(buf117, (6272, 256), (256, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf118, reinterpret_tensor(primals_72, (256, 256), (1, 256), 0), out=buf119)
        buf120 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_4], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf120, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf120, 0.9869565209373832)
        buf126 = empty((8, 784, 256), device='cuda', dtype=torch.float32)
        buf127 = empty((6272, 256), device='cuda', dtype=torch.float32)
        buf844 = empty((8, 784, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm2, x_75], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_27.run(buf102, buf119, primals_73, buf120, primals_74, primals_75, buf126, buf127, buf844, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_75
        buf128 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_77, buf127, reinterpret_tensor(primals_76, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf128)
        del primals_77
        buf129 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76, x_79], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_23.run(buf128, buf129, 6422528, grid=grid(6422528), stream=stream0)
        buf130 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf129, reinterpret_tensor(primals_78, (1024, 256), (1, 1024), 0), out=buf130)
        buf131 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_5], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf131, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf131, 0.9869565209373832)
        buf134 = reinterpret_tensor(buf130, (8, 14, 14, 2, 2, 256), (200704, 14336, 512, 256, 7168, 1), 0); del buf130  # reuse
        # Source Nodes: [x_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf134, buf102, buf119, primals_73, buf120, primals_79, buf131, 1605632, grid=grid(1605632), stream=stream0)
        del primals_73
        del primals_79
        buf138 = reinterpret_tensor(buf119, (8, 14, 14, 1024), (200704, 14336, 1024, 1), 0); del buf119  # reuse
        buf139 = reinterpret_tensor(buf102, (1568, 1024), (1024, 1), 0); del buf102  # reuse
        buf843 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85, x_87], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_29.run(buf134, primals_80, primals_81, buf138, buf139, buf843, 1568, 1024, grid=grid(1568), stream=stream0)
        del primals_81
        buf140 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.mm]
        extern_kernels.mm(buf139, reinterpret_tensor(primals_82, (1024, 512), (1, 1024), 0), out=buf140)
        buf141 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf144 = reinterpret_tensor(buf142, (8, 14, 14, 1), (196, 14, 1, 1), 0); del buf142  # reuse
        # Source Nodes: [shifted_x_16], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_30.run(buf144, buf140, buf141, 1568, 512, grid=grid(1568), stream=stream0)
        buf145 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_31.run(buf140, buf141, buf144, primals_83, primals_84, buf145, 802816, grid=grid(802816), stream=stream0)
        del primals_84
        buf146 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf145, reinterpret_tensor(primals_85, (512, 1536), (1, 512), 0), out=buf146)
        buf147 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20, q_9], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf146, primals_86, buf147, 802816, grid=grid(802816), stream=stream0)
        buf148 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf146, primals_86, buf148, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf149 = empty((512, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf148, (512, 32, 49), (1568, 49, 1), 0), out=buf149)
        buf152 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf154 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_21, attn_22, attn_23], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf149, primals_336, primals_5, buf152, buf154, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_5
        buf153 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf146, primals_86, buf153, 802816, grid=grid(802816), stream=stream0)
        del primals_86
        buf155 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf153, (512, 49, 32), (1568, 32, 1), 0), out=buf155)
        buf156 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf155, buf156, 802816, grid=grid(802816), stream=stream0)
        buf157 = reinterpret_tensor(buf155, (1568, 512), (512, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf156, reinterpret_tensor(primals_87, (512, 512), (1, 512), 0), out=buf157)
        buf158 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_6], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf158, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf158, 0.9826086945831776)
        buf164 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf165 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf842 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___norm2, x_98], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_37.run(buf140, buf157, primals_88, buf158, primals_89, primals_90, buf164, buf165, buf842, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_90
        buf166 = reinterpret_tensor(buf60, (1568, 2048), (2048, 1), 0); del buf60  # reuse
        # Source Nodes: [x_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf165, reinterpret_tensor(primals_91, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf166)
        del primals_92
        buf167 = reinterpret_tensor(buf52, (1568, 2048), (2048, 1), 0); del buf52  # reuse
        # Source Nodes: [x_102, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf166, buf167, 3211264, grid=grid(3211264), stream=stream0)
        buf168 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf167, reinterpret_tensor(primals_93, (2048, 512), (1, 2048), 0), out=buf168)
        buf169 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_7], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf169, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf169, 0.9826086945831776)
        buf172 = reinterpret_tensor(buf168, (8, 196, 512), (100352, 512, 1), 0); del buf168  # reuse
        buf176 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf841 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__7, getattr_getattr_l__mod___layers___2___blocks___1___norm1, mul_12, x_104], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_39.run(buf172, buf140, buf157, primals_88, buf158, primals_94, buf169, buf176, buf841, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_88
        del primals_94
        buf177 = buf157; del buf157  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf176, primals_95, primals_96, buf177, 802816, grid=grid(802816), stream=stream0)
        del primals_96
        buf178 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf177, reinterpret_tensor(primals_97, (512, 1536), (1, 512), 0), out=buf178)
        buf179 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24, q_11], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf178, primals_98, buf179, 802816, grid=grid(802816), stream=stream0)
        buf180 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf178, primals_98, buf180, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf181 = buf149; del buf149  # reuse
        # Source Nodes: [attn_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf179, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf180, (512, 32, 49), (1568, 49, 1), 0), out=buf181)
        buf185 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf840 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_28, attn_29], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf181, primals_338, primals_6, primals_337, buf185, buf840, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_337
        del primals_6
        buf186 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf178, primals_98, buf186, 802816, grid=grid(802816), stream=stream0)
        del primals_98
        buf187 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf185, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf186, (512, 49, 32), (1568, 32, 1), 0), out=buf187)
        buf188 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf187, buf188, 802816, grid=grid(802816), stream=stream0)
        buf189 = reinterpret_tensor(buf187, (1568, 512), (512, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf188, reinterpret_tensor(primals_99, (512, 512), (1, 512), 0), out=buf189)
        buf190 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_8], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf190, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf190, 0.9782608672976494)
        buf196 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf197 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf839 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___norm2, x_116], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_42.run(buf172, buf189, primals_100, buf190, primals_101, primals_102, buf196, buf197, buf839, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_102
        buf198 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_104, buf197, reinterpret_tensor(primals_103, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf198)
        del primals_104
        buf199 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_120], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf198, buf199, 3211264, grid=grid(3211264), stream=stream0)
        buf200 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf199, reinterpret_tensor(primals_105, (2048, 512), (1, 2048), 0), out=buf200)
        buf201 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_9], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf201, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf201, 0.9782608672976494)
        buf204 = reinterpret_tensor(buf200, (8, 196, 512), (100352, 512, 1), 0); del buf200  # reuse
        buf208 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf838 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__9, mul_15, shifted_x_24, x_122], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_43.run(buf204, buf172, buf189, primals_100, buf190, primals_106, buf201, buf208, buf838, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_100
        del primals_106
        buf209 = buf189; del buf189  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf208, primals_107, primals_108, buf209, 802816, grid=grid(802816), stream=stream0)
        del primals_108
        buf210 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf209, reinterpret_tensor(primals_109, (512, 1536), (1, 512), 0), out=buf210)
        buf211 = reinterpret_tensor(buf172, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf172  # reuse
        # Source Nodes: [attn_30, q_13], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf210, primals_110, buf211, 802816, grid=grid(802816), stream=stream0)
        buf212 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf210, primals_110, buf212, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf213 = buf181; del buf181  # reuse
        # Source Nodes: [attn_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf212, (512, 32, 49), (1568, 49, 1), 0), out=buf213)
        buf216 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf218 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_31, attn_32, attn_33], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf213, primals_339, primals_7, buf216, buf218, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_7
        buf217 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf210, primals_110, buf217, 802816, grid=grid(802816), stream=stream0)
        del primals_110
        buf219 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf218, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf217, (512, 49, 32), (1568, 32, 1), 0), out=buf219)
        buf220 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf219, buf220, 802816, grid=grid(802816), stream=stream0)
        buf221 = reinterpret_tensor(buf219, (1568, 512), (512, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf220, reinterpret_tensor(primals_111, (512, 512), (1, 512), 0), out=buf221)
        buf222 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_10], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf222, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf222, 0.9739130418747663)
        buf228 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf229 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf837 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___2___norm2, x_134], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_45.run(buf204, buf221, primals_112, buf222, primals_113, primals_114, buf228, buf229, buf837, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_114
        buf230 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf229, reinterpret_tensor(primals_115, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf230)
        del primals_116
        buf231 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_138], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf230, buf231, 3211264, grid=grid(3211264), stream=stream0)
        buf232 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf231, reinterpret_tensor(primals_117, (2048, 512), (1, 2048), 0), out=buf232)
        buf233 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_11], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf233, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf233, 0.9739130418747663)
        buf236 = reinterpret_tensor(buf232, (8, 196, 512), (100352, 512, 1), 0); del buf232  # reuse
        buf240 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf836 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__11, getattr_getattr_l__mod___layers___2___blocks___3___norm1, mul_18, x_140], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_46.run(buf236, buf204, buf221, primals_112, buf222, primals_118, buf233, buf240, buf836, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_112
        del primals_118
        buf241 = buf221; del buf221  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___3___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf240, primals_119, primals_120, buf241, 802816, grid=grid(802816), stream=stream0)
        del primals_120
        buf242 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf241, reinterpret_tensor(primals_121, (512, 1536), (1, 512), 0), out=buf242)
        buf243 = reinterpret_tensor(buf204, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf204  # reuse
        # Source Nodes: [attn_34, q_15], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf242, primals_122, buf243, 802816, grid=grid(802816), stream=stream0)
        buf244 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf242, primals_122, buf244, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf245 = buf213; del buf213  # reuse
        # Source Nodes: [attn_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf243, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf244, (512, 32, 49), (1568, 49, 1), 0), out=buf245)
        buf249 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf835 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_38, attn_39], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf245, primals_341, primals_8, primals_340, buf249, buf835, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_340
        del primals_8
        buf250 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf242, primals_122, buf250, 802816, grid=grid(802816), stream=stream0)
        del primals_122
        buf251 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf250, (512, 49, 32), (1568, 32, 1), 0), out=buf251)
        buf252 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf251, buf252, 802816, grid=grid(802816), stream=stream0)
        buf253 = reinterpret_tensor(buf251, (1568, 512), (512, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf252, reinterpret_tensor(primals_123, (512, 512), (1, 512), 0), out=buf253)
        buf254 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_12], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf254, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf254, 0.9695652164518833)
        buf260 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf261 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf834 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___3___norm2, x_152], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_47.run(buf236, buf253, primals_124, buf254, primals_125, primals_126, buf260, buf261, buf834, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_126
        buf262 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_128, buf261, reinterpret_tensor(primals_127, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf262)
        del primals_128
        buf263 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153, x_156], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf262, buf263, 3211264, grid=grid(3211264), stream=stream0)
        buf264 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf263, reinterpret_tensor(primals_129, (2048, 512), (1, 2048), 0), out=buf264)
        buf265 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_13], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf265, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf265, 0.9695652164518833)
        buf268 = reinterpret_tensor(buf264, (8, 196, 512), (100352, 512, 1), 0); del buf264  # reuse
        buf272 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf833 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__13, mul_21, shifted_x_32, x_158], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_48.run(buf268, buf236, buf253, primals_124, buf254, primals_130, buf265, buf272, buf833, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_124
        del primals_130
        buf273 = buf253; del buf253  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___4___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf272, primals_131, primals_132, buf273, 802816, grid=grid(802816), stream=stream0)
        del primals_132
        buf274 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf273, reinterpret_tensor(primals_133, (512, 1536), (1, 512), 0), out=buf274)
        buf275 = reinterpret_tensor(buf236, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf236  # reuse
        # Source Nodes: [attn_40, q_17], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf274, primals_134, buf275, 802816, grid=grid(802816), stream=stream0)
        buf276 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf274, primals_134, buf276, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf277 = buf245; del buf245  # reuse
        # Source Nodes: [attn_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf275, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf276, (512, 32, 49), (1568, 49, 1), 0), out=buf277)
        buf280 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf282 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_41, attn_42, attn_43], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf277, primals_342, primals_9, buf280, buf282, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_9
        buf281 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf274, primals_134, buf281, 802816, grid=grid(802816), stream=stream0)
        del primals_134
        buf283 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf281, (512, 49, 32), (1568, 32, 1), 0), out=buf283)
        buf284 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf283, buf284, 802816, grid=grid(802816), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (1568, 512), (512, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf284, reinterpret_tensor(primals_135, (512, 512), (1, 512), 0), out=buf285)
        buf286 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_14], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf286, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf286, 0.9652173891663551)
        buf292 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf293 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf832 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___4___norm2, x_170], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_49.run(buf268, buf285, primals_136, buf286, primals_137, primals_138, buf292, buf293, buf832, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_138
        buf294 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_140, buf293, reinterpret_tensor(primals_139, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf294)
        del primals_140
        buf295 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171, x_174], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf294, buf295, 3211264, grid=grid(3211264), stream=stream0)
        buf296 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf295, reinterpret_tensor(primals_141, (2048, 512), (1, 2048), 0), out=buf296)
        buf297 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_15], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf297, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf297, 0.9652173891663551)
        buf300 = reinterpret_tensor(buf296, (8, 196, 512), (100352, 512, 1), 0); del buf296  # reuse
        buf304 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf831 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__15, getattr_getattr_l__mod___layers___2___blocks___5___norm1, mul_24, x_176], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_50.run(buf300, buf268, buf285, primals_136, buf286, primals_142, buf297, buf304, buf831, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_136
        del primals_142
        buf305 = buf285; del buf285  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___5___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf304, primals_143, primals_144, buf305, 802816, grid=grid(802816), stream=stream0)
        del primals_144
        buf306 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf305, reinterpret_tensor(primals_145, (512, 1536), (1, 512), 0), out=buf306)
        buf307 = reinterpret_tensor(buf268, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf268  # reuse
        # Source Nodes: [attn_44, q_19], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf306, primals_146, buf307, 802816, grid=grid(802816), stream=stream0)
        buf308 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf306, primals_146, buf308, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf309 = buf277; del buf277  # reuse
        # Source Nodes: [attn_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf308, (512, 32, 49), (1568, 49, 1), 0), out=buf309)
        buf313 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf830 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_48, attn_49], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf309, primals_344, primals_10, primals_343, buf313, buf830, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_10
        del primals_343
        buf314 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf306, primals_146, buf314, 802816, grid=grid(802816), stream=stream0)
        del primals_146
        buf315 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf313, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf314, (512, 49, 32), (1568, 32, 1), 0), out=buf315)
        buf316 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf315, buf316, 802816, grid=grid(802816), stream=stream0)
        buf317 = reinterpret_tensor(buf315, (1568, 512), (512, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf316, reinterpret_tensor(primals_147, (512, 512), (1, 512), 0), out=buf317)
        buf318 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_16], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf318, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf318, 0.960869561880827)
        buf324 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf325 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf829 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___5___norm2, x_188], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_51.run(buf300, buf317, primals_148, buf318, primals_149, primals_150, buf324, buf325, buf829, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_150
        buf326 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf325, reinterpret_tensor(primals_151, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf326)
        del primals_152
        buf327 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189, x_192], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf326, buf327, 3211264, grid=grid(3211264), stream=stream0)
        buf328 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf327, reinterpret_tensor(primals_153, (2048, 512), (1, 2048), 0), out=buf328)
        buf329 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_17], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf329, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf329, 0.960869561880827)
        buf332 = reinterpret_tensor(buf328, (8, 196, 512), (100352, 512, 1), 0); del buf328  # reuse
        buf336 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf828 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__17, mul_27, shifted_x_40, x_194], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_52.run(buf332, buf300, buf317, primals_148, buf318, primals_154, buf329, buf336, buf828, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_148
        del primals_154
        buf337 = buf317; del buf317  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___6___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf336, primals_155, primals_156, buf337, 802816, grid=grid(802816), stream=stream0)
        del primals_156
        buf338 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf337, reinterpret_tensor(primals_157, (512, 1536), (1, 512), 0), out=buf338)
        buf339 = reinterpret_tensor(buf300, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf300  # reuse
        # Source Nodes: [attn_50, q_21], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf338, primals_158, buf339, 802816, grid=grid(802816), stream=stream0)
        buf340 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf338, primals_158, buf340, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf341 = buf309; del buf309  # reuse
        # Source Nodes: [attn_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf339, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf340, (512, 32, 49), (1568, 49, 1), 0), out=buf341)
        buf344 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf346 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_51, attn_52, attn_53], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf341, primals_345, primals_11, buf344, buf346, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_11
        buf345 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf338, primals_158, buf345, 802816, grid=grid(802816), stream=stream0)
        del primals_158
        buf347 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf346, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf345, (512, 49, 32), (1568, 32, 1), 0), out=buf347)
        buf348 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf347, buf348, 802816, grid=grid(802816), stream=stream0)
        buf349 = reinterpret_tensor(buf347, (1568, 512), (512, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf348, reinterpret_tensor(primals_159, (512, 512), (1, 512), 0), out=buf349)
        buf350 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_18], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf350, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf350, 0.9565217345952988)
        buf356 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf357 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf827 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___6___norm2, x_206], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_53.run(buf332, buf349, primals_160, buf350, primals_161, primals_162, buf356, buf357, buf827, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_162
        buf358 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_164, buf357, reinterpret_tensor(primals_163, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf358)
        del primals_164
        buf359 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207, x_210], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf358, buf359, 3211264, grid=grid(3211264), stream=stream0)
        buf360 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf359, reinterpret_tensor(primals_165, (2048, 512), (1, 2048), 0), out=buf360)
        buf361 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_19], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf361, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf361, 0.9565217345952988)
        buf364 = reinterpret_tensor(buf360, (8, 196, 512), (100352, 512, 1), 0); del buf360  # reuse
        buf368 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf826 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__19, getattr_getattr_l__mod___layers___2___blocks___7___norm1, mul_30, x_212], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_54.run(buf364, buf332, buf349, primals_160, buf350, primals_166, buf361, buf368, buf826, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_160
        del primals_166
        buf369 = buf349; del buf349  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___7___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf368, primals_167, primals_168, buf369, 802816, grid=grid(802816), stream=stream0)
        del primals_168
        buf370 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf369, reinterpret_tensor(primals_169, (512, 1536), (1, 512), 0), out=buf370)
        buf371 = reinterpret_tensor(buf332, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf332  # reuse
        # Source Nodes: [attn_54, q_23], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf370, primals_170, buf371, 802816, grid=grid(802816), stream=stream0)
        buf372 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf370, primals_170, buf372, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf373 = buf341; del buf341  # reuse
        # Source Nodes: [attn_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf371, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf372, (512, 32, 49), (1568, 49, 1), 0), out=buf373)
        buf377 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf825 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_58, attn_59], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf373, primals_347, primals_12, primals_346, buf377, buf825, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_12
        del primals_346
        buf378 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf370, primals_170, buf378, 802816, grid=grid(802816), stream=stream0)
        del primals_170
        buf379 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf377, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf378, (512, 49, 32), (1568, 32, 1), 0), out=buf379)
        buf380 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf379, buf380, 802816, grid=grid(802816), stream=stream0)
        buf381 = reinterpret_tensor(buf379, (1568, 512), (512, 1), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf380, reinterpret_tensor(primals_171, (512, 512), (1, 512), 0), out=buf381)
        buf382 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_20], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf382, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf382, 0.9521739110350609)
        buf388 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf389 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf824 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___7___norm2, x_224], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_55.run(buf364, buf381, primals_172, buf382, primals_173, primals_174, buf388, buf389, buf824, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_174
        buf390 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf389, reinterpret_tensor(primals_175, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf390)
        del primals_176
        buf391 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_225, x_228], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf390, buf391, 3211264, grid=grid(3211264), stream=stream0)
        buf392 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf391, reinterpret_tensor(primals_177, (2048, 512), (1, 2048), 0), out=buf392)
        buf393 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_21], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf393, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf393, 0.9521739110350609)
        buf396 = reinterpret_tensor(buf392, (8, 196, 512), (100352, 512, 1), 0); del buf392  # reuse
        buf400 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf823 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__21, mul_33, shifted_x_48, x_230], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_56.run(buf396, buf364, buf381, primals_172, buf382, primals_178, buf393, buf400, buf823, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_172
        del primals_178
        buf401 = buf381; del buf381  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___8___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf400, primals_179, primals_180, buf401, 802816, grid=grid(802816), stream=stream0)
        del primals_180
        buf402 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf401, reinterpret_tensor(primals_181, (512, 1536), (1, 512), 0), out=buf402)
        buf403 = reinterpret_tensor(buf364, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf364  # reuse
        # Source Nodes: [attn_60, q_25], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf402, primals_182, buf403, 802816, grid=grid(802816), stream=stream0)
        buf404 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf402, primals_182, buf404, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf405 = buf373; del buf373  # reuse
        # Source Nodes: [attn_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf403, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf404, (512, 32, 49), (1568, 49, 1), 0), out=buf405)
        buf408 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf410 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_61, attn_62, attn_63], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf405, primals_348, primals_13, buf408, buf410, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_13
        buf409 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf402, primals_182, buf409, 802816, grid=grid(802816), stream=stream0)
        del primals_182
        buf411 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf410, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf409, (512, 49, 32), (1568, 32, 1), 0), out=buf411)
        buf412 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf411, buf412, 802816, grid=grid(802816), stream=stream0)
        buf413 = reinterpret_tensor(buf411, (1568, 512), (512, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf412, reinterpret_tensor(primals_183, (512, 512), (1, 512), 0), out=buf413)
        buf414 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_22], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf414, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf414, 0.947826087474823)
        buf420 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf421 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf822 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___8___norm2, x_242], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_57.run(buf396, buf413, primals_184, buf414, primals_185, primals_186, buf420, buf421, buf822, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_186
        buf422 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_188, buf421, reinterpret_tensor(primals_187, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf422)
        del primals_188
        buf423 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243, x_246], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf422, buf423, 3211264, grid=grid(3211264), stream=stream0)
        buf424 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf423, reinterpret_tensor(primals_189, (2048, 512), (1, 2048), 0), out=buf424)
        buf425 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_23], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf425, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf425, 0.947826087474823)
        buf428 = reinterpret_tensor(buf424, (8, 196, 512), (100352, 512, 1), 0); del buf424  # reuse
        buf432 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf821 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__23, getattr_getattr_l__mod___layers___2___blocks___9___norm1, mul_36, x_248], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_58.run(buf428, buf396, buf413, primals_184, buf414, primals_190, buf425, buf432, buf821, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_184
        del primals_190
        buf433 = buf413; del buf413  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___9___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf432, primals_191, primals_192, buf433, 802816, grid=grid(802816), stream=stream0)
        del primals_192
        buf434 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf433, reinterpret_tensor(primals_193, (512, 1536), (1, 512), 0), out=buf434)
        buf435 = reinterpret_tensor(buf396, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf396  # reuse
        # Source Nodes: [attn_64, q_27], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf434, primals_194, buf435, 802816, grid=grid(802816), stream=stream0)
        buf436 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf434, primals_194, buf436, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf437 = buf405; del buf405  # reuse
        # Source Nodes: [attn_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf435, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf436, (512, 32, 49), (1568, 49, 1), 0), out=buf437)
        buf441 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf820 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_68, attn_69], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf437, primals_350, primals_14, primals_349, buf441, buf820, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_14
        del primals_349
        buf442 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf434, primals_194, buf442, 802816, grid=grid(802816), stream=stream0)
        del primals_194
        buf443 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf442, (512, 49, 32), (1568, 32, 1), 0), out=buf443)
        buf444 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf443, buf444, 802816, grid=grid(802816), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (1568, 512), (512, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf444, reinterpret_tensor(primals_195, (512, 512), (1, 512), 0), out=buf445)
        buf446 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_24], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf446, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf446, 0.9434782639145851)
        buf452 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf453 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf819 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___9___norm2, x_260], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_59.run(buf428, buf445, primals_196, buf446, primals_197, primals_198, buf452, buf453, buf819, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_198
        buf454 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_200, buf453, reinterpret_tensor(primals_199, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf454)
        del primals_200
        buf455 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_261, x_264], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf454, buf455, 3211264, grid=grid(3211264), stream=stream0)
        buf456 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf455, reinterpret_tensor(primals_201, (2048, 512), (1, 2048), 0), out=buf456)
        buf457 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_25], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf457, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf457, 0.9434782639145851)
        buf460 = reinterpret_tensor(buf456, (8, 196, 512), (100352, 512, 1), 0); del buf456  # reuse
        buf464 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf818 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__25, mul_39, shifted_x_56, x_266], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_60.run(buf460, buf428, buf445, primals_196, buf446, primals_202, buf457, buf464, buf818, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_196
        del primals_202
        buf465 = buf445; del buf445  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___10___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf464, primals_203, primals_204, buf465, 802816, grid=grid(802816), stream=stream0)
        del primals_204
        buf466 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf465, reinterpret_tensor(primals_205, (512, 1536), (1, 512), 0), out=buf466)
        buf467 = reinterpret_tensor(buf428, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf428  # reuse
        # Source Nodes: [attn_70, q_29], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf466, primals_206, buf467, 802816, grid=grid(802816), stream=stream0)
        buf468 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf466, primals_206, buf468, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf469 = buf437; del buf437  # reuse
        # Source Nodes: [attn_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf467, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf468, (512, 32, 49), (1568, 49, 1), 0), out=buf469)
        buf472 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf474 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_71, attn_72, attn_73], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf469, primals_351, primals_15, buf472, buf474, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_15
        buf473 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf466, primals_206, buf473, 802816, grid=grid(802816), stream=stream0)
        del primals_206
        buf475 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf474, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf473, (512, 49, 32), (1568, 32, 1), 0), out=buf475)
        buf476 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf475, buf476, 802816, grid=grid(802816), stream=stream0)
        buf477 = reinterpret_tensor(buf475, (1568, 512), (512, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf476, reinterpret_tensor(primals_207, (512, 512), (1, 512), 0), out=buf477)
        buf478 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_26], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf478, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf478, 0.9391304366290569)
        buf484 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf485 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf817 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___10___norm2, x_278], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_61.run(buf460, buf477, primals_208, buf478, primals_209, primals_210, buf484, buf485, buf817, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_210
        buf486 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_212, buf485, reinterpret_tensor(primals_211, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf486)
        del primals_212
        buf487 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279, x_282], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf486, buf487, 3211264, grid=grid(3211264), stream=stream0)
        buf488 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf487, reinterpret_tensor(primals_213, (2048, 512), (1, 2048), 0), out=buf488)
        buf489 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_27], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf489, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf489, 0.9391304366290569)
        buf492 = reinterpret_tensor(buf488, (8, 196, 512), (100352, 512, 1), 0); del buf488  # reuse
        buf496 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf816 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__27, getattr_getattr_l__mod___layers___2___blocks___11___norm1, mul_42, x_284], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_62.run(buf492, buf460, buf477, primals_208, buf478, primals_214, buf489, buf496, buf816, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_208
        del primals_214
        buf497 = buf477; del buf477  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___11___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf496, primals_215, primals_216, buf497, 802816, grid=grid(802816), stream=stream0)
        del primals_216
        buf498 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf497, reinterpret_tensor(primals_217, (512, 1536), (1, 512), 0), out=buf498)
        buf499 = reinterpret_tensor(buf460, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf460  # reuse
        # Source Nodes: [attn_74, q_31], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf498, primals_218, buf499, 802816, grid=grid(802816), stream=stream0)
        buf500 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf498, primals_218, buf500, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf501 = buf469; del buf469  # reuse
        # Source Nodes: [attn_74], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf500, (512, 32, 49), (1568, 49, 1), 0), out=buf501)
        buf505 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf815 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_78, attn_79], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf501, primals_353, primals_16, primals_352, buf505, buf815, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_16
        del primals_352
        buf506 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf498, primals_218, buf506, 802816, grid=grid(802816), stream=stream0)
        del primals_218
        buf507 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf505, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf506, (512, 49, 32), (1568, 32, 1), 0), out=buf507)
        buf508 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_289], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf507, buf508, 802816, grid=grid(802816), stream=stream0)
        buf509 = reinterpret_tensor(buf507, (1568, 512), (512, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf508, reinterpret_tensor(primals_219, (512, 512), (1, 512), 0), out=buf509)
        buf510 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_28], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf510, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf510, 0.9347826093435287)
        buf516 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf517 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf814 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___11___norm2, x_296], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_63.run(buf492, buf509, primals_220, buf510, primals_221, primals_222, buf516, buf517, buf814, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_222
        buf518 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_224, buf517, reinterpret_tensor(primals_223, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf518)
        del primals_224
        buf519 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_297, x_300], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf518, buf519, 3211264, grid=grid(3211264), stream=stream0)
        buf520 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf519, reinterpret_tensor(primals_225, (2048, 512), (1, 2048), 0), out=buf520)
        buf521 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_29], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf521, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf521, 0.9347826093435287)
        buf524 = reinterpret_tensor(buf520, (8, 196, 512), (100352, 512, 1), 0); del buf520  # reuse
        buf528 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf813 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__29, mul_45, shifted_x_64, x_302], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_64.run(buf524, buf492, buf509, primals_220, buf510, primals_226, buf521, buf528, buf813, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_220
        del primals_226
        buf529 = buf509; del buf509  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___12___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf528, primals_227, primals_228, buf529, 802816, grid=grid(802816), stream=stream0)
        del primals_228
        buf530 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf529, reinterpret_tensor(primals_229, (512, 1536), (1, 512), 0), out=buf530)
        buf531 = reinterpret_tensor(buf492, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf492  # reuse
        # Source Nodes: [attn_80, q_33], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf530, primals_230, buf531, 802816, grid=grid(802816), stream=stream0)
        buf532 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf530, primals_230, buf532, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf533 = buf501; del buf501  # reuse
        # Source Nodes: [attn_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf531, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf532, (512, 32, 49), (1568, 49, 1), 0), out=buf533)
        buf536 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf538 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_81, attn_82, attn_83], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf533, primals_354, primals_17, buf536, buf538, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_17
        buf537 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_305], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf530, primals_230, buf537, 802816, grid=grid(802816), stream=stream0)
        del primals_230
        buf539 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_305], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf538, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf537, (512, 49, 32), (1568, 32, 1), 0), out=buf539)
        buf540 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf539, buf540, 802816, grid=grid(802816), stream=stream0)
        buf541 = reinterpret_tensor(buf539, (1568, 512), (512, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf540, reinterpret_tensor(primals_231, (512, 512), (1, 512), 0), out=buf541)
        buf542 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_30], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf542, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf542, 0.9304347857832909)
        buf548 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf549 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf812 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___12___norm2, x_314], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_65.run(buf524, buf541, primals_232, buf542, primals_233, primals_234, buf548, buf549, buf812, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_234
        buf550 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_236, buf549, reinterpret_tensor(primals_235, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf550)
        del primals_236
        buf551 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315, x_318], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf550, buf551, 3211264, grid=grid(3211264), stream=stream0)
        buf552 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf551, reinterpret_tensor(primals_237, (2048, 512), (1, 2048), 0), out=buf552)
        buf553 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_31], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf553, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf553, 0.9304347857832909)
        buf556 = reinterpret_tensor(buf552, (8, 196, 512), (100352, 512, 1), 0); del buf552  # reuse
        buf560 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf811 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__31, getattr_getattr_l__mod___layers___2___blocks___13___norm1, mul_48, x_320], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_66.run(buf556, buf524, buf541, primals_232, buf542, primals_238, buf553, buf560, buf811, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_232
        del primals_238
        buf561 = buf541; del buf541  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___13___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf560, primals_239, primals_240, buf561, 802816, grid=grid(802816), stream=stream0)
        del primals_240
        buf562 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf561, reinterpret_tensor(primals_241, (512, 1536), (1, 512), 0), out=buf562)
        buf563 = reinterpret_tensor(buf524, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf524  # reuse
        # Source Nodes: [attn_84, q_35], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf562, primals_242, buf563, 802816, grid=grid(802816), stream=stream0)
        buf564 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf562, primals_242, buf564, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf565 = buf533; del buf533  # reuse
        # Source Nodes: [attn_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf563, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf564, (512, 32, 49), (1568, 49, 1), 0), out=buf565)
        buf569 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf810 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_88, attn_89], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf565, primals_356, primals_18, primals_355, buf569, buf810, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_18
        del primals_355
        buf570 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf562, primals_242, buf570, 802816, grid=grid(802816), stream=stream0)
        del primals_242
        buf571 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf569, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf570, (512, 49, 32), (1568, 32, 1), 0), out=buf571)
        buf572 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf571, buf572, 802816, grid=grid(802816), stream=stream0)
        buf573 = reinterpret_tensor(buf571, (1568, 512), (512, 1), 0); del buf571  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf572, reinterpret_tensor(primals_243, (512, 512), (1, 512), 0), out=buf573)
        buf574 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_32], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf574, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf574, 0.9260869547724724)
        buf580 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf581 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf809 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___13___norm2, x_332], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_67.run(buf556, buf573, primals_244, buf574, primals_245, primals_246, buf580, buf581, buf809, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_246
        buf582 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_332], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_248, buf581, reinterpret_tensor(primals_247, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf582)
        del primals_248
        buf583 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_333, x_336], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf582, buf583, 3211264, grid=grid(3211264), stream=stream0)
        buf584 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf583, reinterpret_tensor(primals_249, (2048, 512), (1, 2048), 0), out=buf584)
        buf585 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_33], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf585, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf585, 0.9260869547724724)
        buf588 = reinterpret_tensor(buf584, (8, 196, 512), (100352, 512, 1), 0); del buf584  # reuse
        buf592 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf808 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__33, mul_51, shifted_x_72, x_338], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_68.run(buf588, buf556, buf573, primals_244, buf574, primals_250, buf585, buf592, buf808, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_244
        del primals_250
        buf593 = buf573; del buf573  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___14___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf592, primals_251, primals_252, buf593, 802816, grid=grid(802816), stream=stream0)
        del primals_252
        buf594 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf593, reinterpret_tensor(primals_253, (512, 1536), (1, 512), 0), out=buf594)
        buf595 = reinterpret_tensor(buf556, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf556  # reuse
        # Source Nodes: [attn_90, q_37], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf594, primals_254, buf595, 802816, grid=grid(802816), stream=stream0)
        buf596 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf594, primals_254, buf596, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf597 = buf565; del buf565  # reuse
        # Source Nodes: [attn_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf595, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf596, (512, 32, 49), (1568, 49, 1), 0), out=buf597)
        buf600 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf602 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_91, attn_92, attn_93], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf597, primals_357, primals_19, buf600, buf602, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_19
        buf601 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf594, primals_254, buf601, 802816, grid=grid(802816), stream=stream0)
        del primals_254
        buf603 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_341], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf602, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf601, (512, 49, 32), (1568, 32, 1), 0), out=buf603)
        buf604 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf603, buf604, 802816, grid=grid(802816), stream=stream0)
        buf605 = reinterpret_tensor(buf603, (1568, 512), (512, 1), 0); del buf603  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf604, reinterpret_tensor(primals_255, (512, 512), (1, 512), 0), out=buf605)
        buf606 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_34], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf606, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf606, 0.9217391312122345)
        buf612 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf613 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf807 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___14___norm2, x_350], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_69.run(buf588, buf605, primals_256, buf606, primals_257, primals_258, buf612, buf613, buf807, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_258
        buf614 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_260, buf613, reinterpret_tensor(primals_259, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf614)
        del primals_260
        buf615 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_351, x_354], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf614, buf615, 3211264, grid=grid(3211264), stream=stream0)
        buf616 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf615, reinterpret_tensor(primals_261, (2048, 512), (1, 2048), 0), out=buf616)
        buf617 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_35], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf617, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf617, 0.9217391312122345)
        buf620 = reinterpret_tensor(buf616, (8, 196, 512), (100352, 512, 1), 0); del buf616  # reuse
        buf624 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf806 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__35, getattr_getattr_l__mod___layers___2___blocks___15___norm1, mul_54, x_356], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_70.run(buf620, buf588, buf605, primals_256, buf606, primals_262, buf617, buf624, buf806, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_256
        del primals_262
        buf625 = buf605; del buf605  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___15___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf624, primals_263, primals_264, buf625, 802816, grid=grid(802816), stream=stream0)
        del primals_264
        buf626 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf625, reinterpret_tensor(primals_265, (512, 1536), (1, 512), 0), out=buf626)
        buf627 = reinterpret_tensor(buf588, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf588  # reuse
        # Source Nodes: [attn_94, q_39], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf626, primals_266, buf627, 802816, grid=grid(802816), stream=stream0)
        buf628 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf626, primals_266, buf628, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf629 = buf597; del buf597  # reuse
        # Source Nodes: [attn_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf627, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf628, (512, 32, 49), (1568, 49, 1), 0), out=buf629)
        buf633 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf805 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_98, attn_99], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf629, primals_359, primals_20, primals_358, buf633, buf805, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_20
        del primals_358
        buf634 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_359], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf626, primals_266, buf634, 802816, grid=grid(802816), stream=stream0)
        del primals_266
        buf635 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_359], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf633, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf634, (512, 49, 32), (1568, 32, 1), 0), out=buf635)
        buf636 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_361], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf635, buf636, 802816, grid=grid(802816), stream=stream0)
        buf637 = reinterpret_tensor(buf635, (1568, 512), (512, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf636, reinterpret_tensor(primals_267, (512, 512), (1, 512), 0), out=buf637)
        buf638 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_36], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf638, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf638, 0.917391300201416)
        buf644 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf645 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf804 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___15___norm2, x_368], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_71.run(buf620, buf637, primals_268, buf638, primals_269, primals_270, buf644, buf645, buf804, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_270
        buf646 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_368], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_272, buf645, reinterpret_tensor(primals_271, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf646)
        del primals_272
        buf647 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_369, x_372], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf646, buf647, 3211264, grid=grid(3211264), stream=stream0)
        buf648 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf647, reinterpret_tensor(primals_273, (2048, 512), (1, 2048), 0), out=buf648)
        buf649 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_37], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf649, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf649, 0.917391300201416)
        buf652 = reinterpret_tensor(buf648, (8, 196, 512), (100352, 512, 1), 0); del buf648  # reuse
        buf656 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf803 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__37, mul_57, shifted_x_80, x_374], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_72.run(buf652, buf620, buf637, primals_268, buf638, primals_274, buf649, buf656, buf803, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_268
        del primals_274
        buf657 = buf637; del buf637  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___16___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_44.run(buf656, primals_275, primals_276, buf657, 802816, grid=grid(802816), stream=stream0)
        del primals_276
        buf658 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf657, reinterpret_tensor(primals_277, (512, 1536), (1, 512), 0), out=buf658)
        buf659 = reinterpret_tensor(buf620, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf620  # reuse
        # Source Nodes: [attn_100, q_41], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf658, primals_278, buf659, 802816, grid=grid(802816), stream=stream0)
        buf660 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_100], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf658, primals_278, buf660, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf661 = buf629; del buf629  # reuse
        # Source Nodes: [attn_100], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf659, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf660, (512, 32, 49), (1568, 49, 1), 0), out=buf661)
        buf664 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf666 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_101, attn_102, attn_103], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_34.run(buf661, primals_360, primals_21, buf664, buf666, 25088, 49, grid=grid(25088), stream=stream0)
        del primals_21
        buf665 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf658, primals_278, buf665, 802816, grid=grid(802816), stream=stream0)
        del primals_278
        buf667 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf666, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf665, (512, 49, 32), (1568, 32, 1), 0), out=buf667)
        buf668 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_379], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf667, buf668, 802816, grid=grid(802816), stream=stream0)
        buf669 = reinterpret_tensor(buf667, (1568, 512), (512, 1), 0); del buf667  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf668, reinterpret_tensor(primals_279, (512, 512), (1, 512), 0), out=buf669)
        buf670 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_38], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf670, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf670, 0.9130434766411781)
        buf676 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf677 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf802 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___16___norm2, x_386], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_73.run(buf652, buf669, primals_280, buf670, primals_281, primals_282, buf676, buf677, buf802, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_282
        buf678 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_386], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_284, buf677, reinterpret_tensor(primals_283, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf678)
        del primals_284
        buf679 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_387, x_390], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf678, buf679, 3211264, grid=grid(3211264), stream=stream0)
        buf680 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf679, reinterpret_tensor(primals_285, (2048, 512), (1, 2048), 0), out=buf680)
        buf681 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_39], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf681, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf681, 0.9130434766411781)
        buf684 = reinterpret_tensor(buf680, (8, 196, 512), (100352, 512, 1), 0); del buf680  # reuse
        buf688 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf801 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__39, getattr_getattr_l__mod___layers___2___blocks___17___norm1, mul_60, x_392], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_74.run(buf684, buf652, buf669, primals_280, buf670, primals_286, buf681, buf688, buf801, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_280
        del primals_286
        buf689 = buf669; del buf669  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___17___attn_qkv], Original ATen: [aten.view]
        triton_poi_fused_view_40.run(buf688, primals_287, primals_288, buf689, 802816, grid=grid(802816), stream=stream0)
        del primals_288
        buf690 = buf658; del buf658  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf689, reinterpret_tensor(primals_289, (512, 1536), (1, 512), 0), out=buf690)
        buf691 = reinterpret_tensor(buf652, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf652  # reuse
        # Source Nodes: [attn_104, q_43], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_32.run(buf690, primals_290, buf691, 802816, grid=grid(802816), stream=stream0)
        buf692 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_104], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf690, primals_290, buf692, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf693 = buf661; del buf661  # reuse
        # Source Nodes: [attn_104], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf691, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf692, (512, 32, 49), (1568, 49, 1), 0), out=buf693)
        buf697 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        buf800 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_108, attn_109], Original ATen: [aten._softmax, aten.clone, aten.detach]
        triton_per_fused__softmax_clone_detach_41.run(buf693, primals_362, primals_22, primals_361, buf697, buf800, 25088, 49, grid=grid(25088), stream=stream0)
        del buf693
        del primals_22
        del primals_361
        buf698 = empty((32, 16, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_395], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf690, primals_290, buf698, 802816, grid=grid(802816), stream=stream0)
        del buf690
        del primals_290
        buf699 = empty((512, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_395], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf697, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf698, (512, 49, 32), (1568, 32, 1), 0), out=buf699)
        buf700 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_397], Original ATen: [aten.view]
        triton_poi_fused_view_36.run(buf699, buf700, 802816, grid=grid(802816), stream=stream0)
        buf701 = reinterpret_tensor(buf699, (1568, 512), (512, 1), 0); del buf699  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf700, reinterpret_tensor(primals_291, (512, 512), (1, 512), 0), out=buf701)
        buf702 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_40], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf702, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf702, 0.9086956530809402)
        buf708 = empty((8, 196, 512), device='cuda', dtype=torch.float32)
        buf709 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf799 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___17___norm2, x_404], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_75.run(buf684, buf701, primals_292, buf702, primals_293, primals_294, buf708, buf709, buf799, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_294
        buf710 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_404], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_296, buf709, reinterpret_tensor(primals_295, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf710)
        del primals_296
        buf711 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_405, x_408], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_38.run(buf710, buf711, 3211264, grid=grid(3211264), stream=stream0)
        buf712 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf711, reinterpret_tensor(primals_297, (2048, 512), (1, 2048), 0), out=buf712)
        buf713 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_41], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf713, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf713, 0.9086956530809402)
        buf716 = reinterpret_tensor(buf712, (8, 7, 7, 2, 2, 512), (100352, 14336, 1024, 512, 7168, 1), 0); del buf712  # reuse
        # Source Nodes: [x_413], Original ATen: [aten.clone]
        triton_poi_fused_clone_76.run(buf716, buf684, buf701, primals_292, buf702, primals_298, buf713, 802816, grid=grid(802816), stream=stream0)
        del primals_292
        del primals_298
        buf720 = reinterpret_tensor(buf701, (8, 7, 7, 2048), (100352, 14336, 2048, 1), 0); del buf701  # reuse
        buf721 = reinterpret_tensor(buf684, (392, 2048), (2048, 1), 0); del buf684  # reuse
        buf798 = empty((8, 7, 7, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_414, x_416], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_view_77.run(buf716, primals_299, primals_300, buf720, buf721, buf798, 392, 2048, grid=grid(392), stream=stream0)
        del buf716
        del primals_300
        buf722 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_416], Original ATen: [aten.mm]
        extern_kernels.mm(buf721, reinterpret_tensor(primals_301, (2048, 1024), (1, 2048), 0), out=buf722)
        buf723 = empty((8, 7, 7, 1), device='cuda', dtype=torch.float32)
        buf724 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        buf726 = reinterpret_tensor(buf724, (8, 7, 7, 1), (49, 7, 1, 1), 0); del buf724  # reuse
        buf727 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv, shifted_x_88], Original ATen: [aten.native_layer_norm, aten.view]
        triton_per_fused_native_layer_norm_view_78.run(buf726, buf722, primals_302, primals_303, buf723, buf727, 392, 1024, grid=grid(392), stream=stream0)
        del primals_303
        buf728 = empty((392, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf727, reinterpret_tensor(primals_304, (1024, 3072), (1, 1024), 0), out=buf728)
        buf729 = empty((8, 32, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_110, q_45], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_79.run(buf728, primals_305, buf729, 401408, grid=grid(401408), stream=stream0)
        buf730 = empty((8, 32, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_110], Original ATen: [aten.clone]
        triton_poi_fused_clone_80.run(buf728, primals_305, buf730, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf731 = empty((256, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_110], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf729, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf730, (256, 32, 49), (1568, 49, 1), 0), out=buf731)
        buf734 = empty((8, 32, 49, 49), device='cuda', dtype=torch.float32)
        buf736 = empty((8, 32, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_111, attn_112, attn_113], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_81.run(buf731, primals_363, primals_23, buf734, buf736, 12544, 49, grid=grid(12544), stream=stream0)
        del primals_23
        buf735 = empty((8, 32, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_418], Original ATen: [aten.clone]
        triton_poi_fused_clone_82.run(buf728, primals_305, buf735, 401408, grid=grid(401408), stream=stream0)
        del primals_305
        buf737 = empty((256, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_418], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf736, (256, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf735, (256, 49, 32), (1568, 32, 1), 0), out=buf737)
        buf738 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_420], Original ATen: [aten.view]
        triton_poi_fused_view_83.run(buf737, buf738, 401408, grid=grid(401408), stream=stream0)
        buf739 = reinterpret_tensor(buf737, (392, 1024), (1024, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf738, reinterpret_tensor(primals_306, (1024, 1024), (1, 1024), 0), out=buf739)
        buf740 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_42], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf740, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf740, 0.9043478220701218)
        buf746 = empty((8, 49, 1024), device='cuda', dtype=torch.float32)
        buf747 = empty((392, 1024), device='cuda', dtype=torch.float32)
        buf797 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___norm2, x_427], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_84.run(buf722, buf739, primals_307, buf740, primals_308, primals_309, buf746, buf747, buf797, 392, 1024, grid=grid(392), stream=stream0)
        del primals_309
        buf748 = reinterpret_tensor(buf134, (392, 4096), (4096, 1), 0); del buf134  # reuse
        # Source Nodes: [x_427], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_311, buf747, reinterpret_tensor(primals_310, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf748)
        del primals_311
        buf749 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_428, x_431], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_85.run(buf748, buf749, 1605632, grid=grid(1605632), stream=stream0)
        buf750 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf749, reinterpret_tensor(primals_312, (4096, 1024), (1, 4096), 0), out=buf750)
        buf751 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_43], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf751, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf751, 0.9043478220701218)
        buf754 = reinterpret_tensor(buf750, (8, 49, 1024), (50176, 1024, 1), 0); del buf750  # reuse
        buf758 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        buf759 = empty((392, 1024), device='cuda', dtype=torch.float32)
        buf796 = empty((8, 7, 7, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__43, getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv, mul_66, shifted_x_92, x_433], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_86.run(buf754, buf722, buf739, primals_307, buf740, primals_313, buf751, primals_314, primals_315, buf758, buf759, buf796, 392, 1024, grid=grid(392), stream=stream0)
        del primals_307
        del primals_313
        del primals_315
        buf760 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf759, reinterpret_tensor(primals_316, (1024, 3072), (1, 1024), 0), out=buf760)
        buf761 = reinterpret_tensor(buf739, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf739  # reuse
        # Source Nodes: [attn_114, q_47], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_79.run(buf760, primals_317, buf761, 401408, grid=grid(401408), stream=stream0)
        buf762 = empty((8, 32, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_114], Original ATen: [aten.clone]
        triton_poi_fused_clone_80.run(buf760, primals_317, buf762, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf763 = buf731; del buf731  # reuse
        # Source Nodes: [attn_114], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf761, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf762, (256, 32, 49), (1568, 49, 1), 0), out=buf763)
        buf766 = empty((8, 32, 49, 49), device='cuda', dtype=torch.float32)
        buf768 = empty((8, 32, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_115, attn_116, attn_117], Original ATen: [aten._softmax, aten.add, aten.clone]
        triton_per_fused__softmax_add_clone_81.run(buf763, primals_364, primals_24, buf766, buf768, 12544, 49, grid=grid(12544), stream=stream0)
        del buf763
        del primals_24
        buf767 = empty((8, 32, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_436], Original ATen: [aten.clone]
        triton_poi_fused_clone_82.run(buf760, primals_317, buf767, 401408, grid=grid(401408), stream=stream0)
        del buf760
        del primals_317
        buf769 = empty((256, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_436], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf768, (256, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf767, (256, 49, 32), (1568, 32, 1), 0), out=buf769)
        buf770 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_438], Original ATen: [aten.view]
        triton_poi_fused_view_83.run(buf769, buf770, 401408, grid=grid(401408), stream=stream0)
        buf771 = reinterpret_tensor(buf769, (392, 1024), (1024, 1), 0); del buf769  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf770, reinterpret_tensor(primals_318, (1024, 1024), (1, 1024), 0), out=buf771)
        buf772 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_44], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf772, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf772, 0.8999999985098839)
        buf778 = empty((8, 49, 1024), device='cuda', dtype=torch.float32)
        buf779 = empty((392, 1024), device='cuda', dtype=torch.float32)
        buf795 = empty((8, 49, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___1___norm2, x_445], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_87.run(buf754, buf771, primals_319, buf772, primals_320, primals_321, buf778, buf779, buf795, 392, 1024, grid=grid(392), stream=stream0)
        del primals_321
        buf780 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_445], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_323, buf779, reinterpret_tensor(primals_322, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf780)
        del primals_323
        buf781 = empty((392, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_446, x_449], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_85.run(buf780, buf781, 1605632, grid=grid(1605632), stream=stream0)
        buf782 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf781, reinterpret_tensor(primals_324, (4096, 1024), (1, 4096), 0), out=buf782)
        buf783 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_45], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_12.run(buf783, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf783, 0.8999999985098839)
        buf786 = reinterpret_tensor(buf782, (8, 49, 1024), (50176, 1024, 1), 0); del buf782  # reuse
        buf790 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        buf794 = empty((8, 7, 7, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__45, mul_69, x_451, x_456], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_88.run(buf786, buf754, buf771, primals_319, buf772, primals_325, buf783, buf790, buf794, 392, 1024, grid=grid(392), stream=stream0)
        del buf754
        del buf771
        del buf786
        del primals_319
        del primals_325
        buf791 = empty((8, 1024), device='cuda', dtype=torch.float32)
        buf792 = buf791; del buf791  # reuse
        # Source Nodes: [x_456, x_457], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_89.run(buf792, buf790, primals_326, primals_327, 8192, 49, grid=grid(8192), stream=stream0)
        del primals_327
        buf793 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_461], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_329, buf792, reinterpret_tensor(primals_328, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf793)
        del primals_329
        return (buf793, primals_25, primals_27, primals_29, primals_35, primals_41, primals_47, primals_53, primals_56, primals_62, primals_68, primals_74, primals_80, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_155, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_302, primals_308, primals_314, primals_320, primals_326, primals_365, buf4, buf8, buf9, reinterpret_tensor(primals_330, (2401, ), (1, ), 0), buf20, buf26, buf27, buf28, buf29, buf34, buf35, reinterpret_tensor(primals_332, (2401, ), (1, ), 0), buf46, buf49, buf56, buf57, buf58, buf59, buf62, buf68, buf69, buf70, buf71, buf74, buf75, reinterpret_tensor(primals_333, (2401, ), (1, ), 0), buf86, buf88, buf94, buf95, buf96, buf97, buf99, buf106, buf107, reinterpret_tensor(primals_335, (2401, ), (1, ), 0), buf118, buf120, buf126, buf127, buf128, buf129, buf131, buf138, buf139, buf140, buf141, buf144, buf145, reinterpret_tensor(primals_336, (2401, ), (1, ), 0), buf156, buf158, buf164, buf165, buf166, buf167, buf169, buf176, buf177, reinterpret_tensor(primals_338, (2401, ), (1, ), 0), buf188, buf190, buf196, buf197, buf198, buf199, buf201, buf208, buf209, reinterpret_tensor(primals_339, (2401, ), (1, ), 0), buf220, buf222, buf228, buf229, buf230, buf231, buf233, buf240, buf241, reinterpret_tensor(primals_341, (2401, ), (1, ), 0), buf252, buf254, buf260, buf261, buf262, buf263, buf265, buf272, buf273, reinterpret_tensor(primals_342, (2401, ), (1, ), 0), buf284, buf286, buf292, buf293, buf294, buf295, buf297, buf304, buf305, reinterpret_tensor(primals_344, (2401, ), (1, ), 0), buf316, buf318, buf324, buf325, buf326, buf327, buf329, buf336, buf337, reinterpret_tensor(primals_345, (2401, ), (1, ), 0), buf348, buf350, buf356, buf357, buf358, buf359, buf361, buf368, buf369, reinterpret_tensor(primals_347, (2401, ), (1, ), 0), buf380, buf382, buf388, buf389, buf390, buf391, buf393, buf400, buf401, reinterpret_tensor(primals_348, (2401, ), (1, ), 0), buf412, buf414, buf420, buf421, buf422, buf423, buf425, buf432, buf433, reinterpret_tensor(primals_350, (2401, ), (1, ), 0), buf444, buf446, buf452, buf453, buf454, buf455, buf457, buf464, buf465, reinterpret_tensor(primals_351, (2401, ), (1, ), 0), buf476, buf478, buf484, buf485, buf486, buf487, buf489, buf496, buf497, reinterpret_tensor(primals_353, (2401, ), (1, ), 0), buf508, buf510, buf516, buf517, buf518, buf519, buf521, buf528, buf529, reinterpret_tensor(primals_354, (2401, ), (1, ), 0), buf540, buf542, buf548, buf549, buf550, buf551, buf553, buf560, buf561, reinterpret_tensor(primals_356, (2401, ), (1, ), 0), buf572, buf574, buf580, buf581, buf582, buf583, buf585, buf592, buf593, reinterpret_tensor(primals_357, (2401, ), (1, ), 0), buf604, buf606, buf612, buf613, buf614, buf615, buf617, buf624, buf625, reinterpret_tensor(primals_359, (2401, ), (1, ), 0), buf636, buf638, buf644, buf645, buf646, buf647, buf649, buf656, buf657, reinterpret_tensor(primals_360, (2401, ), (1, ), 0), buf668, buf670, buf676, buf677, buf678, buf679, buf681, buf688, buf689, reinterpret_tensor(primals_362, (2401, ), (1, ), 0), buf700, buf702, buf708, buf709, buf710, buf711, buf713, buf720, buf721, buf722, buf723, buf726, buf727, reinterpret_tensor(primals_363, (2401, ), (1, ), 0), buf738, buf740, buf746, buf747, buf748, buf749, buf751, buf758, buf759, reinterpret_tensor(primals_364, (2401, ), (1, ), 0), buf770, buf772, buf778, buf779, buf780, buf781, buf783, buf790, buf792, reinterpret_tensor(primals_328, (1000, 1024), (1024, 1), 0), buf794, reinterpret_tensor(primals_324, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_322, (4096, 1024), (1024, 1), 0), buf795, reinterpret_tensor(primals_318, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf768, (256, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf767, (256, 32, 49), (1568, 1, 32), 0), buf766, reinterpret_tensor(buf761, (256, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf762, (256, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_316, (3072, 1024), (1024, 1), 0), buf796, reinterpret_tensor(primals_312, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_310, (4096, 1024), (1024, 1), 0), buf797, reinterpret_tensor(primals_306, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf736, (256, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf735, (256, 32, 49), (1568, 1, 32), 0), buf734, reinterpret_tensor(buf729, (256, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf730, (256, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_304, (3072, 1024), (1024, 1), 0), reinterpret_tensor(primals_301, (1024, 2048), (2048, 1), 0), buf798, reinterpret_tensor(primals_297, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_295, (2048, 512), (512, 1), 0), buf799, reinterpret_tensor(primals_291, (512, 512), (512, 1), 0), reinterpret_tensor(buf697, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf698, (512, 32, 49), (1568, 1, 32), 0), buf800, reinterpret_tensor(buf691, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf692, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_289, (1536, 512), (512, 1), 0), buf801, reinterpret_tensor(primals_285, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_283, (2048, 512), (512, 1), 0), buf802, reinterpret_tensor(primals_279, (512, 512), (512, 1), 0), reinterpret_tensor(buf666, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf665, (512, 32, 49), (1568, 1, 32), 0), buf664, reinterpret_tensor(buf659, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf660, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_277, (1536, 512), (512, 1), 0), buf803, reinterpret_tensor(primals_273, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_271, (2048, 512), (512, 1), 0), buf804, reinterpret_tensor(primals_267, (512, 512), (512, 1), 0), reinterpret_tensor(buf633, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf634, (512, 32, 49), (1568, 1, 32), 0), buf805, reinterpret_tensor(buf627, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf628, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_265, (1536, 512), (512, 1), 0), buf806, reinterpret_tensor(primals_261, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_259, (2048, 512), (512, 1), 0), buf807, reinterpret_tensor(primals_255, (512, 512), (512, 1), 0), reinterpret_tensor(buf602, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf601, (512, 32, 49), (1568, 1, 32), 0), buf600, reinterpret_tensor(buf595, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf596, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_253, (1536, 512), (512, 1), 0), buf808, reinterpret_tensor(primals_249, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_247, (2048, 512), (512, 1), 0), buf809, reinterpret_tensor(primals_243, (512, 512), (512, 1), 0), reinterpret_tensor(buf569, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf570, (512, 32, 49), (1568, 1, 32), 0), buf810, reinterpret_tensor(buf563, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf564, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_241, (1536, 512), (512, 1), 0), buf811, reinterpret_tensor(primals_237, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_235, (2048, 512), (512, 1), 0), buf812, reinterpret_tensor(primals_231, (512, 512), (512, 1), 0), reinterpret_tensor(buf538, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf537, (512, 32, 49), (1568, 1, 32), 0), buf536, reinterpret_tensor(buf531, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf532, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_229, (1536, 512), (512, 1), 0), buf813, reinterpret_tensor(primals_225, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_223, (2048, 512), (512, 1), 0), buf814, reinterpret_tensor(primals_219, (512, 512), (512, 1), 0), reinterpret_tensor(buf505, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf506, (512, 32, 49), (1568, 1, 32), 0), buf815, reinterpret_tensor(buf499, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf500, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_217, (1536, 512), (512, 1), 0), buf816, reinterpret_tensor(primals_213, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_211, (2048, 512), (512, 1), 0), buf817, reinterpret_tensor(primals_207, (512, 512), (512, 1), 0), reinterpret_tensor(buf474, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf473, (512, 32, 49), (1568, 1, 32), 0), buf472, reinterpret_tensor(buf467, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf468, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_205, (1536, 512), (512, 1), 0), buf818, reinterpret_tensor(primals_201, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_199, (2048, 512), (512, 1), 0), buf819, reinterpret_tensor(primals_195, (512, 512), (512, 1), 0), reinterpret_tensor(buf441, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf442, (512, 32, 49), (1568, 1, 32), 0), buf820, reinterpret_tensor(buf435, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf436, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_193, (1536, 512), (512, 1), 0), buf821, reinterpret_tensor(primals_189, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_187, (2048, 512), (512, 1), 0), buf822, reinterpret_tensor(primals_183, (512, 512), (512, 1), 0), reinterpret_tensor(buf410, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf409, (512, 32, 49), (1568, 1, 32), 0), buf408, reinterpret_tensor(buf403, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf404, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_181, (1536, 512), (512, 1), 0), buf823, reinterpret_tensor(primals_177, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_175, (2048, 512), (512, 1), 0), buf824, reinterpret_tensor(primals_171, (512, 512), (512, 1), 0), reinterpret_tensor(buf377, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf378, (512, 32, 49), (1568, 1, 32), 0), buf825, reinterpret_tensor(buf371, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf372, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_169, (1536, 512), (512, 1), 0), buf826, reinterpret_tensor(primals_165, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_163, (2048, 512), (512, 1), 0), buf827, reinterpret_tensor(primals_159, (512, 512), (512, 1), 0), reinterpret_tensor(buf346, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf345, (512, 32, 49), (1568, 1, 32), 0), buf344, reinterpret_tensor(buf339, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf340, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_157, (1536, 512), (512, 1), 0), buf828, reinterpret_tensor(primals_153, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_151, (2048, 512), (512, 1), 0), buf829, reinterpret_tensor(primals_147, (512, 512), (512, 1), 0), reinterpret_tensor(buf313, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf314, (512, 32, 49), (1568, 1, 32), 0), buf830, reinterpret_tensor(buf307, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf308, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_145, (1536, 512), (512, 1), 0), buf831, reinterpret_tensor(primals_141, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_139, (2048, 512), (512, 1), 0), buf832, reinterpret_tensor(primals_135, (512, 512), (512, 1), 0), reinterpret_tensor(buf282, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf281, (512, 32, 49), (1568, 1, 32), 0), buf280, reinterpret_tensor(buf275, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf276, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_133, (1536, 512), (512, 1), 0), buf833, reinterpret_tensor(primals_129, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_127, (2048, 512), (512, 1), 0), buf834, reinterpret_tensor(primals_123, (512, 512), (512, 1), 0), reinterpret_tensor(buf249, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf250, (512, 32, 49), (1568, 1, 32), 0), buf835, reinterpret_tensor(buf243, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf244, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_121, (1536, 512), (512, 1), 0), buf836, reinterpret_tensor(primals_117, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_115, (2048, 512), (512, 1), 0), buf837, reinterpret_tensor(primals_111, (512, 512), (512, 1), 0), reinterpret_tensor(buf218, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf217, (512, 32, 49), (1568, 1, 32), 0), buf216, reinterpret_tensor(buf211, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf212, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_109, (1536, 512), (512, 1), 0), buf838, reinterpret_tensor(primals_105, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_103, (2048, 512), (512, 1), 0), buf839, reinterpret_tensor(primals_99, (512, 512), (512, 1), 0), reinterpret_tensor(buf185, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf186, (512, 32, 49), (1568, 1, 32), 0), buf840, reinterpret_tensor(buf179, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf180, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_97, (1536, 512), (512, 1), 0), buf841, reinterpret_tensor(primals_93, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_91, (2048, 512), (512, 1), 0), buf842, reinterpret_tensor(primals_87, (512, 512), (512, 1), 0), reinterpret_tensor(buf154, (512, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf153, (512, 32, 49), (1568, 1, 32), 0), buf152, reinterpret_tensor(buf147, (512, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf148, (512, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_85, (1536, 512), (512, 1), 0), reinterpret_tensor(primals_82, (512, 1024), (1024, 1), 0), buf843, reinterpret_tensor(primals_78, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_76, (1024, 256), (256, 1), 0), buf844, reinterpret_tensor(primals_72, (256, 256), (256, 1), 0), reinterpret_tensor(buf115, (1024, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf116, (1024, 32, 49), (1568, 1, 32), 0), buf845, reinterpret_tensor(buf109, (1024, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf110, (1024, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_70, (768, 256), (256, 1), 0), buf846, reinterpret_tensor(primals_66, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_64, (1024, 256), (256, 1), 0), buf847, reinterpret_tensor(primals_60, (256, 256), (256, 1), 0), reinterpret_tensor(buf84, (1024, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf83, (1024, 32, 49), (1568, 1, 32), 0), buf82, reinterpret_tensor(buf77, (1024, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf78, (1024, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_58, (768, 256), (256, 1), 0), reinterpret_tensor(primals_55, (256, 512), (512, 1), 0), buf848, reinterpret_tensor(primals_51, (128, 512), (512, 1), 0), reinterpret_tensor(primals_49, (512, 128), (128, 1), 0), buf849, reinterpret_tensor(primals_45, (128, 128), (128, 1), 0), reinterpret_tensor(buf43, (2048, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf44, (2048, 32, 49), (1568, 1, 32), 0), buf850, reinterpret_tensor(buf37, (2048, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf38, (2048, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_43, (384, 128), (128, 1), 0), buf851, reinterpret_tensor(primals_39, (128, 512), (512, 1), 0), reinterpret_tensor(primals_37, (512, 128), (128, 1), 0), buf852, reinterpret_tensor(primals_33, (128, 128), (128, 1), 0), reinterpret_tensor(buf18, (2048, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf17, (2048, 32, 49), (1568, 1, 32), 0), buf16, reinterpret_tensor(buf11, (2048, 32, 49), (1568, 1, 32), 0), reinterpret_tensor(buf12, (2048, 49, 32), (1568, 1, 49), 0), reinterpret_tensor(primals_31, (384, 128), (128, 1), 0), buf853, buf854, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((169, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((169, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((169, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((169, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((169, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((169, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((64, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((16, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_349 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_352 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_355 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_358 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_364 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_365 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swin_base_patch4_window7_224', benchmark_compiled_module)
