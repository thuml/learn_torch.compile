
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


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2jiarujvy6azncpmon5w3egryo5lmdw65q7kmj7peztqriucal.py
# Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___blocks___0___norm => clone, var_mean
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 2
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


# kernel path: /tmp/torchinductor_youkaichao/4o/c4ozffaznmpcnwotcha2pkjafb6vdu27cpettxoflxdx3gvrrgnw.py
# Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks___0___norm => add, clone, rsqrt, var_mean
triton_per_fused_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (392*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (392*x1)), rmask & xmask, other=0.0)
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
    tmp16 = 256.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gqtcijovvarfyq35roi4moeb43hxaus7arnpzy6gznemahrkbd.py
# Source Nodes: [getattr_l__mod___blocks___0___norm, x_4], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_l__mod___blocks___0___norm => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
# x_4 => view_1
triton_poi_fused_native_layer_norm_view_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp11, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5rlrpjqfxls4kekfj26awdtpn3y2zqg6ujoeerb33lvxojrlnz.py
# Source Nodes: [v_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# v_1 => add_3, clone_2, mul_5, rsqrt_1, sub_1, var_mean_1
triton_per_fused_native_layer_norm_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + r1 + (1536*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tmp30 / tmp26
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2wimt7zdibdh5jxvskawsy4vpinuzqgwjfsgmvjkyg6dk53rvi.py
# Source Nodes: [v_2], Original ATen: [aten._unsafe_view, aten.clone]
# v_2 => clone_3, view_3
triton_poi_fused__unsafe_view_clone_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((768*x1) + (150528*(y0 // 768)) + (y0 % 768)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 % 768), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 % 768), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x1 + (196*y0)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce47dzaylceomi56g3wcsqgljtlv5xl4avi2l66zxmoqtfem4t2r.py
# Source Nodes: [x_7, x_9], Original ATen: [aten.mul, aten.view]
# x_7 => mul_7
# x_9 => view_5
triton_poi_fused_mul_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + (1536*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + ((196*x1) + (150528*(y0 // 196)) + (y0 % 196)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0 % 196), ymask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a3kwjkghx5ited3skaulnsviilmx7xhqejldr2d54jy74vhsfq.py
# Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks___1___norm => clone_5, var_mean_2
# x_11 => add_6
triton_red_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 196
    x2 = (xindex // 392)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (50176*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4s/c4sh32olqqpmwvi3zhhvvb7uqgb7iworozittepesfvu3xxi6vga.py
# Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks___1___norm => add_7, clone_5, rsqrt_2, var_mean_2
# x_11 => add_6
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
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
    tmp16 = 256.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xpmgc57etjxbyhmkfbmzmsxp4nohczd6npxqrtl3nzwzcm2vsj.py
# Source Nodes: [getattr_l__mod___blocks___1___norm, x_11, x_12], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_l__mod___blocks___1___norm => add_7, add_8, clone_5, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
# x_11 => add_6
# x_12 => view_7
triton_poi_fused_add_native_layer_norm_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 256.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6g6dn7jw2cl3qjqqiipmcw4koxl2rhadzp2qsjyukokf3kwq5yn.py
# Source Nodes: [getattr_l__mod___blocks___2___norm, x_11, x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___2___norm => add_14, add_15, clone_10, mul_16, mul_17, rsqrt_4, sub_4, var_mean_4
# x_11 => add_6
# x_19 => add_13
# x_20 => view_13
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (50176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 256, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 256.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-06
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp32 / tmp28
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp33, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2h6loqstkzbk6l3qcccbd56j2wewbduraxqt6cbdmo647zr574v.py
# Source Nodes: [getattr_l__mod___blocks___3___norm, x_27, x_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___3___norm => add_21, add_22, clone_15, mul_24, mul_25, rsqrt_6, sub_6, var_mean_6
# x_27 => add_20
# x_28 => view_19
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rwibldvb74v7pq2xqz4zeppngn6witjxqsiqbps2loeui374m6.py
# Source Nodes: [getattr_l__mod___blocks___4___norm, x_27, x_35, x_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___4___norm => add_28, add_29, clone_20, mul_32, mul_33, rsqrt_8, sub_8, var_mean_8
# x_27 => add_20
# x_35 => add_27
# x_36 => view_25
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1568
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bplifcmmkkecqf6eankytrnlwbntch5iuqxwk4oxx7ykmp63ky.py
# Source Nodes: [x_235, x_244, x_246], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_235 => add_202
# x_244 => add_209
# x_246 => add_210, clone_150, mul_240, rsqrt_60, sub_60, var_mean_60
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
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
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clnembjhkxmxt5mlp25zqpbragpsoqpww6mzhuw6ezdmg6evpj6n.py
# Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
# x_246 => add_211, mul_241
# x_247 => mean
triton_red_fused_mean_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhzs3ul6ay5apbvnbglnv5yatb7qjyxri7u22gjmdrmff4leq3j.py
# Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
# x_246 => add_211, mul_241
# x_247 => mean
triton_per_fused_mean_native_layer_norm_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (512*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coypg4g5voc3ymrrcigopcszen3euyhbkki2qb3osqfibwf62vns.py
# Source Nodes: [x_5], Original ATen: [aten.gelu]
# x_5 => add_2, erf, mul_2, mul_3, mul_4
triton_poi_fused_gelu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307 = args
    args.clear()
    assert_size_stride(primals_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_2, (256, ), (1, ))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (1536, 256), (256, 1))
    assert_size_stride(primals_6, (1536, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (196, 196), (196, 1))
    assert_size_stride(primals_10, (196, ), (1, ))
    assert_size_stride(primals_11, (256, 768), (768, 1))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (1536, 256), (256, 1))
    assert_size_stride(primals_16, (1536, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (196, 196), (196, 1))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_21, (256, 768), (768, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (1536, 256), (256, 1))
    assert_size_stride(primals_26, (1536, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (196, 196), (196, 1))
    assert_size_stride(primals_30, (196, ), (1, ))
    assert_size_stride(primals_31, (256, 768), (768, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (1536, 256), (256, 1))
    assert_size_stride(primals_36, (1536, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (196, 196), (196, 1))
    assert_size_stride(primals_40, (196, ), (1, ))
    assert_size_stride(primals_41, (256, 768), (768, 1))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (1536, 256), (256, 1))
    assert_size_stride(primals_46, (1536, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (196, 196), (196, 1))
    assert_size_stride(primals_50, (196, ), (1, ))
    assert_size_stride(primals_51, (256, 768), (768, 1))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (1536, 256), (256, 1))
    assert_size_stride(primals_56, (1536, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (196, 196), (196, 1))
    assert_size_stride(primals_60, (196, ), (1, ))
    assert_size_stride(primals_61, (256, 768), (768, 1))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (1536, 256), (256, 1))
    assert_size_stride(primals_66, (1536, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (196, 196), (196, 1))
    assert_size_stride(primals_70, (196, ), (1, ))
    assert_size_stride(primals_71, (256, 768), (768, 1))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (1536, 256), (256, 1))
    assert_size_stride(primals_76, (1536, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (196, 196), (196, 1))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_81, (256, 768), (768, 1))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (1536, 256), (256, 1))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (196, 196), (196, 1))
    assert_size_stride(primals_90, (196, ), (1, ))
    assert_size_stride(primals_91, (256, 768), (768, 1))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (1536, 256), (256, 1))
    assert_size_stride(primals_96, (1536, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (196, 196), (196, 1))
    assert_size_stride(primals_100, (196, ), (1, ))
    assert_size_stride(primals_101, (256, 768), (768, 1))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (1536, 256), (256, 1))
    assert_size_stride(primals_106, (1536, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (196, 196), (196, 1))
    assert_size_stride(primals_110, (196, ), (1, ))
    assert_size_stride(primals_111, (256, 768), (768, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (1536, 256), (256, 1))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (196, 196), (196, 1))
    assert_size_stride(primals_120, (196, ), (1, ))
    assert_size_stride(primals_121, (256, 768), (768, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (1536, 256), (256, 1))
    assert_size_stride(primals_126, (1536, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (196, 196), (196, 1))
    assert_size_stride(primals_130, (196, ), (1, ))
    assert_size_stride(primals_131, (256, 768), (768, 1))
    assert_size_stride(primals_132, (256, ), (1, ))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (1536, 256), (256, 1))
    assert_size_stride(primals_136, (1536, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (196, 196), (196, 1))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_141, (256, 768), (768, 1))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (1536, 256), (256, 1))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (196, 196), (196, 1))
    assert_size_stride(primals_150, (196, ), (1, ))
    assert_size_stride(primals_151, (256, 768), (768, 1))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (256, ), (1, ))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (1536, 256), (256, 1))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (196, 196), (196, 1))
    assert_size_stride(primals_160, (196, ), (1, ))
    assert_size_stride(primals_161, (256, 768), (768, 1))
    assert_size_stride(primals_162, (256, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_164, (256, ), (1, ))
    assert_size_stride(primals_165, (1536, 256), (256, 1))
    assert_size_stride(primals_166, (1536, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (196, 196), (196, 1))
    assert_size_stride(primals_170, (196, ), (1, ))
    assert_size_stride(primals_171, (256, 768), (768, 1))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (256, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (1536, 256), (256, 1))
    assert_size_stride(primals_176, (1536, ), (1, ))
    assert_size_stride(primals_177, (768, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (196, 196), (196, 1))
    assert_size_stride(primals_180, (196, ), (1, ))
    assert_size_stride(primals_181, (256, 768), (768, 1))
    assert_size_stride(primals_182, (256, ), (1, ))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (1536, 256), (256, 1))
    assert_size_stride(primals_186, (1536, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (196, 196), (196, 1))
    assert_size_stride(primals_190, (196, ), (1, ))
    assert_size_stride(primals_191, (256, 768), (768, 1))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (256, ), (1, ))
    assert_size_stride(primals_195, (1536, 256), (256, 1))
    assert_size_stride(primals_196, (1536, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (196, 196), (196, 1))
    assert_size_stride(primals_200, (196, ), (1, ))
    assert_size_stride(primals_201, (256, 768), (768, 1))
    assert_size_stride(primals_202, (256, ), (1, ))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_204, (256, ), (1, ))
    assert_size_stride(primals_205, (1536, 256), (256, 1))
    assert_size_stride(primals_206, (1536, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (196, 196), (196, 1))
    assert_size_stride(primals_210, (196, ), (1, ))
    assert_size_stride(primals_211, (256, 768), (768, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (1536, 256), (256, 1))
    assert_size_stride(primals_216, (1536, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (196, 196), (196, 1))
    assert_size_stride(primals_220, (196, ), (1, ))
    assert_size_stride(primals_221, (256, 768), (768, 1))
    assert_size_stride(primals_222, (256, ), (1, ))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (1536, 256), (256, 1))
    assert_size_stride(primals_226, (1536, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (196, 196), (196, 1))
    assert_size_stride(primals_230, (196, ), (1, ))
    assert_size_stride(primals_231, (256, 768), (768, 1))
    assert_size_stride(primals_232, (256, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (1536, 256), (256, 1))
    assert_size_stride(primals_236, (1536, ), (1, ))
    assert_size_stride(primals_237, (768, ), (1, ))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (196, 196), (196, 1))
    assert_size_stride(primals_240, (196, ), (1, ))
    assert_size_stride(primals_241, (256, 768), (768, 1))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (1536, 256), (256, 1))
    assert_size_stride(primals_246, (1536, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (196, 196), (196, 1))
    assert_size_stride(primals_250, (196, ), (1, ))
    assert_size_stride(primals_251, (256, 768), (768, 1))
    assert_size_stride(primals_252, (256, ), (1, ))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (1536, 256), (256, 1))
    assert_size_stride(primals_256, (1536, ), (1, ))
    assert_size_stride(primals_257, (768, ), (1, ))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (196, 196), (196, 1))
    assert_size_stride(primals_260, (196, ), (1, ))
    assert_size_stride(primals_261, (256, 768), (768, 1))
    assert_size_stride(primals_262, (256, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (1536, 256), (256, 1))
    assert_size_stride(primals_266, (1536, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_268, (768, ), (1, ))
    assert_size_stride(primals_269, (196, 196), (196, 1))
    assert_size_stride(primals_270, (196, ), (1, ))
    assert_size_stride(primals_271, (256, 768), (768, 1))
    assert_size_stride(primals_272, (256, ), (1, ))
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (1536, 256), (256, 1))
    assert_size_stride(primals_276, (1536, ), (1, ))
    assert_size_stride(primals_277, (768, ), (1, ))
    assert_size_stride(primals_278, (768, ), (1, ))
    assert_size_stride(primals_279, (196, 196), (196, 1))
    assert_size_stride(primals_280, (196, ), (1, ))
    assert_size_stride(primals_281, (256, 768), (768, 1))
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (256, ), (1, ))
    assert_size_stride(primals_285, (1536, 256), (256, 1))
    assert_size_stride(primals_286, (1536, ), (1, ))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_288, (768, ), (1, ))
    assert_size_stride(primals_289, (196, 196), (196, 1))
    assert_size_stride(primals_290, (196, ), (1, ))
    assert_size_stride(primals_291, (256, 768), (768, 1))
    assert_size_stride(primals_292, (256, ), (1, ))
    assert_size_stride(primals_293, (256, ), (1, ))
    assert_size_stride(primals_294, (256, ), (1, ))
    assert_size_stride(primals_295, (1536, 256), (256, 1))
    assert_size_stride(primals_296, (1536, ), (1, ))
    assert_size_stride(primals_297, (768, ), (1, ))
    assert_size_stride(primals_298, (768, ), (1, ))
    assert_size_stride(primals_299, (196, 196), (196, 1))
    assert_size_stride(primals_300, (196, ), (1, ))
    assert_size_stride(primals_301, (256, 768), (768, 1))
    assert_size_stride(primals_302, (256, ), (1, ))
    assert_size_stride(primals_303, (256, ), (1, ))
    assert_size_stride(primals_304, (256, ), (1, ))
    assert_size_stride(primals_305, (1000, 256), (256, 1))
    assert_size_stride(primals_306, (1000, ), (1, ))
    assert_size_stride(primals_307, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_307, primals_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf1 = empty_strided((8, 196, 1, 2), (392, 1, 3136, 196), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 196, 1, 2), (392, 1, 3136, 196), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 196, 1, 2), (392, 1, 3136, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, primals_2, buf1, buf2, buf3, 3136, 128, grid=grid(3136), stream=stream0)
        buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf510 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf1, buf2, buf3, buf4, buf5, buf510, 1568, 2, grid=grid(1568), stream=stream0)
        buf7 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf8 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___norm, x_4], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_2.run(buf0, primals_2, buf4, buf5, primals_3, primals_4, buf7, buf8, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del primals_4
        buf9 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf8, reinterpret_tensor(primals_5, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf9)
        del primals_6
        buf13 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf509 = reinterpret_tensor(buf5, (8, 196, 1), (196, 1, 1), 0); del buf5  # reuse
        # Source Nodes: [v_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf9, buf13, buf509, 1568, 768, grid=grid(1568), stream=stream0)
        buf14 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_2], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf13, primals_7, primals_8, buf14, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_8
        buf15 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, reinterpret_tensor(primals_9, (196, 196), (1, 196), 0), out=buf15)
        buf16 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7, x_9], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf9, buf15, primals_10, buf16, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf17 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf16, reinterpret_tensor(primals_11, (768, 256), (1, 768), 0), out=buf17)
        buf18 = reinterpret_tensor(buf3, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf3  # reuse
        buf19 = reinterpret_tensor(buf2, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf2  # reuse
        buf20 = reinterpret_tensor(buf1, (8, 196, 1, 2), (392, 2, 3136, 1), 0); del buf1  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_6.run(buf0, primals_2, buf17, primals_12, buf18, buf19, buf20, 3136, 128, grid=grid(3136), stream=stream0)
        buf21 = buf4; del buf4  # reuse
        buf22 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf508 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm, x_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf18, buf19, buf20, buf21, buf22, buf508, 1568, 2, grid=grid(1568), stream=stream0)
        del buf18
        del buf19
        del buf20
        buf24 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf25 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm, x_11, x_12], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_8.run(buf0, primals_2, buf17, primals_12, buf21, buf22, primals_13, primals_14, buf24, buf25, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del primals_14
        buf26 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, buf25, reinterpret_tensor(primals_15, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf26)
        del primals_16
        buf30 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf507 = reinterpret_tensor(buf22, (8, 196, 1), (196, 1, 1), 0); del buf22  # reuse
        # Source Nodes: [v_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf26, buf30, buf507, 1568, 768, grid=grid(1568), stream=stream0)
        buf31 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_5], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf30, primals_17, primals_18, buf31, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_18
        buf32 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, reinterpret_tensor(primals_19, (196, 196), (1, 196), 0), out=buf32)
        buf33 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15, x_17], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf26, buf32, primals_20, buf33, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf34 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf33, reinterpret_tensor(primals_21, (768, 256), (1, 768), 0), out=buf34)
        buf35 = reinterpret_tensor(buf17, (8, 196, 256), (50176, 256, 1), 0); del buf17  # reuse
        buf39 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf40 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf506 = reinterpret_tensor(buf21, (8, 196, 1), (196, 1, 1), 0); del buf21  # reuse
        # Source Nodes: [getattr_l__mod___blocks___2___norm, x_11, x_19, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_9.run(buf35, buf0, primals_2, primals_12, buf34, primals_22, primals_23, primals_24, buf39, buf40, buf506, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_12
        del primals_2
        del primals_22
        del primals_24
        buf41 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_26, buf40, reinterpret_tensor(primals_25, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf41)
        del primals_26
        buf45 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf505 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_7], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf41, buf45, buf505, 1568, 768, grid=grid(1568), stream=stream0)
        buf46 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_8], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf45, primals_27, primals_28, buf46, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_28
        buf47 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, reinterpret_tensor(primals_29, (196, 196), (1, 196), 0), out=buf47)
        buf48 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23, x_25], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf41, buf47, primals_30, buf48, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf49 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf48, reinterpret_tensor(primals_31, (768, 256), (1, 768), 0), out=buf49)
        buf53 = reinterpret_tensor(buf0, (8, 196, 256), (50176, 256, 1), 0); del buf0  # reuse
        buf54 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf504 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm, x_27, x_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf35, buf49, primals_32, primals_33, primals_34, buf53, buf54, buf504, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_34
        buf55 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_36, buf54, reinterpret_tensor(primals_35, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf55)
        del primals_36
        buf59 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf503 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_10], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf55, buf59, buf503, 1568, 768, grid=grid(1568), stream=stream0)
        buf60 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_11], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf59, primals_37, primals_38, buf60, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_38
        buf61 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf60, reinterpret_tensor(primals_39, (196, 196), (1, 196), 0), out=buf61)
        buf62 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31, x_33], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf55, buf61, primals_40, buf62, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf63 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf62, reinterpret_tensor(primals_41, (768, 256), (1, 768), 0), out=buf63)
        buf64 = reinterpret_tensor(buf63, (8, 196, 256), (50176, 256, 1), 0); del buf63  # reuse
        buf68 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf69 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf502 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm, x_27, x_35, x_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf64, buf35, buf49, primals_32, primals_42, primals_43, primals_44, buf68, buf69, buf502, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_32
        del primals_42
        del primals_44
        buf70 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_46, buf69, reinterpret_tensor(primals_45, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf70)
        del primals_46
        buf74 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf501 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_13], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf70, buf74, buf501, 1568, 768, grid=grid(1568), stream=stream0)
        buf75 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_14], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf74, primals_47, primals_48, buf75, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_48
        buf76 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf75, reinterpret_tensor(primals_49, (196, 196), (1, 196), 0), out=buf76)
        buf77 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39, x_41], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf70, buf76, primals_50, buf77, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf78 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf77, reinterpret_tensor(primals_51, (768, 256), (1, 768), 0), out=buf78)
        buf82 = buf35; del buf35  # reuse
        buf83 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf500 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm, x_43, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf64, buf78, primals_52, primals_53, primals_54, buf82, buf83, buf500, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_54
        buf84 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_56, buf83, reinterpret_tensor(primals_55, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf84)
        del primals_56
        buf88 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf499 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_16], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf84, buf88, buf499, 1568, 768, grid=grid(1568), stream=stream0)
        buf89 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_17], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf88, primals_57, primals_58, buf89, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_58
        buf90 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf89, reinterpret_tensor(primals_59, (196, 196), (1, 196), 0), out=buf90)
        buf91 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_47, x_49], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf84, buf90, primals_60, buf91, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf92 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf91, reinterpret_tensor(primals_61, (768, 256), (1, 768), 0), out=buf92)
        buf93 = reinterpret_tensor(buf92, (8, 196, 256), (50176, 256, 1), 0); del buf92  # reuse
        buf97 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf98 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf498 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm, x_43, x_51, x_52], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf93, buf64, buf78, primals_52, primals_62, primals_63, primals_64, buf97, buf98, buf498, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_52
        del primals_62
        del primals_64
        buf99 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_66, buf98, reinterpret_tensor(primals_65, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf99)
        del primals_66
        buf103 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf497 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_19], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf99, buf103, buf497, 1568, 768, grid=grid(1568), stream=stream0)
        buf104 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_20], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf103, primals_67, primals_68, buf104, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_68
        buf105 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf104, reinterpret_tensor(primals_69, (196, 196), (1, 196), 0), out=buf105)
        buf106 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55, x_57], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf99, buf105, primals_70, buf106, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf107 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf106, reinterpret_tensor(primals_71, (768, 256), (1, 768), 0), out=buf107)
        buf111 = buf64; del buf64  # reuse
        buf112 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf496 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm, x_59, x_60], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf93, buf107, primals_72, primals_73, primals_74, buf111, buf112, buf496, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_74
        buf113 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_76, buf112, reinterpret_tensor(primals_75, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf113)
        del primals_76
        buf117 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf495 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_22], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf113, buf117, buf495, 1568, 768, grid=grid(1568), stream=stream0)
        buf118 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_23], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf117, primals_77, primals_78, buf118, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_78
        buf119 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf118, reinterpret_tensor(primals_79, (196, 196), (1, 196), 0), out=buf119)
        buf120 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_65], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf113, buf119, primals_80, buf120, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf121 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf120, reinterpret_tensor(primals_81, (768, 256), (1, 768), 0), out=buf121)
        buf122 = reinterpret_tensor(buf121, (8, 196, 256), (50176, 256, 1), 0); del buf121  # reuse
        buf126 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf127 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf494 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm, x_59, x_67, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf122, buf93, buf107, primals_72, primals_82, primals_83, primals_84, buf126, buf127, buf494, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_72
        del primals_82
        del primals_84
        buf128 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_86, buf127, reinterpret_tensor(primals_85, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf128)
        del primals_86
        buf132 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf493 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_25], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf128, buf132, buf493, 1568, 768, grid=grid(1568), stream=stream0)
        buf133 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_26], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf132, primals_87, primals_88, buf133, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_88
        buf134 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf133, reinterpret_tensor(primals_89, (196, 196), (1, 196), 0), out=buf134)
        buf135 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_73], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf128, buf134, primals_90, buf135, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf136 = reinterpret_tensor(buf93, (1568, 256), (256, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf135, reinterpret_tensor(primals_91, (768, 256), (1, 768), 0), out=buf136)
        buf140 = reinterpret_tensor(buf107, (8, 196, 256), (50176, 256, 1), 0); del buf107  # reuse
        buf141 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf492 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm, x_75, x_76], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf122, buf136, primals_92, primals_93, primals_94, buf140, buf141, buf492, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_94
        buf142 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf141, reinterpret_tensor(primals_95, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf142)
        del primals_96
        buf146 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf491 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_28], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf142, buf146, buf491, 1568, 768, grid=grid(1568), stream=stream0)
        buf147 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_29], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf146, primals_97, primals_98, buf147, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_98
        buf148 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf147, reinterpret_tensor(primals_99, (196, 196), (1, 196), 0), out=buf148)
        buf149 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79, x_81], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf142, buf148, primals_100, buf149, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf150 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf149, reinterpret_tensor(primals_101, (768, 256), (1, 768), 0), out=buf150)
        buf151 = reinterpret_tensor(buf150, (8, 196, 256), (50176, 256, 1), 0); del buf150  # reuse
        buf155 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf156 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf490 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm, x_75, x_83, x_84], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf151, buf122, buf136, primals_92, primals_102, primals_103, primals_104, buf155, buf156, buf490, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_102
        del primals_104
        del primals_92
        buf157 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_106, buf156, reinterpret_tensor(primals_105, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf157)
        del primals_106
        buf161 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf489 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_31], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf157, buf161, buf489, 1568, 768, grid=grid(1568), stream=stream0)
        buf162 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_32], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf161, primals_107, primals_108, buf162, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_108
        buf163 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_32], Original ATen: [aten.mm]
        extern_kernels.mm(buf162, reinterpret_tensor(primals_109, (196, 196), (1, 196), 0), out=buf163)
        buf164 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87, x_89], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf157, buf163, primals_110, buf164, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf165 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf164, reinterpret_tensor(primals_111, (768, 256), (1, 768), 0), out=buf165)
        buf169 = buf122; del buf122  # reuse
        buf170 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf488 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm, x_91, x_92], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf151, buf165, primals_112, primals_113, primals_114, buf169, buf170, buf488, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_114
        buf171 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf170, reinterpret_tensor(primals_115, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf171)
        del primals_116
        buf175 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf487 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_34], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf171, buf175, buf487, 1568, 768, grid=grid(1568), stream=stream0)
        buf176 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_35], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf175, primals_117, primals_118, buf176, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_118
        buf177 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_35], Original ATen: [aten.mm]
        extern_kernels.mm(buf176, reinterpret_tensor(primals_119, (196, 196), (1, 196), 0), out=buf177)
        buf178 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95, x_97], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf171, buf177, primals_120, buf178, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf179 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_121, (768, 256), (1, 768), 0), out=buf179)
        buf180 = reinterpret_tensor(buf179, (8, 196, 256), (50176, 256, 1), 0); del buf179  # reuse
        buf184 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf185 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf486 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___12___norm, x_100, x_91, x_99], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf180, buf151, buf165, primals_112, primals_122, primals_123, primals_124, buf184, buf185, buf486, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_112
        del primals_122
        del primals_124
        buf186 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, buf185, reinterpret_tensor(primals_125, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf186)
        del primals_126
        buf190 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf485 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_37], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf186, buf190, buf485, 1568, 768, grid=grid(1568), stream=stream0)
        buf191 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_38], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf190, primals_127, primals_128, buf191, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_128
        buf192 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_38], Original ATen: [aten.mm]
        extern_kernels.mm(buf191, reinterpret_tensor(primals_129, (196, 196), (1, 196), 0), out=buf192)
        buf193 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103, x_105], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf186, buf192, primals_130, buf193, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf194 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf193, reinterpret_tensor(primals_131, (768, 256), (1, 768), 0), out=buf194)
        buf198 = buf151; del buf151  # reuse
        buf199 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf484 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___13___norm, x_107, x_108], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf180, buf194, primals_132, primals_133, primals_134, buf198, buf199, buf484, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_134
        buf200 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_136, buf199, reinterpret_tensor(primals_135, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf200)
        del primals_136
        buf204 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf483 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_40], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf200, buf204, buf483, 1568, 768, grid=grid(1568), stream=stream0)
        buf205 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_41], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf204, primals_137, primals_138, buf205, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_138
        buf206 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_41], Original ATen: [aten.mm]
        extern_kernels.mm(buf205, reinterpret_tensor(primals_139, (196, 196), (1, 196), 0), out=buf206)
        buf207 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111, x_113], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf200, buf206, primals_140, buf207, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf208 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf207, reinterpret_tensor(primals_141, (768, 256), (1, 768), 0), out=buf208)
        buf209 = reinterpret_tensor(buf208, (8, 196, 256), (50176, 256, 1), 0); del buf208  # reuse
        buf213 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf214 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf482 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___14___norm, x_107, x_115, x_116], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf209, buf180, buf194, primals_132, primals_142, primals_143, primals_144, buf213, buf214, buf482, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_132
        del primals_142
        del primals_144
        buf215 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_146, buf214, reinterpret_tensor(primals_145, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf215)
        del primals_146
        buf219 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf481 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_43], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf215, buf219, buf481, 1568, 768, grid=grid(1568), stream=stream0)
        buf220 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_44], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf219, primals_147, primals_148, buf220, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_148
        buf221 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_44], Original ATen: [aten.mm]
        extern_kernels.mm(buf220, reinterpret_tensor(primals_149, (196, 196), (1, 196), 0), out=buf221)
        buf222 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_121], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf215, buf221, primals_150, buf222, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf223 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf222, reinterpret_tensor(primals_151, (768, 256), (1, 768), 0), out=buf223)
        buf227 = buf180; del buf180  # reuse
        buf228 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf480 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___15___norm, x_123, x_124], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf209, buf223, primals_152, primals_153, primals_154, buf227, buf228, buf480, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_154
        buf229 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, buf228, reinterpret_tensor(primals_155, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf229)
        del primals_156
        buf233 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf479 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_46], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf229, buf233, buf479, 1568, 768, grid=grid(1568), stream=stream0)
        buf234 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_47], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf233, primals_157, primals_158, buf234, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_158
        buf235 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_47], Original ATen: [aten.mm]
        extern_kernels.mm(buf234, reinterpret_tensor(primals_159, (196, 196), (1, 196), 0), out=buf235)
        buf236 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127, x_129], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf229, buf235, primals_160, buf236, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf237 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_161, (768, 256), (1, 768), 0), out=buf237)
        buf238 = reinterpret_tensor(buf237, (8, 196, 256), (50176, 256, 1), 0); del buf237  # reuse
        buf242 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf243 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf478 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___16___norm, x_123, x_131, x_132], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf238, buf209, buf223, primals_152, primals_162, primals_163, primals_164, buf242, buf243, buf478, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_152
        del primals_162
        del primals_164
        buf244 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_166, buf243, reinterpret_tensor(primals_165, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf244)
        del primals_166
        buf248 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf477 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_49], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf244, buf248, buf477, 1568, 768, grid=grid(1568), stream=stream0)
        buf249 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_50], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf248, primals_167, primals_168, buf249, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_168
        buf250 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_50], Original ATen: [aten.mm]
        extern_kernels.mm(buf249, reinterpret_tensor(primals_169, (196, 196), (1, 196), 0), out=buf250)
        buf251 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_137], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf244, buf250, primals_170, buf251, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf252 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf251, reinterpret_tensor(primals_171, (768, 256), (1, 768), 0), out=buf252)
        buf256 = buf209; del buf209  # reuse
        buf257 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf476 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___17___norm, x_139, x_140], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf238, buf252, primals_172, primals_173, primals_174, buf256, buf257, buf476, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_174
        buf258 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf257, reinterpret_tensor(primals_175, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf258)
        del primals_176
        buf262 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf475 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_52], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf258, buf262, buf475, 1568, 768, grid=grid(1568), stream=stream0)
        buf263 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_53], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf262, primals_177, primals_178, buf263, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_178
        buf264 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_53], Original ATen: [aten.mm]
        extern_kernels.mm(buf263, reinterpret_tensor(primals_179, (196, 196), (1, 196), 0), out=buf264)
        buf265 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143, x_145], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf258, buf264, primals_180, buf265, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf266 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf265, reinterpret_tensor(primals_181, (768, 256), (1, 768), 0), out=buf266)
        buf267 = reinterpret_tensor(buf266, (8, 196, 256), (50176, 256, 1), 0); del buf266  # reuse
        buf271 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf272 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf474 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___18___norm, x_139, x_147, x_148], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf267, buf238, buf252, primals_172, primals_182, primals_183, primals_184, buf271, buf272, buf474, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_172
        del primals_182
        del primals_184
        buf273 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_186, buf272, reinterpret_tensor(primals_185, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf273)
        del primals_186
        buf277 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf473 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_55], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf273, buf277, buf473, 1568, 768, grid=grid(1568), stream=stream0)
        buf278 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_56], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf277, primals_187, primals_188, buf278, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_188
        buf279 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_56], Original ATen: [aten.mm]
        extern_kernels.mm(buf278, reinterpret_tensor(primals_189, (196, 196), (1, 196), 0), out=buf279)
        buf280 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151, x_153], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf273, buf279, primals_190, buf280, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf281 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf280, reinterpret_tensor(primals_191, (768, 256), (1, 768), 0), out=buf281)
        buf285 = buf238; del buf238  # reuse
        buf286 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf472 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___19___norm, x_155, x_156], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf267, buf281, primals_192, primals_193, primals_194, buf285, buf286, buf472, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_194
        buf287 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_196, buf286, reinterpret_tensor(primals_195, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf287)
        del primals_196
        buf291 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf471 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_58], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf287, buf291, buf471, 1568, 768, grid=grid(1568), stream=stream0)
        buf292 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_59], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf291, primals_197, primals_198, buf292, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_198
        buf293 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_59], Original ATen: [aten.mm]
        extern_kernels.mm(buf292, reinterpret_tensor(primals_199, (196, 196), (1, 196), 0), out=buf293)
        buf294 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159, x_161], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf287, buf293, primals_200, buf294, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf295 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf294, reinterpret_tensor(primals_201, (768, 256), (1, 768), 0), out=buf295)
        buf296 = reinterpret_tensor(buf295, (8, 196, 256), (50176, 256, 1), 0); del buf295  # reuse
        buf300 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf301 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf470 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___20___norm, x_155, x_163, x_164], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf296, buf267, buf281, primals_192, primals_202, primals_203, primals_204, buf300, buf301, buf470, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_192
        del primals_202
        del primals_204
        buf302 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_206, buf301, reinterpret_tensor(primals_205, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf302)
        del primals_206
        buf306 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf469 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_61], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf302, buf306, buf469, 1568, 768, grid=grid(1568), stream=stream0)
        buf307 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_62], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf306, primals_207, primals_208, buf307, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_208
        buf308 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_62], Original ATen: [aten.mm]
        extern_kernels.mm(buf307, reinterpret_tensor(primals_209, (196, 196), (1, 196), 0), out=buf308)
        buf309 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167, x_169], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf302, buf308, primals_210, buf309, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf310 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf309, reinterpret_tensor(primals_211, (768, 256), (1, 768), 0), out=buf310)
        buf314 = buf267; del buf267  # reuse
        buf315 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf468 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___21___norm, x_171, x_172], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf296, buf310, primals_212, primals_213, primals_214, buf314, buf315, buf468, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_214
        buf316 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_216, buf315, reinterpret_tensor(primals_215, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf316)
        del primals_216
        buf320 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf467 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_64], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf316, buf320, buf467, 1568, 768, grid=grid(1568), stream=stream0)
        buf321 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_65], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf320, primals_217, primals_218, buf321, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_218
        buf322 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_65], Original ATen: [aten.mm]
        extern_kernels.mm(buf321, reinterpret_tensor(primals_219, (196, 196), (1, 196), 0), out=buf322)
        buf323 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175, x_177], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf316, buf322, primals_220, buf323, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf324 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf323, reinterpret_tensor(primals_221, (768, 256), (1, 768), 0), out=buf324)
        buf325 = reinterpret_tensor(buf324, (8, 196, 256), (50176, 256, 1), 0); del buf324  # reuse
        buf329 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf330 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf466 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___22___norm, x_171, x_179, x_180], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf325, buf296, buf310, primals_212, primals_222, primals_223, primals_224, buf329, buf330, buf466, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_212
        del primals_222
        del primals_224
        buf331 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_180], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_226, buf330, reinterpret_tensor(primals_225, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf331)
        del primals_226
        buf335 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf465 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_67], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf331, buf335, buf465, 1568, 768, grid=grid(1568), stream=stream0)
        buf336 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_68], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf335, primals_227, primals_228, buf336, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_228
        buf337 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_68], Original ATen: [aten.mm]
        extern_kernels.mm(buf336, reinterpret_tensor(primals_229, (196, 196), (1, 196), 0), out=buf337)
        buf338 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183, x_185], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf331, buf337, primals_230, buf338, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf339 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf338, reinterpret_tensor(primals_231, (768, 256), (1, 768), 0), out=buf339)
        buf343 = buf296; del buf296  # reuse
        buf344 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf464 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___23___norm, x_187, x_188], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf325, buf339, primals_232, primals_233, primals_234, buf343, buf344, buf464, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_234
        buf345 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_236, buf344, reinterpret_tensor(primals_235, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf345)
        del primals_236
        buf349 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf463 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_70], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf345, buf349, buf463, 1568, 768, grid=grid(1568), stream=stream0)
        buf350 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_71], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf349, primals_237, primals_238, buf350, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_238
        buf351 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_71], Original ATen: [aten.mm]
        extern_kernels.mm(buf350, reinterpret_tensor(primals_239, (196, 196), (1, 196), 0), out=buf351)
        buf352 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191, x_193], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf345, buf351, primals_240, buf352, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf353 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf352, reinterpret_tensor(primals_241, (768, 256), (1, 768), 0), out=buf353)
        buf354 = reinterpret_tensor(buf353, (8, 196, 256), (50176, 256, 1), 0); del buf353  # reuse
        buf358 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf359 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf462 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___24___norm, x_187, x_195, x_196], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf354, buf325, buf339, primals_232, primals_242, primals_243, primals_244, buf358, buf359, buf462, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_232
        del primals_242
        del primals_244
        buf360 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_246, buf359, reinterpret_tensor(primals_245, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf360)
        del primals_246
        buf364 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf461 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_73], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf360, buf364, buf461, 1568, 768, grid=grid(1568), stream=stream0)
        buf365 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_74], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf364, primals_247, primals_248, buf365, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_248
        buf366 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_74], Original ATen: [aten.mm]
        extern_kernels.mm(buf365, reinterpret_tensor(primals_249, (196, 196), (1, 196), 0), out=buf366)
        buf367 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199, x_201], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf360, buf366, primals_250, buf367, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf368 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf367, reinterpret_tensor(primals_251, (768, 256), (1, 768), 0), out=buf368)
        buf372 = buf325; del buf325  # reuse
        buf373 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf460 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___25___norm, x_203, x_204], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf354, buf368, primals_252, primals_253, primals_254, buf372, buf373, buf460, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_254
        buf374 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_256, buf373, reinterpret_tensor(primals_255, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf374)
        del primals_256
        buf378 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf459 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_76], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf374, buf378, buf459, 1568, 768, grid=grid(1568), stream=stream0)
        buf379 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_77], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf378, primals_257, primals_258, buf379, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_258
        buf380 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_77], Original ATen: [aten.mm]
        extern_kernels.mm(buf379, reinterpret_tensor(primals_259, (196, 196), (1, 196), 0), out=buf380)
        buf381 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207, x_209], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf374, buf380, primals_260, buf381, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf382 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf381, reinterpret_tensor(primals_261, (768, 256), (1, 768), 0), out=buf382)
        buf383 = reinterpret_tensor(buf382, (8, 196, 256), (50176, 256, 1), 0); del buf382  # reuse
        buf387 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf388 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf458 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___26___norm, x_203, x_211, x_212], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf383, buf354, buf368, primals_252, primals_262, primals_263, primals_264, buf387, buf388, buf458, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_252
        del primals_262
        del primals_264
        buf389 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_266, buf388, reinterpret_tensor(primals_265, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf389)
        del primals_266
        buf393 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf457 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_79], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf389, buf393, buf457, 1568, 768, grid=grid(1568), stream=stream0)
        buf394 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_80], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf393, primals_267, primals_268, buf394, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_268
        buf395 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_80], Original ATen: [aten.mm]
        extern_kernels.mm(buf394, reinterpret_tensor(primals_269, (196, 196), (1, 196), 0), out=buf395)
        buf396 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215, x_217], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf389, buf395, primals_270, buf396, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf397 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf396, reinterpret_tensor(primals_271, (768, 256), (1, 768), 0), out=buf397)
        buf401 = buf354; del buf354  # reuse
        buf402 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf456 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___27___norm, x_219, x_220], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf383, buf397, primals_272, primals_273, primals_274, buf401, buf402, buf456, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_274
        buf403 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_276, buf402, reinterpret_tensor(primals_275, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf403)
        del primals_276
        buf407 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf455 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_82], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf403, buf407, buf455, 1568, 768, grid=grid(1568), stream=stream0)
        buf408 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_83], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf407, primals_277, primals_278, buf408, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_278
        buf409 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_83], Original ATen: [aten.mm]
        extern_kernels.mm(buf408, reinterpret_tensor(primals_279, (196, 196), (1, 196), 0), out=buf409)
        buf410 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223, x_225], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf403, buf409, primals_280, buf410, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf411 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf410, reinterpret_tensor(primals_281, (768, 256), (1, 768), 0), out=buf411)
        buf412 = reinterpret_tensor(buf411, (8, 196, 256), (50176, 256, 1), 0); del buf411  # reuse
        buf416 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf417 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf454 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___28___norm, x_219, x_227, x_228], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_11.run(buf412, buf383, buf397, primals_272, primals_282, primals_283, primals_284, buf416, buf417, buf454, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_272
        del primals_282
        del primals_284
        buf418 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_286, buf417, reinterpret_tensor(primals_285, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf418)
        del primals_286
        buf422 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf453 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_85], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf418, buf422, buf453, 1568, 768, grid=grid(1568), stream=stream0)
        buf423 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_86], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf422, primals_287, primals_288, buf423, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_288
        buf424 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_86], Original ATen: [aten.mm]
        extern_kernels.mm(buf423, reinterpret_tensor(primals_289, (196, 196), (1, 196), 0), out=buf424)
        buf425 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231, x_233], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf418, buf424, primals_290, buf425, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf426 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf425, reinterpret_tensor(primals_291, (768, 256), (1, 768), 0), out=buf426)
        buf430 = buf383; del buf383  # reuse
        buf431 = empty((1568, 256), device='cuda', dtype=torch.float32)
        buf452 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___29___norm, x_235, x_236], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_10.run(buf412, buf426, primals_292, primals_293, primals_294, buf430, buf431, buf452, 1568, 256, grid=grid(1568), stream=stream0)
        del primals_294
        buf432 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_296, buf431, reinterpret_tensor(primals_295, (256, 1536), (1, 256), 0), alpha=1, beta=1, out=buf432)
        del primals_296
        buf436 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        buf451 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_88], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_3.run(buf432, buf436, buf451, 1568, 768, grid=grid(1568), stream=stream0)
        buf437 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_89], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_4.run(buf436, primals_297, primals_298, buf437, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del primals_298
        buf438 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [v_89], Original ATen: [aten.mm]
        extern_kernels.mm(buf437, reinterpret_tensor(primals_299, (196, 196), (1, 196), 0), out=buf438)
        buf439 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239, x_241], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_5.run(buf432, buf438, primals_300, buf439, 1568, 768, grid=grid(1568, 768), stream=stream0)
        buf440 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf439, reinterpret_tensor(primals_301, (768, 256), (1, 768), 0), out=buf440)
        buf441 = reinterpret_tensor(buf440, (8, 196, 256), (50176, 256, 1), 0); del buf440  # reuse
        buf445 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf450 = empty((8, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235, x_244, x_246], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf441, buf412, buf426, primals_292, primals_302, buf445, buf450, 1568, 256, grid=grid(1568), stream=stream0)
        del buf412
        del buf426
        del buf441
        del primals_292
        del primals_302
        buf446 = empty_strided((8, 256, 2), (512, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_red_fused_mean_native_layer_norm_13.run(buf445, primals_303, primals_304, buf446, 4096, 98, grid=grid(4096), stream=stream0)
        del primals_304
        buf447 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf448 = buf447; del buf447  # reuse
        # Source Nodes: [x_246, x_247], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_14.run(buf448, buf446, 2048, 2, grid=grid(2048), stream=stream0)
        del buf446
        buf449 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_306, buf448, reinterpret_tensor(primals_305, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf449)
        del primals_306
        buf511 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf9, buf511, 2408448, grid=grid(2408448), stream=stream0)
        buf512 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf26, buf512, 2408448, grid=grid(2408448), stream=stream0)
        buf513 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf41, buf513, 2408448, grid=grid(2408448), stream=stream0)
        buf514 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf55, buf514, 2408448, grid=grid(2408448), stream=stream0)
        buf515 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf70, buf515, 2408448, grid=grid(2408448), stream=stream0)
        buf516 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf84, buf516, 2408448, grid=grid(2408448), stream=stream0)
        buf517 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf99, buf517, 2408448, grid=grid(2408448), stream=stream0)
        buf518 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf113, buf518, 2408448, grid=grid(2408448), stream=stream0)
        buf519 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf128, buf519, 2408448, grid=grid(2408448), stream=stream0)
        buf520 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf142, buf520, 2408448, grid=grid(2408448), stream=stream0)
        buf521 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf157, buf521, 2408448, grid=grid(2408448), stream=stream0)
        buf522 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf171, buf522, 2408448, grid=grid(2408448), stream=stream0)
        buf523 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf186, buf523, 2408448, grid=grid(2408448), stream=stream0)
        buf524 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf200, buf524, 2408448, grid=grid(2408448), stream=stream0)
        buf525 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf215, buf525, 2408448, grid=grid(2408448), stream=stream0)
        buf526 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf229, buf526, 2408448, grid=grid(2408448), stream=stream0)
        buf527 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf244, buf527, 2408448, grid=grid(2408448), stream=stream0)
        buf528 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_141], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf258, buf528, 2408448, grid=grid(2408448), stream=stream0)
        buf529 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf273, buf529, 2408448, grid=grid(2408448), stream=stream0)
        buf530 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf287, buf530, 2408448, grid=grid(2408448), stream=stream0)
        buf531 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf302, buf531, 2408448, grid=grid(2408448), stream=stream0)
        buf532 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf316, buf532, 2408448, grid=grid(2408448), stream=stream0)
        buf533 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_181], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf331, buf533, 2408448, grid=grid(2408448), stream=stream0)
        buf534 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf345, buf534, 2408448, grid=grid(2408448), stream=stream0)
        buf535 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf360, buf535, 2408448, grid=grid(2408448), stream=stream0)
        buf536 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf374, buf536, 2408448, grid=grid(2408448), stream=stream0)
        buf537 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_213], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf389, buf537, 2408448, grid=grid(2408448), stream=stream0)
        buf538 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf403, buf538, 2408448, grid=grid(2408448), stream=stream0)
        buf539 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_229], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf418, buf539, 2408448, grid=grid(2408448), stream=stream0)
        buf540 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf432, buf540, 2408448, grid=grid(2408448), stream=stream0)
        return (buf449, primals_1, primals_3, primals_7, primals_10, primals_13, primals_17, primals_20, primals_23, primals_27, primals_30, primals_33, primals_37, primals_40, primals_43, primals_47, primals_50, primals_53, primals_57, primals_60, primals_63, primals_67, primals_70, primals_73, primals_77, primals_80, primals_83, primals_87, primals_90, primals_93, primals_97, primals_100, primals_103, primals_107, primals_110, primals_113, primals_117, primals_120, primals_123, primals_127, primals_130, primals_133, primals_137, primals_140, primals_143, primals_147, primals_150, primals_153, primals_157, primals_160, primals_163, primals_167, primals_170, primals_173, primals_177, primals_180, primals_183, primals_187, primals_190, primals_193, primals_197, primals_200, primals_203, primals_207, primals_210, primals_213, primals_217, primals_220, primals_223, primals_227, primals_230, primals_233, primals_237, primals_240, primals_243, primals_247, primals_250, primals_253, primals_257, primals_260, primals_263, primals_267, primals_270, primals_273, primals_277, primals_280, primals_283, primals_287, primals_290, primals_293, primals_297, primals_300, primals_303, primals_307, buf7, buf8, buf9, reinterpret_tensor(buf511, (8, 196, 768), (301056, 1536, 1), 0), buf13, buf14, buf15, buf16, buf24, buf25, buf26, reinterpret_tensor(buf512, (8, 196, 768), (301056, 1536, 1), 0), buf30, buf31, buf32, buf33, buf39, buf40, buf41, reinterpret_tensor(buf513, (8, 196, 768), (301056, 1536, 1), 0), buf45, buf46, buf47, buf48, buf53, buf54, buf55, reinterpret_tensor(buf514, (8, 196, 768), (301056, 1536, 1), 0), buf59, buf60, buf61, buf62, buf68, buf69, buf70, reinterpret_tensor(buf515, (8, 196, 768), (301056, 1536, 1), 0), buf74, buf75, buf76, buf77, buf82, buf83, buf84, reinterpret_tensor(buf516, (8, 196, 768), (301056, 1536, 1), 0), buf88, buf89, buf90, buf91, buf97, buf98, buf99, reinterpret_tensor(buf517, (8, 196, 768), (301056, 1536, 1), 0), buf103, buf104, buf105, buf106, buf111, buf112, buf113, reinterpret_tensor(buf518, (8, 196, 768), (301056, 1536, 1), 0), buf117, buf118, buf119, buf120, buf126, buf127, buf128, reinterpret_tensor(buf519, (8, 196, 768), (301056, 1536, 1), 0), buf132, buf133, buf134, buf135, buf140, buf141, buf142, reinterpret_tensor(buf520, (8, 196, 768), (301056, 1536, 1), 0), buf146, buf147, buf148, buf149, buf155, buf156, buf157, reinterpret_tensor(buf521, (8, 196, 768), (301056, 1536, 1), 0), buf161, buf162, buf163, buf164, buf169, buf170, buf171, reinterpret_tensor(buf522, (8, 196, 768), (301056, 1536, 1), 0), buf175, buf176, buf177, buf178, buf184, buf185, buf186, reinterpret_tensor(buf523, (8, 196, 768), (301056, 1536, 1), 0), buf190, buf191, buf192, buf193, buf198, buf199, buf200, reinterpret_tensor(buf524, (8, 196, 768), (301056, 1536, 1), 0), buf204, buf205, buf206, buf207, buf213, buf214, buf215, reinterpret_tensor(buf525, (8, 196, 768), (301056, 1536, 1), 0), buf219, buf220, buf221, buf222, buf227, buf228, buf229, reinterpret_tensor(buf526, (8, 196, 768), (301056, 1536, 1), 0), buf233, buf234, buf235, buf236, buf242, buf243, buf244, reinterpret_tensor(buf527, (8, 196, 768), (301056, 1536, 1), 0), buf248, buf249, buf250, buf251, buf256, buf257, buf258, reinterpret_tensor(buf528, (8, 196, 768), (301056, 1536, 1), 0), buf262, buf263, buf264, buf265, buf271, buf272, buf273, reinterpret_tensor(buf529, (8, 196, 768), (301056, 1536, 1), 0), buf277, buf278, buf279, buf280, buf285, buf286, buf287, reinterpret_tensor(buf530, (8, 196, 768), (301056, 1536, 1), 0), buf291, buf292, buf293, buf294, buf300, buf301, buf302, reinterpret_tensor(buf531, (8, 196, 768), (301056, 1536, 1), 0), buf306, buf307, buf308, buf309, buf314, buf315, buf316, reinterpret_tensor(buf532, (8, 196, 768), (301056, 1536, 1), 0), buf320, buf321, buf322, buf323, buf329, buf330, buf331, reinterpret_tensor(buf533, (8, 196, 768), (301056, 1536, 1), 0), buf335, buf336, buf337, buf338, buf343, buf344, buf345, reinterpret_tensor(buf534, (8, 196, 768), (301056, 1536, 1), 0), buf349, buf350, buf351, buf352, buf358, buf359, buf360, reinterpret_tensor(buf535, (8, 196, 768), (301056, 1536, 1), 0), buf364, buf365, buf366, buf367, buf372, buf373, buf374, reinterpret_tensor(buf536, (8, 196, 768), (301056, 1536, 1), 0), buf378, buf379, buf380, buf381, buf387, buf388, buf389, reinterpret_tensor(buf537, (8, 196, 768), (301056, 1536, 1), 0), buf393, buf394, buf395, buf396, buf401, buf402, buf403, reinterpret_tensor(buf538, (8, 196, 768), (301056, 1536, 1), 0), buf407, buf408, buf409, buf410, buf416, buf417, buf418, reinterpret_tensor(buf539, (8, 196, 768), (301056, 1536, 1), 0), buf422, buf423, buf424, buf425, buf430, buf431, buf432, reinterpret_tensor(buf540, (8, 196, 768), (301056, 1536, 1), 0), buf436, buf437, buf438, buf439, buf445, buf448, reinterpret_tensor(primals_305, (1000, 256), (256, 1), 0), buf450, reinterpret_tensor(primals_301, (256, 768), (768, 1), 0), reinterpret_tensor(primals_299, (196, 196), (196, 1), 0), buf451, reinterpret_tensor(primals_295, (1536, 256), (256, 1), 0), buf452, reinterpret_tensor(primals_291, (256, 768), (768, 1), 0), reinterpret_tensor(primals_289, (196, 196), (196, 1), 0), buf453, reinterpret_tensor(primals_285, (1536, 256), (256, 1), 0), buf454, reinterpret_tensor(primals_281, (256, 768), (768, 1), 0), reinterpret_tensor(primals_279, (196, 196), (196, 1), 0), buf455, reinterpret_tensor(primals_275, (1536, 256), (256, 1), 0), buf456, reinterpret_tensor(primals_271, (256, 768), (768, 1), 0), reinterpret_tensor(primals_269, (196, 196), (196, 1), 0), buf457, reinterpret_tensor(primals_265, (1536, 256), (256, 1), 0), buf458, reinterpret_tensor(primals_261, (256, 768), (768, 1), 0), reinterpret_tensor(primals_259, (196, 196), (196, 1), 0), buf459, reinterpret_tensor(primals_255, (1536, 256), (256, 1), 0), buf460, reinterpret_tensor(primals_251, (256, 768), (768, 1), 0), reinterpret_tensor(primals_249, (196, 196), (196, 1), 0), buf461, reinterpret_tensor(primals_245, (1536, 256), (256, 1), 0), buf462, reinterpret_tensor(primals_241, (256, 768), (768, 1), 0), reinterpret_tensor(primals_239, (196, 196), (196, 1), 0), buf463, reinterpret_tensor(primals_235, (1536, 256), (256, 1), 0), buf464, reinterpret_tensor(primals_231, (256, 768), (768, 1), 0), reinterpret_tensor(primals_229, (196, 196), (196, 1), 0), buf465, reinterpret_tensor(primals_225, (1536, 256), (256, 1), 0), buf466, reinterpret_tensor(primals_221, (256, 768), (768, 1), 0), reinterpret_tensor(primals_219, (196, 196), (196, 1), 0), buf467, reinterpret_tensor(primals_215, (1536, 256), (256, 1), 0), buf468, reinterpret_tensor(primals_211, (256, 768), (768, 1), 0), reinterpret_tensor(primals_209, (196, 196), (196, 1), 0), buf469, reinterpret_tensor(primals_205, (1536, 256), (256, 1), 0), buf470, reinterpret_tensor(primals_201, (256, 768), (768, 1), 0), reinterpret_tensor(primals_199, (196, 196), (196, 1), 0), buf471, reinterpret_tensor(primals_195, (1536, 256), (256, 1), 0), buf472, reinterpret_tensor(primals_191, (256, 768), (768, 1), 0), reinterpret_tensor(primals_189, (196, 196), (196, 1), 0), buf473, reinterpret_tensor(primals_185, (1536, 256), (256, 1), 0), buf474, reinterpret_tensor(primals_181, (256, 768), (768, 1), 0), reinterpret_tensor(primals_179, (196, 196), (196, 1), 0), buf475, reinterpret_tensor(primals_175, (1536, 256), (256, 1), 0), buf476, reinterpret_tensor(primals_171, (256, 768), (768, 1), 0), reinterpret_tensor(primals_169, (196, 196), (196, 1), 0), buf477, reinterpret_tensor(primals_165, (1536, 256), (256, 1), 0), buf478, reinterpret_tensor(primals_161, (256, 768), (768, 1), 0), reinterpret_tensor(primals_159, (196, 196), (196, 1), 0), buf479, reinterpret_tensor(primals_155, (1536, 256), (256, 1), 0), buf480, reinterpret_tensor(primals_151, (256, 768), (768, 1), 0), reinterpret_tensor(primals_149, (196, 196), (196, 1), 0), buf481, reinterpret_tensor(primals_145, (1536, 256), (256, 1), 0), buf482, reinterpret_tensor(primals_141, (256, 768), (768, 1), 0), reinterpret_tensor(primals_139, (196, 196), (196, 1), 0), buf483, reinterpret_tensor(primals_135, (1536, 256), (256, 1), 0), buf484, reinterpret_tensor(primals_131, (256, 768), (768, 1), 0), reinterpret_tensor(primals_129, (196, 196), (196, 1), 0), buf485, reinterpret_tensor(primals_125, (1536, 256), (256, 1), 0), buf486, reinterpret_tensor(primals_121, (256, 768), (768, 1), 0), reinterpret_tensor(primals_119, (196, 196), (196, 1), 0), buf487, reinterpret_tensor(primals_115, (1536, 256), (256, 1), 0), buf488, reinterpret_tensor(primals_111, (256, 768), (768, 1), 0), reinterpret_tensor(primals_109, (196, 196), (196, 1), 0), buf489, reinterpret_tensor(primals_105, (1536, 256), (256, 1), 0), buf490, reinterpret_tensor(primals_101, (256, 768), (768, 1), 0), reinterpret_tensor(primals_99, (196, 196), (196, 1), 0), buf491, reinterpret_tensor(primals_95, (1536, 256), (256, 1), 0), buf492, reinterpret_tensor(primals_91, (256, 768), (768, 1), 0), reinterpret_tensor(primals_89, (196, 196), (196, 1), 0), buf493, reinterpret_tensor(primals_85, (1536, 256), (256, 1), 0), buf494, reinterpret_tensor(primals_81, (256, 768), (768, 1), 0), reinterpret_tensor(primals_79, (196, 196), (196, 1), 0), buf495, reinterpret_tensor(primals_75, (1536, 256), (256, 1), 0), buf496, reinterpret_tensor(primals_71, (256, 768), (768, 1), 0), reinterpret_tensor(primals_69, (196, 196), (196, 1), 0), buf497, reinterpret_tensor(primals_65, (1536, 256), (256, 1), 0), buf498, reinterpret_tensor(primals_61, (256, 768), (768, 1), 0), reinterpret_tensor(primals_59, (196, 196), (196, 1), 0), buf499, reinterpret_tensor(primals_55, (1536, 256), (256, 1), 0), buf500, reinterpret_tensor(primals_51, (256, 768), (768, 1), 0), reinterpret_tensor(primals_49, (196, 196), (196, 1), 0), buf501, reinterpret_tensor(primals_45, (1536, 256), (256, 1), 0), buf502, reinterpret_tensor(primals_41, (256, 768), (768, 1), 0), reinterpret_tensor(primals_39, (196, 196), (196, 1), 0), buf503, reinterpret_tensor(primals_35, (1536, 256), (256, 1), 0), buf504, reinterpret_tensor(primals_31, (256, 768), (768, 1), 0), reinterpret_tensor(primals_29, (196, 196), (196, 1), 0), buf505, reinterpret_tensor(primals_25, (1536, 256), (256, 1), 0), buf506, reinterpret_tensor(primals_21, (256, 768), (768, 1), 0), reinterpret_tensor(primals_19, (196, 196), (196, 1), 0), buf507, reinterpret_tensor(primals_15, (1536, 256), (256, 1), 0), buf508, reinterpret_tensor(primals_11, (256, 768), (768, 1), 0), reinterpret_tensor(primals_9, (196, 196), (196, 1), 0), buf509, reinterpret_tensor(primals_5, (1536, 256), (256, 1), 0), buf510, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmlp_s16_224', benchmark_compiled_module)
