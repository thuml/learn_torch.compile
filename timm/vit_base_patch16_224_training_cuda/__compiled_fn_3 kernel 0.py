
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2lxzhjn4mibxydfooot26djecrxy7h5jr7gqdg5eu4udye74wz.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm1 => var_mean
# x_5 => add
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9456
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 6) % 197
    x0 = xindex % 6
    x2 = (xindex // 1182)
    x5 = xindex % 1182
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp17 = tl.load(in_ptr3 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 197, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((196*r3) + (25088*x0) + (150528*x2) + (((-1) + x1) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp8, tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp7, tmp15)
        tmp18 = tmp16 + tmp17
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
    tl.store(out_ptr0 + (x6), tmp20, xmask)
    tl.store(out_ptr1 + (x6), tmp21, xmask)
    tl.store(out_ptr2 + (x6), tmp22, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lo6kzmx2elkasd2gekt4gmiw2qiywvo5knkta5wyldfsnheyw3.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm1 => add_1, rsqrt, var_mean
# x_5 => add
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (6*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctua3hxa66zpw6lsvpzc5a37g7pp5xnehafoizjpmmer7jlolucw.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___attn_qkv, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.view]
# cat_1 => cat
# getattr_l__mod___blocks___0___attn_qkv => view_1
# getattr_l__mod___blocks___0___norm1 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
# x_5 => add
triton_poi_fused_add_cat_native_layer_norm_view_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768) % 197
    x0 = xindex % 768
    x2 = (xindex // 151296)
    x3 = xindex % 151296
    x4 = (xindex // 768)
    x5 = xindex
    tmp17 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*x0) + (150528*x2) + (((-1) + x1) % 196)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x5), tmp27, None)
    tl.store(out_ptr1 + (x5), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5soj62fr5lmyft7dbydmp4ciuhkibjulq2oqhlhdetcsal4ejiy.py
# Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm2, x_13, x_14, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cat_1 => cat
# getattr_l__mod___blocks___0___norm2 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# x_13 => add_3
# x_14 => view_7
# x_5 => add
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*r2) + (150528*x1) + (((-1) + x0) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp18 = tmp16 + tmp17
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp44 / tmp40
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp50, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmdowuu34wy72sslbwkqoi6pe7waaj7bbdtapxyzsvo5gcg2t6ub.py
# Source Nodes: [x_15, x_18], Original ATen: [aten.gelu, aten.view]
# x_15 => add_6, erf, mul_4, mul_5, mul_6
# x_18 => view_9
triton_poi_fused_gelu_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4841472
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


# kernel path: /tmp/torchinductor_youkaichao/th/cthoe3rvyl42s2ciqqmpk3ap6qikyunybbfamsqfwuhzujp5z4io.py
# Source Nodes: [getattr_l__mod___blocks___1___attn_qkv, getattr_l__mod___blocks___1___norm1, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___1___attn_qkv => view_11
# getattr_l__mod___blocks___1___norm1 => add_8, add_9, mul_7, mul_8, rsqrt_2, sub_2, var_mean_2
# x_20 => add_7
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tv/ctvui6kqtpnxlc3tzfjcd5f4272akz55j356v2vpau6efleysurt.py
# Source Nodes: [getattr_l__mod___blocks___1___norm2, x_20, x_25, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks___1___norm2 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
# x_20 => add_7
# x_25 => add_10
# x_26 => view_17
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nr/cnrktzhtzgheainudb3h4mnz56v7lmteegy7yz232gbbvjopy4bg.py
# Source Nodes: [x_153, x_155], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_153 => add_84
# x_155 => add_85, mul_84, rsqrt_24, sub_24, var_mean_24
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp28 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3l/c3lwaq63efsdql4uu77nqve23wbkb2mnihhdohahzpz4xjkl5qj2.py
# Source Nodes: [x_158], Original ATen: [aten.clone]
# x_158 => clone_37
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (151296*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153 = args
    args.clear()
    assert_size_stride(primals_1, (1, 197, 768), (151296, 768, 1))
    assert_size_stride(primals_2, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_3, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (2304, 768), (768, 1))
    assert_size_stride(primals_8, (2304, ), (1, ))
    assert_size_stride(primals_9, (768, 768), (768, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (3072, 768), (768, 1))
    assert_size_stride(primals_14, (3072, ), (1, ))
    assert_size_stride(primals_15, (768, 3072), (3072, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (2304, 768), (768, 1))
    assert_size_stride(primals_20, (2304, ), (1, ))
    assert_size_stride(primals_21, (768, 768), (768, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (3072, 768), (768, 1))
    assert_size_stride(primals_26, (3072, ), (1, ))
    assert_size_stride(primals_27, (768, 3072), (3072, 1))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (2304, 768), (768, 1))
    assert_size_stride(primals_32, (2304, ), (1, ))
    assert_size_stride(primals_33, (768, 768), (768, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (3072, 768), (768, 1))
    assert_size_stride(primals_38, (3072, ), (1, ))
    assert_size_stride(primals_39, (768, 3072), (3072, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (2304, 768), (768, 1))
    assert_size_stride(primals_44, (2304, ), (1, ))
    assert_size_stride(primals_45, (768, 768), (768, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (3072, 768), (768, 1))
    assert_size_stride(primals_50, (3072, ), (1, ))
    assert_size_stride(primals_51, (768, 3072), (3072, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (2304, 768), (768, 1))
    assert_size_stride(primals_56, (2304, ), (1, ))
    assert_size_stride(primals_57, (768, 768), (768, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (3072, 768), (768, 1))
    assert_size_stride(primals_62, (3072, ), (1, ))
    assert_size_stride(primals_63, (768, 3072), (3072, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (2304, 768), (768, 1))
    assert_size_stride(primals_68, (2304, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (3072, 768), (768, 1))
    assert_size_stride(primals_74, (3072, ), (1, ))
    assert_size_stride(primals_75, (768, 3072), (3072, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (2304, 768), (768, 1))
    assert_size_stride(primals_80, (2304, ), (1, ))
    assert_size_stride(primals_81, (768, 768), (768, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (3072, 768), (768, 1))
    assert_size_stride(primals_86, (3072, ), (1, ))
    assert_size_stride(primals_87, (768, 3072), (3072, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (2304, 768), (768, 1))
    assert_size_stride(primals_92, (2304, ), (1, ))
    assert_size_stride(primals_93, (768, 768), (768, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (3072, 768), (768, 1))
    assert_size_stride(primals_98, (3072, ), (1, ))
    assert_size_stride(primals_99, (768, 3072), (3072, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (2304, 768), (768, 1))
    assert_size_stride(primals_104, (2304, ), (1, ))
    assert_size_stride(primals_105, (768, 768), (768, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (3072, 768), (768, 1))
    assert_size_stride(primals_110, (3072, ), (1, ))
    assert_size_stride(primals_111, (768, 3072), (3072, 1))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (2304, 768), (768, 1))
    assert_size_stride(primals_116, (2304, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (3072, 768), (768, 1))
    assert_size_stride(primals_122, (3072, ), (1, ))
    assert_size_stride(primals_123, (768, 3072), (3072, 1))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (2304, 768), (768, 1))
    assert_size_stride(primals_128, (2304, ), (1, ))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (3072, 768), (768, 1))
    assert_size_stride(primals_134, (3072, ), (1, ))
    assert_size_stride(primals_135, (768, 3072), (3072, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (2304, 768), (768, 1))
    assert_size_stride(primals_140, (2304, ), (1, ))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (3072, 768), (768, 1))
    assert_size_stride(primals_146, (3072, ), (1, ))
    assert_size_stride(primals_147, (768, 3072), (3072, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (1000, 768), (768, 1))
    assert_size_stride(primals_152, (1000, ), (1, ))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_153, primals_3, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 768, 14, 14), (150528, 196, 14, 1))
        buf1 = empty_strided((8, 197, 1, 6), (1182, 6, 9456, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 197, 1, 6), (1182, 6, 9456, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((8, 197, 1, 6), (1182, 6, 9456, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(primals_2, buf0, primals_4, primals_1, buf1, buf2, buf3, 9456, 128, grid=grid(9456), stream=stream0)
        buf4 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf286 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_1.run(buf1, buf2, buf3, buf4, buf5, buf286, 1576, 6, grid=grid(1576), stream=stream0)
        del buf1
        del buf2
        del buf3
        buf7 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf8 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___attn_qkv, getattr_l__mod___blocks___0___norm1, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_cat_native_layer_norm_view_2.run(primals_2, buf0, primals_4, primals_1, buf4, buf5, primals_5, primals_6, buf7, buf8, 1210368, grid=grid(1210368), stream=stream0)
        del primals_6
        buf9 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf8, reinterpret_tensor(primals_7, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del primals_8
        # Source Nodes: [x_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf10 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf11 = buf10[0]
        buf12 = buf10[1]
        buf13 = buf10[2]
        buf14 = buf10[3]
        del buf10
        buf15 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf11, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), out=buf15)
        buf16 = reinterpret_tensor(buf15, (8, 197, 768), (151296, 768, 1), 0); del buf15  # reuse
        buf20 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf21 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf285 = reinterpret_tensor(buf5, (8, 197, 1), (197, 1, 1), 0); del buf5  # reuse
        # Source Nodes: [cat_1, getattr_l__mod___blocks___0___norm2, x_13, x_14, x_5], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_3.run(buf16, primals_2, buf0, primals_4, primals_1, primals_10, primals_11, primals_12, buf20, buf21, buf285, 1576, 768, grid=grid(1576), stream=stream0)
        del buf0
        del primals_1
        del primals_10
        del primals_12
        del primals_2
        del primals_4
        buf22 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf21, reinterpret_tensor(primals_13, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf22)
        del primals_14
        buf23 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15, x_18], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf22, buf23, 4841472, grid=grid(4841472), stream=stream0)
        buf24 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf23, reinterpret_tensor(primals_15, (3072, 768), (1, 3072), 0), out=buf24)
        buf28 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf29 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf284 = reinterpret_tensor(buf4, (8, 197, 1), (197, 1, 1), 0); del buf4  # reuse
        # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv, getattr_l__mod___blocks___1___norm1, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf16, buf24, primals_16, primals_17, primals_18, buf28, buf29, buf284, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_18
        buf30 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_20, buf29, reinterpret_tensor(primals_19, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf30)
        del primals_20
        # Source Nodes: [x_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf31 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf32 = buf31[0]
        buf33 = buf31[1]
        buf34 = buf31[2]
        buf35 = buf31[3]
        del buf31
        buf36 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_21, (768, 768), (1, 768), 0), out=buf36)
        buf37 = reinterpret_tensor(buf36, (8, 197, 768), (151296, 768, 1), 0); del buf36  # reuse
        buf41 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf42 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf283 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___1___norm2, x_20, x_25, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf37, buf16, buf24, primals_16, primals_22, primals_23, primals_24, buf41, buf42, buf283, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_16
        del primals_22
        del primals_24
        buf43 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_26, buf42, reinterpret_tensor(primals_25, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf43)
        del primals_26
        buf44 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27, x_30], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf43, buf44, 4841472, grid=grid(4841472), stream=stream0)
        buf45 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_27, (3072, 768), (1, 3072), 0), out=buf45)
        buf49 = buf16; del buf16  # reuse
        buf50 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf282 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv, getattr_l__mod___blocks___2___norm1, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf37, buf45, primals_28, primals_29, primals_30, buf49, buf50, buf282, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_30
        buf51 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_32, buf50, reinterpret_tensor(primals_31, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf51)
        del primals_32
        # Source Nodes: [x_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf52 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf53 = buf52[0]
        buf54 = buf52[1]
        buf55 = buf52[2]
        buf56 = buf52[3]
        del buf52
        buf57 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_33, (768, 768), (1, 768), 0), out=buf57)
        buf58 = reinterpret_tensor(buf57, (8, 197, 768), (151296, 768, 1), 0); del buf57  # reuse
        buf62 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf63 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf281 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___2___norm2, x_32, x_37, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf58, buf37, buf45, primals_28, primals_34, primals_35, primals_36, buf62, buf63, buf281, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_28
        del primals_34
        del primals_36
        buf64 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_38, buf63, reinterpret_tensor(primals_37, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf64)
        del primals_38
        buf65 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39, x_42], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf64, buf65, 4841472, grid=grid(4841472), stream=stream0)
        buf66 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf65, reinterpret_tensor(primals_39, (3072, 768), (1, 3072), 0), out=buf66)
        buf70 = buf37; del buf37  # reuse
        buf71 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf280 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv, getattr_l__mod___blocks___3___norm1, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf58, buf66, primals_40, primals_41, primals_42, buf70, buf71, buf280, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_42
        buf72 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_44, buf71, reinterpret_tensor(primals_43, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf72)
        del primals_44
        # Source Nodes: [x_45], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf73 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf74 = buf73[0]
        buf75 = buf73[1]
        buf76 = buf73[2]
        buf77 = buf73[3]
        del buf73
        buf78 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_45, (768, 768), (1, 768), 0), out=buf78)
        buf79 = reinterpret_tensor(buf78, (8, 197, 768), (151296, 768, 1), 0); del buf78  # reuse
        buf83 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf84 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf279 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___3___norm2, x_44, x_49, x_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf79, buf58, buf66, primals_40, primals_46, primals_47, primals_48, buf83, buf84, buf279, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_40
        del primals_46
        del primals_48
        buf85 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_50, buf84, reinterpret_tensor(primals_49, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf85)
        del primals_50
        buf86 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51, x_54], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf85, buf86, 4841472, grid=grid(4841472), stream=stream0)
        buf87 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf86, reinterpret_tensor(primals_51, (3072, 768), (1, 3072), 0), out=buf87)
        buf91 = buf58; del buf58  # reuse
        buf92 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf278 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv, getattr_l__mod___blocks___4___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf79, buf87, primals_52, primals_53, primals_54, buf91, buf92, buf278, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_54
        buf93 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_56, buf92, reinterpret_tensor(primals_55, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf93)
        del primals_56
        # Source Nodes: [x_57], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf94 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf95 = buf94[0]
        buf96 = buf94[1]
        buf97 = buf94[2]
        buf98 = buf94[3]
        del buf94
        buf99 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), out=buf99)
        buf100 = reinterpret_tensor(buf99, (8, 197, 768), (151296, 768, 1), 0); del buf99  # reuse
        buf104 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf105 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf277 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___4___norm2, x_56, x_61, x_62], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf100, buf79, buf87, primals_52, primals_58, primals_59, primals_60, buf104, buf105, buf277, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_52
        del primals_58
        del primals_60
        buf106 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_62, buf105, reinterpret_tensor(primals_61, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf106)
        del primals_62
        buf107 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_66], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf106, buf107, 4841472, grid=grid(4841472), stream=stream0)
        buf108 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf107, reinterpret_tensor(primals_63, (3072, 768), (1, 3072), 0), out=buf108)
        buf112 = buf79; del buf79  # reuse
        buf113 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf276 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv, getattr_l__mod___blocks___5___norm1, x_68], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf100, buf108, primals_64, primals_65, primals_66, buf112, buf113, buf276, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_66
        buf114 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_68, buf113, reinterpret_tensor(primals_67, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf114)
        del primals_68
        # Source Nodes: [x_69], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf115 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf116 = buf115[0]
        buf117 = buf115[1]
        buf118 = buf115[2]
        buf119 = buf115[3]
        del buf115
        buf120 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), out=buf120)
        buf121 = reinterpret_tensor(buf120, (8, 197, 768), (151296, 768, 1), 0); del buf120  # reuse
        buf125 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf126 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf275 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___5___norm2, x_68, x_73, x_74], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf121, buf100, buf108, primals_64, primals_70, primals_71, primals_72, buf125, buf126, buf275, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_64
        del primals_70
        del primals_72
        buf127 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_74, buf126, reinterpret_tensor(primals_73, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf127)
        del primals_74
        buf128 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75, x_78], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf127, buf128, 4841472, grid=grid(4841472), stream=stream0)
        buf129 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf128, reinterpret_tensor(primals_75, (3072, 768), (1, 3072), 0), out=buf129)
        buf133 = buf100; del buf100  # reuse
        buf134 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf274 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv, getattr_l__mod___blocks___6___norm1, x_80], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf121, buf129, primals_76, primals_77, primals_78, buf133, buf134, buf274, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_78
        buf135 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_80, buf134, reinterpret_tensor(primals_79, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf135)
        del primals_80
        # Source Nodes: [x_81], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf136 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf137 = buf136[0]
        buf138 = buf136[1]
        buf139 = buf136[2]
        buf140 = buf136[3]
        del buf136
        buf141 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf137, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 768), (1, 768), 0), out=buf141)
        buf142 = reinterpret_tensor(buf141, (8, 197, 768), (151296, 768, 1), 0); del buf141  # reuse
        buf146 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf147 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf273 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___6___norm2, x_80, x_85, x_86], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf142, buf121, buf129, primals_76, primals_82, primals_83, primals_84, buf146, buf147, buf273, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_76
        del primals_82
        del primals_84
        buf148 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_86, buf147, reinterpret_tensor(primals_85, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf148)
        del primals_86
        buf149 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87, x_90], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf148, buf149, 4841472, grid=grid(4841472), stream=stream0)
        buf150 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf149, reinterpret_tensor(primals_87, (3072, 768), (1, 3072), 0), out=buf150)
        buf154 = buf121; del buf121  # reuse
        buf155 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf272 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv, getattr_l__mod___blocks___7___norm1, x_92], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf142, buf150, primals_88, primals_89, primals_90, buf154, buf155, buf272, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_90
        buf156 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf155, reinterpret_tensor(primals_91, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf156)
        del primals_92
        # Source Nodes: [x_93], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf157 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf158 = buf157[0]
        buf159 = buf157[1]
        buf160 = buf157[2]
        buf161 = buf157[3]
        del buf157
        buf162 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_93, (768, 768), (1, 768), 0), out=buf162)
        buf163 = reinterpret_tensor(buf162, (8, 197, 768), (151296, 768, 1), 0); del buf162  # reuse
        buf167 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf168 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf271 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___7___norm2, x_92, x_97, x_98], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf163, buf142, buf150, primals_88, primals_94, primals_95, primals_96, buf167, buf168, buf271, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_88
        del primals_94
        del primals_96
        buf169 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_98, buf168, reinterpret_tensor(primals_97, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf169)
        del primals_98
        buf170 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf169, buf170, 4841472, grid=grid(4841472), stream=stream0)
        buf171 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf170, reinterpret_tensor(primals_99, (3072, 768), (1, 3072), 0), out=buf171)
        buf175 = buf142; del buf142  # reuse
        buf176 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf270 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv, getattr_l__mod___blocks___8___norm1, x_104], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf163, buf171, primals_100, primals_101, primals_102, buf175, buf176, buf270, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_102
        buf177 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_104, buf176, reinterpret_tensor(primals_103, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf177)
        del primals_104
        # Source Nodes: [x_105], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf178 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf179 = buf178[0]
        buf180 = buf178[1]
        buf181 = buf178[2]
        buf182 = buf178[3]
        del buf178
        buf183 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf179, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_105, (768, 768), (1, 768), 0), out=buf183)
        buf184 = reinterpret_tensor(buf183, (8, 197, 768), (151296, 768, 1), 0); del buf183  # reuse
        buf188 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf189 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf269 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___8___norm2, x_104, x_109, x_110], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf184, buf163, buf171, primals_100, primals_106, primals_107, primals_108, buf188, buf189, buf269, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_100
        del primals_106
        del primals_108
        buf190 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_110, buf189, reinterpret_tensor(primals_109, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf190)
        del primals_110
        buf191 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111, x_114], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf190, buf191, 4841472, grid=grid(4841472), stream=stream0)
        buf192 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf191, reinterpret_tensor(primals_111, (3072, 768), (1, 3072), 0), out=buf192)
        buf196 = buf163; del buf163  # reuse
        buf197 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf268 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv, getattr_l__mod___blocks___9___norm1, x_116], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf184, buf192, primals_112, primals_113, primals_114, buf196, buf197, buf268, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_114
        buf198 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf197, reinterpret_tensor(primals_115, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf198)
        del primals_116
        # Source Nodes: [x_117], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf199 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf200 = buf199[0]
        buf201 = buf199[1]
        buf202 = buf199[2]
        buf203 = buf199[3]
        del buf199
        buf204 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), out=buf204)
        buf205 = reinterpret_tensor(buf204, (8, 197, 768), (151296, 768, 1), 0); del buf204  # reuse
        buf209 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf210 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf267 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___9___norm2, x_116, x_121, x_122], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf205, buf184, buf192, primals_112, primals_118, primals_119, primals_120, buf209, buf210, buf267, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_112
        del primals_118
        del primals_120
        buf211 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_122, buf210, reinterpret_tensor(primals_121, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf211)
        del primals_122
        buf212 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123, x_126], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf211, buf212, 4841472, grid=grid(4841472), stream=stream0)
        buf213 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf212, reinterpret_tensor(primals_123, (3072, 768), (1, 3072), 0), out=buf213)
        buf217 = buf184; del buf184  # reuse
        buf218 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf266 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv, getattr_l__mod___blocks___10___norm1, x_128], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf205, buf213, primals_124, primals_125, primals_126, buf217, buf218, buf266, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_126
        buf219 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_128, buf218, reinterpret_tensor(primals_127, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf219)
        del primals_128
        # Source Nodes: [x_129], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf220 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf221 = buf220[0]
        buf222 = buf220[1]
        buf223 = buf220[2]
        buf224 = buf220[3]
        del buf220
        buf225 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), out=buf225)
        buf226 = reinterpret_tensor(buf225, (8, 197, 768), (151296, 768, 1), 0); del buf225  # reuse
        buf230 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf231 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf265 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___10___norm2, x_128, x_133, x_134], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf226, buf205, buf213, primals_124, primals_130, primals_131, primals_132, buf230, buf231, buf265, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_124
        del primals_130
        del primals_132
        buf232 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_134, buf231, reinterpret_tensor(primals_133, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf232)
        del primals_134
        buf233 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_138], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf232, buf233, 4841472, grid=grid(4841472), stream=stream0)
        buf234 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf233, reinterpret_tensor(primals_135, (3072, 768), (1, 3072), 0), out=buf234)
        buf238 = buf205; del buf205  # reuse
        buf239 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf264 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv, getattr_l__mod___blocks___11___norm1, x_140], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_5.run(buf226, buf234, primals_136, primals_137, primals_138, buf238, buf239, buf264, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_138
        buf240 = empty((1576, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_140, buf239, reinterpret_tensor(primals_139, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf240)
        del primals_140
        # Source Nodes: [x_141], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf241 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), None, True)
        buf242 = buf241[0]
        buf243 = buf241[1]
        buf244 = buf241[2]
        buf245 = buf241[3]
        del buf241
        buf246 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1576, 768), (768, 1), 0), reinterpret_tensor(primals_141, (768, 768), (1, 768), 0), out=buf246)
        buf247 = reinterpret_tensor(buf246, (8, 197, 768), (151296, 768, 1), 0); del buf246  # reuse
        buf251 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf252 = empty((1576, 768), device='cuda', dtype=torch.float32)
        buf263 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks___11___norm2, x_140, x_145, x_146], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf247, buf226, buf234, primals_136, primals_142, primals_143, primals_144, buf251, buf252, buf263, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_136
        del primals_142
        del primals_144
        buf253 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_146, buf252, reinterpret_tensor(primals_145, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf253)
        del primals_146
        buf254 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147, x_150], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_4.run(buf253, buf254, 4841472, grid=grid(4841472), stream=stream0)
        buf255 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf254, reinterpret_tensor(primals_147, (3072, 768), (1, 3072), 0), out=buf255)
        buf259 = buf226; del buf226  # reuse
        buf262 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153, x_155], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf247, buf255, primals_148, buf259, buf262, 1576, 768, grid=grid(1576), stream=stream0)
        del buf247
        del buf255
        del primals_148
        buf260 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf259, primals_149, primals_150, buf260, 6144, grid=grid(6144), stream=stream0)
        del primals_150
        buf261 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_152, buf260, reinterpret_tensor(primals_151, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf261)
        del primals_152
        return (buf261, primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, buf7, buf8, reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf9, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf12, buf13, buf14, reinterpret_tensor(buf11, (1576, 768), (768, 1), 0), buf20, buf21, buf22, buf23, buf28, buf29, reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf30, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf33, buf34, buf35, reinterpret_tensor(buf32, (1576, 768), (768, 1), 0), buf41, buf42, buf43, buf44, buf49, buf50, reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf51, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf54, buf55, buf56, reinterpret_tensor(buf53, (1576, 768), (768, 1), 0), buf62, buf63, buf64, buf65, buf70, buf71, reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf72, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf75, buf76, buf77, reinterpret_tensor(buf74, (1576, 768), (768, 1), 0), buf83, buf84, buf85, buf86, buf91, buf92, reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf93, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf96, buf97, buf98, reinterpret_tensor(buf95, (1576, 768), (768, 1), 0), buf104, buf105, buf106, buf107, buf112, buf113, reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf114, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf117, buf118, buf119, reinterpret_tensor(buf116, (1576, 768), (768, 1), 0), buf125, buf126, buf127, buf128, buf133, buf134, reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf135, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf138, buf139, buf140, reinterpret_tensor(buf137, (1576, 768), (768, 1), 0), buf146, buf147, buf148, buf149, buf154, buf155, reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf156, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf159, buf160, buf161, reinterpret_tensor(buf158, (1576, 768), (768, 1), 0), buf167, buf168, buf169, buf170, buf175, buf176, reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf177, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf180, buf181, buf182, reinterpret_tensor(buf179, (1576, 768), (768, 1), 0), buf188, buf189, buf190, buf191, buf196, buf197, reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf198, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf201, buf202, buf203, reinterpret_tensor(buf200, (1576, 768), (768, 1), 0), buf209, buf210, buf211, buf212, buf217, buf218, reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf219, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf222, buf223, buf224, reinterpret_tensor(buf221, (1576, 768), (768, 1), 0), buf230, buf231, buf232, buf233, buf238, buf239, reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 0), reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 768), reinterpret_tensor(buf240, (8, 12, 197, 64), (453888, 64, 2304, 1), 1536), buf243, buf244, buf245, reinterpret_tensor(buf242, (1576, 768), (768, 1), 0), buf251, buf252, buf253, buf254, buf259, buf260, reinterpret_tensor(primals_151, (1000, 768), (768, 1), 0), buf262, reinterpret_tensor(primals_147, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_145, (3072, 768), (768, 1), 0), buf263, reinterpret_tensor(primals_141, (768, 768), (768, 1), 0), buf242, reinterpret_tensor(primals_139, (2304, 768), (768, 1), 0), buf264, reinterpret_tensor(primals_135, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_133, (3072, 768), (768, 1), 0), buf265, reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), buf221, reinterpret_tensor(primals_127, (2304, 768), (768, 1), 0), buf266, reinterpret_tensor(primals_123, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_121, (3072, 768), (768, 1), 0), buf267, reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), buf200, reinterpret_tensor(primals_115, (2304, 768), (768, 1), 0), buf268, reinterpret_tensor(primals_111, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_109, (3072, 768), (768, 1), 0), buf269, reinterpret_tensor(primals_105, (768, 768), (768, 1), 0), buf179, reinterpret_tensor(primals_103, (2304, 768), (768, 1), 0), buf270, reinterpret_tensor(primals_99, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_97, (3072, 768), (768, 1), 0), buf271, reinterpret_tensor(primals_93, (768, 768), (768, 1), 0), buf158, reinterpret_tensor(primals_91, (2304, 768), (768, 1), 0), buf272, reinterpret_tensor(primals_87, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_85, (3072, 768), (768, 1), 0), buf273, reinterpret_tensor(primals_81, (768, 768), (768, 1), 0), buf137, reinterpret_tensor(primals_79, (2304, 768), (768, 1), 0), buf274, reinterpret_tensor(primals_75, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_73, (3072, 768), (768, 1), 0), buf275, reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), buf116, reinterpret_tensor(primals_67, (2304, 768), (768, 1), 0), buf276, reinterpret_tensor(primals_63, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_61, (3072, 768), (768, 1), 0), buf277, reinterpret_tensor(primals_57, (768, 768), (768, 1), 0), buf95, reinterpret_tensor(primals_55, (2304, 768), (768, 1), 0), buf278, reinterpret_tensor(primals_51, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_49, (3072, 768), (768, 1), 0), buf279, reinterpret_tensor(primals_45, (768, 768), (768, 1), 0), buf74, reinterpret_tensor(primals_43, (2304, 768), (768, 1), 0), buf280, reinterpret_tensor(primals_39, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_37, (3072, 768), (768, 1), 0), buf281, reinterpret_tensor(primals_33, (768, 768), (768, 1), 0), buf53, reinterpret_tensor(primals_31, (2304, 768), (768, 1), 0), buf282, reinterpret_tensor(primals_27, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_25, (3072, 768), (768, 1), 0), buf283, reinterpret_tensor(primals_21, (768, 768), (768, 1), 0), buf32, reinterpret_tensor(primals_19, (2304, 768), (768, 1), 0), buf284, reinterpret_tensor(primals_15, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_13, (3072, 768), (768, 1), 0), buf285, reinterpret_tensor(primals_9, (768, 768), (768, 1), 0), buf11, reinterpret_tensor(primals_7, (2304, 768), (768, 1), 0), buf286, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vit_base_patch16_224', benchmark_compiled_module)
