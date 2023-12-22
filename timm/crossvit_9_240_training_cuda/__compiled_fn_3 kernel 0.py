
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


# kernel path: /tmp/torchinductor_youkaichao/rq/crqmay4cwpok2zveskeyzxtlqoxd2fl3ftmxrdntygh3hiou37rl.py
# Source Nodes: [x__5], Original ATen: [aten.sub]
# x__5 => sub_7
triton_poi_fused_sub_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = tl.math.floor(tmp6)
    tmp8 = tmp6 - tmp7
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = -0.75
    tmp12 = tmp10 * tmp11
    tmp13 = -3.75
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp10
    tmp16 = -6.0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = -3.0
    tmp20 = tmp18 - tmp19
    tl.store(out_ptr0 + (x0), tmp20, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rd/crd45ji3j3da26jjuyhg3ug6iicfv5ee4azu4guze3zg32zwyet4.py
# Source Nodes: [x__5], Original ATen: [aten.add]
# x__5 => add_9
triton_poi_fused_add_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = tl.math.floor(tmp6)
    tmp8 = tmp6 - tmp7
    tmp9 = 1.25
    tmp10 = tmp8 * tmp9
    tmp11 = 2.25
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp8
    tmp14 = tmp13 * tmp8
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtydeznc5dypm4vch6szy633vtcxeuelsnao4zxjwrtfmgcffvs.py
# Source Nodes: [x__5], Original ATen: [aten.add]
# x__5 => add_10
triton_poi_fused_add_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = tl.math.floor(tmp6)
    tmp8 = tmp6 - tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = 1.25
    tmp12 = tmp10 * tmp11
    tmp13 = 2.25
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp10
    tmp16 = tmp15 * tmp10
    tmp17 = tmp16 + tmp9
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvin5tmvdk4mj2sq4ybqwzrgdvhy27kl5kwa6tuj33ohesrs7flv.py
# Source Nodes: [x__5], Original ATen: [aten.sub]
# x__5 => sub_13
triton_poi_fused_sub_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = tl.math.floor(tmp6)
    tmp8 = tmp6 - tmp7
    tmp9 = 2.0
    tmp10 = tmp9 - tmp8
    tmp11 = -0.75
    tmp12 = tmp10 * tmp11
    tmp13 = -3.75
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp10
    tmp16 = -6.0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 * tmp10
    tmp19 = -3.0
    tmp20 = tmp18 - tmp19
    tl.store(out_ptr0 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/cle5aodtvkel7uenidbn2grl4dnd65au5b42gkexzgfpz36cdmsq.py
# Source Nodes: [x__5], Original ATen: [aten._unsafe_index, aten.add, aten.mul]
# x__5 => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add_12, add_13, add_14, add_20, add_21, add_22, add_28, add_29, add_30, add_36, add_37, add_38, add_44, add_45, add_46, mul_14, mul_15, mul_16, mul_17, mul_30, mul_31, mul_32, mul_33, mul_46, mul_47, mul_48, mul_49, mul_62, mul_63, mul_64, mul_65, mul_78, mul_79, mul_80, mul_81
triton_poi_fused__unsafe_index_add_mul_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_4', 'mutated_arg_names': ['in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 224) % 224
    x0 = xindex % 224
    x2 = (xindex // 50176)
    x4 = xindex
    tmp25 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = tl.math.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1], 239, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp4
    tmp17 = tmp16 - tmp2
    tmp18 = tl.math.floor(tmp17)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp19 - tmp20
    tmp22 = triton_helpers.maximum(tmp21, tmp9)
    tmp23 = triton_helpers.minimum(tmp22, tmp11)
    tmp24 = tl.load(in_ptr0 + (tmp23 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp24 * tmp25
    tmp27 = triton_helpers.maximum(tmp19, tmp9)
    tmp28 = triton_helpers.minimum(tmp27, tmp11)
    tmp29 = tl.load(in_ptr0 + (tmp28 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp31 = tmp29 * tmp30
    tmp32 = tmp26 + tmp31
    tmp33 = tmp19 + tmp20
    tmp34 = triton_helpers.maximum(tmp33, tmp9)
    tmp35 = triton_helpers.minimum(tmp34, tmp11)
    tmp36 = tl.load(in_ptr0 + (tmp35 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 + tmp38
    tmp40 = tl.full([1], 2, tl.int64)
    tmp41 = tmp19 + tmp40
    tmp42 = triton_helpers.maximum(tmp41, tmp9)
    tmp43 = triton_helpers.minimum(tmp42, tmp11)
    tmp44 = tl.load(in_ptr0 + (tmp43 + (240*tmp12) + (57600*x2)), None, eviction_policy='evict_last')
    tmp46 = tmp44 * tmp45
    tmp47 = tmp39 + tmp46
    tmp48 = tmp8 - tmp20
    tmp49 = triton_helpers.maximum(tmp48, tmp9)
    tmp50 = triton_helpers.minimum(tmp49, tmp11)
    tmp51 = tl.load(in_ptr0 + (tmp23 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp52 = tmp51 * tmp25
    tmp53 = tl.load(in_ptr0 + (tmp28 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp54 = tmp53 * tmp30
    tmp55 = tmp52 + tmp54
    tmp56 = tmp8 + tmp20
    tmp57 = triton_helpers.maximum(tmp56, tmp9)
    tmp58 = triton_helpers.minimum(tmp57, tmp11)
    tmp59 = tl.load(in_ptr0 + (tmp23 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp60 = tmp59 * tmp25
    tmp61 = tl.load(in_ptr0 + (tmp28 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp62 = tmp61 * tmp30
    tmp63 = tmp60 + tmp62
    tmp64 = tmp8 + tmp40
    tmp65 = triton_helpers.maximum(tmp64, tmp9)
    tmp66 = triton_helpers.minimum(tmp65, tmp11)
    tmp67 = tl.load(in_ptr0 + (tmp23 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp68 = tmp67 * tmp25
    tmp69 = tl.load(in_ptr0 + (tmp28 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp70 = tmp69 * tmp30
    tmp71 = tmp68 + tmp70
    tmp72 = tl.load(in_ptr0 + (tmp35 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp73 = tmp72 * tmp37
    tmp74 = tmp55 + tmp73
    tmp75 = tl.load(in_ptr0 + (tmp43 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp76 = tmp75 * tmp45
    tmp77 = tmp74 + tmp76
    tmp78 = tl.load(in_ptr0 + (tmp35 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp79 = tmp78 * tmp37
    tmp80 = tmp63 + tmp79
    tmp81 = tl.load(in_ptr0 + (tmp43 + (240*tmp58) + (57600*x2)), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp45
    tmp83 = tmp80 + tmp82
    tmp84 = tl.load(in_ptr0 + (tmp35 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp85 = tmp84 * tmp37
    tmp86 = tmp71 + tmp85
    tmp87 = tl.load(in_ptr0 + (tmp43 + (240*tmp66) + (57600*x2)), None, eviction_policy='evict_last')
    tmp88 = tmp87 * tmp45
    tmp89 = tmp86 + tmp88
    tmp91 = tmp77 * tmp90
    tmp93 = tmp47 * tmp92
    tmp94 = tmp91 + tmp93
    tmp96 = tmp83 * tmp95
    tmp97 = tmp94 + tmp96
    tmp99 = tmp89 * tmp98
    tmp100 = tmp97 + tmp99
    tl.store(in_out_ptr1 + (x4), tmp100, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cocu6cnxxv6kp7cntt4mp6xbethfoodjfbbt3u34jwqbfstiq6o7.py
# Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___attn_qkv, getattr_l__mod___blocks_0_blocks_0___0___norm1, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cat_27 => cat
# getattr_l__mod___blocks_0_blocks_0___0___attn_qkv => view_6
# getattr_l__mod___blocks_0_blocks_0___0___norm1 => add_48, add_49, mul_82, mul_83, rsqrt, sub_46, var_mean
# x__3 => add
triton_red_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 401
    x1 = (xindex // 401)
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 401, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((400*r2) + (51200*x1) + (((-1) + x0) % 400)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp40 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = x0
        tmp24 = tl.full([1, 1], 0, tl.int64)
        tmp25 = tmp23 >= tmp24
        tmp26 = tl.full([1, 1], 1, tl.int64)
        tmp27 = tmp23 < tmp26
        tmp28 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp27 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
        tmp30 = tl.where(tmp27, tmp28, tmp29)
        tmp31 = tmp23 >= tmp26
        tmp32 = tl.full([1, 1], 401, tl.int64)
        tmp33 = tmp23 < tmp32
        tmp34 = tl.load(in_ptr1 + ((400*r2) + (51200*x1) + (((-1) + x0) % 400)), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 + tmp35
        tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
        tmp38 = tl.where(tmp31, tmp36, tmp37)
        tmp39 = tl.where(tmp27, tmp30, tmp38)
        tmp41 = tmp39 + tmp40
        tmp42 = tmp41 - tmp20
        tmp43 = 128.0
        tmp44 = tmp21 / tmp43
        tmp45 = 1e-06
        tmp46 = tmp44 + tmp45
        tmp47 = tl.math.rsqrt(tmp46)
        tmp48 = tmp42 * tmp47
        tmp50 = tmp48 * tmp49
        tmp52 = tmp50 + tmp51
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp48, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp52, rmask & xmask)
    tmp53 = 128.0
    tmp54 = tmp21 / tmp53
    tmp55 = 1e-06
    tmp56 = tmp54 + tmp55
    tmp57 = tl.math.rsqrt(tmp56)
    tmp58 = tmp57 / tmp53
    tl.store(out_ptr4 + (x3), tmp58, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofhb3r5vwgv3yzzl5pp75g5f2ybrzsh45rjlcxgve6dkpq4svk7.py
# Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___norm2, x_7, x_8, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cat_27 => cat
# getattr_l__mod___blocks_0_blocks_0___0___norm2 => add_51, add_52, mul_84, mul_85, rsqrt_1, sub_47, var_mean_1
# x_7 => add_50
# x_8 => view_12
# x__3 => add
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 401, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((400*r2) + (51200*x1) + (((-1) + x0) % 400)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp18 = tmp16 + tmp17
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp22 - tmp32
    tmp40 = 128.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp44 / tmp40
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp50, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f7/cf7jvvytprzsbhw37dzfjs32wdmsxobfev5kba5dfhc2oeounhm6.py
# Source Nodes: [x_12, x_9], Original ATen: [aten.gelu, aten.view]
# x_12 => view_14
# x_9 => add_53, erf, mul_86, mul_87, mul_88
triton_poi_fused_gelu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1231872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwx2h3p6aoo4bnbqfj7udojd7qlda7djnnakap5vtxa72kzx4hq.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm1 => var_mean_2
# x__8 => add_47
triton_red_fused_add_cat_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2) % 197
    x0 = xindex % 2
    x2 = (xindex // 394)
    x5 = xindex % 394
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
        tmp11 = tl.load(in_ptr1 + ((196*r3) + (25088*x0) + (50176*x2) + (((-1) + x1) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3t/c3t6vvgswlo5vi4s3s725ge5rikxuitzlkth77nauc2gglpvt2dz.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm1 => add_55, rsqrt_2, var_mean_2
# x__8 => add_47
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
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


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhbevd46ubl4sutf2uo2hszoj7fnafmz76tk6cbs34eepyl5izb.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___attn_qkv, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.view]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___attn_qkv => view_16
# getattr_l__mod___blocks_0_blocks_1___0___norm1 => add_55, add_56, mul_89, mul_90, rsqrt_2, sub_48, var_mean_2
# x__8 => add_47
triton_poi_fused_add_cat_native_layer_norm_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 197
    x0 = xindex % 256
    x2 = (xindex // 50432)
    x3 = xindex % 50432
    x4 = (xindex // 256)
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
    tmp11 = tl.load(in_ptr1 + ((196*x0) + (50176*x2) + (((-1) + x1) % 196)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = 256.0
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcgjctwjirjutlbrmoknbpmxc6tqspmq2gbsnj2nxjkr2lw3bbc.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm2, x_19, x_20, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm2 => add_58, add_59, mul_91, mul_92, rsqrt_3, sub_49, var_mean_3
# x_19 => add_57
# x_20 => view_22
# x__8 => add_47
triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (r2 + (256*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
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
    tmp11 = tl.load(in_ptr1 + ((196*r2) + (50176*x1) + (((-1) + x0) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp30 = tl.full([1], 256, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 256.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp44 / tmp40
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp49, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp50, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/du/cduy35ftaveg7ajlyofjemjv5lr4loywuro4mtxw77qjajn56zxm.py
# Source Nodes: [x_21, x_24], Original ATen: [aten.gelu, aten.view]
# x_21 => add_60, erf_1, mul_93, mul_94, mul_95
# x_24 => view_24
triton_poi_fused_gelu_view_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
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


# kernel path: /tmp/torchinductor_youkaichao/zj/czj3255ewwgg4n7l3bbw4fy2af5ch5zufcm67e7qsfolqjdajmzk.py
# Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___attn_qkv, getattr_l__mod___blocks_0_blocks_1___1___norm1, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks_0_blocks_1___1___attn_qkv => view_26
# getattr_l__mod___blocks_0_blocks_1___1___norm1 => add_62, add_63, mul_96, mul_97, rsqrt_4, sub_50, var_mean_4
# x_26 => add_61
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
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


# kernel path: /tmp/torchinductor_youkaichao/rp/crpebfbisjunoqb2rcaebbuhw7x5hhhsopdo4hiqjrqouxssh5zr.py
# Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___norm2, x_26, x_31, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks_0_blocks_1___1___norm2 => add_65, add_66, mul_98, mul_99, rsqrt_5, sub_51, var_mean_5
# x_26 => add_61
# x_31 => add_64
# x_32 => view_32
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1576
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


# kernel path: /tmp/torchinductor_youkaichao/27/c27hiz7rjvj3dvrpqlctcnb6stc7vzfw5bt6we6bmnihrdlulghd.py
# Source Nodes: [l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1, l__mod___blocks_0_projs_0_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_projs_0_0 => add_76, add_77, clone_14, mul_110, mul_111, rsqrt_8, sub_54, var_mean_8
# l__mod___blocks_0_projs_0_1 => add_78, erf_4, mul_112, mul_113, mul_114
# l__mod___blocks_0_projs_0_2 => view_46
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (51328*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (51328*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = 0.7071067811865476
    tmp35 = tmp31 * tmp34
    tmp36 = tl.math.erf(tmp35)
    tmp37 = 1.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp33 * tmp38
    tmp40 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6sqvdd2rqi3ocvnyctppa3dska3eich2y57cgd4w5ibgklqakvr.py
# Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1, l__mod___blocks_0_projs_1_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_projs_1_0 => add_79, add_80, clone_15, mul_115, mul_116, rsqrt_9, sub_55, var_mean_9
# l__mod___blocks_0_projs_1_1 => add_81, erf_5, mul_117, mul_118, mul_119
# l__mod___blocks_0_projs_1_2 => view_48
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1 + (50432*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (50432*x0)), rmask & xmask, other=0.0)
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
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = 0.7071067811865476
    tmp35 = tmp31 * tmp34
    tmp36 = tl.math.erf(tmp35)
    tmp37 = 1.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp33 * tmp38
    tmp40 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpiuqkkc4tmtkl23auj5x3337p4dii7mmwqjg46xx2w3n3cb2uj.py
# Source Nodes: [cat_25, l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_25 => cat_2
# l__mod___blocks_0_fusion_0_norm1 => add_82, add_83, mul_120, mul_121, rsqrt_10, sub_56, var_mean_10
triton_per_fused_cat_native_layer_norm_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 256, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = 256.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp18 - tmp28
    tmp41 = tmp40 * tmp39
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (256*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp39, xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujpnbetnhrk24dvn2rgl5x6vyikvepymbwnzdcwxq6l35pofeob.py
# Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.add]
# l__mod___blocks_0_fusion_0_attn_wq => add_84
triton_poi_fused_add_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/cai3rituzmixd4u5wiea7t4pzuqrategx4bcjyv5tkaqzb4szz2g.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_16
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 197
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
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (50432*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrigm5gdjuaow2drjankg5wppjqcfb5krszunfjz2fl4nbqbppj.py
# Source Nodes: [attn, attn_1, attn_2], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn => mul_122
# attn_1 => amax, div, exp, sub_57, sum_1
# attn_2 => clone_17
triton_per_fused__softmax_clone_detach_mul_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (197*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (197*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpz2wbnqarew7ipb6rdrbtyxsoogm4dpmq33o26qamkxafj2dd6g.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_18
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 4
    x3 = (xindex // 50432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (50432*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrfm7pkdhmau65nzbnscuskuqko3codhdvvky74u7rotpp3z5in.py
# Source Nodes: [l__mod___blocks_0_revert_projs_0_0, l__mod___blocks_0_revert_projs_0_1, reverted_proj_cls_token, tmp_1], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_revert_projs_0_0 => add_86, add_87, mul_123, mul_124, rsqrt_11, sub_58, var_mean_11
# l__mod___blocks_0_revert_projs_0_1 => add_88, erf_6, mul_125, mul_126, mul_127
# reverted_proj_cls_token => view_68
# tmp_1 => add_85
triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1 + (50432*x0)), rmask & xmask, other=0.0)
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
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = 0.7071067811865476
    tmp35 = tmp31 * tmp34
    tmp36 = tl.math.erf(tmp35)
    tmp37 = 1.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp33 * tmp38
    tmp40 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxbvq5evfc3a4opjbehswvzlciikkupkhllscu22lijonhadjd5.py
# Source Nodes: [cat_23, cat_24, getattr_l__mod___blocks_1_blocks_0___0___attn_qkv, getattr_l__mod___blocks_1_blocks_0___0___norm1, l__mod___blocks_0_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_23 => cat_4
# cat_24 => cat_3
# getattr_l__mod___blocks_1_blocks_0___0___attn_qkv => view_90
# getattr_l__mod___blocks_1_blocks_0___0___norm1 => add_96, add_97, mul_136, mul_137, rsqrt_14, sub_62, var_mean_14
# l__mod___blocks_0_fusion_1_norm1 => add_89, add_90, mul_128, mul_129, rsqrt_12, sub_59, var_mean_12
triton_per_fused_cat_native_layer_norm_view_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp63 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 401, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp21, tmp17)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None]
    tmp46 = tmp45 / tmp31
    tmp47 = tmp39 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.where(rmask & xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None]
    tmp53 = 128.0
    tmp54 = tmp38 / tmp53
    tmp55 = 1e-06
    tmp56 = tmp54 + tmp55
    tmp57 = tl.math.rsqrt(tmp56)
    tmp58 = tmp52 / tmp53
    tmp59 = tmp58 + tmp55
    tmp60 = tl.math.rsqrt(tmp59)
    tmp61 = tmp22 - tmp32
    tmp62 = tmp61 * tmp57
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp18 - tmp46
    tmp68 = tmp67 * tmp60
    tmp70 = tmp68 * tmp69
    tmp72 = tmp70 + tmp71
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp18, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (128*x3)), tmp22, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp57, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp60, xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp66, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp72, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp32, xmask)
    tl.store(out_ptr3 + (x3), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdfy6iuz57m2foy3o4uhjzxlgm7jiugpydeyt4rbrzgfvzjnbnk.py
# Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.add]
# l__mod___blocks_0_fusion_1_attn_wq => add_91
triton_poi_fused_add_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqamhvjoqybgliqw7w55ir5cd3rairp6tfvaigcjscfye3pqzvzx.py
# Source Nodes: [matmul_2], Original ATen: [aten.clone]
# matmul_2 => clone_20
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 401
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
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (51328*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (401*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zb/czbiy2ee3xgt5xnhmbsqfgdtwksccymleei6ba5s7gpdfck5trhl.py
# Source Nodes: [attn_3, attn_4, attn_5], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn_3 => mul_130
# attn_4 => amax_1, div_1, exp_1, sub_60, sum_2
# attn_5 => clone_21
triton_per_fused__softmax_clone_detach_mul_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 401
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (401*x0)), rmask & xmask, other=0.0)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp5, 0))
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (401*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (401*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7w7gnr2yvvqmiaogsbqttwhjpprzgyoxwhafvyre7nfvlkrohg6.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_22
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 410624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 401
    x2 = (xindex // 12832) % 4
    x3 = (xindex // 51328)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (128*x1) + (51328*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crclhtetqcv7gub7rognookvnizv54uxbkyugageg5dlrk7mimd6.py
# Source Nodes: [l__mod___blocks_0_revert_projs_1_0, l__mod___blocks_0_revert_projs_1_1, reverted_proj_cls_token_1, tmp_4], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_0_revert_projs_1_0 => add_93, add_94, mul_131, mul_132, rsqrt_13, sub_61, var_mean_13
# l__mod___blocks_0_revert_projs_1_1 => add_95, erf_7, mul_133, mul_134, mul_135
# reverted_proj_cls_token_1 => view_88
# tmp_4 => add_92
triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (51328*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = 0.7071067811865476
    tmp35 = tmp31 * tmp34
    tmp36 = tl.math.erf(tmp35)
    tmp37 = 1.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp33 * tmp38
    tmp40 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbx5imxvg34iriupxvki35r7iah7gah3ob6cg5utaezch3hg5izv.py
# Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___norm2, x_63, x_64], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_l__mod___blocks_1_blocks_0___0___norm2 => add_100, add_99, mul_138, mul_139, rsqrt_15, sub_63, var_mean_15
# x_63 => add_98
# x_64 => view_96
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
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
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkyrj5c6sxa3duvsqmi6po3shxmwnm6jragutq2op4zqfak2tad.py
# Source Nodes: [x_63, x_70], Original ATen: [aten.add]
# x_63 => add_98
# x_70 => add_102
triton_poi_fused_add_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 410624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eppyisrrsb3xwub7nc6t32w7mzwwv73cr5e54df6uxdgmos4gq.py
# Source Nodes: [l__mod___blocks_1_projs_0_0, l__mod___blocks_1_projs_0_1, l__mod___blocks_1_projs_0_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_projs_0_0 => add_124, add_125, clone_36, mul_164, mul_165, rsqrt_22, sub_70, var_mean_22
# l__mod___blocks_1_projs_0_1 => add_126, erf_12, mul_166, mul_167, mul_168
# l__mod___blocks_1_projs_0_2 => view_130
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (51328*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = 0.5
    tmp29 = tmp27 * tmp28
    tmp30 = 0.7071067811865476
    tmp31 = tmp27 * tmp30
    tmp32 = tl.math.erf(tmp31)
    tmp33 = 1.0
    tmp34 = tmp32 + tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6p7d57fwdanfopgf4ijyhz5cqbepldxceo72leniun3xtkl3ne.py
# Source Nodes: [cat_21, l__mod___blocks_1_fusion_0_norm1, x_106, x_99], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_21 => cat_6
# l__mod___blocks_1_fusion_0_norm1 => add_130, add_131, mul_174, mul_175, rsqrt_24, sub_72, var_mean_24
# x_106 => add_123
# x_99 => add_119
triton_per_fused_add_cat_native_layer_norm_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    x2 = xindex % 197
    x3 = (xindex // 197)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = x2
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(in_ptr4 + (r1 + (256*x3)), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp9 >= tmp12
    tmp18 = tl.full([1], 197, tl.int64)
    tmp19 = tmp9 < tmp18
    tmp20 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp21 = tl.where(tmp17, tmp8, tmp20)
    tmp22 = tl.where(tmp13, tmp16, tmp21)
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 256, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = 256.0
    tmp40 = tmp38 / tmp39
    tmp41 = 1e-06
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp22 - tmp32
    tmp45 = tmp44 * tmp43
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp22, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp43, xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp49, rmask & xmask)
    tl.store(out_ptr1 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csgudqr5dwn56bm7om3ufaf2vbmtsdyfjztbsntltrqrdq6sh3aw.py
# Source Nodes: [l__mod___blocks_1_projs_1_0, l__mod___blocks_1_projs_1_1, l__mod___blocks_1_projs_1_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# l__mod___blocks_1_projs_1_0 => add_127, add_128, clone_37, mul_169, mul_170, rsqrt_23, sub_71, var_mean_23
# l__mod___blocks_1_projs_1_1 => add_129, erf_13, mul_171, mul_172, mul_173
# l__mod___blocks_1_projs_1_2 => view_132
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1 + (50432*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp17 = tmp0 - tmp10
    tmp18 = 256.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = 0.5
    tmp29 = tmp27 * tmp28
    tmp30 = 0.7071067811865476
    tmp31 = tmp27 * tmp30
    tmp32 = tl.math.erf(tmp31)
    tmp33 = 1.0
    tmp34 = tmp32 + tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hx/chxszsbnq6hm27dva2cprhruv2s4zmsoxm7m4uen2fjptyz6bt7s.py
# Source Nodes: [cat_19, cat_20, getattr_l__mod___blocks_2_blocks_0___0___attn_qkv, getattr_l__mod___blocks_2_blocks_0___0___norm1, l__mod___blocks_1_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_19 => cat_8
# cat_20 => cat_7
# getattr_l__mod___blocks_2_blocks_0___0___attn_qkv => view_174
# getattr_l__mod___blocks_2_blocks_0___0___norm1 => add_144, add_145, mul_190, mul_191, rsqrt_28, sub_78, var_mean_28
# l__mod___blocks_1_fusion_1_norm1 => add_137, add_138, mul_182, mul_183, rsqrt_26, sub_75, var_mean_26
triton_per_fused_cat_native_layer_norm_view_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp59 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 401, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp17, tmp13)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp41 / tmp27
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp49 = 128.0
    tmp50 = tmp34 / tmp49
    tmp51 = 1e-06
    tmp52 = tmp50 + tmp51
    tmp53 = tl.math.rsqrt(tmp52)
    tmp54 = tmp48 / tmp49
    tmp55 = tmp54 + tmp51
    tmp56 = tl.math.rsqrt(tmp55)
    tmp57 = tmp18 - tmp28
    tmp58 = tmp57 * tmp53
    tmp60 = tmp58 * tmp59
    tmp62 = tmp60 + tmp61
    tmp63 = tmp14 - tmp42
    tmp64 = tmp63 * tmp56
    tmp66 = tmp64 * tmp65
    tmp68 = tmp66 + tmp67
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp14, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (128*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp53, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp56, xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp62, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp68, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp28, xmask)
    tl.store(out_ptr3 + (x3), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7oqgpfvtdgbftn6easqnaoyevnru5qarhn3u37fjawrdpfwo7kb.py
# Source Nodes: [cat_18, getattr_l__mod___blocks_2_blocks_1___0___attn_qkv, getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
# cat_18 => cat_9
# getattr_l__mod___blocks_2_blocks_1___0___attn_qkv => view_184
# getattr_l__mod___blocks_2_blocks_1___0___norm1 => add_151, add_152, mul_197, mul_198, rsqrt_30, sub_80, var_mean_30
triton_per_fused_cat_native_layer_norm_view_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp38 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 256, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = 256.0
    tmp32 = tmp30 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = tl.math.rsqrt(tmp34)
    tmp36 = tmp14 - tmp24
    tmp37 = tmp36 * tmp35
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tl.store(out_ptr0 + (r2 + (256*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp41, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbell2ecog2ym4r6zdjkukhjah2xhdm6zuxbljgyczhpc5kxkhk7.py
# Source Nodes: [cat_15, cat_16, l__mod___blocks_2_fusion_1_norm1, x_171], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_15 => cat_12
# cat_16 => cat_11
# l__mod___blocks_2_fusion_1_norm1 => add_185, add_186, mul_236, mul_237, rsqrt_40, sub_91, var_mean_40
# x_171 => add_192, rsqrt_42, var_mean_42
triton_per_fused_cat_native_layer_norm_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp59 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 401, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp17, tmp13)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp41 / tmp27
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp49 = 128.0
    tmp50 = tmp34 / tmp49
    tmp51 = 1e-06
    tmp52 = tmp50 + tmp51
    tmp53 = tl.math.rsqrt(tmp52)
    tmp54 = tmp48 / tmp49
    tmp55 = tmp54 + tmp51
    tmp56 = tl.math.rsqrt(tmp55)
    tmp57 = tmp18 - tmp28
    tmp58 = tmp57 * tmp53
    tmp60 = tmp58 * tmp59
    tmp62 = tmp60 + tmp61
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp14, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (128*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp53, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp56, xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp62, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp28, xmask)
    tl.store(out_ptr3 + (x3), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coevy6byasgiutju6eshegzodiby3s4eqb4laqhtzlcye3ztknlm.py
# Source Nodes: [cat_14, x_172], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_14 => cat_13
# x_172 => add_194, rsqrt_43, var_mean_43
triton_per_fused_cat_native_layer_norm_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_37', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = tl.full([1], 256, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = 256.0
    tmp32 = tmp30 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = tl.math.rsqrt(tmp34)
    tl.store(out_ptr0 + (r2 + (256*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrkdbybvrd6pdnppviazmskdn5pdrb4b4btmbwupirpbioczi7x.py
# Source Nodes: [l__mod___head_drop], Original ATen: [aten.clone]
# l__mod___head_drop => clone_68
triton_poi_fused_clone_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (51328*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (401*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (401*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5hih76vn2pybnw6vzkns75wn3nzt6i2dhrmq74jucoe3uvvzkl.py
# Source Nodes: [l__mod___head_drop_1], Original ATen: [aten.clone]
# l__mod___head_drop_1 => clone_69
triton_poi_fused_clone_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (50432*x1)), None)
    tmp1 = tl.load(in_ptr1 + (197*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (197*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjzwwoz3io54lnd35gtrakcc63x4mloaokuabuiw2zpaky33zskn.py
# Source Nodes: [pred], Original ATen: [aten.mean]
# pred => mean
triton_poi_fused_mean_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1000)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-8000) + x2), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = 8 + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tl.load(in_ptr0 + (8000 + x2), tmp17 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp15 >= tmp3
    tmp22 = tmp15 < tmp9
    tmp23 = tl.load(in_ptr1 + (x2), tmp21 & xmask, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp21, tmp23, tmp24)
    tmp26 = tl.where(tmp17, tmp20, tmp25)
    tmp27 = tmp14 + tmp26
    tmp28 = 2.0
    tmp29 = tmp27 / tmp28
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_2, (1, 401, 128), (51328, 128, 1))
    assert_size_stride(primals_3, (1, 1, 256), (256, 256, 1))
    assert_size_stride(primals_4, (1, 197, 256), (50432, 256, 1))
    assert_size_stride(primals_5, (128, 3, 12, 12), (432, 144, 12, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (384, 128), (128, 1))
    assert_size_stride(primals_12, (384, ), (1, ))
    assert_size_stride(primals_13, (128, 128), (128, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (384, 128), (128, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (128, 384), (384, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (768, 256), (256, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (256, 256), (256, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (768, 256), (256, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (256, 768), (768, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (768, 256), (256, 1))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (256, 256), (256, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (768, 256), (256, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (256, 768), (768, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (768, 256), (256, 1))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (256, 256), (256, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (768, 256), (256, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (256, 768), (768, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (256, 128), (128, 1))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (128, 256), (256, 1))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 256), (256, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, 256), (256, 1))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, 256), (256, 1))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, 256), (256, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (128, 256), (256, 1))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (128, 128), (128, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, 128), (128, 1))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, 128), (128, 1))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, 128), (128, 1))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (256, 128), (128, 1))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (384, 128), (128, 1))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (128, 128), (128, 1))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (384, 128), (128, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (128, 384), (384, 1))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (768, 256), (256, 1))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (256, 256), (256, 1))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (768, 256), (256, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (256, 768), (768, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (768, 256), (256, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (256, 256), (256, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (768, 256), (256, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (256, 768), (768, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (768, 256), (256, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (256, 256), (256, 1))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (768, 256), (256, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (256, 768), (768, 1))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (256, 128), (128, 1))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (128, 256), (256, 1))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (256, 256), (256, 1))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (256, 256), (256, 1))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (256, 256), (256, 1))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_157, (256, 256), (256, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_161, (128, 256), (256, 1))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (128, 128), (128, 1))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (128, 128), (128, 1))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, 128), (128, 1))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, 128), (128, 1))
    assert_size_stride(primals_172, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (256, 128), (128, 1))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (384, 128), (128, 1))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (128, 128), (128, 1))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (384, 128), (128, 1))
    assert_size_stride(primals_186, (384, ), (1, ))
    assert_size_stride(primals_187, (128, 384), (384, 1))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (768, 256), (256, 1))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (256, 256), (256, 1))
    assert_size_stride(primals_194, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (768, 256), (256, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (256, 768), (768, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (256, ), (1, ))
    assert_size_stride(primals_203, (768, 256), (256, 1))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (256, 256), (256, 1))
    assert_size_stride(primals_206, (256, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (768, 256), (256, 1))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (256, 768), (768, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (768, 256), (256, 1))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (256, 256), (256, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (768, 256), (256, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (256, 768), (768, 1))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (256, 128), (128, 1))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (128, 256), (256, 1))
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, 256), (256, 1))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, 256), (256, 1))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, 256), (256, 1))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, 256), (256, 1))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (128, 256), (256, 1))
    assert_size_stride(primals_246, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (128, 128), (128, 1))
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, 128), (128, 1))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, 128), (128, 1))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (128, 128), (128, 1))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (128, ), (1, ))
    assert_size_stride(primals_259, (256, 128), (128, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (1000, 128), (128, 1))
    assert_size_stride(primals_266, (1000, ), (1, ))
    assert_size_stride(primals_267, (1000, 256), (256, 1))
    assert_size_stride(primals_268, (1000, ), (1, ))
    assert_size_stride(primals_269, (8, 3, 240, 240), (172800, 57600, 240, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___patch_embed_0_proj], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_269, primals_5, stride=(12, 12), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 20, 20), (51200, 400, 20, 1))
        buf2 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sub_0.run(buf2, 224, grid=grid(224), stream=stream0)
        buf4 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf4, 224, grid=grid(224), stream=stream0)
        buf6 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf6, 224, grid=grid(224), stream=stream0)
        buf8 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_3.run(buf8, 224, grid=grid(224), stream=stream0)
        buf18 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_0.run(buf18, 224, grid=grid(224), stream=stream0)
        buf20 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf20, 224, grid=grid(224), stream=stream0)
        buf22 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf22, 224, grid=grid(224), stream=stream0)
        buf24 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_3.run(buf24, 224, grid=grid(224), stream=stream0)
        buf9 = empty((8, 3, 224, 224), device='cuda', dtype=torch.float32)
        buf10 = buf9; del buf9  # reuse
        buf25 = buf10; del buf10  # reuse
        # Source Nodes: [x__5], Original ATen: [aten._unsafe_index, aten.add, aten.mul]
        triton_poi_fused__unsafe_index_add_mul_4.run(buf25, primals_269, buf2, buf4, buf6, buf8, buf18, buf20, buf22, buf24, 1204224, grid=grid(1204224), stream=stream0)
        del buf18
        del buf2
        del buf20
        del buf22
        del buf24
        del buf4
        del buf6
        del buf8
        # Source Nodes: [l__mod___patch_embed_1_proj], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_7, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf30 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf31 = empty((3208, 128), device='cuda', dtype=torch.float32)
        buf518 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___attn_qkv, getattr_l__mod___blocks_0_blocks_0___0___norm1, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_red_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_5.run(primals_1, buf0, primals_6, primals_2, primals_9, primals_10, buf30, buf31, buf518, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_10
        buf32 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, buf31, reinterpret_tensor(primals_11, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf32)
        del primals_12
        # Source Nodes: [x_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf33 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf32, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf32, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf32, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        buf36 = buf33[2]
        buf37 = buf33[3]
        del buf33
        buf38 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_13, (128, 128), (1, 128), 0), out=buf38)
        buf39 = reinterpret_tensor(buf38, (8, 401, 128), (51328, 128, 1), 0); del buf38  # reuse
        buf43 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf44 = empty((3208, 128), device='cuda', dtype=torch.float32)
        buf517 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___norm2, x_7, x_8, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_6.run(buf39, primals_1, buf0, primals_6, primals_2, primals_14, primals_15, primals_16, buf43, buf44, buf517, 3208, 128, grid=grid(3208), stream=stream0)
        del buf0
        del primals_1
        del primals_14
        del primals_16
        del primals_2
        del primals_6
        buf45 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_18, buf44, reinterpret_tensor(primals_17, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf45)
        del primals_18
        buf46 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_9], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf45, buf46, 1231872, grid=grid(1231872), stream=stream0)
        buf47 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf46, reinterpret_tensor(primals_19, (384, 128), (1, 384), 0), out=buf47)
        buf48 = empty_strided((8, 197, 1, 2), (394, 2, 3152, 1), device='cuda', dtype=torch.float32)
        buf49 = empty_strided((8, 197, 1, 2), (394, 2, 3152, 1), device='cuda', dtype=torch.float32)
        buf50 = empty_strided((8, 197, 1, 2), (394, 2, 3152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_8.run(primals_3, buf26, primals_8, primals_4, buf48, buf49, buf50, 3152, 128, grid=grid(3152), stream=stream0)
        buf51 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf516 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_9.run(buf48, buf49, buf50, buf51, buf52, buf516, 1576, 2, grid=grid(1576), stream=stream0)
        del buf48
        del buf49
        del buf50
        buf54 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf55 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___attn_qkv, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_cat_native_layer_norm_view_10.run(primals_3, buf26, primals_8, primals_4, buf51, buf52, primals_21, primals_22, buf54, buf55, 403456, grid=grid(403456), stream=stream0)
        del primals_22
        buf56 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_24, buf55, reinterpret_tensor(primals_23, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf56)
        del primals_24
        # Source Nodes: [x_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf57 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf56, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf56, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf56, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf58 = buf57[0]
        buf59 = buf57[1]
        buf60 = buf57[2]
        buf61 = buf57[3]
        del buf57
        buf62 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_25, (256, 256), (1, 256), 0), out=buf62)
        buf63 = reinterpret_tensor(buf62, (8, 197, 256), (50432, 256, 1), 0); del buf62  # reuse
        buf67 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf68 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf515 = reinterpret_tensor(buf52, (8, 197, 1), (197, 1, 1), 0); del buf52  # reuse
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm2, x_19, x_20, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_cat_native_layer_norm_native_layer_norm_backward_view_11.run(buf63, primals_3, buf26, primals_8, primals_4, primals_26, primals_27, primals_28, buf67, buf68, buf515, 1576, 256, grid=grid(1576), stream=stream0)
        del buf26
        del primals_26
        del primals_28
        del primals_3
        del primals_4
        del primals_8
        buf69 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_30, buf68, reinterpret_tensor(primals_29, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf69)
        del primals_30
        buf70 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_24], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf69, buf70, 1210368, grid=grid(1210368), stream=stream0)
        buf71 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf70, reinterpret_tensor(primals_31, (768, 256), (1, 768), 0), out=buf71)
        buf75 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf76 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf514 = reinterpret_tensor(buf51, (8, 197, 1), (197, 1, 1), 0); del buf51  # reuse
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___attn_qkv, getattr_l__mod___blocks_0_blocks_1___1___norm1, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf63, buf71, primals_32, primals_33, primals_34, buf75, buf76, buf514, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_34
        buf77 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_36, buf76, reinterpret_tensor(primals_35, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf77)
        del primals_36
        # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf78 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf77, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf77, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf77, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf79 = buf78[0]
        buf80 = buf78[1]
        buf81 = buf78[2]
        buf82 = buf78[3]
        del buf78
        buf83 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_37, (256, 256), (1, 256), 0), out=buf83)
        buf84 = reinterpret_tensor(buf83, (8, 197, 256), (50432, 256, 1), 0); del buf83  # reuse
        buf88 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf89 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf513 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___norm2, x_26, x_31, x_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf84, buf63, buf71, primals_32, primals_38, primals_39, primals_40, buf88, buf89, buf513, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_32
        del primals_38
        del primals_40
        buf90 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, buf89, reinterpret_tensor(primals_41, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf90)
        del primals_42
        buf91 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf90, buf91, 1210368, grid=grid(1210368), stream=stream0)
        buf92 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf91, reinterpret_tensor(primals_43, (768, 256), (1, 768), 0), out=buf92)
        buf96 = buf63; del buf63  # reuse
        buf97 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf512 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___attn_qkv, getattr_l__mod___blocks_0_blocks_1___2___norm1, x_38], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf84, buf92, primals_44, primals_45, primals_46, buf96, buf97, buf512, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_46
        buf98 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, buf97, reinterpret_tensor(primals_47, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf98)
        del primals_48
        # Source Nodes: [x_39], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf99 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf98, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf98, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf98, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf100 = buf99[0]
        buf101 = buf99[1]
        buf102 = buf99[2]
        buf103 = buf99[3]
        del buf99
        buf104 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf100, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_49, (256, 256), (1, 256), 0), out=buf104)
        buf105 = reinterpret_tensor(buf104, (8, 197, 256), (50432, 256, 1), 0); del buf104  # reuse
        buf109 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf110 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf511 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___norm2, x_38, x_43, x_44], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf105, buf84, buf92, primals_44, primals_50, primals_51, primals_52, buf109, buf110, buf511, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_44
        del primals_50
        del primals_52
        buf111 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_54, buf110, reinterpret_tensor(primals_53, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf111)
        del primals_54
        buf112 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_48], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf111, buf112, 1210368, grid=grid(1210368), stream=stream0)
        buf113 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf112, reinterpret_tensor(primals_55, (768, 256), (1, 768), 0), out=buf113)
        buf117 = empty((8, 1, 128), device='cuda', dtype=torch.float32)
        buf118 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf510 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1, l__mod___blocks_0_projs_0_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_15.run(buf39, buf47, primals_20, primals_57, primals_58, buf117, buf118, buf510, 8, 128, grid=grid(8), stream=stream0)
        buf119 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_0_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_60, buf118, reinterpret_tensor(primals_59, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf119)
        del primals_60
        buf123 = empty((8, 1, 256), device='cuda', dtype=torch.float32)
        buf124 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf509 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1, l__mod___blocks_0_projs_1_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_16.run(buf105, buf113, primals_56, primals_61, primals_62, buf123, buf124, buf509, 8, 256, grid=grid(8), stream=stream0)
        buf125 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_1_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, buf124, reinterpret_tensor(primals_63, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf125)
        del primals_64
        buf126 = buf84; del buf84  # reuse
        buf127 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf128 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf130 = reinterpret_tensor(buf128, (8, 197, 1), (197, 1, 1), 0); del buf128  # reuse
        buf131 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_25, l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_17.run(buf130, buf119, buf105, buf113, primals_56, primals_65, primals_66, buf126, buf127, buf131, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_66
        buf132 = buf119; del buf119  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (8, 256), (50432, 1), 0), reinterpret_tensor(primals_67, (256, 256), (1, 256), 0), out=buf132)
        buf133 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_69, (256, 256), (1, 256), 0), out=buf133)
        buf134 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_71, (256, 256), (1, 256), 0), out=buf134)
        buf135 = reinterpret_tensor(buf132, (8, 1, 256), (256, 256, 1), 0); del buf132  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf135, primals_68, 2048, grid=grid(2048), stream=stream0)
        del primals_68
        buf136 = empty((8, 4, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf133, primals_70, buf136, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del primals_70
        buf137 = empty((32, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf136, (32, 64, 197), (12608, 197, 1), 0), out=buf137)
        buf140 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        buf508 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, attn_2], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_20.run(buf137, buf140, buf508, 32, 197, grid=grid(32), stream=stream0)
        buf141 = reinterpret_tensor(buf133, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf134, primals_72, buf141, 403456, grid=grid(403456), stream=stream0)
        del primals_72
        buf142 = empty((32, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf140, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf141, (32, 197, 64), (12608, 64, 1), 0), out=buf142)
        buf143 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (8, 256), (256, 1), 0), reinterpret_tensor(primals_73, (256, 256), (1, 256), 0), out=buf143)
        buf147 = empty((8, 1, 256), device='cuda', dtype=torch.float32)
        buf148 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf507 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_revert_projs_0_0, l__mod___blocks_0_revert_projs_0_1, reverted_proj_cls_token, tmp_1], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_22.run(buf126, buf143, primals_74, primals_75, primals_76, buf147, buf148, buf507, 8, 256, grid=grid(8), stream=stream0)
        del primals_74
        buf149 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [reverted_proj_cls_token], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_78, buf148, reinterpret_tensor(primals_77, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf149)
        del primals_78
        buf150 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf151 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf152 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        buf153 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf176 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        buf177 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf155 = reinterpret_tensor(buf153, (8, 401, 1), (401, 1, 1), 0); del buf153  # reuse
        buf179 = reinterpret_tensor(buf177, (8, 401, 1), (401, 1, 1), 0); del buf177  # reuse
        buf156 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf180 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, cat_24, getattr_l__mod___blocks_1_blocks_0___0___attn_qkv, getattr_l__mod___blocks_1_blocks_0___0___norm1, l__mod___blocks_0_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_23.run(buf155, buf179, buf149, buf39, buf47, primals_20, buf125, primals_79, primals_80, primals_93, primals_94, buf150, buf151, buf152, buf176, buf156, buf180, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_20
        del primals_80
        del primals_94
        buf157 = buf149; del buf149  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (8, 128), (51328, 1), 0), reinterpret_tensor(primals_81, (128, 128), (1, 128), 0), out=buf157)
        buf158 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_83, (128, 128), (1, 128), 0), out=buf158)
        buf159 = reinterpret_tensor(buf39, (3208, 128), (128, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_85, (128, 128), (1, 128), 0), out=buf159)
        buf160 = reinterpret_tensor(buf157, (8, 1, 128), (128, 128, 1), 0); del buf157  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_24.run(buf160, primals_82, 1024, grid=grid(1024), stream=stream0)
        del primals_82
        buf161 = empty((8, 4, 32, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf158, primals_84, buf161, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del primals_84
        buf162 = empty((32, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf161, (32, 32, 401), (12832, 401, 1), 0), out=buf162)
        buf165 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        buf506 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4, attn_5], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf162, buf165, buf506, 32, 401, grid=grid(32), stream=stream0)
        buf166 = reinterpret_tensor(buf158, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf158  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf159, primals_86, buf166, 410624, grid=grid(410624), stream=stream0)
        del primals_86
        buf167 = reinterpret_tensor(buf125, (32, 1, 32), (32, 32, 1), 0); del buf125  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf165, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf166, (32, 401, 32), (12832, 32, 1), 0), out=buf167)
        buf168 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (8, 128), (128, 1), 0), reinterpret_tensor(primals_87, (128, 128), (1, 128), 0), out=buf168)
        buf172 = empty((8, 1, 128), device='cuda', dtype=torch.float32)
        buf173 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf505 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_revert_projs_1_0, l__mod___blocks_0_revert_projs_1_1, reverted_proj_cls_token_1, tmp_4], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_28.run(buf151, buf168, primals_88, primals_89, primals_90, buf172, buf173, buf505, 8, 128, grid=grid(8), stream=stream0)
        del primals_88
        buf174 = buf143; del buf143  # reuse
        # Source Nodes: [reverted_proj_cls_token_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf173, reinterpret_tensor(primals_91, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf174)
        del primals_92
        buf175 = reinterpret_tensor(buf134, (8, 197, 256), (50432, 256, 1), 0); del buf134  # reuse
        buf197 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf200 = reinterpret_tensor(buf198, (8, 197, 1), (197, 1, 1), 0); del buf198  # reuse
        buf201 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_22, getattr_l__mod___blocks_1_blocks_1___0___attn_qkv, getattr_l__mod___blocks_1_blocks_1___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_17.run(buf200, buf174, buf105, buf113, primals_56, primals_105, primals_106, buf175, buf197, buf201, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_106
        del primals_56
        buf181 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_96, buf180, reinterpret_tensor(primals_95, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf181)
        del primals_96
        # Source Nodes: [x_59], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf182 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf181, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf181, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf181, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, True)
        buf183 = buf182[0]
        buf184 = buf182[1]
        buf185 = buf182[2]
        buf186 = buf182[3]
        del buf182
        buf187 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_97, (128, 128), (1, 128), 0), out=buf187)
        buf191 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf192 = empty((3208, 128), device='cuda', dtype=torch.float32)
        buf504 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___norm2, x_63, x_64], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29.run(buf150, buf187, primals_98, primals_99, primals_100, buf191, buf192, buf504, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_100
        buf193 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, buf192, reinterpret_tensor(primals_101, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf193)
        del primals_102
        buf194 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65, x_68], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf193, buf194, 1231872, grid=grid(1231872), stream=stream0)
        buf195 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf194, reinterpret_tensor(primals_103, (384, 128), (1, 384), 0), out=buf195)
        buf196 = reinterpret_tensor(buf195, (8, 401, 128), (51328, 128, 1), 0); del buf195  # reuse
        # Source Nodes: [x_63, x_70], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf196, buf150, buf187, primals_98, primals_104, 410624, grid=grid(410624), stream=stream0)
        del primals_104
        del primals_98
        buf202 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_108, buf201, reinterpret_tensor(primals_107, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf202)
        del primals_108
        # Source Nodes: [x_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf203 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf202, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf202, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf202, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf204 = buf203[0]
        buf205 = buf203[1]
        buf206 = buf203[2]
        buf207 = buf203[3]
        del buf203
        buf208 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_109, (256, 256), (1, 256), 0), out=buf208)
        buf212 = buf105; del buf105  # reuse
        buf213 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf503 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___norm2, x_75, x_76], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf175, buf208, primals_110, primals_111, primals_112, buf212, buf213, buf503, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_112
        buf214 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_114, buf213, reinterpret_tensor(primals_113, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf214)
        del primals_114
        buf215 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77, x_80], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf214, buf215, 1210368, grid=grid(1210368), stream=stream0)
        buf216 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf215, reinterpret_tensor(primals_115, (768, 256), (1, 768), 0), out=buf216)
        buf217 = reinterpret_tensor(buf216, (8, 197, 256), (50432, 256, 1), 0); del buf216  # reuse
        buf221 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf222 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf502 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___attn_qkv, getattr_l__mod___blocks_1_blocks_1___1___norm1, x_75, x_82], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf217, buf175, buf208, primals_110, primals_116, primals_117, primals_118, buf221, buf222, buf502, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_110
        del primals_116
        del primals_118
        buf223 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf222, reinterpret_tensor(primals_119, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf223)
        del primals_120
        # Source Nodes: [x_83], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf224 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf223, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf223, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf223, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf225 = buf224[0]
        buf226 = buf224[1]
        buf227 = buf224[2]
        buf228 = buf224[3]
        del buf224
        buf229 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf225, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_121, (256, 256), (1, 256), 0), out=buf229)
        buf233 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf234 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf501 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___norm2, x_87, x_88], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf217, buf229, primals_122, primals_123, primals_124, buf233, buf234, buf501, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_124
        buf235 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, buf234, reinterpret_tensor(primals_125, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf235)
        del primals_126
        buf236 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89, x_92], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf235, buf236, 1210368, grid=grid(1210368), stream=stream0)
        buf237 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf236, reinterpret_tensor(primals_127, (768, 256), (1, 768), 0), out=buf237)
        buf238 = reinterpret_tensor(buf237, (8, 197, 256), (50432, 256, 1), 0); del buf237  # reuse
        buf242 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf243 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf500 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___attn_qkv, getattr_l__mod___blocks_1_blocks_1___2___norm1, x_87, x_94], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf238, buf217, buf229, primals_122, primals_128, primals_129, primals_130, buf242, buf243, buf500, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_122
        del primals_128
        del primals_130
        buf244 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_132, buf243, reinterpret_tensor(primals_131, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf244)
        del primals_132
        # Source Nodes: [x_95], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf245 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf244, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf244, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf244, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf246 = buf245[0]
        buf247 = buf245[1]
        buf248 = buf245[2]
        buf249 = buf245[3]
        del buf245
        buf250 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_133, (256, 256), (1, 256), 0), out=buf250)
        buf254 = buf217; del buf217  # reuse
        buf255 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf499 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___norm2, x_100, x_99], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf238, buf250, primals_134, primals_135, primals_136, buf254, buf255, buf499, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_136
        buf256 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_138, buf255, reinterpret_tensor(primals_137, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf256)
        del primals_138
        buf257 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_104], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf256, buf257, 1210368, grid=grid(1210368), stream=stream0)
        buf258 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf257, reinterpret_tensor(primals_139, (768, 256), (1, 768), 0), out=buf258)
        buf263 = reinterpret_tensor(buf168, (8, 1, 128), (128, 128, 1), 0); del buf168  # reuse
        buf264 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf498 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_projs_0_0, l__mod___blocks_1_projs_0_1, l__mod___blocks_1_projs_0_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_31.run(buf196, primals_141, primals_142, buf263, buf264, buf498, 8, 128, grid=grid(8), stream=stream0)
        buf265 = buf174; del buf174  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_0_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_144, buf264, reinterpret_tensor(primals_143, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf265)
        del primals_144
        buf259 = reinterpret_tensor(buf258, (8, 197, 256), (50432, 256, 1), 0); del buf258  # reuse
        buf272 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf273 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf274 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf276 = reinterpret_tensor(buf274, (8, 197, 1), (197, 1, 1), 0); del buf274  # reuse
        buf277 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_21, l__mod___blocks_1_fusion_0_norm1, x_106, x_99], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_32.run(buf259, buf276, buf238, buf250, primals_134, primals_140, buf265, primals_149, primals_150, buf272, buf273, buf277, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_134
        del primals_140
        del primals_150
        buf269 = reinterpret_tensor(buf265, (8, 1, 256), (256, 256, 1), 0); del buf265  # reuse
        buf270 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf497 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_projs_1_0, l__mod___blocks_1_projs_1_1, l__mod___blocks_1_projs_1_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_33.run(buf259, primals_145, primals_146, buf269, buf270, buf497, 8, 256, grid=grid(8), stream=stream0)
        buf271 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_projs_1_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_148, buf270, reinterpret_tensor(primals_147, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf271)
        del primals_148
        buf278 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (8, 256), (50432, 1), 0), reinterpret_tensor(primals_151, (256, 256), (1, 256), 0), out=buf278)
        buf279 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_153, (256, 256), (1, 256), 0), out=buf279)
        buf280 = reinterpret_tensor(buf238, (1576, 256), (256, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_155, (256, 256), (1, 256), 0), out=buf280)
        buf281 = reinterpret_tensor(buf278, (8, 1, 256), (256, 256, 1), 0); del buf278  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf281, primals_152, 2048, grid=grid(2048), stream=stream0)
        del primals_152
        buf282 = empty((8, 4, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf279, primals_154, buf282, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del primals_154
        buf283 = buf137; del buf137  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf281, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf282, (32, 64, 197), (12608, 197, 1), 0), out=buf283)
        buf286 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        buf496 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7, attn_8], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_20.run(buf283, buf286, buf496, 32, 197, grid=grid(32), stream=stream0)
        buf287 = reinterpret_tensor(buf279, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf279  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf280, primals_156, buf287, 403456, grid=grid(403456), stream=stream0)
        del primals_156
        buf288 = empty((32, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf286, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf287, (32, 197, 64), (12608, 64, 1), 0), out=buf288)
        buf289 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (8, 256), (256, 1), 0), reinterpret_tensor(primals_157, (256, 256), (1, 256), 0), out=buf289)
        buf293 = empty((8, 1, 256), device='cuda', dtype=torch.float32)
        buf294 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf495 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_revert_projs_0_0, l__mod___blocks_1_revert_projs_0_1, reverted_proj_cls_token_2, tmp_7], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_22.run(buf272, buf289, primals_158, primals_159, primals_160, buf293, buf294, buf495, 8, 256, grid=grid(8), stream=stream0)
        del primals_158
        buf295 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [reverted_proj_cls_token_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_162, buf294, reinterpret_tensor(primals_161, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf295)
        del primals_162
        buf296 = reinterpret_tensor(buf187, (8, 401, 128), (51328, 128, 1), 0); del buf187  # reuse
        buf297 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf298 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        buf299 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf322 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        buf323 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf301 = reinterpret_tensor(buf299, (8, 401, 1), (401, 1, 1), 0); del buf299  # reuse
        buf325 = reinterpret_tensor(buf323, (8, 401, 1), (401, 1, 1), 0); del buf323  # reuse
        buf302 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf326 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_19, cat_20, getattr_l__mod___blocks_2_blocks_0___0___attn_qkv, getattr_l__mod___blocks_2_blocks_0___0___norm1, l__mod___blocks_1_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_34.run(buf301, buf325, buf295, buf196, buf271, primals_163, primals_164, primals_177, primals_178, buf296, buf297, buf298, buf322, buf302, buf326, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_164
        del primals_178
        buf303 = buf295; del buf295  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (8, 128), (51328, 1), 0), reinterpret_tensor(primals_165, (128, 128), (1, 128), 0), out=buf303)
        buf304 = reinterpret_tensor(buf196, (3208, 128), (128, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_167, (128, 128), (1, 128), 0), out=buf304)
        buf305 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_169, (128, 128), (1, 128), 0), out=buf305)
        buf306 = reinterpret_tensor(buf303, (8, 1, 128), (128, 128, 1), 0); del buf303  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_24.run(buf306, primals_166, 1024, grid=grid(1024), stream=stream0)
        del primals_166
        buf307 = empty((8, 4, 32, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf304, primals_168, buf307, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del primals_168
        buf308 = buf162; del buf162  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf307, (32, 32, 401), (12832, 401, 1), 0), out=buf308)
        buf311 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        buf494 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_11, attn_9], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf308, buf311, buf494, 32, 401, grid=grid(32), stream=stream0)
        buf312 = reinterpret_tensor(buf304, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf304  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf305, primals_170, buf312, 410624, grid=grid(410624), stream=stream0)
        del primals_170
        buf313 = reinterpret_tensor(buf271, (32, 1, 32), (32, 32, 1), 0); del buf271  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf311, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf312, (32, 401, 32), (12832, 32, 1), 0), out=buf313)
        buf314 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf313, (8, 128), (128, 1), 0), reinterpret_tensor(primals_171, (128, 128), (1, 128), 0), out=buf314)
        buf318 = empty((8, 1, 128), device='cuda', dtype=torch.float32)
        buf319 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf493 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_revert_projs_1_0, l__mod___blocks_1_revert_projs_1_1, reverted_proj_cls_token_3, tmp_10], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_28.run(buf297, buf314, primals_172, primals_173, primals_174, buf318, buf319, buf493, 8, 128, grid=grid(8), stream=stream0)
        del primals_172
        buf320 = buf289; del buf289  # reuse
        # Source Nodes: [reverted_proj_cls_token_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf319, reinterpret_tensor(primals_175, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf320)
        del primals_176
        buf321 = reinterpret_tensor(buf280, (8, 197, 256), (50432, 256, 1), 0); del buf280  # reuse
        buf343 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf344 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf346 = reinterpret_tensor(buf344, (8, 197, 1), (197, 1, 1), 0); del buf344  # reuse
        buf347 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_18, getattr_l__mod___blocks_2_blocks_1___0___attn_qkv, getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm, aten.view]
        triton_per_fused_cat_native_layer_norm_view_35.run(buf346, buf320, buf259, primals_189, primals_190, buf321, buf343, buf347, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_190
        buf327 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_180, buf326, reinterpret_tensor(primals_179, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf327)
        del primals_180
        # Source Nodes: [x_115], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf328 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf327, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf327, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf327, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, True)
        buf329 = buf328[0]
        buf330 = buf328[1]
        buf331 = buf328[2]
        buf332 = buf328[3]
        del buf328
        buf333 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_181, (128, 128), (1, 128), 0), out=buf333)
        buf337 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf338 = empty((3208, 128), device='cuda', dtype=torch.float32)
        buf492 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___norm2, x_119, x_120], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_29.run(buf296, buf333, primals_182, primals_183, primals_184, buf337, buf338, buf492, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_184
        buf339 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_186, buf338, reinterpret_tensor(primals_185, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf339)
        del primals_186
        buf340 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121, x_124], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf339, buf340, 1231872, grid=grid(1231872), stream=stream0)
        buf341 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf340, reinterpret_tensor(primals_187, (384, 128), (1, 384), 0), out=buf341)
        buf342 = reinterpret_tensor(buf341, (8, 401, 128), (51328, 128, 1), 0); del buf341  # reuse
        # Source Nodes: [x_119, x_126], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf342, buf296, buf333, primals_182, primals_188, 410624, grid=grid(410624), stream=stream0)
        del primals_182
        del primals_188
        buf348 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_192, buf347, reinterpret_tensor(primals_191, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf348)
        del primals_192
        # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf349 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf348, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf348, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf348, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf350 = buf349[0]
        buf351 = buf349[1]
        buf352 = buf349[2]
        buf353 = buf349[3]
        del buf349
        buf354 = reinterpret_tensor(buf259, (1576, 256), (256, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_193, (256, 256), (1, 256), 0), out=buf354)
        buf358 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf359 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf491 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___norm2, x_131, x_132], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf321, buf354, primals_194, primals_195, primals_196, buf358, buf359, buf491, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_196
        buf360 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_198, buf359, reinterpret_tensor(primals_197, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf360)
        del primals_198
        buf361 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133, x_136], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf360, buf361, 1210368, grid=grid(1210368), stream=stream0)
        buf362 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf361, reinterpret_tensor(primals_199, (768, 256), (1, 768), 0), out=buf362)
        buf363 = reinterpret_tensor(buf362, (8, 197, 256), (50432, 256, 1), 0); del buf362  # reuse
        buf367 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf368 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf490 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___attn_qkv, getattr_l__mod___blocks_2_blocks_1___1___norm1, x_131, x_138], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf363, buf321, buf354, primals_194, primals_200, primals_201, primals_202, buf367, buf368, buf490, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_194
        del primals_200
        del primals_202
        buf369 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_204, buf368, reinterpret_tensor(primals_203, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf369)
        del primals_204
        # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf370 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf369, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf369, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf369, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf371 = buf370[0]
        buf372 = buf370[1]
        buf373 = buf370[2]
        buf374 = buf370[3]
        del buf370
        buf375 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf371, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_205, (256, 256), (1, 256), 0), out=buf375)
        buf379 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf380 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf489 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___norm2, x_143, x_144], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf363, buf375, primals_206, primals_207, primals_208, buf379, buf380, buf489, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_208
        buf381 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_210, buf380, reinterpret_tensor(primals_209, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf381)
        del primals_210
        buf382 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145, x_148], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf381, buf382, 1210368, grid=grid(1210368), stream=stream0)
        buf383 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf382, reinterpret_tensor(primals_211, (768, 256), (1, 768), 0), out=buf383)
        buf384 = reinterpret_tensor(buf383, (8, 197, 256), (50432, 256, 1), 0); del buf383  # reuse
        buf388 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf389 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf488 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___attn_qkv, getattr_l__mod___blocks_2_blocks_1___2___norm1, x_143, x_150], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_14.run(buf384, buf363, buf375, primals_206, primals_212, primals_213, primals_214, buf388, buf389, buf488, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_206
        del primals_212
        del primals_214
        buf390 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_216, buf389, reinterpret_tensor(primals_215, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf390)
        del primals_216
        # Source Nodes: [x_151], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf391 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf390, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf390, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf390, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, True)
        buf392 = buf391[0]
        buf393 = buf391[1]
        buf394 = buf391[2]
        buf395 = buf391[3]
        del buf391
        buf396 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_217, (256, 256), (1, 256), 0), out=buf396)
        buf400 = buf363; del buf363  # reuse
        buf401 = empty((1576, 256), device='cuda', dtype=torch.float32)
        buf487 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___norm2, x_155, x_156], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_13.run(buf384, buf396, primals_218, primals_219, primals_220, buf400, buf401, buf487, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_220
        buf402 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_222, buf401, reinterpret_tensor(primals_221, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf402)
        del primals_222
        buf403 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157, x_160], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf402, buf403, 1210368, grid=grid(1210368), stream=stream0)
        buf404 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf403, reinterpret_tensor(primals_223, (768, 256), (1, 768), 0), out=buf404)
        buf409 = reinterpret_tensor(buf314, (8, 1, 128), (128, 128, 1), 0); del buf314  # reuse
        buf410 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf486 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1, l__mod___blocks_2_projs_0_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_31.run(buf342, primals_225, primals_226, buf409, buf410, buf486, 8, 128, grid=grid(8), stream=stream0)
        buf411 = buf320; del buf320  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_0_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_228, buf410, reinterpret_tensor(primals_227, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf411)
        del primals_228
        buf405 = reinterpret_tensor(buf404, (8, 197, 256), (50432, 256, 1), 0); del buf404  # reuse
        buf418 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf419 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf420 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf422 = reinterpret_tensor(buf420, (8, 197, 1), (197, 1, 1), 0); del buf420  # reuse
        buf423 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_17, l__mod___blocks_2_fusion_0_norm1, x_155, x_162], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_32.run(buf405, buf422, buf384, buf396, primals_218, primals_224, buf411, primals_233, primals_234, buf418, buf419, buf423, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_218
        del primals_224
        del primals_234
        buf415 = reinterpret_tensor(buf411, (8, 1, 256), (256, 256, 1), 0); del buf411  # reuse
        buf416 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf485 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1, l__mod___blocks_2_projs_1_2], Original ATen: [aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_33.run(buf405, primals_229, primals_230, buf415, buf416, buf485, 8, 256, grid=grid(8), stream=stream0)
        buf417 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_projs_1_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_232, buf416, reinterpret_tensor(primals_231, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf417)
        del primals_232
        buf424 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (8, 256), (50432, 1), 0), reinterpret_tensor(primals_235, (256, 256), (1, 256), 0), out=buf424)
        buf425 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_237, (256, 256), (1, 256), 0), out=buf425)
        buf426 = reinterpret_tensor(buf384, (1576, 256), (256, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (1576, 256), (256, 1), 0), reinterpret_tensor(primals_239, (256, 256), (1, 256), 0), out=buf426)
        buf427 = reinterpret_tensor(buf424, (8, 1, 256), (256, 256, 1), 0); del buf424  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf427, primals_236, 2048, grid=grid(2048), stream=stream0)
        del primals_236
        buf428 = empty((8, 4, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf425, primals_238, buf428, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del primals_238
        buf429 = buf283; del buf283  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf427, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf428, (32, 64, 197), (12608, 197, 1), 0), out=buf429)
        buf432 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        buf484 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13, attn_14], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_20.run(buf429, buf432, buf484, 32, 197, grid=grid(32), stream=stream0)
        del buf429
        buf433 = reinterpret_tensor(buf425, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf425  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf426, primals_240, buf433, 403456, grid=grid(403456), stream=stream0)
        del primals_240
        buf434 = empty((32, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf433, (32, 197, 64), (12608, 64, 1), 0), out=buf434)
        buf435 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf434, (8, 256), (256, 1), 0), reinterpret_tensor(primals_241, (256, 256), (1, 256), 0), out=buf435)
        buf439 = empty((8, 1, 256), device='cuda', dtype=torch.float32)
        buf440 = empty((8, 256), device='cuda', dtype=torch.float32)
        buf483 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_revert_projs_0_0, l__mod___blocks_2_revert_projs_0_1, reverted_proj_cls_token_4, tmp_13], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_22.run(buf418, buf435, primals_242, primals_243, primals_244, buf439, buf440, buf483, 8, 256, grid=grid(8), stream=stream0)
        del primals_242
        buf441 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [reverted_proj_cls_token_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_246, buf440, reinterpret_tensor(primals_245, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf441)
        del primals_246
        buf442 = reinterpret_tensor(buf333, (8, 401, 128), (51328, 128, 1), 0); del buf333  # reuse
        buf443 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf444 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        buf445 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf468 = empty((8, 401, 1), device='cuda', dtype=torch.float32)
        buf469 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf447 = reinterpret_tensor(buf445, (8, 401, 1), (401, 1, 1), 0); del buf445  # reuse
        buf471 = reinterpret_tensor(buf469, (8, 401, 1), (401, 1, 1), 0); del buf469  # reuse
        buf448 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_15, cat_16, l__mod___blocks_2_fusion_1_norm1, x_171], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_36.run(buf447, buf471, buf441, buf342, buf417, primals_247, primals_248, buf442, buf443, buf444, buf468, buf448, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_248
        buf449 = buf441; del buf441  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (8, 128), (51328, 1), 0), reinterpret_tensor(primals_249, (128, 128), (1, 128), 0), out=buf449)
        buf450 = reinterpret_tensor(buf342, (3208, 128), (128, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf448, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_251, (128, 128), (1, 128), 0), out=buf450)
        buf451 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf448, (3208, 128), (128, 1), 0), reinterpret_tensor(primals_253, (128, 128), (1, 128), 0), out=buf451)
        buf452 = reinterpret_tensor(buf449, (8, 1, 128), (128, 128, 1), 0); del buf449  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_24.run(buf452, primals_250, 1024, grid=grid(1024), stream=stream0)
        del primals_250
        buf453 = empty((8, 4, 32, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf450, primals_252, buf453, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del primals_252
        buf454 = buf308; del buf308  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf452, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf453, (32, 32, 401), (12832, 401, 1), 0), out=buf454)
        buf457 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        buf482 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_15, attn_16, attn_17], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_26.run(buf454, buf457, buf482, 32, 401, grid=grid(32), stream=stream0)
        del buf454
        buf458 = reinterpret_tensor(buf450, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf450  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf451, primals_254, buf458, 410624, grid=grid(410624), stream=stream0)
        del buf451
        del primals_254
        buf459 = reinterpret_tensor(buf417, (32, 1, 32), (32, 32, 1), 0); del buf417  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf457, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf458, (32, 401, 32), (12832, 32, 1), 0), out=buf459)
        buf460 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (8, 128), (128, 1), 0), reinterpret_tensor(primals_255, (128, 128), (1, 128), 0), out=buf460)
        buf464 = empty((8, 1, 128), device='cuda', dtype=torch.float32)
        buf465 = empty((8, 128), device='cuda', dtype=torch.float32)
        buf481 = empty((8, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1, reverted_proj_cls_token_5, tmp_16], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_gelu_native_layer_norm_native_layer_norm_backward_view_28.run(buf443, buf460, primals_256, primals_257, primals_258, buf464, buf465, buf481, 8, 128, grid=grid(8), stream=stream0)
        del primals_256
        buf466 = buf435; del buf435  # reuse
        # Source Nodes: [reverted_proj_cls_token_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_260, buf465, reinterpret_tensor(primals_259, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf466)
        del primals_260
        buf467 = reinterpret_tensor(buf426, (8, 197, 256), (50432, 256, 1), 0); del buf426  # reuse
        buf472 = empty((8, 197, 1), device='cuda', dtype=torch.float32)
        buf473 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf475 = reinterpret_tensor(buf473, (8, 197, 1), (197, 1, 1), 0); del buf473  # reuse
        # Source Nodes: [cat_14, x_172], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_37.run(buf475, buf466, buf405, buf467, buf472, 1576, 256, grid=grid(1576), stream=stream0)
        del buf405
        buf476 = buf460; del buf460  # reuse
        # Source Nodes: [l__mod___head_drop], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf442, buf468, buf471, primals_261, primals_262, buf476, 1024, grid=grid(1024), stream=stream0)
        del primals_262
        buf477 = buf466; del buf466  # reuse
        # Source Nodes: [l__mod___head_drop_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf467, buf472, buf475, primals_263, primals_264, buf477, 2048, grid=grid(2048), stream=stream0)
        del primals_264
        buf478 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_266, buf476, reinterpret_tensor(primals_265, (128, 1000), (1, 128), 0), alpha=1, beta=1, out=buf478)
        del primals_266
        buf479 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_268, buf477, reinterpret_tensor(primals_267, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf479)
        del primals_268
        buf480 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.mean]
        triton_poi_fused_mean_40.run(buf478, buf479, buf480, 8000, grid=grid(8000), stream=stream0)
        return (buf480, primals_5, primals_7, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, primals_269, buf25, buf30, buf31, reinterpret_tensor(buf32, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf32, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf32, (8, 4, 401, 32), (153984, 32, 384, 1), 256), buf35, buf36, buf37, reinterpret_tensor(buf34, (3208, 128), (128, 1), 0), buf43, buf44, buf45, buf46, buf54, buf55, reinterpret_tensor(buf56, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf56, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf56, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf59, buf60, buf61, reinterpret_tensor(buf58, (1576, 256), (256, 1), 0), buf67, buf68, buf69, buf70, buf75, buf76, reinterpret_tensor(buf77, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf77, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf77, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf80, buf81, buf82, reinterpret_tensor(buf79, (1576, 256), (256, 1), 0), buf88, buf89, buf90, buf91, buf96, buf97, reinterpret_tensor(buf98, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf98, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf98, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf101, buf102, buf103, reinterpret_tensor(buf100, (1576, 256), (256, 1), 0), buf109, buf110, buf111, buf112, buf117, buf118, buf123, buf124, buf126, buf127, buf130, reinterpret_tensor(buf131, (8, 256), (50432, 1), 0), reinterpret_tensor(buf131, (1576, 256), (256, 1), 0), reinterpret_tensor(buf142, (8, 256), (256, 1), 0), buf147, buf148, buf150, buf151, buf152, buf155, reinterpret_tensor(buf156, (8, 128), (51328, 1), 0), reinterpret_tensor(buf156, (3208, 128), (128, 1), 0), reinterpret_tensor(buf167, (8, 128), (128, 1), 0), buf172, buf173, buf175, buf176, buf179, buf180, reinterpret_tensor(buf181, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf181, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf181, (8, 4, 401, 32), (153984, 32, 384, 1), 256), buf184, buf185, buf186, reinterpret_tensor(buf183, (3208, 128), (128, 1), 0), buf191, buf192, buf193, buf194, buf197, buf200, buf201, reinterpret_tensor(buf202, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf202, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf202, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf205, buf206, buf207, reinterpret_tensor(buf204, (1576, 256), (256, 1), 0), buf212, buf213, buf214, buf215, buf221, buf222, reinterpret_tensor(buf223, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf223, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf223, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf226, buf227, buf228, reinterpret_tensor(buf225, (1576, 256), (256, 1), 0), buf233, buf234, buf235, buf236, buf242, buf243, reinterpret_tensor(buf244, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf244, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf244, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf247, buf248, buf249, reinterpret_tensor(buf246, (1576, 256), (256, 1), 0), buf254, buf255, buf256, buf257, buf263, buf264, buf269, buf270, buf272, buf273, buf276, reinterpret_tensor(buf277, (8, 256), (50432, 1), 0), reinterpret_tensor(buf277, (1576, 256), (256, 1), 0), reinterpret_tensor(buf288, (8, 256), (256, 1), 0), buf293, buf294, buf296, buf297, buf298, buf301, reinterpret_tensor(buf302, (8, 128), (51328, 1), 0), reinterpret_tensor(buf302, (3208, 128), (128, 1), 0), reinterpret_tensor(buf313, (8, 128), (128, 1), 0), buf318, buf319, buf321, buf322, buf325, buf326, reinterpret_tensor(buf327, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf327, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf327, (8, 4, 401, 32), (153984, 32, 384, 1), 256), buf330, buf331, buf332, reinterpret_tensor(buf329, (3208, 128), (128, 1), 0), buf337, buf338, buf339, buf340, buf343, buf346, buf347, reinterpret_tensor(buf348, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf348, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf348, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf351, buf352, buf353, reinterpret_tensor(buf350, (1576, 256), (256, 1), 0), buf358, buf359, buf360, buf361, buf367, buf368, reinterpret_tensor(buf369, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf369, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf369, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf372, buf373, buf374, reinterpret_tensor(buf371, (1576, 256), (256, 1), 0), buf379, buf380, buf381, buf382, buf388, buf389, reinterpret_tensor(buf390, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf390, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf390, (8, 4, 197, 64), (151296, 64, 768, 1), 512), buf393, buf394, buf395, reinterpret_tensor(buf392, (1576, 256), (256, 1), 0), buf400, buf401, buf402, buf403, buf409, buf410, buf415, buf416, buf418, buf419, buf422, reinterpret_tensor(buf423, (8, 256), (50432, 1), 0), reinterpret_tensor(buf423, (1576, 256), (256, 1), 0), reinterpret_tensor(buf434, (8, 256), (256, 1), 0), buf439, buf440, buf442, buf443, buf444, buf447, reinterpret_tensor(buf448, (8, 128), (51328, 1), 0), reinterpret_tensor(buf448, (3208, 128), (128, 1), 0), reinterpret_tensor(buf459, (8, 128), (128, 1), 0), buf464, buf465, buf467, buf468, buf471, buf472, buf475, buf476, buf477, reinterpret_tensor(primals_267, (1000, 256), (256, 1), 0), reinterpret_tensor(primals_265, (1000, 128), (128, 1), 0), reinterpret_tensor(primals_259, (256, 128), (128, 1), 0), buf481, reinterpret_tensor(primals_255, (128, 128), (128, 1), 0), reinterpret_tensor(buf457, (32, 401, 1), (401, 1, 0), 0), reinterpret_tensor(buf458, (32, 32, 401), (12832, 1, 32), 0), buf482, reinterpret_tensor(buf452, (32, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf453, (32, 401, 32), (12832, 1, 401), 0), reinterpret_tensor(primals_253, (128, 128), (128, 1), 0), reinterpret_tensor(primals_251, (128, 128), (128, 1), 0), reinterpret_tensor(primals_249, (128, 128), (128, 1), 0), reinterpret_tensor(primals_245, (128, 256), (256, 1), 0), buf483, reinterpret_tensor(primals_241, (256, 256), (256, 1), 0), reinterpret_tensor(buf432, (32, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf433, (32, 64, 197), (12608, 1, 64), 0), buf484, reinterpret_tensor(buf427, (32, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf428, (32, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_239, (256, 256), (256, 1), 0), reinterpret_tensor(primals_237, (256, 256), (256, 1), 0), reinterpret_tensor(primals_235, (256, 256), (256, 1), 0), reinterpret_tensor(primals_231, (128, 256), (256, 1), 0), buf485, reinterpret_tensor(primals_227, (256, 128), (128, 1), 0), buf486, reinterpret_tensor(primals_223, (256, 768), (768, 1), 0), reinterpret_tensor(primals_221, (768, 256), (256, 1), 0), buf487, reinterpret_tensor(primals_217, (256, 256), (256, 1), 0), buf392, reinterpret_tensor(primals_215, (768, 256), (256, 1), 0), buf488, reinterpret_tensor(primals_211, (256, 768), (768, 1), 0), reinterpret_tensor(primals_209, (768, 256), (256, 1), 0), buf489, reinterpret_tensor(primals_205, (256, 256), (256, 1), 0), buf371, reinterpret_tensor(primals_203, (768, 256), (256, 1), 0), buf490, reinterpret_tensor(primals_199, (256, 768), (768, 1), 0), reinterpret_tensor(primals_197, (768, 256), (256, 1), 0), buf491, reinterpret_tensor(primals_193, (256, 256), (256, 1), 0), buf350, reinterpret_tensor(primals_191, (768, 256), (256, 1), 0), reinterpret_tensor(primals_187, (128, 384), (384, 1), 0), reinterpret_tensor(primals_185, (384, 128), (128, 1), 0), buf492, reinterpret_tensor(primals_181, (128, 128), (128, 1), 0), buf329, reinterpret_tensor(primals_179, (384, 128), (128, 1), 0), reinterpret_tensor(primals_175, (256, 128), (128, 1), 0), buf493, reinterpret_tensor(primals_171, (128, 128), (128, 1), 0), reinterpret_tensor(buf311, (32, 401, 1), (401, 1, 0), 0), reinterpret_tensor(buf312, (32, 32, 401), (12832, 1, 32), 0), buf494, reinterpret_tensor(buf306, (32, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf307, (32, 401, 32), (12832, 1, 401), 0), reinterpret_tensor(primals_169, (128, 128), (128, 1), 0), reinterpret_tensor(primals_167, (128, 128), (128, 1), 0), reinterpret_tensor(primals_165, (128, 128), (128, 1), 0), reinterpret_tensor(primals_161, (128, 256), (256, 1), 0), buf495, reinterpret_tensor(primals_157, (256, 256), (256, 1), 0), reinterpret_tensor(buf286, (32, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf287, (32, 64, 197), (12608, 1, 64), 0), buf496, reinterpret_tensor(buf281, (32, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf282, (32, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_155, (256, 256), (256, 1), 0), reinterpret_tensor(primals_153, (256, 256), (256, 1), 0), reinterpret_tensor(primals_151, (256, 256), (256, 1), 0), reinterpret_tensor(primals_147, (128, 256), (256, 1), 0), buf497, reinterpret_tensor(primals_143, (256, 128), (128, 1), 0), buf498, reinterpret_tensor(primals_139, (256, 768), (768, 1), 0), reinterpret_tensor(primals_137, (768, 256), (256, 1), 0), buf499, reinterpret_tensor(primals_133, (256, 256), (256, 1), 0), buf246, reinterpret_tensor(primals_131, (768, 256), (256, 1), 0), buf500, reinterpret_tensor(primals_127, (256, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 256), (256, 1), 0), buf501, reinterpret_tensor(primals_121, (256, 256), (256, 1), 0), buf225, reinterpret_tensor(primals_119, (768, 256), (256, 1), 0), buf502, reinterpret_tensor(primals_115, (256, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 256), (256, 1), 0), buf503, reinterpret_tensor(primals_109, (256, 256), (256, 1), 0), buf204, reinterpret_tensor(primals_107, (768, 256), (256, 1), 0), reinterpret_tensor(primals_103, (128, 384), (384, 1), 0), reinterpret_tensor(primals_101, (384, 128), (128, 1), 0), buf504, reinterpret_tensor(primals_97, (128, 128), (128, 1), 0), buf183, reinterpret_tensor(primals_95, (384, 128), (128, 1), 0), reinterpret_tensor(primals_91, (256, 128), (128, 1), 0), buf505, reinterpret_tensor(primals_87, (128, 128), (128, 1), 0), reinterpret_tensor(buf165, (32, 401, 1), (401, 1, 0), 0), reinterpret_tensor(buf166, (32, 32, 401), (12832, 1, 32), 0), buf506, reinterpret_tensor(buf160, (32, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf161, (32, 401, 32), (12832, 1, 401), 0), reinterpret_tensor(primals_85, (128, 128), (128, 1), 0), reinterpret_tensor(primals_83, (128, 128), (128, 1), 0), reinterpret_tensor(primals_81, (128, 128), (128, 1), 0), reinterpret_tensor(primals_77, (128, 256), (256, 1), 0), buf507, reinterpret_tensor(primals_73, (256, 256), (256, 1), 0), reinterpret_tensor(buf140, (32, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf141, (32, 64, 197), (12608, 1, 64), 0), buf508, reinterpret_tensor(buf135, (32, 64, 1), (64, 1, 0), 0), reinterpret_tensor(buf136, (32, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_71, (256, 256), (256, 1), 0), reinterpret_tensor(primals_69, (256, 256), (256, 1), 0), reinterpret_tensor(primals_67, (256, 256), (256, 1), 0), reinterpret_tensor(primals_63, (128, 256), (256, 1), 0), buf509, reinterpret_tensor(primals_59, (256, 128), (128, 1), 0), buf510, reinterpret_tensor(primals_55, (256, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 256), (256, 1), 0), buf511, reinterpret_tensor(primals_49, (256, 256), (256, 1), 0), buf100, reinterpret_tensor(primals_47, (768, 256), (256, 1), 0), buf512, reinterpret_tensor(primals_43, (256, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 256), (256, 1), 0), buf513, reinterpret_tensor(primals_37, (256, 256), (256, 1), 0), buf79, reinterpret_tensor(primals_35, (768, 256), (256, 1), 0), buf514, reinterpret_tensor(primals_31, (256, 768), (768, 1), 0), reinterpret_tensor(primals_29, (768, 256), (256, 1), 0), buf515, reinterpret_tensor(primals_25, (256, 256), (256, 1), 0), buf58, reinterpret_tensor(primals_23, (768, 256), (256, 1), 0), buf516, reinterpret_tensor(primals_19, (128, 384), (384, 1), 0), reinterpret_tensor(primals_17, (384, 128), (128, 1), 0), buf517, reinterpret_tensor(primals_13, (128, 128), (128, 1), 0), buf34, reinterpret_tensor(primals_11, (384, 128), (128, 1), 0), buf518, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 3, 12, 12), (432, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((8, 3, 240, 240), (172800, 57600, 240, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
