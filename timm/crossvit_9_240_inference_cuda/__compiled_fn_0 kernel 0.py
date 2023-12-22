
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


# kernel path: /tmp/torchinductor_youkaichao/av/cav6jwcifjrsnblkdqihmg3ha4werwyinaohr6qthmzthuf44jwb.py
# Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___norm1, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_27 => cat
# getattr_l__mod___blocks_0_blocks_0___0___norm1 => add_48, add_49, mul_82, mul_83, rsqrt, sub_46, var_mean
# x__3 => add
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp52, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmu4yg37gvshkydn2gxwiro7h5gslvrq625bcegzrpgq5koemu4.py
# Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___norm2, x_7, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_27 => cat
# getattr_l__mod___blocks_0_blocks_0___0___norm2 => add_51, add_52, mul_84, mul_85, rsqrt_1, sub_47, var_mean_1
# x_7 => add_50
# x__3 => add
triton_per_fused_add_cat_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxw3mdn2gbgdfczhfztcodf6amxx45a5ziug3fnuowxv5nanrgun.py
# Source Nodes: [x__5], Original ATen: [aten.sub]
# x__5 => sub_7
triton_poi_fused_sub_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/db/cdbvqjktyjs7tr7cwzfrubliccl7rky6otifhp2jszlqesyqubb4.py
# Source Nodes: [x__5], Original ATen: [aten.add]
# x__5 => add_9
triton_poi_fused_add_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/24/c24bmcot7uzprinzvxx2ae5a7qcrojh5j2k6ppsjdpjf75kmlgdw.py
# Source Nodes: [x__5], Original ATen: [aten.add]
# x__5 => add_10
triton_poi_fused_add_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tm/ctm6nmyrpubniimnzwuel5bromyglvwhqbk2z7nqzwzmxzd3uxow.py
# Source Nodes: [x__5], Original ATen: [aten.sub]
# x__5 => sub_13
triton_poi_fused_sub_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2lnc5sgg4kjstsj4blz4eiby4cqz5pnsyls6as225dszskwhkk.py
# Source Nodes: [l__mod___patch_embed_1_proj, x__5], Original ATen: [aten._unsafe_index, aten.add, aten.convolution, aten.mul]
# l__mod___patch_embed_1_proj => convolution_1
# x__5 => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add_12, add_13, add_14, add_20, add_21, add_22, add_28, add_29, add_30, add_36, add_37, add_38, add_44, add_45, add_46, mul_14, mul_15, mul_16, mul_17, mul_30, mul_31, mul_32, mul_33, mul_46, mul_47, mul_48, mul_49, mul_62, mul_63, mul_64, mul_65, mul_78, mul_79, mul_80, mul_81
triton_poi_fused__unsafe_index_add_convolution_mul_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_6', 'mutated_arg_names': ['in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, xnumel, XBLOCK : tl.constexpr):
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
    tmp52 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr18 + (x1), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr20 + (x1), None, eviction_policy='evict_last')
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
    tmp53 = tmp51 * tmp52
    tmp54 = tl.load(in_ptr0 + (tmp28 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp56 = tmp54 * tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = tmp8 + tmp20
    tmp59 = triton_helpers.maximum(tmp58, tmp9)
    tmp60 = triton_helpers.minimum(tmp59, tmp11)
    tmp61 = tl.load(in_ptr0 + (tmp23 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp63 = tmp61 * tmp62
    tmp64 = tl.load(in_ptr0 + (tmp28 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp66 = tmp64 * tmp65
    tmp67 = tmp63 + tmp66
    tmp68 = tmp8 + tmp40
    tmp69 = triton_helpers.maximum(tmp68, tmp9)
    tmp70 = triton_helpers.minimum(tmp69, tmp11)
    tmp71 = tl.load(in_ptr0 + (tmp23 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp73 = tmp71 * tmp72
    tmp74 = tl.load(in_ptr0 + (tmp28 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp76 = tmp74 * tmp75
    tmp77 = tmp73 + tmp76
    tmp78 = tl.load(in_ptr0 + (tmp35 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp80 = tmp78 * tmp79
    tmp81 = tmp57 + tmp80
    tmp82 = tl.load(in_ptr0 + (tmp43 + (240*tmp50) + (57600*x2)), None, eviction_policy='evict_last')
    tmp84 = tmp82 * tmp83
    tmp85 = tmp81 + tmp84
    tmp86 = tl.load(in_ptr0 + (tmp35 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp88 = tmp86 * tmp87
    tmp89 = tmp67 + tmp88
    tmp90 = tl.load(in_ptr0 + (tmp43 + (240*tmp60) + (57600*x2)), None, eviction_policy='evict_last')
    tmp92 = tmp90 * tmp91
    tmp93 = tmp89 + tmp92
    tmp94 = tl.load(in_ptr0 + (tmp35 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp96 = tmp94 * tmp95
    tmp97 = tmp77 + tmp96
    tmp98 = tl.load(in_ptr0 + (tmp43 + (240*tmp70) + (57600*x2)), None, eviction_policy='evict_last')
    tmp100 = tmp98 * tmp99
    tmp101 = tmp97 + tmp100
    tmp103 = tmp85 * tmp102
    tmp105 = tmp47 * tmp104
    tmp106 = tmp103 + tmp105
    tmp108 = tmp93 * tmp107
    tmp109 = tmp106 + tmp108
    tmp111 = tmp101 * tmp110
    tmp112 = tmp109 + tmp111
    tl.store(in_out_ptr1 + (x4), tmp112, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpotwk2d34xpskjoxjxw2rjnhd7oxffclajayvinorsn54grjea.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm1 => var_mean_2
# x__8 => add_47
triton_red_fused_add_cat_native_layer_norm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ya/cya6fmyyczu5ilxgpxjppjfvi246ci5rdxrb6oyjtclewnvqya2n.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm1 => var_mean_2
# x__8 => add_47
triton_per_fused_add_cat_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxxei6wfdnp55efavpbok3janppkle3rfmbn5wucrttq7hu5yao.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm1 => add_55, add_56, mul_89, mul_90, rsqrt_2, sub_48, var_mean_2
# x__8 => add_47
triton_poi_fused_add_cat_native_layer_norm_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x5), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c563yb4nw3vzxwtznrlwyeqwez6nfwobtxnuoznwpnioubdldb74.py
# Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm2, x_19, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_26 => cat_1
# getattr_l__mod___blocks_0_blocks_1___0___norm2 => add_58, add_59, mul_91, mul_92, rsqrt_3, sub_49, var_mean_3
# x_19 => add_57
# x__8 => add_47
triton_per_fused_add_cat_native_layer_norm_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwq2libt547u76bn2wfe4eb23i4wjwaaquzt5xkusctxco5zprm.py
# Source Nodes: [x_21], Original ATen: [aten.gelu]
# x_21 => add_60, erf_1, mul_93, mul_94, mul_95
triton_poi_fused_gelu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
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


# kernel path: /tmp/torchinductor_youkaichao/xu/cxue4q3kdazlkaeqbjdeoyrtqbi3eiwjl3c2maknzlcbdi7z6j5n.py
# Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___norm1, x_26], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks_0_blocks_1___1___norm1 => add_62, add_63, mul_96, mul_97, rsqrt_4, sub_50, var_mean_4
# x_26 => add_61
triton_per_fused_add_native_layer_norm_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfrgtdsho6bfgiyrjtx6f6k5hxrl4grz7u62gbadzldj2ig2yjsz.py
# Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___norm2, x_26, x_31], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___blocks_0_blocks_1___1___norm2 => add_65, add_66, mul_98, mul_99, rsqrt_5, sub_51, var_mean_5
# x_26 => add_61
# x_31 => add_64
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbxk6ff63ymoomgx4a3ubxnqnp5avigkftrx6yobgxg46pysxdx.py
# Source Nodes: [x_9], Original ATen: [aten.gelu]
# x_9 => add_53, erf, mul_86, mul_87, mul_88
triton_poi_fused_gelu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1231872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvak7nhqogwh5wbip4wg2hkofeelrjzghub75srf55lgesozf2yn.py
# Source Nodes: [l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1], Original ATen: [aten.gelu, aten.native_layer_norm]
# l__mod___blocks_0_projs_0_0 => add_76, add_77, clone_14, mul_110, mul_111, rsqrt_8, sub_54, var_mean_8
# l__mod___blocks_0_projs_0_1 => add_78, erf_4, mul_112, mul_113, mul_114
triton_per_fused_gelu_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_15', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czk2herd73nen25o26ixbvzyd7ji33476f7wjkc3fue6uhjhm4rs.py
# Source Nodes: [cat_25, l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_25 => cat_2
# l__mod___blocks_0_fusion_0_norm1 => add_82, add_83, mul_120, mul_121, rsqrt_10, sub_56, var_mean_10
triton_per_fused_cat_native_layer_norm_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp35 = tmp18 - tmp28
    tmp36 = 256.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcuxjug2xdzpkd6p24vp72afi32326hpoqxrerywg23oe6z5vm2.py
# Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.add]
# l__mod___blocks_0_fusion_0_attn_wq => add_84
triton_poi_fused_add_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_17', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7doidhyoydkhc7ma37z5x6dwufdcbrzcgc55rgdyneqv2xv4ju.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_16
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjbbloarbsm4xbaxnszpxx6scl4dtfyhoxxl3vfnip4uqr7f6qq.py
# Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
# attn => mul_122
# attn_1 => amax, div, exp, sub_57, sum_1
triton_per_fused__softmax_mul_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2oulbvtnkafnhrgfygwyshg5g5366uk2uilgjsubigpikyn4ml.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_18
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkfud42zk5soj7qrohjtbnitl66i6vszd3aefky4sek7fthdzk6.py
# Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1, l__mod___blocks_0_revert_projs_0_0, l__mod___blocks_0_revert_projs_0_1, tmp_1], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
# l__mod___blocks_0_projs_1_0 => add_79, add_80, clone_15, mul_115, mul_116, rsqrt_9, sub_55, var_mean_9
# l__mod___blocks_0_projs_1_1 => add_81, erf_5, mul_117, mul_118, mul_119
# l__mod___blocks_0_revert_projs_0_0 => add_86, add_87, mul_123, mul_124, rsqrt_11, sub_58, var_mean_11
# l__mod___blocks_0_revert_projs_0_1 => add_88, erf_6, mul_125, mul_126, mul_127
# tmp_1 => add_85
triton_per_fused_add_gelu_native_layer_norm_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel):
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
    tmp39 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp40 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tmp21 >= tmp21
    tmp23 = tl.full([1], 1, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & tmp24 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tmp21 >= tmp23
    tmp29 = tl.full([1], 197, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tl.load(in_ptr0 + (r1 + (50432*x0)), rmask & tmp28 & xmask, other=0.0)
    tmp32 = tl.load(in_ptr1 + (r1 + (50432*x0)), rmask & tmp28 & xmask, other=0.0)
    tmp33 = tl.load(in_ptr2 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp28, tmp35, tmp36)
    tmp38 = tl.where(tmp24, tmp27, tmp37)
    tmp41 = tmp39 + tmp40
    tmp42 = tmp38 + tmp41
    tmp43 = tmp4 - tmp14
    tmp44 = 256.0
    tmp45 = tmp20 / tmp44
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = 0.5
    tmp55 = tmp53 * tmp54
    tmp56 = 0.7071067811865476
    tmp57 = tmp53 * tmp56
    tmp58 = tl.math.erf(tmp57)
    tmp59 = 1.0
    tmp60 = tmp58 + tmp59
    tmp61 = tmp55 * tmp60
    tmp62 = tl.broadcast_to(tmp42, [RBLOCK])
    tmp64 = tl.where(rmask & xmask, tmp62, 0)
    tmp65 = tl.broadcast_to(tmp62, [RBLOCK])
    tmp67 = tl.where(rmask & xmask, tmp65, 0)
    tmp68 = triton_helpers.promote_to_tensor(tl.sum(tmp67, 0))
    tmp69 = tmp68 / tmp13
    tmp70 = tmp62 - tmp69
    tmp71 = tmp70 * tmp70
    tmp72 = tl.broadcast_to(tmp71, [RBLOCK])
    tmp74 = tl.where(rmask & xmask, tmp72, 0)
    tmp75 = triton_helpers.promote_to_tensor(tl.sum(tmp74, 0))
    tmp76 = tmp42 - tmp69
    tmp77 = tmp75 / tmp44
    tmp78 = tmp77 + tmp46
    tmp79 = tl.math.rsqrt(tmp78)
    tmp80 = tmp76 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp84 * tmp54
    tmp86 = tmp84 * tmp56
    tmp87 = tl.math.erf(tmp86)
    tmp88 = tmp87 + tmp59
    tmp89 = tmp85 * tmp88
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp42, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + (256*x0)), tmp61, rmask & xmask)
    tl.store(in_out_ptr2 + (r1 + (256*x0)), tmp89, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcuxi372jmgfozvfqw27b523qgnlvcmvc4nqj7ebo7ui7jsootr.py
# Source Nodes: [cat_23, cat_24, getattr_l__mod___blocks_1_blocks_0___0___norm1, l__mod___blocks_0_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_23 => cat_4
# cat_24 => cat_3
# getattr_l__mod___blocks_1_blocks_0___0___norm1 => add_96, add_97, mul_136, mul_137, rsqrt_14, sub_62, var_mean_14
# l__mod___blocks_0_fusion_1_norm1 => add_89, add_90, mul_128, mul_129, rsqrt_12, sub_59, var_mean_12
triton_per_fused_cat_native_layer_norm_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp60 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp35 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp4, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp37, tmp17)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None]
    tmp46 = tmp45 / tmp27
    tmp47 = tmp39 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.where(rmask & xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None]
    tmp53 = tmp18 - tmp28
    tmp54 = 128.0
    tmp55 = tmp34 / tmp54
    tmp56 = 1e-06
    tmp57 = tmp55 + tmp56
    tmp58 = tl.math.rsqrt(tmp57)
    tmp59 = tmp53 * tmp58
    tmp61 = tmp59 * tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = tmp38 - tmp46
    tmp65 = tmp52 / tmp54
    tmp66 = tmp65 + tmp56
    tmp67 = tl.math.rsqrt(tmp66)
    tmp68 = tmp64 * tmp67
    tmp70 = tmp68 * tmp69
    tmp72 = tmp70 + tmp71
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp63, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp72, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yu/cyupabgq5o5en6faxffyft5gvmpbv3bzxajww2wwgeyujgr6hqd5.py
# Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.add]
# l__mod___blocks_0_fusion_1_attn_wq => add_91
triton_poi_fused_add_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_23', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/yi/cyicl3zjfh2cg36isml7mmlizgagimdr33y4pqbuot637rtoy77c.py
# Source Nodes: [matmul_2], Original ATen: [aten.clone]
# matmul_2 => clone_20
triton_poi_fused_clone_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnhae5yfjnw7cejz2k32jqhv7w7anj2fsllejymrn5cjqk7dabr.py
# Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
# attn_3 => mul_130
# attn_4 => amax_1, div_1, exp_1, sub_60, sum_2
triton_per_fused__softmax_mul_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cev43sxiwyr6o7bozisnp4g3p4hh4kwb3jbw6rarpmutfkfvdshm.py
# Source Nodes: [matmul_3], Original ATen: [aten.clone]
# matmul_3 => clone_22
triton_poi_fused_clone_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_26', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzw536whcwkq7bya4ktuxh72fkzxppxhw2yh5i2dy3f4wrgprua.py
# Source Nodes: [l__mod___blocks_0_revert_projs_1_0, l__mod___blocks_0_revert_projs_1_1, tmp_4], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
# l__mod___blocks_0_revert_projs_1_0 => add_93, add_94, mul_131, mul_132, rsqrt_13, sub_61, var_mean_13
# l__mod___blocks_0_revert_projs_1_1 => add_95, erf_7, mul_133, mul_134, mul_135
# tmp_4 => add_92
triton_per_fused_add_gelu_native_layer_norm_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & tmp3 & xmask, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1, 1], 401, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp6, tmp16)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tmp32 = tmp22 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None]
    tmp38 = tmp21 - tmp31
    tmp39 = 128.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-06
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = 0.5
    tmp50 = tmp48 * tmp49
    tmp51 = 0.7071067811865476
    tmp52 = tmp48 * tmp51
    tmp53 = tl.math.erf(tmp52)
    tmp54 = 1.0
    tmp55 = tmp53 + tmp54
    tmp56 = tmp50 * tmp55
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + (128*x0)), tmp56, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/csh6zgjnobvjbucgi3ezkejjpbbjkuqbv5qrhxvffzgix56adw43.py
# Source Nodes: [cat_24, getattr_l__mod___blocks_1_blocks_0___0___norm2, x_63], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_24 => cat_3
# getattr_l__mod___blocks_1_blocks_0___0___norm2 => add_100, add_99, mul_138, mul_139, rsqrt_15, sub_63, var_mean_15
# x_63 => add_98
triton_per_fused_add_cat_native_layer_norm_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwkvlghsbozfjk7myv5u3ypy76u4zclar5igobbln34lc5xnmzk.py
# Source Nodes: [cat_22, getattr_l__mod___blocks_1_blocks_1___0___norm2, x_75], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_22 => cat_5
# getattr_l__mod___blocks_1_blocks_1___0___norm2 => add_106, add_107, mul_145, mul_146, rsqrt_17, sub_65, var_mean_17
# x_75 => add_105
triton_per_fused_add_cat_native_layer_norm_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    tmp19 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgja3xgweibff43hfm2cqan362ctzt7c67cbtxnrpexgpai3cyn3.py
# Source Nodes: [cat_15, cat_16, l__mod___blocks_2_fusion_1_norm1, x_171], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_15 => cat_12
# cat_16 => cat_11
# l__mod___blocks_2_fusion_1_norm1 => add_185, add_186, mul_236, mul_237, rsqrt_40, sub_91, var_mean_40
# x_171 => var_mean_42
triton_per_fused_cat_native_layer_norm_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp60 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp35 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp4, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp37, tmp17)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None]
    tmp46 = tmp45 / tmp27
    tmp47 = tmp39 - tmp46
    tmp48 = tmp47 * tmp47
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.where(rmask & xmask, tmp49, 0)
    tmp52 = tl.sum(tmp51, 1)[:, None]
    tmp53 = tmp18 - tmp28
    tmp54 = 128.0
    tmp55 = tmp34 / tmp54
    tmp56 = 1e-06
    tmp57 = tmp55 + tmp56
    tmp58 = tl.math.rsqrt(tmp57)
    tmp59 = tmp53 * tmp58
    tmp61 = tmp59 * tmp60
    tmp63 = tmp61 + tmp62
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp63, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp46, xmask)
    tl.store(out_ptr3 + (x3), tmp52, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvgsc77yzlbt5z74xevwdocnct3glpssohvww4pz54tvssduql3i.py
# Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1, l__mod___head_drop, tmp_16], Original ATen: [aten.add, aten.clone, aten.gelu, aten.native_layer_norm]
# l__mod___blocks_2_revert_projs_1_0 => add_189, add_190, mul_239, mul_240, rsqrt_41, sub_93, var_mean_41
# l__mod___blocks_2_revert_projs_1_1 => add_191, erf_23, mul_241, mul_242, mul_243
# l__mod___head_drop => clone_68
# tmp_16 => add_188
triton_per_fused_add_clone_gelu_native_layer_norm_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_gelu_native_layer_norm_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr6 + (401*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (401*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr11 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & tmp3 & xmask, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1, 1], 401, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp6, tmp16)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tl.load(in_ptr5 + (r1 + (128*x0)), rmask & tmp3 & xmask, other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp3, tmp22, tmp23)
    tmp25 = tl.where(tmp3, tmp24, tmp16)
    tmp27 = tmp25 - tmp26
    tmp29 = 128.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp27 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp44 = tl.where(rmask & xmask, tmp42, 0)
    tmp45 = tl.sum(tmp44, 1)[:, None]
    tmp46 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 / tmp47
    tmp49 = tmp39 - tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp53 = tl.where(rmask & xmask, tmp51, 0)
    tmp54 = tl.sum(tmp53, 1)[:, None]
    tmp55 = tmp21 - tmp48
    tmp56 = tmp54 / tmp29
    tmp57 = tmp56 + tmp31
    tmp58 = tl.math.rsqrt(tmp57)
    tmp59 = tmp55 * tmp58
    tmp61 = tmp59 * tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = 0.5
    tmp65 = tmp63 * tmp64
    tmp66 = 0.7071067811865476
    tmp67 = tmp63 * tmp66
    tmp68 = tl.math.erf(tmp67)
    tmp69 = 1.0
    tmp70 = tmp68 + tmp69
    tmp71 = tmp65 * tmp70
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr0 + (r1 + (128*x0)), tmp38, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + (128*x0)), tmp71, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy6chowrlfwls7te6givexp6srp6q73sekkaeytuxrru2jkyf3ul.py
# Source Nodes: [cat_14, x_172], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_14 => cat_13
# x_172 => var_mean_43
triton_per_fused_cat_native_layer_norm_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6g5mjunhfdjba5ivktnxmagp7gjisneljwh6yvb5mx5oi4dnacm.py
# Source Nodes: [l__mod___head_drop_1], Original ATen: [aten.clone]
# l__mod___head_drop_1 => clone_69
triton_poi_fused_clone_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp18 = tl.load(in_ptr4 + (197*x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (197*x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (x2), tmp3, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1], 197, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + (50432*x1)), tmp7, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (50432*x1)), tmp7, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), tmp7, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp6, tmp16)
    tmp19 = tmp17 - tmp18
    tmp21 = 256.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp19 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (x2), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrlou7ialdaezc6ikgyargsvnaj627o7jv2tg6uterhjbec6wug.py
# Source Nodes: [x_175], Original ATen: [aten.mean]
# x_175 => mean
triton_poi_fused_mean_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_34', 'mutated_arg_names': []},
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(arg1_1, (1, 401, 128), (51328, 128, 1))
    assert_size_stride(arg2_1, (1, 1, 256), (256, 256, 1))
    assert_size_stride(arg3_1, (1, 197, 256), (50432, 256, 1))
    assert_size_stride(arg4_1, (128, 3, 12, 12), (432, 144, 12, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg7_1, (256, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (384, 128), (128, 1))
    assert_size_stride(arg11_1, (384, ), (1, ))
    assert_size_stride(arg12_1, (128, 128), (128, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (384, 128), (128, 1))
    assert_size_stride(arg17_1, (384, ), (1, ))
    assert_size_stride(arg18_1, (128, 384), (384, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (768, 256), (256, 1))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (256, 256), (256, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (768, 256), (256, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (256, 768), (768, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (768, 256), (256, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (256, 256), (256, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (768, 256), (256, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (256, 768), (768, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (768, 256), (256, 1))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (256, 256), (256, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (768, 256), (256, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (256, 768), (768, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (256, 128), (128, 1))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (128, 256), (256, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (256, 256), (256, 1))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, 256), (256, 1))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, 256), (256, 1))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, 256), (256, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (128, 256), (256, 1))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, 128), (128, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, 128), (128, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, 128), (128, 1))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, 128), (128, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (256, 128), (128, 1))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (384, 128), (128, 1))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (128, 128), (128, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (384, 128), (128, 1))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (128, 384), (384, 1))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (768, 256), (256, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (256, 256), (256, 1))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (768, 256), (256, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (256, 768), (768, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (768, 256), (256, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (256, 256), (256, 1))
    assert_size_stride(arg121_1, (256, ), (1, ))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (768, 256), (256, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (256, 768), (768, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (768, 256), (256, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (256, 256), (256, 1))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (768, 256), (256, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (256, 768), (768, 1))
    assert_size_stride(arg139_1, (256, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (256, 128), (128, 1))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (128, 256), (256, 1))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, 256), (256, 1))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (256, 256), (256, 1))
    assert_size_stride(arg153_1, (256, ), (1, ))
    assert_size_stride(arg154_1, (256, 256), (256, 1))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (256, 256), (256, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (128, 256), (256, 1))
    assert_size_stride(arg161_1, (128, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, 128), (128, 1))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, 128), (128, 1))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, 128), (128, 1))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, 128), (128, 1))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (256, 128), (128, 1))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (384, 128), (128, 1))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (128, 128), (128, 1))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (384, 128), (128, 1))
    assert_size_stride(arg185_1, (384, ), (1, ))
    assert_size_stride(arg186_1, (128, 384), (384, 1))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (768, 256), (256, 1))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (256, 256), (256, 1))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (768, 256), (256, 1))
    assert_size_stride(arg197_1, (768, ), (1, ))
    assert_size_stride(arg198_1, (256, 768), (768, 1))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (768, 256), (256, 1))
    assert_size_stride(arg203_1, (768, ), (1, ))
    assert_size_stride(arg204_1, (256, 256), (256, 1))
    assert_size_stride(arg205_1, (256, ), (1, ))
    assert_size_stride(arg206_1, (256, ), (1, ))
    assert_size_stride(arg207_1, (256, ), (1, ))
    assert_size_stride(arg208_1, (768, 256), (256, 1))
    assert_size_stride(arg209_1, (768, ), (1, ))
    assert_size_stride(arg210_1, (256, 768), (768, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (768, 256), (256, 1))
    assert_size_stride(arg215_1, (768, ), (1, ))
    assert_size_stride(arg216_1, (256, 256), (256, 1))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (256, ), (1, ))
    assert_size_stride(arg220_1, (768, 256), (256, 1))
    assert_size_stride(arg221_1, (768, ), (1, ))
    assert_size_stride(arg222_1, (256, 768), (768, 1))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (256, 128), (128, 1))
    assert_size_stride(arg227_1, (256, ), (1, ))
    assert_size_stride(arg228_1, (256, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (128, 256), (256, 1))
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, 256), (256, 1))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, 256), (256, 1))
    assert_size_stride(arg237_1, (256, ), (1, ))
    assert_size_stride(arg238_1, (256, 256), (256, 1))
    assert_size_stride(arg239_1, (256, ), (1, ))
    assert_size_stride(arg240_1, (256, 256), (256, 1))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (128, 256), (256, 1))
    assert_size_stride(arg245_1, (128, ), (1, ))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (128, 128), (128, 1))
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, 128), (128, 1))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, 128), (128, 1))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (128, 128), (128, 1))
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (128, ), (1, ))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (256, 128), (128, 1))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (1000, 128), (128, 1))
    assert_size_stride(arg265_1, (1000, ), (1, ))
    assert_size_stride(arg266_1, (1000, 256), (256, 1))
    assert_size_stride(arg267_1, (1000, ), (1, ))
    assert_size_stride(arg268_1, (8, 3, 240, 240), (172800, 57600, 240, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___patch_embed_0_proj], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg268_1, arg4_1, stride=(12, 12), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 20, 20), (51200, 400, 20, 1))
        del arg4_1
        buf4 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___norm1, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(arg0_1, buf0, arg5_1, arg1_1, arg8_1, arg9_1, buf4, 3208, 128, grid=grid(3208), stream=stream0)
        del arg8_1
        del arg9_1
        buf5 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf4, (3208, 128), (128, 1), 0), reinterpret_tensor(arg10_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg10_1
        del arg11_1
        # Source Nodes: [x_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf6 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf5, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf5, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf5, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, False)
        buf7 = buf6[0]
        del buf6
        buf11 = reinterpret_tensor(buf4, (3208, 128), (128, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf7, (3208, 128), (128, 1), 0), reinterpret_tensor(arg12_1, (128, 128), (1, 128), 0), out=buf11)
        del arg12_1
        buf12 = reinterpret_tensor(buf11, (8, 401, 128), (51328, 128, 1), 0); del buf11  # reuse
        buf122 = reinterpret_tensor(buf7, (8, 401, 128), (51328, 128, 1), 0); del buf7  # reuse
        # Source Nodes: [cat_27, getattr_l__mod___blocks_0_blocks_0___0___norm2, x_7, x__3], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_1.run(buf12, arg0_1, buf0, arg5_1, arg1_1, arg13_1, arg14_1, arg15_1, buf122, 3208, 128, grid=grid(3208), stream=stream0)
        del arg0_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg1_1
        del arg5_1
        del buf0
        buf17 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_2.run(buf17, 224, grid=grid(224), stream=stream0)
        buf19 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf19, 224, grid=grid(224), stream=stream0)
        buf22 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf22, 224, grid=grid(224), stream=stream0)
        buf24 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_5.run(buf24, 224, grid=grid(224), stream=stream0)
        buf27 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_2.run(buf27, 224, grid=grid(224), stream=stream0)
        buf29 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_2.run(buf29, 224, grid=grid(224), stream=stream0)
        buf31 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf31, 224, grid=grid(224), stream=stream0)
        buf34 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf34, 224, grid=grid(224), stream=stream0)
        buf36 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_5.run(buf36, 224, grid=grid(224), stream=stream0)
        buf39 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf39, 224, grid=grid(224), stream=stream0)
        buf41 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_2.run(buf41, 224, grid=grid(224), stream=stream0)
        buf43 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf43, 224, grid=grid(224), stream=stream0)
        buf46 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf46, 224, grid=grid(224), stream=stream0)
        buf48 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_5.run(buf48, 224, grid=grid(224), stream=stream0)
        buf51 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf51, 224, grid=grid(224), stream=stream0)
        buf53 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_2.run(buf53, 224, grid=grid(224), stream=stream0)
        buf55 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_3.run(buf55, 224, grid=grid(224), stream=stream0)
        buf58 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf58, 224, grid=grid(224), stream=stream0)
        buf60 = empty((1, 1, 1, 224), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_5.run(buf60, 224, grid=grid(224), stream=stream0)
        buf63 = empty((1, 1, 224, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x__5], Original ATen: [aten.sub]
        triton_poi_fused_sub_5.run(buf63, 224, grid=grid(224), stream=stream0)
        buf20 = empty((8, 3, 224, 224), device='cuda', dtype=torch.float32)
        buf25 = buf20; del buf20  # reuse
        buf64 = buf25; del buf25  # reuse
        # Source Nodes: [l__mod___patch_embed_1_proj, x__5], Original ATen: [aten._unsafe_index, aten.add, aten.convolution, aten.mul]
        triton_poi_fused__unsafe_index_add_convolution_mul_6.run(buf64, arg268_1, buf29, buf31, buf34, buf36, buf17, buf19, buf41, buf43, buf53, buf55, buf22, buf24, buf46, buf48, buf58, buf60, buf27, buf39, buf51, buf63, 1204224, grid=grid(1204224), stream=stream0)
        del arg268_1
        del buf17
        del buf19
        del buf22
        del buf24
        del buf27
        del buf29
        del buf31
        del buf34
        del buf36
        del buf39
        del buf41
        del buf43
        del buf46
        del buf48
        del buf51
        del buf53
        del buf55
        del buf58
        del buf60
        del buf63
        # Source Nodes: [l__mod___patch_embed_1_proj, x__5], Original ATen: [aten.add, aten.convolution, aten.mul]
        buf65 = extern_kernels.convolution(buf64, arg6_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg6_1
        del buf64
        buf66 = empty_strided((8, 197, 1, 2), (394, 2, 3152, 1), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((8, 197, 1, 2), (394, 2, 3152, 1), device='cuda', dtype=torch.float32)
        buf68 = empty_strided((8, 197, 1, 2), (394, 2, 3152, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_7.run(arg2_1, buf65, arg7_1, arg3_1, buf66, buf67, buf68, 3152, 128, grid=grid(3152), stream=stream0)
        buf69 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_8.run(buf66, buf67, buf68, buf69, buf70, 1576, 2, grid=grid(1576), stream=stream0)
        del buf66
        del buf67
        del buf68
        buf72 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm1, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_poi_fused_add_cat_native_layer_norm_9.run(arg2_1, buf65, arg7_1, arg3_1, buf69, buf70, arg20_1, arg21_1, buf72, 403456, grid=grid(403456), stream=stream0)
        del arg20_1
        del arg21_1
        buf73 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg23_1, reinterpret_tensor(buf72, (1576, 256), (256, 1), 0), reinterpret_tensor(arg22_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf73)
        del arg22_1
        del arg23_1
        # Source Nodes: [x_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf74 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf73, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf73, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf73, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf75 = buf74[0]
        del buf74
        buf79 = reinterpret_tensor(buf72, (1576, 256), (256, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (1576, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 256), (1, 256), 0), out=buf79)
        del arg24_1
        buf80 = reinterpret_tensor(buf79, (8, 197, 256), (50432, 256, 1), 0); del buf79  # reuse
        buf84 = reinterpret_tensor(buf75, (8, 197, 256), (50432, 256, 1), 0); del buf75  # reuse
        # Source Nodes: [cat_26, getattr_l__mod___blocks_0_blocks_1___0___norm2, x_19, x__8], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_10.run(buf80, arg2_1, buf65, arg7_1, arg3_1, arg25_1, arg26_1, arg27_1, buf84, 1576, 256, grid=grid(1576), stream=stream0)
        del arg25_1
        del arg26_1
        del arg27_1
        del arg2_1
        del arg3_1
        del arg7_1
        del buf65
        buf85 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf84, (1576, 256), (256, 1), 0), reinterpret_tensor(arg28_1, (256, 768), (1, 256), 0), out=buf85)
        del arg28_1
        buf86 = reinterpret_tensor(buf85, (8, 197, 768), (151296, 768, 1), 0); del buf85  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf86, arg29_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg29_1
        buf87 = reinterpret_tensor(buf84, (1576, 256), (256, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf86, (1576, 768), (768, 1), 0), reinterpret_tensor(arg30_1, (768, 256), (1, 768), 0), out=buf87)
        del arg30_1
        buf91 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___norm1, x_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf80, buf87, arg31_1, arg32_1, arg33_1, buf91, 1576, 256, grid=grid(1576), stream=stream0)
        del arg32_1
        del arg33_1
        buf92 = reinterpret_tensor(buf86, (1576, 768), (768, 1), 0); del buf86  # reuse
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg35_1, reinterpret_tensor(buf91, (1576, 256), (256, 1), 0), reinterpret_tensor(arg34_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf92)
        del arg34_1
        del arg35_1
        # Source Nodes: [x_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf93 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf92, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf92, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf92, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf94 = buf93[0]
        del buf93
        buf98 = reinterpret_tensor(buf91, (1576, 256), (256, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (1576, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 256), (1, 256), 0), out=buf98)
        del arg36_1
        buf99 = reinterpret_tensor(buf98, (8, 197, 256), (50432, 256, 1), 0); del buf98  # reuse
        buf103 = reinterpret_tensor(buf94, (8, 197, 256), (50432, 256, 1), 0); del buf94  # reuse
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___1___norm2, x_26, x_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf99, buf80, buf87, arg31_1, arg37_1, arg38_1, arg39_1, buf103, 1576, 256, grid=grid(1576), stream=stream0)
        del arg31_1
        del arg37_1
        del arg38_1
        del arg39_1
        buf104 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1576, 256), (256, 1), 0), reinterpret_tensor(arg40_1, (256, 768), (1, 256), 0), out=buf104)
        del arg40_1
        buf105 = reinterpret_tensor(buf104, (8, 197, 768), (151296, 768, 1), 0); del buf104  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf105, arg41_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg41_1
        buf106 = reinterpret_tensor(buf103, (1576, 256), (256, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (1576, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 256), (1, 768), 0), out=buf106)
        del arg42_1
        buf110 = reinterpret_tensor(buf87, (8, 197, 256), (50432, 256, 1), 0); del buf87  # reuse
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___norm1, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf99, buf106, arg43_1, arg44_1, arg45_1, buf110, 1576, 256, grid=grid(1576), stream=stream0)
        del arg44_1
        del arg45_1
        buf111 = reinterpret_tensor(buf105, (1576, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg47_1, reinterpret_tensor(buf110, (1576, 256), (256, 1), 0), reinterpret_tensor(arg46_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf111)
        del arg46_1
        del arg47_1
        # Source Nodes: [x_39], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf112 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf111, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf111, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf111, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf113 = buf112[0]
        del buf112
        buf117 = reinterpret_tensor(buf110, (1576, 256), (256, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf113, (1576, 256), (256, 1), 0), reinterpret_tensor(arg48_1, (256, 256), (1, 256), 0), out=buf117)
        del arg48_1
        buf118 = reinterpret_tensor(buf117, (8, 197, 256), (50432, 256, 1), 0); del buf117  # reuse
        buf129 = reinterpret_tensor(buf113, (8, 197, 256), (50432, 256, 1), 0); del buf113  # reuse
        # Source Nodes: [getattr_l__mod___blocks_0_blocks_1___2___norm2, x_38, x_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf118, buf99, buf106, arg43_1, arg49_1, arg50_1, arg51_1, buf129, 1576, 256, grid=grid(1576), stream=stream0)
        del arg43_1
        del arg49_1
        del arg50_1
        del arg51_1
        buf123 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (3208, 128), (128, 1), 0), reinterpret_tensor(arg16_1, (128, 384), (1, 128), 0), out=buf123)
        del arg16_1
        buf124 = reinterpret_tensor(buf123, (8, 401, 384), (153984, 384, 1), 0); del buf123  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf124, arg17_1, 1231872, grid=grid(1231872), stream=stream0)
        del arg17_1
        buf125 = reinterpret_tensor(buf122, (3208, 128), (128, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (3208, 384), (384, 1), 0), reinterpret_tensor(arg18_1, (384, 128), (1, 384), 0), out=buf125)
        del arg18_1
        buf136 = empty_strided((8, 1, 128), (128, 1024, 1), device='cuda', dtype=torch.float32)
        buf137 = reinterpret_tensor(buf136, (8, 1, 128), (128, 128, 1), 0); del buf136  # reuse
        # Source Nodes: [l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_15.run(buf137, buf12, buf125, arg19_1, arg56_1, arg57_1, 8, 128, grid=grid(8), stream=stream0)
        del arg56_1
        del arg57_1
        buf130 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (1576, 256), (256, 1), 0), reinterpret_tensor(arg52_1, (256, 768), (1, 256), 0), out=buf130)
        del arg52_1
        buf131 = reinterpret_tensor(buf130, (8, 197, 768), (151296, 768, 1), 0); del buf130  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf131, arg53_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg53_1
        buf132 = reinterpret_tensor(buf129, (1576, 256), (256, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (1576, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 256), (1, 768), 0), out=buf132)
        del arg54_1
        buf138 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_0_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf137, (8, 128), (128, 1), 0), reinterpret_tensor(arg58_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf138)
        del arg58_1
        del arg59_1
        buf142 = buf99; del buf99  # reuse
        # Source Nodes: [cat_25, l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf138, buf118, buf132, arg55_1, arg64_1, arg65_1, buf142, 1576, 256, grid=grid(1576), stream=stream0)
        del arg64_1
        del arg65_1
        buf143 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (8, 256), (50432, 1), 0), reinterpret_tensor(arg66_1, (256, 256), (1, 256), 0), out=buf143)
        del arg66_1
        buf145 = reinterpret_tensor(buf143, (8, 1, 256), (256, 256, 1), 0); del buf143  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_0_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf145, arg67_1, 2048, grid=grid(2048), stream=stream0)
        del arg67_1
        buf144 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1576, 256), (256, 1), 0), reinterpret_tensor(arg68_1, (256, 256), (1, 256), 0), out=buf144)
        del arg68_1
        buf146 = reinterpret_tensor(buf80, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf80  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf144, arg69_1, buf146, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del arg69_1
        del buf144
        buf147 = empty((32, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf146, (32, 64, 197), (12608, 197, 1), 0), out=buf147)
        buf151 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_19.run(buf147, buf151, 32, 197, grid=grid(32), stream=stream0)
        buf150 = reinterpret_tensor(buf146, (1576, 256), (256, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (1576, 256), (256, 1), 0), reinterpret_tensor(arg70_1, (256, 256), (1, 256), 0), out=buf150)
        del arg70_1
        buf152 = reinterpret_tensor(buf142, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf142  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf150, arg71_1, buf152, 403456, grid=grid(403456), stream=stream0)
        del arg71_1
        del buf150
        buf153 = reinterpret_tensor(buf145, (32, 1, 64), (64, 64, 1), 0); del buf145  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf152, (32, 197, 64), (12608, 64, 1), 0), out=buf153)
        buf154 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (8, 256), (256, 1), 0), reinterpret_tensor(arg72_1, (256, 256), (1, 256), 0), out=buf154)
        del arg72_1
        buf155 = reinterpret_tensor(buf154, (8, 1, 256), (256, 2048, 1), 0); del buf154  # reuse
        buf159 = reinterpret_tensor(buf153, (8, 1, 256), (256, 2048, 1), 0); del buf153  # reuse
        buf160 = reinterpret_tensor(buf159, (8, 1, 256), (256, 256, 1), 0); del buf159  # reuse
        buf182 = empty_strided((8, 1, 256), (256, 2048, 1), device='cuda', dtype=torch.float32)
        buf183 = reinterpret_tensor(buf182, (8, 1, 256), (256, 256, 1), 0); del buf182  # reuse
        # Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1, l__mod___blocks_0_revert_projs_0_0, l__mod___blocks_0_revert_projs_0_1, tmp_1], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
        triton_per_fused_add_gelu_native_layer_norm_21.run(buf155, buf160, buf183, buf118, buf132, arg55_1, buf138, arg73_1, arg60_1, arg61_1, arg74_1, arg75_1, 8, 256, grid=grid(8), stream=stream0)
        del arg60_1
        del arg61_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf161 = reinterpret_tensor(buf137, (8, 128), (128, 1), 0); del buf137  # reuse
        # Source Nodes: [l__mod___blocks_0_projs_1_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg63_1, reinterpret_tensor(buf160, (8, 256), (256, 1), 0), reinterpret_tensor(arg62_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf161)
        del arg62_1
        del arg63_1
        buf184 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [reverted_proj_cls_token], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg77_1, reinterpret_tensor(buf183, (8, 256), (256, 1), 0), reinterpret_tensor(arg76_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf184)
        del arg76_1
        del arg77_1
        buf165 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        buf188 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, cat_24, getattr_l__mod___blocks_1_blocks_0___0___norm1, l__mod___blocks_0_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_22.run(buf161, buf12, buf125, arg19_1, buf184, arg78_1, arg79_1, arg92_1, arg93_1, buf165, buf188, 3208, 128, grid=grid(3208), stream=stream0)
        del arg78_1
        del arg79_1
        del arg92_1
        del arg93_1
        buf166 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (8, 128), (51328, 1), 0), reinterpret_tensor(arg80_1, (128, 128), (1, 128), 0), out=buf166)
        del arg80_1
        buf167 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (3208, 128), (128, 1), 0), reinterpret_tensor(arg82_1, (128, 128), (1, 128), 0), out=buf167)
        del arg82_1
        buf168 = reinterpret_tensor(buf166, (8, 1, 128), (128, 128, 1), 0); del buf166  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_1_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf168, arg81_1, 1024, grid=grid(1024), stream=stream0)
        del arg81_1
        buf169 = empty((8, 4, 32, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf167, arg83_1, buf169, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del arg83_1
        del buf167
        buf170 = empty((32, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf169, (32, 32, 401), (12832, 401, 1), 0), out=buf170)
        buf174 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_25.run(buf170, buf174, 32, 401, grid=grid(32), stream=stream0)
        buf173 = reinterpret_tensor(buf169, (3208, 128), (128, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (3208, 128), (128, 1), 0), reinterpret_tensor(arg84_1, (128, 128), (1, 128), 0), out=buf173)
        del arg84_1
        buf175 = reinterpret_tensor(buf165, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf165  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf173, arg85_1, buf175, 410624, grid=grid(410624), stream=stream0)
        del arg85_1
        buf176 = reinterpret_tensor(buf168, (32, 1, 32), (32, 32, 1), 0); del buf168  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf175, (32, 401, 32), (12832, 32, 1), 0), out=buf176)
        buf177 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (8, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 128), (1, 128), 0), out=buf177)
        del arg86_1
        buf178 = reinterpret_tensor(buf177, (8, 1, 128), (128, 1024, 1), 0); del buf177  # reuse
        buf200 = reinterpret_tensor(buf176, (8, 1, 128), (128, 1024, 1), 0); del buf176  # reuse
        buf201 = reinterpret_tensor(buf200, (8, 1, 128), (128, 128, 1), 0); del buf200  # reuse
        # Source Nodes: [l__mod___blocks_0_revert_projs_1_0, l__mod___blocks_0_revert_projs_1_1, tmp_4], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
        triton_per_fused_add_gelu_native_layer_norm_27.run(buf178, buf201, buf161, buf12, buf125, arg19_1, arg87_1, arg88_1, arg89_1, 8, 128, grid=grid(8), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        buf189 = reinterpret_tensor(buf124, (3208, 384), (384, 1), 0); del buf124  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf188, (3208, 128), (128, 1), 0), reinterpret_tensor(arg94_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf189)
        del arg94_1
        del arg95_1
        # Source Nodes: [x_59], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf190 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf189, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf189, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf189, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, False)
        buf191 = buf190[0]
        del buf190
        buf195 = reinterpret_tensor(buf188, (3208, 128), (128, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf191, (3208, 128), (128, 1), 0), reinterpret_tensor(arg96_1, (128, 128), (1, 128), 0), out=buf195)
        del arg96_1
        buf196 = reinterpret_tensor(buf195, (8, 401, 128), (51328, 128, 1), 0); del buf195  # reuse
        buf256 = reinterpret_tensor(buf191, (8, 401, 128), (51328, 128, 1), 0); del buf191  # reuse
        # Source Nodes: [cat_24, getattr_l__mod___blocks_1_blocks_0___0___norm2, x_63], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_28.run(buf196, buf184, buf12, buf125, arg19_1, arg97_1, arg98_1, arg99_1, buf256, 3208, 128, grid=grid(3208), stream=stream0)
        del arg19_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf202 = reinterpret_tensor(buf183, (8, 256), (256, 1), 0); del buf183  # reuse
        # Source Nodes: [reverted_proj_cls_token_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg91_1, reinterpret_tensor(buf201, (8, 128), (128, 1), 0), reinterpret_tensor(arg90_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf202)
        del arg90_1
        del arg91_1
        buf206 = reinterpret_tensor(buf152, (8, 197, 256), (50432, 256, 1), 0); del buf152  # reuse
        # Source Nodes: [cat_22, getattr_l__mod___blocks_1_blocks_1___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf202, buf118, buf132, arg55_1, arg104_1, arg105_1, buf206, 1576, 256, grid=grid(1576), stream=stream0)
        del arg104_1
        del arg105_1
        buf207 = reinterpret_tensor(buf131, (1576, 768), (768, 1), 0); del buf131  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg107_1, reinterpret_tensor(buf206, (1576, 256), (256, 1), 0), reinterpret_tensor(arg106_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf207)
        del arg106_1
        del arg107_1
        # Source Nodes: [x_71], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf208 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf207, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf207, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf207, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf209 = buf208[0]
        del buf208
        buf213 = reinterpret_tensor(buf206, (1576, 256), (256, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (1576, 256), (256, 1), 0), reinterpret_tensor(arg108_1, (256, 256), (1, 256), 0), out=buf213)
        del arg108_1
        buf214 = reinterpret_tensor(buf213, (8, 197, 256), (50432, 256, 1), 0); del buf213  # reuse
        buf218 = reinterpret_tensor(buf209, (8, 197, 256), (50432, 256, 1), 0); del buf209  # reuse
        # Source Nodes: [cat_22, getattr_l__mod___blocks_1_blocks_1___0___norm2, x_75], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_29.run(buf214, buf202, buf118, buf132, arg55_1, arg109_1, arg110_1, arg111_1, buf218, 1576, 256, grid=grid(1576), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del arg55_1
        del buf118
        buf219 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (1576, 256), (256, 1), 0), reinterpret_tensor(arg112_1, (256, 768), (1, 256), 0), out=buf219)
        del arg112_1
        buf220 = reinterpret_tensor(buf219, (8, 197, 768), (151296, 768, 1), 0); del buf219  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf220, arg113_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg113_1
        buf221 = reinterpret_tensor(buf218, (1576, 256), (256, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (1576, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 256), (1, 768), 0), out=buf221)
        del arg114_1
        buf225 = reinterpret_tensor(buf132, (8, 197, 256), (50432, 256, 1), 0); del buf132  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___norm1, x_82], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf214, buf221, arg115_1, arg116_1, arg117_1, buf225, 1576, 256, grid=grid(1576), stream=stream0)
        del arg116_1
        del arg117_1
        buf226 = reinterpret_tensor(buf220, (1576, 768), (768, 1), 0); del buf220  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg119_1, reinterpret_tensor(buf225, (1576, 256), (256, 1), 0), reinterpret_tensor(arg118_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf226)
        del arg118_1
        del arg119_1
        # Source Nodes: [x_83], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf227 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf226, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf226, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf226, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf228 = buf227[0]
        del buf227
        buf232 = reinterpret_tensor(buf225, (1576, 256), (256, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1576, 256), (256, 1), 0), reinterpret_tensor(arg120_1, (256, 256), (1, 256), 0), out=buf232)
        del arg120_1
        buf233 = reinterpret_tensor(buf232, (8, 197, 256), (50432, 256, 1), 0); del buf232  # reuse
        buf237 = reinterpret_tensor(buf228, (8, 197, 256), (50432, 256, 1), 0); del buf228  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___1___norm2, x_82, x_87], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf233, buf214, buf221, arg115_1, arg121_1, arg122_1, arg123_1, buf237, 1576, 256, grid=grid(1576), stream=stream0)
        del arg115_1
        del arg121_1
        del arg122_1
        del arg123_1
        buf238 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (1576, 256), (256, 1), 0), reinterpret_tensor(arg124_1, (256, 768), (1, 256), 0), out=buf238)
        del arg124_1
        buf239 = reinterpret_tensor(buf238, (8, 197, 768), (151296, 768, 1), 0); del buf238  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf239, arg125_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg125_1
        buf240 = reinterpret_tensor(buf237, (1576, 256), (256, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (1576, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 256), (1, 768), 0), out=buf240)
        del arg126_1
        buf244 = reinterpret_tensor(buf221, (8, 197, 256), (50432, 256, 1), 0); del buf221  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___norm1, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf233, buf240, arg127_1, arg128_1, arg129_1, buf244, 1576, 256, grid=grid(1576), stream=stream0)
        del arg128_1
        del arg129_1
        buf245 = reinterpret_tensor(buf239, (1576, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg131_1, reinterpret_tensor(buf244, (1576, 256), (256, 1), 0), reinterpret_tensor(arg130_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf245)
        del arg130_1
        del arg131_1
        # Source Nodes: [x_95], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf246 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf245, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf245, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf245, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf247 = buf246[0]
        del buf246
        buf251 = reinterpret_tensor(buf244, (1576, 256), (256, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1576, 256), (256, 1), 0), reinterpret_tensor(arg132_1, (256, 256), (1, 256), 0), out=buf251)
        del arg132_1
        buf252 = reinterpret_tensor(buf251, (8, 197, 256), (50432, 256, 1), 0); del buf251  # reuse
        buf263 = reinterpret_tensor(buf247, (8, 197, 256), (50432, 256, 1), 0); del buf247  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___2___norm2, x_94, x_99], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf252, buf233, buf240, arg127_1, arg133_1, arg134_1, arg135_1, buf263, 1576, 256, grid=grid(1576), stream=stream0)
        del arg127_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf257 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (3208, 128), (128, 1), 0), reinterpret_tensor(arg100_1, (128, 384), (1, 128), 0), out=buf257)
        del arg100_1
        buf258 = reinterpret_tensor(buf257, (8, 401, 384), (153984, 384, 1), 0); del buf257  # reuse
        # Source Nodes: [x_65], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf258, arg101_1, 1231872, grid=grid(1231872), stream=stream0)
        del arg101_1
        buf259 = reinterpret_tensor(buf256, (3208, 128), (128, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (3208, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 128), (1, 384), 0), out=buf259)
        del arg102_1
        buf270 = reinterpret_tensor(buf201, (8, 1, 128), (128, 1024, 1), 0); del buf201  # reuse
        buf271 = reinterpret_tensor(buf270, (8, 1, 128), (128, 128, 1), 0); del buf270  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_0_0, l__mod___blocks_1_projs_0_1], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_15.run(buf271, buf196, buf259, arg103_1, arg140_1, arg141_1, 8, 128, grid=grid(8), stream=stream0)
        del arg140_1
        del arg141_1
        buf264 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (1576, 256), (256, 1), 0), reinterpret_tensor(arg136_1, (256, 768), (1, 256), 0), out=buf264)
        del arg136_1
        buf265 = reinterpret_tensor(buf264, (8, 197, 768), (151296, 768, 1), 0); del buf264  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf265, arg137_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg137_1
        buf266 = reinterpret_tensor(buf263, (1576, 256), (256, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (1576, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 256), (1, 768), 0), out=buf266)
        del arg138_1
        buf272 = buf202; del buf202  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_0_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf271, (8, 128), (128, 1), 0), reinterpret_tensor(arg142_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf272)
        del arg142_1
        del arg143_1
        buf276 = reinterpret_tensor(buf240, (8, 197, 256), (50432, 256, 1), 0); del buf240  # reuse
        # Source Nodes: [cat_21, l__mod___blocks_1_fusion_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf272, buf252, buf266, arg139_1, arg148_1, arg149_1, buf276, 1576, 256, grid=grid(1576), stream=stream0)
        del arg148_1
        del arg149_1
        buf277 = reinterpret_tensor(buf160, (8, 256), (256, 1), 0); del buf160  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (8, 256), (50432, 1), 0), reinterpret_tensor(arg150_1, (256, 256), (1, 256), 0), out=buf277)
        del arg150_1
        buf279 = reinterpret_tensor(buf277, (8, 1, 256), (256, 256, 1), 0); del buf277  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_0_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf279, arg151_1, 2048, grid=grid(2048), stream=stream0)
        del arg151_1
        buf278 = reinterpret_tensor(buf233, (1576, 256), (256, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1576, 256), (256, 1), 0), reinterpret_tensor(arg152_1, (256, 256), (1, 256), 0), out=buf278)
        del arg152_1
        buf280 = reinterpret_tensor(buf214, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf214  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf278, arg153_1, buf280, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del arg153_1
        del buf278
        buf281 = reinterpret_tensor(buf151, (32, 1, 197), (197, 197, 1), 0); del buf151  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf280, (32, 64, 197), (12608, 197, 1), 0), out=buf281)
        buf285 = reinterpret_tensor(buf147, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf147  # reuse
        # Source Nodes: [attn_6, attn_7], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_19.run(buf281, buf285, 32, 197, grid=grid(32), stream=stream0)
        buf284 = reinterpret_tensor(buf280, (1576, 256), (256, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1576, 256), (256, 1), 0), reinterpret_tensor(arg154_1, (256, 256), (1, 256), 0), out=buf284)
        del arg154_1
        buf286 = reinterpret_tensor(buf276, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf276  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf284, arg155_1, buf286, 403456, grid=grid(403456), stream=stream0)
        del arg155_1
        del buf284
        buf287 = reinterpret_tensor(buf279, (32, 1, 64), (64, 64, 1), 0); del buf279  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf285, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf286, (32, 197, 64), (12608, 64, 1), 0), out=buf287)
        buf288 = reinterpret_tensor(buf155, (8, 256), (256, 1), 0); del buf155  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (8, 256), (256, 1), 0), reinterpret_tensor(arg156_1, (256, 256), (1, 256), 0), out=buf288)
        del arg156_1
        buf289 = reinterpret_tensor(buf288, (8, 1, 256), (256, 2048, 1), 0); del buf288  # reuse
        buf293 = reinterpret_tensor(buf287, (8, 1, 256), (256, 2048, 1), 0); del buf287  # reuse
        buf294 = reinterpret_tensor(buf293, (8, 1, 256), (256, 256, 1), 0); del buf293  # reuse
        buf316 = reinterpret_tensor(buf138, (8, 1, 256), (256, 2048, 1), 0); del buf138  # reuse
        buf317 = reinterpret_tensor(buf316, (8, 1, 256), (256, 256, 1), 0); del buf316  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_1_0, l__mod___blocks_1_projs_1_1, l__mod___blocks_1_revert_projs_0_0, l__mod___blocks_1_revert_projs_0_1, tmp_7], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
        triton_per_fused_add_gelu_native_layer_norm_21.run(buf289, buf294, buf317, buf252, buf266, arg139_1, buf272, arg157_1, arg144_1, arg145_1, arg158_1, arg159_1, 8, 256, grid=grid(8), stream=stream0)
        del arg144_1
        del arg145_1
        del arg157_1
        del arg158_1
        del arg159_1
        buf295 = reinterpret_tensor(buf271, (8, 128), (128, 1), 0); del buf271  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_1_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf294, (8, 256), (256, 1), 0), reinterpret_tensor(arg146_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf295)
        del arg146_1
        del arg147_1
        buf318 = buf184; del buf184  # reuse
        # Source Nodes: [reverted_proj_cls_token_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg161_1, reinterpret_tensor(buf317, (8, 256), (256, 1), 0), reinterpret_tensor(arg160_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf318)
        del arg160_1
        del arg161_1
        buf299 = reinterpret_tensor(buf125, (8, 401, 128), (51328, 128, 1), 0); del buf125  # reuse
        buf322 = buf12; del buf12  # reuse
        # Source Nodes: [cat_19, cat_20, getattr_l__mod___blocks_2_blocks_0___0___norm1, l__mod___blocks_1_fusion_1_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_22.run(buf295, buf196, buf259, arg103_1, buf318, arg162_1, arg163_1, arg176_1, arg177_1, buf299, buf322, 3208, 128, grid=grid(3208), stream=stream0)
        del arg162_1
        del arg163_1
        del arg176_1
        del arg177_1
        buf300 = reinterpret_tensor(buf178, (8, 128), (128, 1), 0); del buf178  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (8, 128), (51328, 1), 0), reinterpret_tensor(arg164_1, (128, 128), (1, 128), 0), out=buf300)
        del arg164_1
        buf301 = reinterpret_tensor(buf175, (3208, 128), (128, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (3208, 128), (128, 1), 0), reinterpret_tensor(arg166_1, (128, 128), (1, 128), 0), out=buf301)
        del arg166_1
        buf302 = reinterpret_tensor(buf300, (8, 1, 128), (128, 128, 1), 0); del buf300  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_1_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf302, arg165_1, 1024, grid=grid(1024), stream=stream0)
        del arg165_1
        buf303 = reinterpret_tensor(buf173, (8, 4, 32, 401), (51328, 12832, 401, 1), 0); del buf173  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf301, arg167_1, buf303, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del arg167_1
        del buf301
        buf304 = reinterpret_tensor(buf174, (32, 1, 401), (401, 401, 1), 0); del buf174  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf303, (32, 32, 401), (12832, 401, 1), 0), out=buf304)
        buf308 = reinterpret_tensor(buf170, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf170  # reuse
        # Source Nodes: [attn_10, attn_9], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_25.run(buf304, buf308, 32, 401, grid=grid(32), stream=stream0)
        buf307 = reinterpret_tensor(buf303, (3208, 128), (128, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (3208, 128), (128, 1), 0), reinterpret_tensor(arg168_1, (128, 128), (1, 128), 0), out=buf307)
        del arg168_1
        buf309 = reinterpret_tensor(buf299, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf299  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf307, arg169_1, buf309, 410624, grid=grid(410624), stream=stream0)
        del arg169_1
        del buf307
        buf310 = reinterpret_tensor(buf302, (32, 1, 32), (32, 32, 1), 0); del buf302  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf308, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf309, (32, 401, 32), (12832, 32, 1), 0), out=buf310)
        buf311 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (8, 128), (128, 1), 0), reinterpret_tensor(arg170_1, (128, 128), (1, 128), 0), out=buf311)
        del arg170_1
        buf312 = reinterpret_tensor(buf311, (8, 1, 128), (128, 1024, 1), 0); del buf311  # reuse
        buf334 = reinterpret_tensor(buf310, (8, 1, 128), (128, 1024, 1), 0); del buf310  # reuse
        buf335 = reinterpret_tensor(buf334, (8, 1, 128), (128, 128, 1), 0); del buf334  # reuse
        # Source Nodes: [l__mod___blocks_1_revert_projs_1_0, l__mod___blocks_1_revert_projs_1_1, tmp_10], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
        triton_per_fused_add_gelu_native_layer_norm_27.run(buf312, buf335, buf295, buf196, buf259, arg103_1, arg171_1, arg172_1, arg173_1, 8, 128, grid=grid(8), stream=stream0)
        del arg171_1
        del arg172_1
        del arg173_1
        buf323 = reinterpret_tensor(buf258, (3208, 384), (384, 1), 0); del buf258  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg179_1, reinterpret_tensor(buf322, (3208, 128), (128, 1), 0), reinterpret_tensor(arg178_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf323)
        del arg178_1
        del arg179_1
        # Source Nodes: [x_115], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf324 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf323, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf323, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf323, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, False)
        buf325 = buf324[0]
        del buf324
        buf329 = reinterpret_tensor(buf322, (3208, 128), (128, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf325, (3208, 128), (128, 1), 0), reinterpret_tensor(arg180_1, (128, 128), (1, 128), 0), out=buf329)
        del arg180_1
        buf330 = reinterpret_tensor(buf329, (8, 401, 128), (51328, 128, 1), 0); del buf329  # reuse
        buf390 = reinterpret_tensor(buf325, (8, 401, 128), (51328, 128, 1), 0); del buf325  # reuse
        # Source Nodes: [cat_20, getattr_l__mod___blocks_2_blocks_0___0___norm2, x_119], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_28.run(buf330, buf318, buf196, buf259, arg103_1, arg181_1, arg182_1, arg183_1, buf390, 3208, 128, grid=grid(3208), stream=stream0)
        del arg103_1
        del arg181_1
        del arg182_1
        del arg183_1
        buf336 = reinterpret_tensor(buf317, (8, 256), (256, 1), 0); del buf317  # reuse
        # Source Nodes: [reverted_proj_cls_token_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg175_1, reinterpret_tensor(buf335, (8, 128), (128, 1), 0), reinterpret_tensor(arg174_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf336)
        del arg174_1
        del arg175_1
        buf340 = reinterpret_tensor(buf286, (8, 197, 256), (50432, 256, 1), 0); del buf286  # reuse
        # Source Nodes: [cat_18, getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf336, buf252, buf266, arg139_1, arg188_1, arg189_1, buf340, 1576, 256, grid=grid(1576), stream=stream0)
        del arg188_1
        del arg189_1
        buf341 = reinterpret_tensor(buf265, (1576, 768), (768, 1), 0); del buf265  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg191_1, reinterpret_tensor(buf340, (1576, 256), (256, 1), 0), reinterpret_tensor(arg190_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf341)
        del arg190_1
        del arg191_1
        # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf342 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf341, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf341, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf341, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf343 = buf342[0]
        del buf342
        buf347 = reinterpret_tensor(buf340, (1576, 256), (256, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (1576, 256), (256, 1), 0), reinterpret_tensor(arg192_1, (256, 256), (1, 256), 0), out=buf347)
        del arg192_1
        buf348 = reinterpret_tensor(buf347, (8, 197, 256), (50432, 256, 1), 0); del buf347  # reuse
        buf352 = reinterpret_tensor(buf343, (8, 197, 256), (50432, 256, 1), 0); del buf343  # reuse
        # Source Nodes: [cat_18, getattr_l__mod___blocks_2_blocks_1___0___norm2, x_131], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_29.run(buf348, buf336, buf252, buf266, arg139_1, arg193_1, arg194_1, arg195_1, buf352, 1576, 256, grid=grid(1576), stream=stream0)
        del arg139_1
        del arg193_1
        del arg194_1
        del arg195_1
        del buf252
        buf353 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf352, (1576, 256), (256, 1), 0), reinterpret_tensor(arg196_1, (256, 768), (1, 256), 0), out=buf353)
        del arg196_1
        buf354 = reinterpret_tensor(buf353, (8, 197, 768), (151296, 768, 1), 0); del buf353  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf354, arg197_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg197_1
        buf355 = reinterpret_tensor(buf352, (1576, 256), (256, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1576, 768), (768, 1), 0), reinterpret_tensor(arg198_1, (768, 256), (1, 768), 0), out=buf355)
        del arg198_1
        buf359 = reinterpret_tensor(buf266, (8, 197, 256), (50432, 256, 1), 0); del buf266  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___norm1, x_138], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf348, buf355, arg199_1, arg200_1, arg201_1, buf359, 1576, 256, grid=grid(1576), stream=stream0)
        del arg200_1
        del arg201_1
        buf360 = reinterpret_tensor(buf354, (1576, 768), (768, 1), 0); del buf354  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg203_1, reinterpret_tensor(buf359, (1576, 256), (256, 1), 0), reinterpret_tensor(arg202_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf360)
        del arg202_1
        del arg203_1
        # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf361 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf360, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf360, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf360, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf362 = buf361[0]
        del buf361
        buf366 = reinterpret_tensor(buf359, (1576, 256), (256, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (1576, 256), (256, 1), 0), reinterpret_tensor(arg204_1, (256, 256), (1, 256), 0), out=buf366)
        del arg204_1
        buf367 = reinterpret_tensor(buf366, (8, 197, 256), (50432, 256, 1), 0); del buf366  # reuse
        buf371 = reinterpret_tensor(buf362, (8, 197, 256), (50432, 256, 1), 0); del buf362  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___1___norm2, x_138, x_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf367, buf348, buf355, arg199_1, arg205_1, arg206_1, arg207_1, buf371, 1576, 256, grid=grid(1576), stream=stream0)
        del arg199_1
        del arg205_1
        del arg206_1
        del arg207_1
        buf372 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf371, (1576, 256), (256, 1), 0), reinterpret_tensor(arg208_1, (256, 768), (1, 256), 0), out=buf372)
        del arg208_1
        buf373 = reinterpret_tensor(buf372, (8, 197, 768), (151296, 768, 1), 0); del buf372  # reuse
        # Source Nodes: [x_145], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf373, arg209_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg209_1
        buf374 = reinterpret_tensor(buf371, (1576, 256), (256, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (1576, 768), (768, 1), 0), reinterpret_tensor(arg210_1, (768, 256), (1, 768), 0), out=buf374)
        del arg210_1
        buf378 = reinterpret_tensor(buf355, (8, 197, 256), (50432, 256, 1), 0); del buf355  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___norm1, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_12.run(buf367, buf374, arg211_1, arg212_1, arg213_1, buf378, 1576, 256, grid=grid(1576), stream=stream0)
        del arg212_1
        del arg213_1
        buf379 = reinterpret_tensor(buf373, (1576, 768), (768, 1), 0); del buf373  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg215_1, reinterpret_tensor(buf378, (1576, 256), (256, 1), 0), reinterpret_tensor(arg214_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf379)
        del arg214_1
        del arg215_1
        # Source Nodes: [x_151], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf380 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf379, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf379, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf379, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf381 = buf380[0]
        del buf380
        buf385 = reinterpret_tensor(buf378, (1576, 256), (256, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (1576, 256), (256, 1), 0), reinterpret_tensor(arg216_1, (256, 256), (1, 256), 0), out=buf385)
        del arg216_1
        buf386 = reinterpret_tensor(buf385, (8, 197, 256), (50432, 256, 1), 0); del buf385  # reuse
        buf397 = reinterpret_tensor(buf381, (8, 197, 256), (50432, 256, 1), 0); del buf381  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___2___norm2, x_150, x_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf386, buf367, buf374, arg211_1, arg217_1, arg218_1, arg219_1, buf397, 1576, 256, grid=grid(1576), stream=stream0)
        del arg211_1
        del arg217_1
        del arg218_1
        del arg219_1
        buf391 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf390, (3208, 128), (128, 1), 0), reinterpret_tensor(arg184_1, (128, 384), (1, 128), 0), out=buf391)
        del arg184_1
        buf392 = reinterpret_tensor(buf391, (8, 401, 384), (153984, 384, 1), 0); del buf391  # reuse
        # Source Nodes: [x_121], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_14.run(buf392, arg185_1, 1231872, grid=grid(1231872), stream=stream0)
        del arg185_1
        buf393 = reinterpret_tensor(buf390, (3208, 128), (128, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (3208, 384), (384, 1), 0), reinterpret_tensor(arg186_1, (384, 128), (1, 384), 0), out=buf393)
        del arg186_1
        del buf392
        buf404 = reinterpret_tensor(buf335, (8, 1, 128), (128, 1024, 1), 0); del buf335  # reuse
        buf405 = reinterpret_tensor(buf404, (8, 1, 128), (128, 128, 1), 0); del buf404  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1], Original ATen: [aten.gelu, aten.native_layer_norm]
        triton_per_fused_gelu_native_layer_norm_15.run(buf405, buf330, buf393, arg187_1, arg224_1, arg225_1, 8, 128, grid=grid(8), stream=stream0)
        del arg224_1
        del arg225_1
        buf398 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (1576, 256), (256, 1), 0), reinterpret_tensor(arg220_1, (256, 768), (1, 256), 0), out=buf398)
        del arg220_1
        buf399 = reinterpret_tensor(buf398, (8, 197, 768), (151296, 768, 1), 0); del buf398  # reuse
        # Source Nodes: [x_157], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_11.run(buf399, arg221_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg221_1
        buf400 = reinterpret_tensor(buf397, (1576, 256), (256, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (1576, 768), (768, 1), 0), reinterpret_tensor(arg222_1, (768, 256), (1, 768), 0), out=buf400)
        del arg222_1
        del buf399
        buf406 = buf336; del buf336  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_0_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg227_1, reinterpret_tensor(buf405, (8, 128), (128, 1), 0), reinterpret_tensor(arg226_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf406)
        del arg226_1
        del arg227_1
        buf410 = reinterpret_tensor(buf374, (8, 197, 256), (50432, 256, 1), 0); del buf374  # reuse
        # Source Nodes: [cat_17, l__mod___blocks_2_fusion_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_16.run(buf406, buf386, buf400, arg223_1, arg232_1, arg233_1, buf410, 1576, 256, grid=grid(1576), stream=stream0)
        del arg232_1
        del arg233_1
        buf411 = reinterpret_tensor(buf294, (8, 256), (256, 1), 0); del buf294  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (8, 256), (50432, 1), 0), reinterpret_tensor(arg234_1, (256, 256), (1, 256), 0), out=buf411)
        del arg234_1
        buf413 = reinterpret_tensor(buf411, (8, 1, 256), (256, 256, 1), 0); del buf411  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_0_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf413, arg235_1, 2048, grid=grid(2048), stream=stream0)
        del arg235_1
        buf412 = reinterpret_tensor(buf367, (1576, 256), (256, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (1576, 256), (256, 1), 0), reinterpret_tensor(arg236_1, (256, 256), (1, 256), 0), out=buf412)
        del arg236_1
        buf414 = reinterpret_tensor(buf348, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf348  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf412, arg237_1, buf414, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del arg237_1
        del buf412
        buf415 = reinterpret_tensor(buf285, (32, 1, 197), (197, 197, 1), 0); del buf285  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf413, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf414, (32, 64, 197), (12608, 197, 1), 0), out=buf415)
        buf419 = reinterpret_tensor(buf281, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf281  # reuse
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_19.run(buf415, buf419, 32, 197, grid=grid(32), stream=stream0)
        del buf415
        buf418 = reinterpret_tensor(buf414, (1576, 256), (256, 1), 0); del buf414  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (1576, 256), (256, 1), 0), reinterpret_tensor(arg238_1, (256, 256), (1, 256), 0), out=buf418)
        del arg238_1
        buf420 = reinterpret_tensor(buf410, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf410  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf418, arg239_1, buf420, 403456, grid=grid(403456), stream=stream0)
        del arg239_1
        del buf418
        buf421 = reinterpret_tensor(buf413, (32, 1, 64), (64, 64, 1), 0); del buf413  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf420, (32, 197, 64), (12608, 64, 1), 0), out=buf421)
        del buf419
        del buf420
        buf422 = reinterpret_tensor(buf289, (8, 256), (256, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (8, 256), (256, 1), 0), reinterpret_tensor(arg240_1, (256, 256), (1, 256), 0), out=buf422)
        del arg240_1
        buf423 = reinterpret_tensor(buf422, (8, 1, 256), (256, 2048, 1), 0); del buf422  # reuse
        buf427 = reinterpret_tensor(buf421, (8, 1, 256), (256, 2048, 1), 0); del buf421  # reuse
        buf428 = reinterpret_tensor(buf427, (8, 1, 256), (256, 256, 1), 0); del buf427  # reuse
        buf450 = reinterpret_tensor(buf272, (8, 1, 256), (256, 2048, 1), 0); del buf272  # reuse
        buf451 = reinterpret_tensor(buf450, (8, 1, 256), (256, 256, 1), 0); del buf450  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1, l__mod___blocks_2_revert_projs_0_0, l__mod___blocks_2_revert_projs_0_1, tmp_13], Original ATen: [aten.add, aten.gelu, aten.native_layer_norm]
        triton_per_fused_add_gelu_native_layer_norm_21.run(buf423, buf428, buf451, buf386, buf400, arg223_1, buf406, arg241_1, arg228_1, arg229_1, arg242_1, arg243_1, 8, 256, grid=grid(8), stream=stream0)
        del arg228_1
        del arg229_1
        del arg241_1
        del arg242_1
        del arg243_1
        del buf406
        del buf423
        buf429 = reinterpret_tensor(buf405, (8, 128), (128, 1), 0); del buf405  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_1_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg231_1, reinterpret_tensor(buf428, (8, 256), (256, 1), 0), reinterpret_tensor(arg230_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf429)
        del arg230_1
        del arg231_1
        buf452 = buf318; del buf318  # reuse
        # Source Nodes: [reverted_proj_cls_token_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg245_1, reinterpret_tensor(buf451, (8, 256), (256, 1), 0), reinterpret_tensor(arg244_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf452)
        del arg244_1
        del arg245_1
        buf453 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf454 = empty_strided((8, 401, 1), (401, 1, 3208), device='cuda', dtype=torch.float32)
        buf433 = reinterpret_tensor(buf259, (8, 401, 128), (51328, 128, 1), 0); del buf259  # reuse
        # Source Nodes: [cat_15, cat_16, l__mod___blocks_2_fusion_1_norm1, x_171], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_30.run(buf429, buf330, buf393, arg187_1, buf452, arg246_1, arg247_1, buf453, buf454, buf433, 3208, 128, grid=grid(3208), stream=stream0)
        del arg246_1
        del arg247_1
        buf434 = reinterpret_tensor(buf312, (8, 128), (128, 1), 0); del buf312  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wq], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (8, 128), (51328, 1), 0), reinterpret_tensor(arg248_1, (128, 128), (1, 128), 0), out=buf434)
        del arg248_1
        buf435 = reinterpret_tensor(buf196, (3208, 128), (128, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (3208, 128), (128, 1), 0), reinterpret_tensor(arg250_1, (128, 128), (1, 128), 0), out=buf435)
        del arg250_1
        buf436 = reinterpret_tensor(buf434, (8, 1, 128), (128, 128, 1), 0); del buf434  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_1_attn_wq], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf436, arg249_1, 1024, grid=grid(1024), stream=stream0)
        del arg249_1
        buf437 = reinterpret_tensor(buf309, (8, 4, 32, 401), (51328, 12832, 401, 1), 0); del buf309  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf435, arg251_1, buf437, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del arg251_1
        del buf435
        buf438 = reinterpret_tensor(buf308, (32, 1, 401), (401, 401, 1), 0); del buf308  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf437, (32, 32, 401), (12832, 401, 1), 0), out=buf438)
        buf442 = reinterpret_tensor(buf304, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf304  # reuse
        # Source Nodes: [attn_15, attn_16], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_25.run(buf438, buf442, 32, 401, grid=grid(32), stream=stream0)
        del buf438
        buf441 = reinterpret_tensor(buf437, (3208, 128), (128, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (3208, 128), (128, 1), 0), reinterpret_tensor(arg252_1, (128, 128), (1, 128), 0), out=buf441)
        del arg252_1
        buf443 = reinterpret_tensor(buf433, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf433  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf441, arg253_1, buf443, 410624, grid=grid(410624), stream=stream0)
        del arg253_1
        del buf441
        buf444 = reinterpret_tensor(buf436, (32, 1, 32), (32, 32, 1), 0); del buf436  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf442, (32, 1, 401), (401, 0, 1), 0), reinterpret_tensor(buf443, (32, 401, 32), (12832, 32, 1), 0), out=buf444)
        del buf442
        del buf443
        buf445 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf444, (8, 128), (128, 1), 0), reinterpret_tensor(arg254_1, (128, 128), (1, 128), 0), out=buf445)
        del arg254_1
        buf446 = reinterpret_tensor(buf445, (8, 1, 128), (128, 1024, 1), 0); del buf445  # reuse
        buf462 = reinterpret_tensor(buf444, (8, 128), (128, 1), 0); del buf444  # reuse
        buf456 = empty_strided((8, 1, 128), (128, 1024, 1), device='cuda', dtype=torch.float32)
        buf457 = reinterpret_tensor(buf456, (8, 1, 128), (128, 128, 1), 0); del buf456  # reuse
        # Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1, l__mod___head_drop, tmp_16], Original ATen: [aten.add, aten.clone, aten.gelu, aten.native_layer_norm]
        triton_per_fused_add_clone_gelu_native_layer_norm_31.run(buf446, buf457, buf429, buf330, buf393, arg187_1, arg255_1, buf452, buf453, buf454, arg260_1, arg261_1, arg256_1, arg257_1, buf462, 8, 128, grid=grid(8), stream=stream0)
        del arg187_1
        del arg255_1
        del arg256_1
        del arg257_1
        del arg260_1
        del arg261_1
        del buf330
        del buf393
        del buf429
        del buf446
        del buf452
        del buf453
        del buf454
        buf458 = reinterpret_tensor(buf451, (8, 256), (256, 1), 0); del buf451  # reuse
        # Source Nodes: [reverted_proj_cls_token_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg259_1, reinterpret_tensor(buf457, (8, 128), (128, 1), 0), reinterpret_tensor(arg258_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf458)
        del arg258_1
        del arg259_1
        del buf457
        buf459 = buf70; del buf70  # reuse
        buf460 = buf69; del buf69  # reuse
        # Source Nodes: [cat_14, x_172], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_32.run(buf458, buf386, buf400, arg223_1, buf459, buf460, 1576, 256, grid=grid(1576), stream=stream0)
        buf463 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_0, l__mod___head_drop], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg265_1, buf462, reinterpret_tensor(arg264_1, (128, 1000), (1, 128), 0), alpha=1, beta=1, out=buf463)
        del arg264_1
        del arg265_1
        del buf462
        buf464 = reinterpret_tensor(buf428, (8, 256), (256, 1), 0); del buf428  # reuse
        # Source Nodes: [l__mod___head_drop_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf458, buf386, buf400, arg223_1, buf459, buf460, arg262_1, arg263_1, buf464, 2048, grid=grid(2048), stream=stream0)
        del arg223_1
        del arg262_1
        del arg263_1
        del buf386
        del buf400
        del buf458
        del buf459
        del buf460
        buf465 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_1, l__mod___head_drop_1], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg267_1, buf464, reinterpret_tensor(arg266_1, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf465)
        del arg266_1
        del arg267_1
        del buf464
        buf466 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten.mean]
        triton_poi_fused_mean_34.run(buf463, buf465, buf466, 8000, grid=grid(8000), stream=stream0)
        return (buf466, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 3, 12, 12), (432, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((8, 3, 240, 240), (172800, 57600, 240, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
